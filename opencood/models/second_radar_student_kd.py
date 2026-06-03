import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class SecondRadarStudentKd(nn.Module):
    """
    Radar-only SECOND student for TruckScenes LiDAR->radar KD.

    The model consumes the TruckScenes dataset's existing ``processed_radar``
    dict, emits normal OpenCOOD prediction keys, and exposes ``feature`` plus a
    radar-derived ``kd_motion_mask`` for motion-aware feature distillation.
    """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args['batch_size']
        self.return_kd_motion_mask = args.get('return_kd_motion_mask', True)
        self.radar_velocity_channel = int(args.get('radar_velocity_channel', 3))
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 7))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))
        self.motion_threshold = float(args.get('motion_threshold', 0.15))
        self.grid_size = args['grid_size']

        radar_num_point_features = args.get('radar_num_point_features', 4)
        self.mean_vfe = MeanVFE(args.get('mean_vfe', {}), radar_num_point_features)
        self.backbone_3d = VoxelBackBone8x(
            args.get('backbone_3d', {}),
            radar_num_point_features,
            self.grid_size
        )
        self.height_compression = HeightCompression(args['height_compression'])
        bev_backbone_type = args.get('bev_backbone_type', 'base').lower()
        if bev_backbone_type in ('resnet', 'resnet_bev'):
            bev_cfg = dict(args['base_bev_backbone'])
            bev_cfg.setdefault('inplanes', args['height_compression']['feature_num'])
            self.backbone_2d = ResNetBEVBackbone(
                bev_cfg,
                args['height_compression']['feature_num']
            )
        else:
            self.backbone_2d = BaseBEVBackbone(
                args['base_bev_backbone'],
                args['height_compression']['feature_num']
            )
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        anchor_number = args['anchor_number']
        self.cls_head = nn.Conv2d(self.out_channel, anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * anchor_number, kernel_size=1)

        self.use_dir = 'dir_args' in args
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * anchor_number,
                kernel_size=1,
            )

    def _infer_batch_size(self, processed_radar, data_dict=None):
        if data_dict is not None:
            if 'label_dict' in data_dict and 'pos_equal_one' in data_dict['label_dict']:
                return int(data_dict['label_dict']['pos_equal_one'].shape[0])
            if 'object_bbx_center' in data_dict:
                return int(data_dict['object_bbx_center'].shape[0])
            if 'record_len' in data_dict:
                return int(data_dict['record_len'].sum().item())
        voxel_coords = processed_radar['voxel_coords']
        if voxel_coords.numel() == 0:
            return self.batch_size
        return int(voxel_coords[:, 0].max().item()) + 1

    def _build_motion_mask(self, processed_radar, target_hw, batch_size):
        voxel_features = processed_radar['voxel_features']
        voxel_coords = processed_radar['voxel_coords']
        if voxel_features.shape[-1] <= self.radar_velocity_channel:
            return None

        point_velocity = voxel_features[:, :, self.radar_velocity_channel].abs()
        voxel_motion = point_velocity.max(dim=1)[0]
        voxel_motion = torch.sigmoid((voxel_motion - self.motion_threshold) * 12.0)

        # voxel_coords are [batch_idx, z_idx, y_idx, x_idx]. Build a BEV mask at
        # voxel resolution and downsample/interpolate to the SECOND BEV feature.
        h = int(self.grid_size[1])
        w = int(self.grid_size[0])
        mask = voxel_features.new_zeros((batch_size, 1, h, w))
        b = voxel_coords[:, 0].long().clamp(0, batch_size - 1)
        y = voxel_coords[:, 2].long().clamp(0, h - 1)
        x = voxel_coords[:, 3].long().clamp(0, w - 1)
        mask[b, 0, y, x] = torch.maximum(mask[b, 0, y, x], voxel_motion)

        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.motion_mask_kernel_size, stride=1, padding=pad)
        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)
        mask = mask.clamp_(0.0, 1.0)
        if self.motion_mask_scale != 1.0:
            mask = (mask * self.motion_mask_scale).clamp_(0.0, 1.0)
        return mask

    def _encode_to_bev(self, processed_radar, batch_size):
        batch_dict = {
            'voxel_features': processed_radar['voxel_features'],
            'voxel_coords': processed_radar['voxel_coords'],
            'voxel_num_points': processed_radar['voxel_num_points'],
            'batch_size': batch_size,
        }
        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)
        return bev_features

    def forward(self, data_dict):
        processed_radar = data_dict['processed_radar']
        batch_size = self._infer_batch_size(processed_radar, data_dict)
        bev_features = self._encode_to_bev(processed_radar, batch_size)
        output_dict = {
            'cls_preds': self.cls_head(bev_features),
            'reg_preds': self.reg_head(bev_features),
            'feature': bev_features,
            'batch_size': batch_size,
        }
        if self.return_kd_motion_mask:
            kd_motion_mask = self._build_motion_mask(processed_radar, bev_features.shape[-2:], batch_size)
            if kd_motion_mask is not None:
                output_dict['kd_motion_mask'] = kd_motion_mask
        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(bev_features)
        return output_dict
