import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe_baseline_attention import PillarVFEBaselineAttention
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention import PointPillarScatterBaselineAttention


class PointPillarRadarStudentKd(nn.Module):
    """
    WP3 radar student with motion-aware KD support.

    Key outputs:
    - cls_preds / reg_preds / optional dir_preds
    - feature: BEV feature before detection heads
    - kd_motion_mask: radar-derived motion prior in BEV space
    """

    def __init__(self, args):
        super().__init__()

        radar_num_point_features = args.get('radar_num_point_features', 4)
        self.return_kd_motion_mask = args.get('return_kd_motion_mask', True)
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 7))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))

        self.radar_pillar_vfe = PillarVFEBaselineAttention(
            args['pillar_vfe'],
            num_point_features=radar_num_point_features,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.radar_scatter = PointPillarScatterBaselineAttention(args['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
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

    def _build_motion_mask(self, radar_batch_dict, target_hw):
        velocity_mask = radar_batch_dict.get('velocity_confidence_mask', None)
        if velocity_mask is None:
            return None

        mask = velocity_mask.float()
        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.motion_mask_kernel_size, stride=1, padding=pad)

        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)

        mask = mask.clamp_(0.0, 1.0)
        if self.motion_mask_scale != 1.0:
            mask = (mask * self.motion_mask_scale).clamp_(0.0, 1.0)
        return mask

    def forward(self, data_dict):
        radar_batch_dict = {
            'voxel_features': data_dict['processed_radar']['voxel_features'],
            'voxel_coords': data_dict['processed_radar']['voxel_coords'],
            'voxel_num_points': data_dict['processed_radar']['voxel_num_points'],
        }

        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.radar_scatter(radar_batch_dict)

        batch_dict = {'spatial_features': radar_batch_dict['spatial_features']}
        batch_dict = self.backbone(batch_dict)

        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)

        cls_preds = self.cls_head(bev_features)
        reg_preds = self.reg_head(bev_features)

        output_dict = {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'feature': bev_features,
        }

        if self.return_kd_motion_mask:
            kd_motion_mask = self._build_motion_mask(radar_batch_dict, bev_features.shape[-2:])
            if kd_motion_mask is not None:
                output_dict['kd_motion_mask'] = kd_motion_mask

        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(bev_features)

        return output_dict
