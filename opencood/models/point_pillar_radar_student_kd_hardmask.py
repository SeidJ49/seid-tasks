import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe_baseline_attention import PillarVFEBaselineAttention
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention import PointPillarScatterBaselineAttention


class PointPillarRadarStudentKdHardmask(nn.Module):
    """
    WP3 radar student with explicit hard motion mask from radar Doppler.

    Differences to the soft-mask variant:
    - motion mask is built directly from radar voxel input channel 3 (v_r_norm)
    - hard thresholding instead of relying on velocity_confidence_mask
    - nearest-neighbor resizing keeps the mask binary across scales
    """

    def __init__(self, args):
        super().__init__()

        radar_num_point_features = args.get('radar_num_point_features', 4)
        self.return_kd_motion_mask = args.get('return_kd_motion_mask', True)
        self.motion_threshold = float(args.get('motion_threshold', 0.5))
        self.motion_reduce = args.get('motion_reduce', 'max')
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 1))

        self.grid_size = tuple(int(x) for x in args['point_pillar_scatter']['grid_size'])
        self.nx, self.ny, self.nz = self.grid_size

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

    def _build_hard_motion_mask(self, voxel_features, voxel_num_points, voxel_coords, target_hw):
        if voxel_features.shape[-1] < 4:
            return None

        batch_size = int(voxel_coords[:, 0].max().item()) + 1
        num_points = voxel_features.shape[1]
        valid_mask = torch.arange(num_points, device=voxel_num_points.device).unsqueeze(0) < voxel_num_points.unsqueeze(1)

        v_r = voxel_features[:, :, 3].abs()
        v_r = v_r * valid_mask.float()

        if self.motion_reduce == 'mean':
            denom = voxel_num_points.clamp_min(1).float()
            voxel_motion_score = v_r.sum(dim=1) / denom
        else:
            voxel_motion_score = v_r.max(dim=1)[0]

        voxel_is_dynamic = (voxel_motion_score > self.motion_threshold).float()

        bev_mask = voxel_features.new_zeros((batch_size, 1, self.ny, self.nx))
        batch_idx = voxel_coords[:, 0].long()
        y_idx = voxel_coords[:, 2].long()
        x_idx = voxel_coords[:, 3].long()
        bev_mask[batch_idx, 0, y_idx, x_idx] = voxel_is_dynamic

        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            bev_mask = F.max_pool2d(bev_mask, self.motion_mask_kernel_size, stride=1, padding=pad)
            bev_mask = (bev_mask > 0).float()

        if tuple(bev_mask.shape[-2:]) != tuple(target_hw):
            bev_mask = F.interpolate(bev_mask, size=target_hw, mode='nearest')

        return (bev_mask > 0).float()

    def forward(self, data_dict):
        voxel_features = data_dict['processed_radar']['voxel_features']
        voxel_coords = data_dict['processed_radar']['voxel_coords']
        voxel_num_points = data_dict['processed_radar']['voxel_num_points']

        radar_batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
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
            kd_motion_mask = self._build_hard_motion_mask(
                voxel_features,
                voxel_num_points,
                voxel_coords,
                bev_features.shape[-2:],
            )
            if kd_motion_mask is not None:
                output_dict['kd_motion_mask'] = kd_motion_mask

        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(bev_features)

        return output_dict
