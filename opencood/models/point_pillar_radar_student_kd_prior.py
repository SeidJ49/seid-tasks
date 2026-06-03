import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe_baseline_attention import PillarVFEBaselineAttention
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention import PointPillarScatterBaselineAttention


class PointPillarRadarStudentKdPrior(nn.Module):
    """
    WP3 radar student with explicit radar BEV priors.

    The point-pillar branch learns from radar pillars as before. In parallel,
    simple radar-specific BEV maps are built directly from the voxel tensor:
    occupancy, point density, absolute Doppler max, signed Doppler mean, and a
    soft dynamic score. These maps are fused into the BEV feature before the
    detection heads and before feature KD.
    """

    def __init__(self, args):
        super().__init__()

        radar_num_point_features = args.get('radar_num_point_features', 4)
        self.return_kd_motion_mask = args.get('return_kd_motion_mask', True)
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 3))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))
        self.prior_static_weight = float(args.get('prior_static_weight', 0.20))
        self.doppler_norm = float(args.get('doppler_norm', 2.0))

        self.grid_size = tuple(int(x) for x in args['point_pillar_scatter']['grid_size'])
        self.nx, self.ny, self.nz = self.grid_size
        assert self.nz == 1

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

        prior_channels = int(args.get('prior_channels', 32))
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(5, prior_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prior_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prior_channels),
            nn.ReLU(inplace=True),
        )
        self.prior_fusion = nn.Sequential(
            nn.Conv2d(self.out_channel + prior_channels, self.out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
        )

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

    def _build_radar_priors(self, voxel_features, voxel_num_points, voxel_coords, target_hw):
        batch_size = int(voxel_coords[:, 0].max().item()) + 1
        num_points = voxel_features.shape[1]
        valid = torch.arange(num_points, device=voxel_num_points.device).unsqueeze(0) < voxel_num_points.unsqueeze(1)
        valid_f = valid.float()

        v_r = voxel_features[:, :, 3] if voxel_features.shape[-1] > 3 else voxel_features.new_zeros(valid.shape)
        abs_v = v_r.abs() * valid_f
        signed_v = v_r * valid_f

        density = voxel_num_points.float().clamp_min(1.0)
        density_score = torch.log1p(density) / torch.log(voxel_features.new_tensor(float(num_points + 1)))
        abs_v_max = (abs_v.max(dim=1)[0] / self.doppler_norm).clamp(0.0, 1.0)
        signed_v_mean = (signed_v.sum(dim=1) / density).clamp(-self.doppler_norm, self.doppler_norm) / self.doppler_norm

        vm = self.radar_pillar_vfe.model_cfg.get('velocity_mask', {})
        thresh = float(vm.get('thresh', 0.15))
        temp = float(vm.get('temp', 12.0))
        dynamic_score = torch.sigmoid(temp * (abs_v_max * self.doppler_norm - thresh))

        priors = voxel_features.new_zeros((batch_size, 5, self.ny, self.nx))
        batch_idx = voxel_coords[:, 0].long()
        y_idx = voxel_coords[:, 2].long()
        x_idx = voxel_coords[:, 3].long()

        priors[batch_idx, 0, y_idx, x_idx] = 1.0
        priors[batch_idx, 1, y_idx, x_idx] = density_score
        priors[batch_idx, 2, y_idx, x_idx] = abs_v_max
        priors[batch_idx, 3, y_idx, x_idx] = signed_v_mean
        priors[batch_idx, 4, y_idx, x_idx] = dynamic_score

        if tuple(priors.shape[-2:]) != tuple(target_hw):
            priors = F.interpolate(priors, size=target_hw, mode='bilinear', align_corners=False)
        return priors

    def _build_motion_mask(self, priors, target_hw):
        occupancy = priors[:, 0:1].clamp(0.0, 1.0)
        dynamic = priors[:, 4:5].clamp(0.0, 1.0)
        mask = torch.maximum(dynamic, occupancy * self.prior_static_weight)

        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.motion_mask_kernel_size, stride=1, padding=pad)

        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)

        return (mask * self.motion_mask_scale).clamp(0.0, 1.0)

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

        radar_priors = self._build_radar_priors(
            voxel_features,
            voxel_num_points,
            voxel_coords,
            bev_features.shape[-2:],
        )
        prior_features = self.prior_encoder(radar_priors)
        bev_features = self.prior_fusion(torch.cat([bev_features, prior_features], dim=1))

        cls_preds = self.cls_head(bev_features)
        reg_preds = self.reg_head(bev_features)

        output_dict = {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'feature': bev_features,
            'radar_priors': radar_priors,
        }

        if self.return_kd_motion_mask:
            output_dict['kd_motion_mask'] = self._build_motion_mask(radar_priors, bev_features.shape[-2:])

        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(bev_features)

        return output_dict
