import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.pillar_vfe_baseline_attention import PillarVFEBaselineAttention
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention import PointPillarScatterBaselineAttention


class PointPillarLidarBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_point_features = args.get('num_point_features', 4)
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=num_point_features,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(
            args['base_bev_backbone'],
            args['point_pillar_scatter']['num_features'],
        )
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

    def forward(self, processed_lidar):
        batch_dict = {
            'voxel_features': processed_lidar['voxel_features'],
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        raw_spatial = batch_dict['spatial_features']
        batch_dict = self.backbone({'spatial_features': raw_spatial})
        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)
        return {
            'feature': bev_features,
            'raw_spatial_feature': raw_spatial,
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
        }


class PointPillarRadarBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_point_features = args.get('radar_num_point_features', 4)
        self.return_velocity_mask = args.get('return_velocity_mask', True)
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 7))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))

        self.pillar_vfe = PillarVFEBaselineAttention(
            args['pillar_vfe'],
            num_point_features=num_point_features,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.scatter = PointPillarScatterBaselineAttention(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(
            args['base_bev_backbone'],
            args['point_pillar_scatter']['num_features'],
        )
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

    def _build_motion_mask(self, scatter_dict, target_hw):
        velocity_mask = scatter_dict.get('velocity_confidence_mask', None)
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

    def forward(self, processed_radar):
        batch_dict = {
            'voxel_features': processed_radar['voxel_features'],
            'voxel_coords': processed_radar['voxel_coords'],
            'voxel_num_points': processed_radar['voxel_num_points'],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        raw_spatial = batch_dict['spatial_features']
        backbone_dict = self.backbone({'spatial_features': raw_spatial})
        bev_features = backbone_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)

        output = {
            'feature': bev_features,
            'raw_spatial_feature': raw_spatial,
            'voxel_coords': processed_radar['voxel_coords'],
            'voxel_num_points': processed_radar['voxel_num_points'],
        }
        if self.return_velocity_mask:
            motion_mask = self._build_motion_mask(batch_dict, bev_features.shape[-2:])
            if motion_mask is not None:
                output['velocity_confidence_mask'] = motion_mask
        return output


class DetectionHead(nn.Module):
    def __init__(self, in_channels, anchor_number, dir_args=None):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 7 * anchor_number, kernel_size=1)
        self.use_dir = dir_args is not None
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                in_channels,
                dir_args['num_bins'] * anchor_number,
                kernel_size=1,
            )

    def forward(self, feature, prefix=''):
        output = {
            f'{prefix}cls_preds': self.cls_head(feature),
            f'{prefix}reg_preds': self.reg_head(feature),
        }
        if self.use_dir:
            output[f'{prefix}dir_preds'] = self.dir_head(feature)
        return output


class LidarReliabilityEstimator(nn.Module):
    def __init__(self, in_channels, grid_size, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])
        self.mode = cfg.get('mode', 'heuristic')
        self.pool_kernel = int(cfg.get('pool_kernel', 5))
        self.density_weight = float(cfg.get('density_weight', 0.7))
        self.distance_weight = float(cfg.get('distance_weight', 0.3))
        self.eps = float(cfg.get('eps', 1e-6))

        hidden_channels = int(cfg.get('hidden_channels', max(16, in_channels // 4)))
        if self.mode == 'learned':
            self.learned_head = nn.Sequential(
                nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.learned_head = None

        ys = torch.linspace(-1.0, 1.0, self.ny)
        xs = torch.linspace(-1.0, 1.0, self.nx)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(xx.pow(2) + yy.pow(2)).clamp_(0.0, 1.0)
        self.register_buffer('distance_prior', dist.view(1, 1, self.ny, self.nx), persistent=False)

    def _density_map(self, voxel_coords, voxel_num_points, batch_size, device, dtype):
        flat_size = batch_size * self.ny * self.nx
        density = torch.zeros(flat_size, device=device, dtype=dtype)
        batch_idx = voxel_coords[:, 0].long()
        y_idx = voxel_coords[:, 2].long().clamp_(0, self.ny - 1)
        x_idx = voxel_coords[:, 3].long().clamp_(0, self.nx - 1)
        flat_idx = batch_idx * (self.ny * self.nx) + y_idx * self.nx + x_idx
        density.scatter_add_(0, flat_idx, voxel_num_points.to(dtype).view(-1))
        density = density.view(batch_size, 1, self.ny, self.nx)
        if self.pool_kernel > 1:
            padding = self.pool_kernel // 2
            density = F.avg_pool2d(density, kernel_size=self.pool_kernel, stride=1, padding=padding)
        density_max = density.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (density / density_max).clamp_(0.0, 1.0)

    def forward(self, lidar_feature, voxel_coords, voxel_num_points):
        batch_size = lidar_feature.shape[0]
        density_map = self._density_map(
            voxel_coords=voxel_coords,
            voxel_num_points=voxel_num_points,
            batch_size=batch_size,
            device=lidar_feature.device,
            dtype=lidar_feature.dtype,
        )
        distance_prior = self.distance_prior.to(device=lidar_feature.device, dtype=lidar_feature.dtype)
        distance_prior = distance_prior.expand(batch_size, -1, -1, -1)

        if self.learned_head is not None:
            reliability = torch.sigmoid(
                self.learned_head(torch.cat([lidar_feature, density_map, 1.0 - distance_prior], dim=1))
            )
        else:
            reliability = self.density_weight * density_map + self.distance_weight * (1.0 - distance_prior)
            reliability = reliability.clamp_(0.0, 1.0)

        return reliability, density_map


class SpatialReliabilityFusion(nn.Module):
    def __init__(self, feature_channels, cfg=None):
        super().__init__()
        cfg = cfg or {}
        hidden_channels = int(cfg.get('hidden_channels', max(32, feature_channels // 2)))
        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_feature, radar_feature, reliability_map):
        gate_input = torch.cat([lidar_feature, radar_feature, reliability_map], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        fused = gate * lidar_feature + (1.0 - gate) * radar_feature
        fused = self.post_conv(fused)
        return fused, gate

class LidarUnreliabilityEstimator(nn.Module):
    """
    WP4 Stage-3 LiDAR unreliability estimator.

    Produces U_L in [0, 1] over BEV cells where high means LiDAR is likely
    unreliable/degraded. The heuristic intentionally uses only geometric LiDAR
    evidence: point count and local occupancy sparsity. It does not use feature
    variance, because feature variance alone is not a robust degradation cue.
    """

    def __init__(self, grid_size, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])
        self.pool_kernel = int(cfg.get('pool_kernel', 7))
        self.expected_points_per_cell = float(cfg.get('expected_points_per_cell', 6.0))
        self.point_count_weight = float(cfg.get('point_count_weight', 0.6))
        self.occupancy_sparsity_weight = float(cfg.get('occupancy_sparsity_weight', 0.4))
        self.eps = float(cfg.get('eps', 1e-6))

    def _bev_point_count(self, voxel_coords, voxel_num_points, batch_size, device, dtype):
        flat_size = batch_size * self.ny * self.nx
        point_count = torch.zeros(flat_size, device=device, dtype=dtype)
        if voxel_coords.numel() == 0:
            return point_count.view(batch_size, 1, self.ny, self.nx)

        batch_idx = voxel_coords[:, 0].long().clamp(0, batch_size - 1)
        y_idx = voxel_coords[:, 2].long().clamp(0, self.ny - 1)
        x_idx = voxel_coords[:, 3].long().clamp(0, self.nx - 1)
        flat_idx = batch_idx * (self.ny * self.nx) + y_idx * self.nx + x_idx
        point_count.scatter_add_(0, flat_idx, voxel_num_points.to(dtype).view(-1))
        return point_count.view(batch_size, 1, self.ny, self.nx)

    def forward(self, voxel_coords, voxel_num_points, batch_size, device=None, dtype=None):
        if device is None:
            device = voxel_num_points.device
        if dtype is None:
            dtype = torch.float32

        point_count_map = self._bev_point_count(
            voxel_coords=voxel_coords,
            voxel_num_points=voxel_num_points,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        occupied_map = (point_count_map > 0).to(dtype)
        if self.pool_kernel > 1:
            padding = self.pool_kernel // 2
            occupancy_ratio = F.avg_pool2d(
                occupied_map,
                kernel_size=self.pool_kernel,
                stride=1,
                padding=padding,
            )
        else:
            occupancy_ratio = occupied_map

        point_count_score = (point_count_map / max(self.expected_points_per_cell, self.eps)).clamp(0.0, 1.0)
        point_count_unreliability = 1.0 - point_count_score
        occupancy_sparsity = 1.0 - occupancy_ratio.clamp(0.0, 1.0)

        weight_sum = max(self.point_count_weight + self.occupancy_sparsity_weight, self.eps)
        unreliability_map = (
            self.point_count_weight * point_count_unreliability
            + self.occupancy_sparsity_weight * occupancy_sparsity
        ) / weight_sum
        unreliability_map = unreliability_map.clamp(0.0, 1.0)

        return {
            'lidar_unreliability': unreliability_map,
            'lidar_point_count_map': point_count_map,
            'lidar_occupancy_map': occupied_map,
            'lidar_occupancy_sparsity': occupancy_sparsity,
        }