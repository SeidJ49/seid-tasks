import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.thesis_modules import (
    DetectionHead,
    LidarReliabilityEstimator,
    SpatialReliabilityFusion,
)


class _SecondBranch(nn.Module):
    """Shared SECOND encoder branch for LiDAR or radar BEV features."""

    def __init__(self, args, num_point_features_key):
        super().__init__()
        num_point_features = args.get(num_point_features_key, 4)
        self.grid_size = args['grid_size']
        self.mean_vfe = MeanVFE(args.get('mean_vfe', {}), num_point_features)
        self.backbone_3d = VoxelBackBone8x(
            args.get('backbone_3d', {}),
            num_point_features,
            self.grid_size,
        )
        self.height_compression = HeightCompression(args['height_compression'])

        bev_backbone_type = args.get('bev_backbone_type', 'base').lower()
        if bev_backbone_type in ('resnet', 'resnet_bev'):
            bev_cfg = dict(args['base_bev_backbone'])
            bev_cfg.setdefault('inplanes', args['height_compression']['feature_num'])
            self.backbone_2d = ResNetBEVBackbone(
                bev_cfg,
                args['height_compression']['feature_num'],
            )
        else:
            self.backbone_2d = BaseBEVBackbone(
                args['base_bev_backbone'],
                args['height_compression']['feature_num'],
            )

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

    def forward(self, processed_dict, batch_size):
        batch_dict = {
            'voxel_features': processed_dict['voxel_features'],
            'voxel_coords': processed_dict['voxel_coords'],
            'voxel_num_points': processed_dict['voxel_num_points'],
            'batch_size': batch_size,
        }
        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)
        return {
            'feature': bev_features,
            'voxel_coords': processed_dict['voxel_coords'],
            'voxel_num_points': processed_dict['voxel_num_points'],
        }


class SecondReliabilityFusion(nn.Module):
    """
    WP4 reliability-guided LiDAR-radar fusion using SECOND sparse voxel encoders.

    The LiDAR branch provides geometry and an explicit BEV reliability map. The
    radar branch provides a weather-robust fallback feature. A spatial gate fuses
    both features so unreliable LiDAR cells can lean more on radar.
    """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args['batch_size']
        self.lidar_branch = _SecondBranch(args, 'num_point_features')
        self.radar_branch = _SecondBranch(args, 'radar_num_point_features')

        feature_channels = self.lidar_branch.out_channel
        self.reliability_estimator = LidarReliabilityEstimator(
            in_channels=feature_channels,
            grid_size=args['grid_size'],
            cfg=args.get('reliability', {}),
        )
        self.fusion_gate = SpatialReliabilityFusion(
            feature_channels=feature_channels,
            cfg=args.get('fusion_gate', {}),
        )
        self.head = DetectionHead(
            in_channels=feature_channels,
            anchor_number=args['anchor_number'],
            dir_args=args.get('dir_args'),
        )

    def _infer_batch_size(self, data_dict):
        if 'label_dict' in data_dict and 'pos_equal_one' in data_dict['label_dict']:
            return int(data_dict['label_dict']['pos_equal_one'].shape[0])
        if 'object_bbx_center' in data_dict:
            return int(data_dict['object_bbx_center'].shape[0])
        if 'record_len' in data_dict:
            return int(data_dict['record_len'].sum().item())

        max_batch = -1
        for key in ('processed_lidar', 'processed_radar'):
            if key in data_dict and data_dict[key]['voxel_coords'].numel() > 0:
                max_batch = max(max_batch, int(data_dict[key]['voxel_coords'][:, 0].max().item()))
        return max_batch + 1 if max_batch >= 0 else self.batch_size

    def _align_spatial(self, feature, target_hw):
        if tuple(feature.shape[-2:]) != tuple(target_hw):
            return F.interpolate(feature, size=target_hw, mode='bilinear', align_corners=False)
        return feature

    def forward(self, data_dict):
        batch_size = self._infer_batch_size(data_dict)
        lidar_dict = self.lidar_branch(data_dict['processed_lidar'], batch_size)
        radar_dict = self.radar_branch(data_dict['processed_radar'], batch_size)

        lidar_feature = lidar_dict['feature']
        radar_feature = self._align_spatial(radar_dict['feature'], lidar_feature.shape[-2:])

        reliability_map, density_map = self.reliability_estimator(
            lidar_feature,
            lidar_dict['voxel_coords'],
            lidar_dict['voxel_num_points'],
        )
        reliability_map = self._align_spatial(reliability_map, lidar_feature.shape[-2:])
        density_map = self._align_spatial(density_map, lidar_feature.shape[-2:])

        fused_feature, fusion_gate = self.fusion_gate(
            lidar_feature,
            radar_feature,
            reliability_map,
        )

        output_dict = self.head(fused_feature)
        output_dict.update({
            'feature': fused_feature,
            'lidar_feature': lidar_feature,
            'radar_feature': radar_feature,
            'lidar_reliability': reliability_map,
            'lidar_density_map': density_map,
            'fusion_gate': fusion_gate,
            'batch_size': batch_size,
        })
        return output_dict
