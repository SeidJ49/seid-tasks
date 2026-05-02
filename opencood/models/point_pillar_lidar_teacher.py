import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarLidarTeacher(nn.Module):
    """
    LiDAR-only PointPillars teacher for WP-2.

    Design goals:
    - stay close to the existing single-lidar baseline
    - expose a clean BEV feature tap right before the detection heads
    - support prefixed outputs for later KD use (e.g. teacher_feature)
    - avoid debug side effects during normal training/inference
    """

    def __init__(self, args):
        super().__init__()

        num_point_features = args.get('num_point_features', 4)
        self.prediction_key_prefix = args.get('prediction_key_prefix', '')
        self.feature_key = args.get(
            'feature_key',
            f"{self.prediction_key_prefix}feature" if self.prediction_key_prefix else 'feature'
        )

        self.lidar_pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=num_point_features,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])

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

    def _encode_lidar_to_bev(self, processed_lidar):
        batch_dict = {
            'voxel_features': processed_lidar['voxel_features'],
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
        }
        batch_dict = self.lidar_pillar_vfe(batch_dict)
        batch_dict = self.lidar_scatter(batch_dict)
        batch_dict = {'spatial_features': batch_dict['spatial_features']}
        batch_dict = self.backbone(batch_dict)

        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)

        return bev_features

    def forward(self, data_dict):
        bev_features = self._encode_lidar_to_bev(data_dict['processed_lidar'])

        cls_preds = self.cls_head(bev_features)
        reg_preds = self.reg_head(bev_features)

        prefix = self.prediction_key_prefix
        output_dict = {
            f'{prefix}cls_preds': cls_preds,
            f'{prefix}reg_preds': reg_preds,
            self.feature_key: bev_features,
        }

        if self.use_dir:
            output_dict[f'{prefix}dir_preds'] = self.dir_head(bev_features)

        return output_dict
