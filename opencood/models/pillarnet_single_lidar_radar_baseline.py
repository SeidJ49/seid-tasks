import torch
import torch.nn as nn

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import CenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import (
    PillarRes18BackBone8x,
    RadarPillarRes18BackBone8x,
)
from opencood.models.sub_modules.radardistill_vfe import (
    DynamicPillarVFESimple2D,
    RadarDynamicPillarVFESimple2D,
)


class PillarnetSingleLidarRadarBaseline(nn.Module):
    """WP1 naive PillarNet LiDAR-radar BEV fusion baseline.

    Both modalities are encoded independently with the same PillarNet-style
    branches used by the later workpackages. Features are fused by channel
    concatenation plus a 1x1 projection, intentionally without reliability
    gates, KD losses, or radar cue priors.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.class_names = args['class_names']
        self.loss_weight = float(args.get('fusion_loss_weight', 1.0))
        self.feature_key = args.get('feature_key', 'feature')

        self.grid_size = args['grid_size']
        lidar_args = args['lidar_branch_args']
        radar_args = args['radar_branch_args']

        self.lidar_vfe = DynamicPillarVFESimple2D(
            lidar_args['lidar_vfe'],
            num_point_features=int(lidar_args.get('lidar_num_point_features', 5)),
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.lidar_backbone = PillarRes18BackBone8x(self.grid_size)
        self.lidar_bev_backbone = BaseBEVBackboneV2(lidar_args['lidar_bev_backbone'])

        self.radar_vfe = RadarDynamicPillarVFESimple2D(
            radar_args['radar_vfe'],
            num_point_features=int(radar_args.get('radar_num_point_features', 4)),
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.radar_backbone = RadarPillarRes18BackBone8x(self.grid_size)
        self.radar_bev_backbone = BaseBEVBackboneV2(radar_args['radar_bev_backbone'])

        lidar_channels = int(args.get('lidar_feature_channels', args['fusion_head']['input_channels']))
        radar_channels = int(args.get('radar_feature_channels', args['fusion_head']['input_channels']))
        fusion_channels = int(args['fusion_head']['input_channels'])
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(lidar_channels + radar_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.head = CenterHead(
            args['fusion_head'],
            input_channels=fusion_channels,
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
        )

    @staticmethod
    def _masked_gt_boxes(object_bbx_center, object_bbx_mask):
        if object_bbx_center.shape[-1] == 8:
            gt_boxes = object_bbx_center.float().clone()
            gt_boxes[:, :, 7] = gt_boxes[:, :, 7] * object_bbx_mask.float()
            return gt_boxes
        gt_boxes = object_bbx_center.new_zeros(object_bbx_center.shape[0], object_bbx_center.shape[1], 8)
        gt_boxes[:, :, :7] = object_bbx_center.float()
        gt_boxes[:, :, 7] = object_bbx_mask.float()
        return gt_boxes

    @staticmethod
    def _select_final_box_dict(batch_dict):
        return batch_dict.get('lidar_final_box_dict', batch_dict.get('final_box_dict', []))

    def _encode_lidar(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        batch_dict = {'batch_size': batch_size, 'points': data_dict['lidar_points']}
        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.lidar_backbone(batch_dict)
        batch_dict = self.lidar_bev_backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def _encode_radar(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        batch_dict = {'batch_size': batch_size, 'radar_points': data_dict['radar_points']}
        batch_dict = self.radar_vfe(batch_dict)
        batch_dict = self.radar_backbone(batch_dict)
        batch_dict['multi_scale_2d_features'] = batch_dict['radar_multi_scale_2d_features']
        batch_dict['multi_scale_2d_strides'] = batch_dict['radar_multi_scale_2d_strides']
        batch_dict = self.radar_bev_backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        lidar_feature = self._encode_lidar(data_dict)
        radar_feature = self._encode_radar(data_dict)
        fused_feature = self.fusion_proj(torch.cat([lidar_feature, radar_feature], dim=1))

        batch_dict = {
            'batch_size': batch_size,
            'spatial_features_2d': fused_feature,
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': compute_loss,
        }
        batch_dict = self.head(batch_dict)

        output_dict = {
            self.feature_key: fused_feature,
            'lidar_feature': lidar_feature,
            'radar_feature': radar_feature,
            'final_box_dict': self._select_final_box_dict(batch_dict),
        }
        if self.training or compute_loss:
            head_loss, head_tb = self.head.get_loss()
            total_loss = self.loss_weight * head_loss
            output_dict.update({
                'loss': total_loss,
                'tb_dict': {
                    'total_loss': total_loss.item(),
                    'fusion_head_loss': head_loss.item(),
                    **head_tb,
                },
                'fusion_head_loss': head_loss,
            })
        return output_dict
