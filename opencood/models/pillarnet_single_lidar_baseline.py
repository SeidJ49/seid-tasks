import torch
import torch.nn as nn

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import CenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import DynamicPillarVFESimple2D


class PillarnetSingleLidarBaseline(nn.Module):
    """WP1 PillarNet LiDAR-only baseline.

    This mirrors the WP2 LiDAR teacher architecture but is intentionally named
    and configured as a baseline so final experiments do not rely on legacy
    PointPillar configs after the PillarNet decision.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = args['grid_size']
        self.class_names = args['class_names']
        self.num_point_features = int(args.get('lidar_num_point_features', args.get('num_point_features', 5)))
        self.loss_weight = float(args.get('loss_weight', 1.0))
        self.feature_key = args.get('feature_key', 'feature')

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=self.num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.backbone = PillarRes18BackBone8x(self.grid_size)
        self.bev_backbone = BaseBEVBackboneV2(args['lidar_bev_backbone'])
        self.head = CenterHead(
            args['lidar_head'],
            input_channels=args['lidar_head']['input_channels'],
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

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        batch_dict = {
            'batch_size': batch_size,
            'points': data_dict['lidar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': compute_loss,
        }

        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.bev_backbone(batch_dict)
        feature = batch_dict['spatial_features_2d']
        batch_dict = self.head(batch_dict)

        output_dict = {
            self.feature_key: feature,
            'final_box_dict': self._select_final_box_dict(batch_dict),
        }
        if self.training or compute_loss:
            head_loss, head_tb = self.head.get_loss()
            total_loss = self.loss_weight * head_loss
            output_dict.update({
                'loss': total_loss,
                'tb_dict': {
                    'total_loss': total_loss.item(),
                    'lidar_head_loss': head_loss.item(),
                    **head_tb,
                },
                'lidar_head_loss': head_loss,
            })
        return output_dict
