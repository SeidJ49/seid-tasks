import torch
import torch.nn as nn

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import CenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import DynamicPillarVFESimple2D


class PillarnetLidarTeacher(nn.Module):
    """
    LiDAR-only PillarNet teacher for WP-2.

    This is the PillarNet/CenterHead counterpart of
    ``point_pillar_lidar_teacher.py``.  It intentionally uses only LiDAR
    points and exposes a feature tap before the CenterHead so the checkpoint can
    later be reused as a stronger teacher for radar distillation.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = args['grid_size']
        self.class_names = args['class_names']
        self.num_point_features = args.get('lidar_num_point_features', args.get('num_point_features', 5))
        self.loss_weight = args.get('teacher_loss_weight', 1.0)
        self.feature_key = args.get('feature_key', 'feature')
        self.prediction_key_prefix = args.get('prediction_key_prefix', '')

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=self.num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.backbone = PillarRes18BackBone8x(self.grid_size)
        self.bev_backbone = BaseBEVBackboneV2(args['teacher_bev_backbone'])
        self.head = CenterHead(
            args['teacher_head'],
            input_channels=args['teacher_head']['input_channels'],
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
        )

    @staticmethod
    def _masked_gt_boxes(object_bbx_center, object_bbx_mask):
        """Convert TruckScenes boxes to CenterHead gt_boxes with class ids."""
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
        # CenterHead stores decoded boxes under lidar_final_box_dict, while
        # RadarCenterHead/distill variants use final_box_dict.  Keep both so
        # inference/post_process always receives a batch-sized list.
        return batch_dict.get('lidar_final_box_dict', batch_dict.get('final_box_dict', []))

    def _encode_lidar_to_bev(self, batch_dict):
        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.bev_backbone(batch_dict)
        return batch_dict

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        batch_dict = {
            'batch_size': batch_size,
            'points': data_dict['lidar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': compute_loss,
        }

        batch_dict = self._encode_lidar_to_bev(batch_dict)
        feature = batch_dict['spatial_features_2d']
        batch_dict = self.head(batch_dict)

        if self.training or compute_loss:
            head_loss, head_tb = self.head.get_loss()
            total_loss = self.loss_weight * head_loss
            tb_dict = {
                'total_loss': total_loss.item(),
                'teacher_head_loss': head_loss.item(),
                **head_tb,
            }
            return {
                'loss': total_loss,
                'tb_dict': tb_dict,
                'teacher_head_loss': head_loss,
                self.feature_key: feature,
                'final_box_dict': self._select_final_box_dict(batch_dict),
            }

        output_dict = {
            self.feature_key: feature,
            'final_box_dict': self._select_final_box_dict(batch_dict),
        }

        # Keep common PointPillars-style aliases when CenterHead internals expose
        # them. This makes downstream KD/debugging code more tolerant.
        prefix = self.prediction_key_prefix
        if 'batch_cls_preds' in batch_dict:
            output_dict[f'{prefix}cls_preds'] = batch_dict['batch_cls_preds']
        if 'batch_box_preds' in batch_dict:
            output_dict[f'{prefix}reg_preds'] = batch_dict['batch_box_preds']

        return output_dict