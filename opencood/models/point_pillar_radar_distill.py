import torch
import torch.nn as nn

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_block import RadarDistill
from opencood.models.sub_modules.radardistill_head import CenterHead, RadarCenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x, RadarPillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import DynamicPillarVFESimple2D, RadarDynamicPillarVFESimple2D


class PointPillarRadarDistill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = args['grid_size']
        self.class_names = args['class_names']
        self.train_stage = args.get('train_stage', 'distill')
        self.freeze_teacher = args.get('freeze_teacher', True)
        self.radar_loss_weight = args.get('radar_loss_weight', 1.0)
        self.distill_loss_weight = args.get('distill_loss_weight', 1.0)
        self.teacher_loss_weight = args.get('teacher_loss_weight', 1.0)
        self.teacher_ckpt = args.get('teacher_ckpt', '')

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=args['lidar_num_point_features'],
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.radar_vfe = RadarDynamicPillarVFESimple2D(
            args['radar_vfe'],
            num_point_features=args['radar_num_point_features'],
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )

        self.teacher_backbone = PillarRes18BackBone8x(self.grid_size)
        self.teacher_bev_backbone = BaseBEVBackboneV2(args['teacher_bev_backbone'])
        self.teacher_head = CenterHead(
            args['teacher_head'],
            input_channels=args['teacher_head']['input_channels'],
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
        )
        self.radar_backbone = RadarPillarRes18BackBone8x(self.grid_size)
        self.radar_distill = RadarDistill(args['radar_distill'])
        self.radar_head = RadarCenterHead(
            args['radar_head'],
            input_channels=args['radar_head']['input_channels'],
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
        )
        if self.teacher_ckpt:
            self._load_teacher_checkpoint(self.teacher_ckpt)

    def _load_teacher_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        teacher_prefixes = ('lidar_vfe.', 'teacher_backbone.', 'teacher_bev_backbone.', 'teacher_head.')
        filtered_state = {}
        model_state = self.state_dict()
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith(teacher_prefixes) and clean_key in model_state and model_state[clean_key].shape == value.shape:
                filtered_state[clean_key] = value

        self.load_state_dict(filtered_state, strict=False)

    @staticmethod
    def _masked_gt_boxes(object_bbx_center, object_bbx_mask):
        gt_boxes = object_bbx_center.new_zeros(object_bbx_center.shape[0], object_bbx_center.shape[1], 8)
        gt_boxes[:, :, :7] = object_bbx_center.float()
        gt_boxes[:, :, 7] = object_bbx_mask.float()
        gt_boxes[:, :, 7][object_bbx_mask > 0] = 1.0
        return gt_boxes

    def _run_teacher(self, batch_dict):
        if self.freeze_teacher:
            with torch.no_grad():
                batch_dict = self.lidar_vfe(batch_dict)
                batch_dict = self.teacher_backbone(batch_dict)
                batch_dict = self.teacher_bev_backbone(batch_dict)
            return batch_dict

        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.teacher_backbone(batch_dict)
        batch_dict = self.teacher_bev_backbone(batch_dict)
        return batch_dict

    def _teacher_forward(self, batch_dict):
        batch_dict = self._run_teacher(batch_dict)
        batch_dict = self.teacher_head(batch_dict)
        return batch_dict

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        batch_dict = {
            'batch_size': batch_size,
            'points': data_dict['lidar_points'],
            'radar_points': data_dict['radar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
        }

        batch_dict = self._teacher_forward(batch_dict)

        if self.train_stage == 'teacher':
            if self.training:
                teacher_head_loss, teacher_tb = self.teacher_head.get_loss()
                total_loss = self.teacher_loss_weight * teacher_head_loss
                tb_dict = {'total_loss': total_loss.item(), 'teacher_head_loss': teacher_head_loss.item(), **teacher_tb}
                return {
                    'loss': total_loss,
                    'tb_dict': tb_dict,
                    'teacher_head_loss': teacher_head_loss,
                    'final_box_dict': batch_dict.get('lidar_final_box_dict', []),
                }
            return {'final_box_dict': batch_dict.get('lidar_final_box_dict', [])}

        batch_dict = self.radar_vfe(batch_dict)
        batch_dict = self.radar_backbone(batch_dict)
        batch_dict = self.radar_distill(batch_dict)
        batch_dict = self.radar_head(batch_dict)

        if self.training:
            radar_head_loss, radar_tb = self.radar_head.get_loss()
            distill_loss, distill_tb = self.radar_distill.get_loss(batch_dict)
            total_loss = self.radar_loss_weight * radar_head_loss + self.distill_loss_weight * distill_loss
            tb_dict = {'total_loss': total_loss.item(), **radar_tb, **distill_tb}
            return {
                'loss': total_loss,
                'tb_dict': tb_dict,
                'radar_head_loss': radar_head_loss,
                'distill_loss': distill_loss,
                'final_box_dict': batch_dict.get('final_box_dict', []),
            }

        return {'final_box_dict': batch_dict.get('final_box_dict', [])}