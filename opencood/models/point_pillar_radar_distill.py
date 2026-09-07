import torch
import torch.nn as nn

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_block import RadarDistill
from opencood.models.sub_modules.radardistill_head import CenterHead, RadarCenterHead
from opencood.models.sub_modules.preservation import (
    FrozenRadarAnchor,
    HeatmapPreservationLoss,
)
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x, RadarPillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import (
    DynamicPillarVFESimple2D,
    FactorizedRadarDynamicPillarVFE,
    RadarDynamicPillarVFESimple2D,
)


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
        self.init_radar_from_teacher = args.get('init_radar_from_teacher', self.train_stage != 'teacher')
        self.radar_loss_weight = args.get('radar_loss_weight', 1.0)
        self.distill_loss_weight = args.get('distill_loss_weight', 1.0)
        self.teacher_loss_weight = args.get('teacher_loss_weight', 1.0)
        self.teacher_ckpt = args.get('teacher_ckpt', '')
        self.preservation_cfg = dict(args.get('wp3_preservation', {}))
        if self.train_stage not in {'teacher', 'distill', 'joint'}:
            raise ValueError(f"Unsupported train_stage '{self.train_stage}'")

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=args['lidar_num_point_features'],
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        radar_vfe_cfg = args['radar_vfe']
        radar_vfe_type = radar_vfe_cfg.get('type', 'dynamic_pillar')
        radar_vfe_cls = FactorizedRadarDynamicPillarVFE if radar_vfe_type == 'factorized' else RadarDynamicPillarVFESimple2D
        self.radar_vfe = radar_vfe_cls(
            radar_vfe_cfg,
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
        radar_distill_cfg = dict(args['radar_distill'])
        radar_distill_cfg.setdefault('point_cloud_range', self.point_cloud_range)
        radar_distill_cfg.setdefault('voxel_size', self.voxel_size)
        radar_distill_cfg.setdefault('grid_size', self.grid_size)
        radar_distill_cfg.setdefault('wp3_guided_pfd', args.get('wp3_guided_pfd', {}))
        radar_distill_cfg.setdefault('class_names', self.class_names)
        radar_distill_cfg.setdefault(
            'class_names_each_head', args['radar_head']['class_names_each_head'])
        target_cfg = args['radar_head'].get('target_assigner_config', {})
        radar_distill_cfg.setdefault(
            'feature_map_stride', target_cfg.get('feature_map_stride', 8))
        radar_distill_cfg.setdefault(
            'gaussian_overlap', target_cfg.get('gaussian_overlap', 0.1))
        radar_distill_cfg.setdefault('min_radius', target_cfg.get('min_radius', 2))
        self.radar_distill = RadarDistill(radar_distill_cfg)
        self.radar_head = RadarCenterHead(
            args['radar_head'],
            input_channels=args['radar_head']['input_channels'],
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
        )
        if self.teacher_ckpt:
            self._load_teacher_checkpoint(self.teacher_ckpt)
        if self.freeze_teacher and self.train_stage != 'teacher':
            self._freeze_teacher_modules()
        self.preservation_loss = HeatmapPreservationLoss(
            self.preservation_cfg,
            args['radar_head']['class_names_each_head'],
        )
        self.radar_anchor = None
        if self.preservation_loss.enabled:
            anchor_checkpoint = self.preservation_cfg.get('anchor_checkpoint', '')
            anchor_sha256 = self.preservation_cfg.get('anchor_sha256', '')
            if not anchor_checkpoint or not anchor_sha256:
                raise ValueError(
                    'enabled AP15 preservation requires anchor_checkpoint and anchor_sha256.'
                )
            self.radar_anchor = FrozenRadarAnchor(
                self.radar_vfe,
                self.radar_backbone,
                self.radar_distill,
                self.radar_head,
                anchor_checkpoint,
                anchor_sha256,
            )

    def _load_teacher_checkpoint(self, ckpt_path, initialize_radar=None):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        teacher_prefixes = ('lidar_vfe.', 'teacher_backbone.', 'teacher_bev_backbone.', 'teacher_head.')
        teacher_to_radar_prefixes = {
            'lidar_vfe.': 'radar_vfe.',
            'teacher_backbone.': 'radar_backbone.',
            'teacher_bev_backbone.': 'radar_distill.',
            'teacher_head.': 'radar_head.',
        }
        filtered_state = {}
        model_state = self.state_dict()
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith(teacher_prefixes) and clean_key in model_state and model_state[clean_key].shape == value.shape:
                filtered_state[clean_key] = value

            initialize_radar = (
                self.init_radar_from_teacher
                if initialize_radar is None else bool(initialize_radar)
            )
            if not initialize_radar:
                continue

            for teacher_prefix, radar_prefix in teacher_to_radar_prefixes.items():
                if not clean_key.startswith(teacher_prefix):
                    continue

                radar_key = radar_prefix + clean_key[len(teacher_prefix):]
                if radar_key in model_state and model_state[radar_key].shape == value.shape:
                    filtered_state[radar_key] = value
                break

        # Loading a partial state dict without module metadata makes older
        # PyTorch BatchNorm loaders synthesize missing num_batches_tracked
        # entries for unrelated modules. Merge into the complete current state
        # so restoring the selected LiDAR teacher cannot mutate the E20
        # student/anchor buffers.
        model_state.update(filtered_state)
        self.load_state_dict(model_state, strict=True)

    def reload_selected_lidar_teacher(self):
        """Restore the selected LiDAR teacher after loading the E20 student."""
        if not self.teacher_ckpt:
            raise ValueError('Selected-teacher reload requires a LiDAR checkpoint.')
        self._load_teacher_checkpoint(self.teacher_ckpt, initialize_radar=False)
        self._freeze_teacher_modules()

    def _freeze_teacher_modules(self):
        for module in (self.lidar_vfe, self.teacher_backbone, self.teacher_bev_backbone, self.teacher_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_teacher and self.train_stage != 'teacher':
            for module in (self.lidar_vfe, self.teacher_backbone, self.teacher_bev_backbone, self.teacher_head):
                module.eval()
        if self.radar_anchor is not None:
            self.radar_anchor.eval()
        return self

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
        if self.freeze_teacher and self.train_stage != 'teacher':
            with torch.no_grad():
                batch_dict = self._run_teacher(batch_dict)
                batch_dict = self.teacher_head(batch_dict)
            return batch_dict

        batch_dict = self._run_teacher(batch_dict)
        batch_dict = self.teacher_head(batch_dict)
        return batch_dict

    @staticmethod
    def _select_final_box_dict(batch_dict):
        if 'final_box_dict' in batch_dict:
            return batch_dict['final_box_dict']
        if 'lidar_final_box_dict' in batch_dict:
            return batch_dict['lidar_final_box_dict']
        return []

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        batch_dict = {
            'batch_size': batch_size,
            'points': data_dict['lidar_points'],
            'radar_points': data_dict['radar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': compute_loss,
        }

        anchor_pred_dicts = None
        if self.radar_anchor is not None and (self.training or compute_loss):
            anchor_pred_dicts = self.radar_anchor(
                data_dict['radar_points'], batch_dict['gt_boxes'])

        batch_dict = self._teacher_forward(batch_dict)

        if self.train_stage == 'teacher':
            if self.training or compute_loss:
                teacher_head_loss, teacher_tb = self.teacher_head.get_loss()
                total_loss = self.teacher_loss_weight * teacher_head_loss
                tb_dict = {'total_loss': total_loss.item(), 'teacher_head_loss': teacher_head_loss.item(), **teacher_tb}
                return {
                    'loss': total_loss,
                    'tb_dict': tb_dict,
                    'teacher_head_loss': teacher_head_loss,
                    'final_box_dict': self._select_final_box_dict(batch_dict),
                }
            return {'final_box_dict': self._select_final_box_dict(batch_dict)}

        batch_dict = self.radar_vfe(batch_dict)
        batch_dict = self.radar_backbone(batch_dict)
        batch_dict = self.radar_distill(batch_dict)
        batch_dict = self.radar_head(batch_dict)

        if self.training or compute_loss:
            teacher_head_loss = None
            teacher_tb = {}
            if self.train_stage == 'joint':
                teacher_head_loss, teacher_tb = self.teacher_head.get_loss()

            radar_head_loss, radar_tb = self.radar_head.get_loss()
            distill_loss, distill_tb = self.radar_distill.get_loss(batch_dict)
            preservation_loss, preservation_tb = self.preservation_loss(
                batch_dict['radar_pred_dicts'],
                anchor_pred_dicts,
                batch_dict['target_dicts'],
            )
            weighted_preservation_loss = (
                self.preservation_loss.loss_weight * preservation_loss
            )
            total_loss = (
                self.radar_loss_weight * radar_head_loss +
                self.distill_loss_weight * distill_loss +
                weighted_preservation_loss
            )
            tb_dict = {
                'total_loss': total_loss.item(),
                'radar_head_loss': radar_head_loss.item(),
                'detection_loss': (self.radar_loss_weight * radar_head_loss).item(),
                'lidar_kd_loss': (self.distill_loss_weight * distill_loss).item(),
                **radar_tb,
                **distill_tb,
                **preservation_tb,
            }
            output_dict = {
                'loss': total_loss,
                'tb_dict': tb_dict,
                'debug_maps': getattr(self.radar_distill, 'debug_maps', {}),
                'radar_head_loss': radar_head_loss,
                'distill_loss': distill_loss,
                'preservation_loss': preservation_loss,
                'weighted_preservation_loss': weighted_preservation_loss,
                'final_box_dict': self._select_final_box_dict(batch_dict),
            }

            if teacher_head_loss is not None:
                total_loss = total_loss + self.teacher_loss_weight * teacher_head_loss
                tb_dict['total_loss'] = total_loss.item()
                tb_dict.update({f'teacher_{k}': v for k, v in teacher_tb.items()})
                tb_dict['teacher_head_loss'] = teacher_head_loss.item()
                output_dict['loss'] = total_loss
                output_dict['teacher_head_loss'] = teacher_head_loss

            return output_dict

        return {'final_box_dict': self._select_final_box_dict(batch_dict)}
