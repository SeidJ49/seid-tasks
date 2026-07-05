import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.pillarnet_radar_student_kd import PillarnetRadarStudentKd
from opencood.models.pillarnet_reliability_fusion import ReliabilityGuidedPillarFusion
from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import CenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import DynamicPillarVFESimple2D


class PillarnetReliabilityFusionRadarDistill(nn.Module):
    """WP5 PillarNet radar-distillation + LiDAR reliability fusion.

    This is the PillarNet version: no SECOND branches, no PointPillar
    VFE/scatter/anchor-head wrappers.  The LiDAR side uses dynamic PillarNet
    encoding.  The radar side reuses ``PillarnetRadarStudentKd`` as
    the distilled radar backbone, optionally initialized from a WP3 checkpoint
    and frozen for first fusion experiments.

    Forward outputs intentionally expose thesis maps/features: ``U_L``,
    ``lidar_gate``, ``radar_gate``, ``fusion_gate``, ``kd_motion_mask`` and
    ``rcs_confidence_mask``.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = np.asarray(args['grid_size'], dtype=np.int64)
        self.class_names = args['class_names']
        self.lidar_num_point_features = int(args.get('lidar_num_point_features', args.get('num_point_features', 5)))
        self.loss_weight = float(args.get('fusion_loss_weight', 1.0))
        self.feature_key = args.get('feature_key', 'feature')
        self.freeze_radar_student = bool(args.get('freeze_radar_student', True))
        self.pretrained_radar_student_checkpoint = args.get('pretrained_radar_student_checkpoint')

        rel_cfg = args.get('unreliability', {})
        self.reliability_pool_kernel = int(rel_cfg.get('pool_kernel', 7))
        self.expected_points_per_cell = float(rel_cfg.get('expected_points_per_cell', 6.0))
        self.point_count_weight = float(rel_cfg.get('point_count_weight', 0.6))
        self.occupancy_sparsity_weight = float(rel_cfg.get('occupancy_sparsity_weight', 0.4))
        self.unreliability_eps = float(rel_cfg.get('eps', 1e-6))

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=self.lidar_num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.lidar_backbone = PillarRes18BackBone8x(self.grid_size)
        self.lidar_bev_backbone = BaseBEVBackboneV2(args['lidar_bev_backbone'])

        radar_student_args = dict(args['radar_student_args'])
        # The YAML parser may leave a nested student grid_size as a plain Python
        # list. Do not use setdefault here: force the model-boundary value to
        # stay a numpy array so RadarDistill/PillarNet sparse backbones can use
        # grid_size[[1, 0]].
        radar_student_args['grid_size'] = self.grid_size
        radar_student_args['voxel_size'] = radar_student_args.get('voxel_size', self.voxel_size)
        radar_student_args['lidar_range'] = radar_student_args.get('lidar_range', self.point_cloud_range)
        radar_student_args.setdefault('class_names', self.class_names)
        radar_student_args.setdefault('feature_key', 'feature')
        radar_student_args.setdefault('return_kd_motion_mask', True)
        radar_student_args.setdefault('return_rcs_confidence_mask', True)
        self.radar_student = PillarnetRadarStudentKd(radar_student_args)

        if self.pretrained_radar_student_checkpoint:
            self._load_radar_student_checkpoint(self.pretrained_radar_student_checkpoint)
        if self.freeze_radar_student:
            for param in self.radar_student.parameters():
                param.requires_grad_(False)

        feature_channels = int(args.get('fusion_channels', args['fusion_head']['input_channels']))
        self.fusion_gate = ReliabilityGuidedPillarFusion(feature_channels, args.get('fusion_gate', {}))
        self.head = CenterHead(
            args['fusion_head'],
            input_channels=args['fusion_head']['input_channels'],
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

    @staticmethod
    def _strip_known_prefix(key):
        for prefix in ('module.', 'student.', 'radar_student.', 'model.'):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def _load_radar_student_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'pretrained_radar_student_checkpoint not found: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        filtered = OrderedDict()
        skipped_prefixes = ('head.', 'module.head.', 'student.head.', 'radar_student.head.', 'model.head.')
        for key, value in state_dict.items():
            if key.startswith(skipped_prefixes):
                continue
            clean_key = self._strip_known_prefix(key)
            if clean_key.startswith('head.'):
                continue
            filtered[clean_key] = value
        missing, unexpected = self.radar_student.load_state_dict(filtered, strict=False)
        self.radar_student_checkpoint_info = {
            'path': ckpt_path,
            'loaded_keys': len(filtered),
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
        }

    def _points_to_bev_counts(self, points, batch_size, value=None, reduce='sum'):
        dtype = points.dtype
        device = points.device
        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])
        flat_size = batch_size * ny * nx
        out = torch.zeros(flat_size, device=device, dtype=dtype)
        if points.numel() == 0:
            return out.view(batch_size, 1, ny, nx)

        batch_idx = points[:, 0].long()
        xyz = points[:, 1:4]
        coords = torch.floor(
            (xyz[:, [0, 1]] - xyz.new_tensor(self.point_cloud_range[:2])) /
            xyz.new_tensor(self.voxel_size[:2])
        ).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < nx) &
            (coords[:, 1] >= 0) & (coords[:, 1] < ny)
        )
        if not valid.any():
            return out.view(batch_size, 1, ny, nx)
        flat_idx = batch_idx[valid] * (ny * nx) + coords[valid, 1] * nx + coords[valid, 0]
        src = torch.ones_like(flat_idx, dtype=dtype, device=device) if value is None else value[valid].to(dtype)
        if reduce == 'max':
            out = points.new_full((flat_size,), float('-inf'))
            if hasattr(out, 'scatter_reduce_'):
                out.scatter_reduce_(0, flat_idx, src, reduce='amax', include_self=True)
                out[out == float('-inf')] = 0.0
            else:
                out = points.new_zeros(flat_size)
                for idx, val in zip(flat_idx.tolist(), src):
                    out[idx] = torch.maximum(out[idx], val)
        else:
            out.scatter_add_(0, flat_idx, src)
        return out.view(batch_size, 1, ny, nx)

    def _build_lidar_unreliability(self, lidar_points, batch_size, target_hw):
        point_count = self._points_to_bev_counts(lidar_points, batch_size)
        occupied = (point_count > 0).to(point_count.dtype)
        if self.reliability_pool_kernel > 1:
            pad = self.reliability_pool_kernel // 2
            occupancy_ratio = F.avg_pool2d(occupied, self.reliability_pool_kernel, stride=1, padding=pad)
        else:
            occupancy_ratio = occupied
        point_count_score = (point_count / max(self.expected_points_per_cell, self.unreliability_eps)).clamp(0.0, 1.0)
        point_count_unreliability = 1.0 - point_count_score
        occupancy_sparsity = 1.0 - occupancy_ratio.clamp(0.0, 1.0)
        weight_sum = max(self.point_count_weight + self.occupancy_sparsity_weight, self.unreliability_eps)
        u_l = (
            self.point_count_weight * point_count_unreliability +
            self.occupancy_sparsity_weight * occupancy_sparsity
        ) / weight_sum
        u_l = u_l.clamp(0.0, 1.0)
        maps = {
            'lidar_unreliability': u_l,
            'U_L': u_l,
            'lidar_point_count_map': point_count,
            'lidar_occupancy_sparsity': occupancy_sparsity,
        }
        for key, val in list(maps.items()):
            if val.shape[-2:] != target_hw:
                maps[key] = F.interpolate(val.float(), size=target_hw, mode='bilinear', align_corners=False)
        return maps

    def _encode_lidar_to_bev(self, data_dict, batch_size):
        batch_dict = {'batch_size': batch_size, 'points': data_dict['lidar_points']}
        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.lidar_backbone(batch_dict)
        batch_dict = self.lidar_bev_backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def _build_radar_motion_mask(self, radar_points, batch_size, target_hw):
        """Build the radar motion cue while tolerating older WP3 student copies."""
        if hasattr(self.radar_student, '_build_motion_mask'):
            return self.radar_student._build_motion_mask(radar_points, batch_size, target_hw)
        return None

    def _build_radar_rcs_mask(self, radar_points, batch_size, target_hw):
        """Build the RCS cue while tolerating older WP3 student copies.

        Some server checkouts still have an older ``PillarnetRadarStudentKd``
        without ``_build_rcs_mask``.  WP5 fusion should not crash at that
        boundary: use the student's implementation when present, otherwise
        construct the same lightweight BEV RCS prior locally from the raw radar
        points and the student's/YAML RCS settings.
        """
        if hasattr(self.radar_student, '_build_rcs_mask'):
            return self.radar_student._build_rcs_mask(radar_points, batch_size, target_hw)

        student_args = getattr(self.radar_student, 'args', {})
        use_rcs_branch = bool(getattr(self.radar_student, 'use_rcs_branch', student_args.get('use_rcs_branch', False)))
        if not use_rcs_branch:
            return None
        if radar_points.numel() == 0:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))

        rcs_cfg = student_args.get('rcs_mask', {})
        rcs_index = int(getattr(self.radar_student, 'rcs_index', rcs_cfg.get('index', 5)))
        if radar_points.shape[1] <= rcs_index:
            raise ValueError(
                f'RCS cue expects radar_points[:, {rcs_index}], '
                f'but got only {radar_points.shape[1]} columns. Set '
                'preprocess.args.radar_num_point_features and model.args.radar_student_args.radar_num_point_features to 5.'
            )

        rcs_thresh = float(getattr(self.radar_student, 'rcs_thresh', rcs_cfg.get('thresh', 0.0)))
        rcs_mode = getattr(self.radar_student, 'rcs_mode', rcs_cfg.get('mode', 'soft'))
        rcs_temp = float(getattr(self.radar_student, 'rcs_temp', rcs_cfg.get('temp', 1.0)))
        rcs_reduce = getattr(self.radar_student, 'rcs_reduce', rcs_cfg.get('reduce', 'max'))
        rcs_mask_kernel_size = int(getattr(self.radar_student, 'rcs_mask_kernel_size', rcs_cfg.get('kernel_size', 5)))
        rcs_mask_scale = float(getattr(self.radar_student, 'rcs_mask_scale', rcs_cfg.get('scale', 1.0)))

        batch_idx = radar_points[:, 0].long()
        xyz = radar_points[:, 1:4]
        rcs_val = radar_points[:, rcs_index]
        if rcs_mode == 'soft':
            score = torch.sigmoid(rcs_temp * (rcs_val - rcs_thresh))
        else:
            score = (rcs_val > rcs_thresh).to(radar_points.dtype)

        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])
        coords = torch.floor(
            (xyz[:, [0, 1]] - xyz.new_tensor(self.point_cloud_range[:2])) /
            xyz.new_tensor(self.voxel_size[:2])
        ).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < nx) &
            (coords[:, 1] >= 0) & (coords[:, 1] < ny)
        )
        if not valid.any():
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))

        batch_idx = batch_idx[valid]
        coords = coords[valid]
        score = score[valid]
        flat_index = batch_idx * ny * nx + coords[:, 1] * nx + coords[:, 0]
        flat_size = batch_size * ny * nx
        if rcs_reduce == 'mean':
            mask_flat = radar_points.new_zeros(flat_size)
            count_flat = radar_points.new_zeros(flat_size)
            mask_flat.index_add_(0, flat_index, score)
            count_flat.index_add_(0, flat_index, torch.ones_like(score))
            mask_flat = mask_flat / count_flat.clamp_min(1.0)
        else:
            mask_flat = radar_points.new_full((flat_size,), float('-inf'))
            if hasattr(mask_flat, 'scatter_reduce_'):
                mask_flat.scatter_reduce_(0, flat_index, score, reduce='amax', include_self=True)
                mask_flat[mask_flat == float('-inf')] = 0.0
            else:
                mask_flat = radar_points.new_zeros(flat_size)
                for idx, val in zip(flat_index.tolist(), score):
                    mask_flat[idx] = torch.maximum(mask_flat[idx], val)

        mask = mask_flat.view(batch_size, 1, ny, nx)
        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)
        if rcs_mask_kernel_size > 1:
            pad = rcs_mask_kernel_size // 2
            mask = F.max_pool2d(mask, rcs_mask_kernel_size, stride=1, padding=pad)
        mask = mask.clamp_(0.0, 1.0)
        if rcs_mask_scale != 1.0:
            mask = (mask * rcs_mask_scale).clamp_(0.0, 1.0)
        return mask

    def _encode_radar_student_to_bev(self, data_dict, batch_size):
        radar_dict = {
            'batch_size': batch_size,
            'radar_points': data_dict['radar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': False,
        }
        if self.freeze_radar_student:
            self.radar_student.eval()
            with torch.no_grad():
                radar_dict = self.radar_student._encode_radar_to_bev(radar_dict)
        else:
            radar_dict = self.radar_student._encode_radar_to_bev(radar_dict)
        radar_feature = radar_dict['spatial_features_2d']
        kd_motion_mask = self._build_radar_motion_mask(data_dict['radar_points'], batch_size, radar_feature.shape[-2:])
        rcs_confidence_mask = self._build_radar_rcs_mask(data_dict['radar_points'], batch_size, radar_feature.shape[-2:])
        return radar_feature, kd_motion_mask, rcs_confidence_mask

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        gt_boxes = self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask'])

        lidar_feature = self._encode_lidar_to_bev(data_dict, batch_size)
        radar_feature, kd_motion_mask, rcs_confidence_mask = self._encode_radar_student_to_bev(data_dict, batch_size)
        target_hw = lidar_feature.shape[-2:]
        if radar_feature.shape[-2:] != target_hw:
            radar_feature = F.interpolate(radar_feature, size=target_hw, mode='bilinear', align_corners=False)
        if kd_motion_mask is not None and kd_motion_mask.shape[-2:] != target_hw:
            kd_motion_mask = F.interpolate(kd_motion_mask.float(), size=target_hw, mode='nearest')
        if rcs_confidence_mask is not None and rcs_confidence_mask.shape[-2:] != target_hw:
            rcs_confidence_mask = F.interpolate(rcs_confidence_mask.float(), size=target_hw, mode='bilinear', align_corners=False)

        reliability_maps = self._build_lidar_unreliability(data_dict['lidar_points'], batch_size, target_hw)
        fused_feature, lidar_gate, radar_gate, learned_lidar_gate = self.fusion_gate(
            lidar_feature,
            radar_feature,
            reliability_maps['U_L'],
            radar_motion_mask=kd_motion_mask,
            rcs_confidence_mask=rcs_confidence_mask,
        )

        head_dict = {
            'batch_size': batch_size,
            'spatial_features_2d': fused_feature,
            'gt_boxes': gt_boxes,
            'compute_loss': compute_loss,
        }
        head_dict = self.head(head_dict)

        output_dict = {
            self.feature_key: fused_feature,
            'fused_feature': fused_feature,
            'lidar_feature': lidar_feature,
            'radar_feature': radar_feature,
            'lidar_unreliability': reliability_maps['U_L'],
            'U_L': reliability_maps['U_L'],
            'lidar_point_count_map': reliability_maps['lidar_point_count_map'],
            'lidar_occupancy_sparsity': reliability_maps['lidar_occupancy_sparsity'],
            'fusion_gate': lidar_gate,
            'lidar_gate': lidar_gate,
            'radar_gate': radar_gate,
            'learned_lidar_gate': learned_lidar_gate,
            'kd_motion_mask': kd_motion_mask,
            'rcs_confidence_mask': rcs_confidence_mask,
            'final_box_dict': self._select_final_box_dict(head_dict),
        }
        if 'batch_cls_preds' in head_dict:
            output_dict['cls_preds'] = head_dict['batch_cls_preds']
        if 'batch_box_preds' in head_dict:
            output_dict['reg_preds'] = head_dict['batch_box_preds']
        if hasattr(self, 'radar_student_checkpoint_info'):
            output_dict['radar_student_checkpoint_info'] = self.radar_student_checkpoint_info

        if self.training or compute_loss:
            head_loss, head_tb = self.head.get_loss()
            total_loss = self.loss_weight * head_loss
            output_dict.update({
                'loss': total_loss,
                'fusion_head_loss': head_loss,
                'tb_dict': {
                    'total_loss': total_loss.item(),
                    'fusion_head_loss': head_loss.item(),
                    **head_tb,
                },
            })
        return output_dict