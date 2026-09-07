import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.guided_pfd import GuidedPFD
from opencood.pcdet_utils.basicblock.modules.Basicblock_convn import ConvNeXtBlock


def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


class RadarDistill(BaseBEVBackboneV2):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.point_cloud_range = model_cfg.get('point_cloud_range', None)
        self.voxel_size = model_cfg.get('voxel_size', None)
        self.grid_size = model_cfg.get('grid_size', None)
        self.kd_mode = model_cfg.get('kd_mode', 'radardistill')
        self.guided_pfd = GuidedPFD(
            model_cfg.get('wp3_guided_pfd', {}),
            model_cfg['class_names'],
            model_cfg['class_names_each_head'],
            self.point_cloud_range,
            self.voxel_size,
            feature_map_stride=model_cfg.get('feature_map_stride', 8),
            gaussian_overlap=model_cfg.get('gaussian_overlap', 0.1),
            min_radius=model_cfg.get('min_radius', 2),
        )
        self.guided_pfd_last_output = None
        evidence_cfg = model_cfg.get('radar_evidence_mask', {})
        self.use_radar_evidence_mask = bool(evidence_cfg.get('enabled', False))
        self.evidence_apply_to = evidence_cfg.get('apply_to', 'high')
        self.evidence_doppler_index = int(evidence_cfg.get('doppler_index', 4))
        self.evidence_rcs_index = int(evidence_cfg.get('rcs_index', 5))
        self.evidence_doppler_center = float(evidence_cfg.get('doppler_center', 5.0))
        self.evidence_doppler_scale = max(float(evidence_cfg.get('doppler_scale', 3.0)), 1e-6)
        self.evidence_rcs_center = float(evidence_cfg.get('rcs_center', -10.0))
        self.evidence_rcs_scale = max(float(evidence_cfg.get('rcs_scale', 4.0)), 1e-6)
        self.evidence_rcs_alpha = float(evidence_cfg.get('rcs_alpha', 0.25))
        self.evidence_combine = evidence_cfg.get('combine', 'max')
        self.evidence_reduce = evidence_cfg.get('reduce', 'max')
        self.evidence_kernel_size = int(evidence_cfg.get('kernel_size', 3))
        self.evidence_rcs_aware_dilation = bool(evidence_cfg.get('rcs_aware_dilation', False))
        self.evidence_rcs_mid_threshold = float(evidence_cfg.get('rcs_mid_threshold', 0.5))
        self.evidence_rcs_high_threshold = float(evidence_cfg.get('rcs_high_threshold', 0.75))
        self.evidence_rcs_mid_kernel_size = int(evidence_cfg.get('rcs_mid_kernel_size', 3))
        self.evidence_rcs_high_kernel_size = int(evidence_cfg.get('rcs_high_kernel_size', 5))
        self.evidence_mask_floor = float(evidence_cfg.get('mask_floor', 0.25))
        self.evidence_weight_boost = float(evidence_cfg.get('weight_boost', 1.0))
        self.evidence_include_teacher = bool(evidence_cfg.get('include_teacher_heatmap', True))
        self.evidence_preserve_weight_sum = bool(evidence_cfg.get('preserve_weight_sum', True))
        self.evidence_use_lidar_intensity = bool(evidence_cfg.get('use_lidar_intensity', False))
        self.evidence_lidar_intensity_index = int(evidence_cfg.get('lidar_intensity_index', 4))
        self.evidence_lidar_intensity_percentile = float(evidence_cfg.get('lidar_intensity_percentile', 0.95))
        self.evidence_lidar_intensity_kernel_size = int(evidence_cfg.get('lidar_intensity_kernel_size', 3))
        self.evidence_fusion = evidence_cfg.get('fusion', 'max')
        object_kd_cfg = model_cfg.get('object_kd', {})
        self.afd_loss_weight = float(model_cfg.get('afd_loss_weight', 5.0))
        self.pfd_loss_weight = float(model_cfg.get('pfd_loss_weight', 25.0))
        self.masked_afd_feature_weight = float(model_cfg.get('masked_afd_feature_weight', 1.0))
        self.masked_afd_mask_weight = float(model_cfg.get('masked_afd_mask_weight', 1.0))
        task_dense_cfg = model_cfg.get('task_dense_kd', {})
        self.use_task_dense_kd = bool(task_dense_cfg.get('enabled', False))
        self.task_dense_feature_weight = float(task_dense_cfg.get('feature_weight', 1.0))
        self.task_dense_teacher_threshold = float(task_dense_cfg.get('teacher_threshold', 0.10))
        self.task_dense_teacher_floor = float(task_dense_cfg.get('teacher_floor', 0.20))
        self.task_dense_radar_floor = float(task_dense_cfg.get('radar_floor', 0.50))
        self.task_dense_use_teacher_confidence = bool(task_dense_cfg.get('use_teacher_confidence', True))
        self.task_dense_use_lidar_intensity = bool(task_dense_cfg.get('use_lidar_intensity', True))
        self.task_dense_use_radar_support = bool(task_dense_cfg.get('use_radar_support', True))
        self.task_dense_min_mask_sum = float(task_dense_cfg.get('min_mask_sum', 1.0))
        learned_mask_cfg = model_cfg.get('learned_reliability_mask', {})
        self.use_learned_reliability_mask = bool(learned_mask_cfg.get('enabled', False))
        self.learned_mask_loss_weight = float(learned_mask_cfg.get('loss_weight', 1.0))
        self.learned_mask_kd_floor = float(learned_mask_cfg.get('kd_floor', 0.05))
        self.learned_mask_teacher_threshold = float(learned_mask_cfg.get('teacher_threshold', self.task_dense_teacher_threshold))
        self.learned_mask_neg_weight = float(learned_mask_cfg.get('neg_weight', 0.25))
        self.learned_mask_use_radar_prior = bool(learned_mask_cfg.get('use_radar_prior', True))
        self.learned_mask_use_lidar_prior = bool(learned_mask_cfg.get('use_lidar_prior', True))
        self.learned_mask_detach_for_kd = bool(learned_mask_cfg.get('detach_for_kd', False))
        self.use_object_kd = bool(object_kd_cfg.get('enabled', False))
        self.object_kd_weight = float(object_kd_cfg.get('loss_weight', 25.0))
        self.object_kd_metric = object_kd_cfg.get('metric', 'l2_norm')
        self.object_kd_foreground_source = object_kd_cfg.get('foreground_source', 'heatmap')
        self.object_kd_box_scale = float(object_kd_cfg.get('box_scale', 1.0))
        self.object_kd_box_dim_order = object_kd_cfg.get('box_dim_order', 'hwl')
        self.object_kd_box_value = object_kd_cfg.get('box_value', 'flat')
        self.object_kd_box_soft_floor = float(object_kd_cfg.get('box_soft_floor', 0.25))
        self.object_kd_teacher_floor = float(object_kd_cfg.get('teacher_floor', 0.5))
        self.object_kd_radar_floor = float(object_kd_cfg.get('radar_floor', 0.25))
        self.object_kd_use_lidar_intensity = bool(object_kd_cfg.get('use_lidar_intensity', True))
        self.object_kd_use_radar_support = bool(object_kd_cfg.get('use_radar_support', True))
        self.object_kd_min_mask_sum = float(object_kd_cfg.get('min_mask_sum', 1.0))
        proto_cfg = model_cfg.get('object_proto_kd', {})
        self.use_object_proto_kd = bool(proto_cfg.get('enabled', False))
        self.object_proto_kd_weight = float(proto_cfg.get('loss_weight', 1.0))
        self.object_proto_top_fraction = float(proto_cfg.get('top_fraction', 0.20))
        self.object_proto_min_box_weight = float(proto_cfg.get('min_box_weight', 1.0))
        self.object_proto_context_scale = float(proto_cfg.get('context_box_scale', self.object_kd_box_scale))
        self.object_proto_radar_floor = float(proto_cfg.get('radar_floor', 0.05))
        self.object_proto_teacher_floor = float(proto_cfg.get('teacher_floor', 0.50))
        self.object_proto_weight_floor = float(proto_cfg.get('weight_floor', 0.30))
        self.object_proto_use_lidar_intensity = bool(proto_cfg.get('use_lidar_intensity', True))
        self.object_proto_use_radar_support = bool(proto_cfg.get('use_radar_support', True))

        semantic_cfg = model_cfg.get('semantic_heatmap_kd', {})
        self.use_semantic_heatmap_kd = bool(semantic_cfg.get('enabled', False))
        self.semantic_heatmap_kd_weight = float(semantic_cfg.get('loss_weight', 1.0))
        self.semantic_heatmap_loss_type = semantic_cfg.get('loss_type', 'mse')
        self.semantic_heatmap_weight_source = semantic_cfg.get('weight_source', 'radar_support')
        self.semantic_heatmap_temperature = max(float(semantic_cfg.get('temperature', 1.0)), 1e-6)
        self.semantic_heatmap_floor = float(semantic_cfg.get('floor', 0.50))
        self.semantic_heatmap_teacher_floor = float(semantic_cfg.get('teacher_floor', self.semantic_heatmap_floor))
        self.semantic_heatmap_radar_boost = float(semantic_cfg.get('radar_boost', 0.0))
        self.semantic_heatmap_use_radar_support = bool(semantic_cfg.get('use_radar_support', True))
        response_cfg = model_cfg.get('response_kd', {})
        self.use_response_kd = bool(response_cfg.get('enabled', False))
        self.response_pos_weight = float(response_cfg.get('pos_weight', 2.0))
        self.response_neg_weight = float(response_cfg.get('neg_weight', 1.0))
        self.response_teacher_threshold = float(response_cfg.get('teacher_threshold', 0.10))
        self.response_teacher_bg_threshold = float(response_cfg.get('teacher_bg_threshold', 0.05))
        self.response_student_fp_threshold = float(response_cfg.get('student_fp_threshold', 0.10))
        self.response_gt_bg_threshold = float(response_cfg.get('gt_bg_threshold', 0.05))
        self.response_teacher_floor = float(response_cfg.get('teacher_floor', 0.20))
        self.response_neg_radar_floor = float(response_cfg.get('neg_radar_floor', 0.25))
        self.response_max_fp_cells_per_sample = int(response_cfg.get('max_fp_cells_per_sample', 256))
        self.response_temperature = max(float(response_cfg.get('temperature', 1.0)), 1e-6)
        proposal_cfg = model_cfg.get('proposal_error_kd', {})
        self.use_proposal_error_kd = bool(proposal_cfg.get('enabled', False))
        self.proposal_fp_weight = float(proposal_cfg.get('fp_weight', 0.1))
        self.proposal_fn_boost = float(proposal_cfg.get('fn_boost', 1.0))
        self.proposal_fp_student_threshold = float(proposal_cfg.get('fp_student_threshold', 0.10))
        self.proposal_fp_teacher_threshold = float(proposal_cfg.get('fp_teacher_threshold', 0.05))
        self.proposal_fn_student_threshold = float(proposal_cfg.get('fn_student_threshold', 0.10))
        self.proposal_fn_teacher_threshold = float(proposal_cfg.get('fn_teacher_threshold', 0.10))
        self.proposal_gt_background_threshold = float(proposal_cfg.get('gt_background_threshold', 0.05))
        self.proposal_gt_positive_threshold = float(proposal_cfg.get('gt_positive_threshold', 0.10))
        self.proposal_observability_floor_fp = float(proposal_cfg.get('observability_floor_fp', 0.20))
        self.proposal_observability_floor_fn = float(proposal_cfg.get('observability_floor_fn', 0.40))
        self.proposal_max_fp_cells_per_sample = int(proposal_cfg.get('max_fp_cells_per_sample', 128))
        # NOTE: These encoder/decoder/aggregation blocks are the OpenCOOD
        # implementation of the paper's CMA step, which densifies radar BEV
        # features before the higher-level distillation losses are applied.
        self.encoder_1 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())

        self.encoder_2 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_2 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())

        self.encoder_3 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_3 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        self.reliability_mask_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.debug_maps = {}
        self.last_loss_terms = {}

    def _build_radar_evidence_mask(self, batch_dict, target_hw):
        if not self.use_radar_evidence_mask:
            return None
        radar_points = batch_dict.get('radar_points', None)
        if radar_points is None:
            return None
        batch_size = int(batch_dict['batch_size'])
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if radar_points.numel() == 0:
            return radar_points.new_zeros((batch_size, 1, target_h, target_w))
        if radar_points.shape[1] <= max(self.evidence_doppler_index, self.evidence_rcs_index):
            raise ValueError(
                'RadarDistill evidence mask expects collated radar_points with '
                f'Doppler index {self.evidence_doppler_index} and RCS index {self.evidence_rcs_index}, '
                f'but got {radar_points.shape[1]} columns.'
            )
        if self.point_cloud_range is None or self.grid_size is None:
            raise ValueError('RadarDistill evidence mask needs point_cloud_range and grid_size in model config.')

        batch_idx = radar_points[:, 0].long()
        xyz = radar_points[:, 1:4]
        doppler = radar_points[:, self.evidence_doppler_index].abs()
        rcs_val = radar_points[:, self.evidence_rcs_index]

        doppler_score = torch.sigmoid(
            (doppler - self.evidence_doppler_center) / self.evidence_doppler_scale
        )
        rcs_confidence = torch.sigmoid(
            (rcs_val - self.evidence_rcs_center) / self.evidence_rcs_scale
        )
        rcs_score = rcs_confidence * self.evidence_rcs_alpha

        if self.evidence_combine == 'sum':
            score = (doppler_score + rcs_score).clamp(0.0, 1.0)
        elif self.evidence_combine == 'mean':
            score = ((doppler_score + rcs_score) * 0.5).clamp(0.0, 1.0)
        elif self.evidence_combine == 'or':
            score = (1.0 - (1.0 - doppler_score) * (1.0 - rcs_score)).clamp(0.0, 1.0)
        else:
            score = torch.maximum(doppler_score, rcs_score).clamp(0.0, 1.0)

        pc_range = xyz.new_tensor(self.point_cloud_range)
        voxel_size = xyz.new_tensor(self.voxel_size)
        coords = torch.floor((xyz[:, [0, 1]] - pc_range[:2]) / voxel_size[:2]).long()
        grid_x, grid_y = int(self.grid_size[0]), int(self.grid_size[1])
        target_x = torch.floor(coords[:, 0].float() * target_w / float(grid_x)).long()
        target_y = torch.floor(coords[:, 1].float() * target_h / float(grid_y)).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < grid_x) &
            (coords[:, 1] >= 0) & (coords[:, 1] < grid_y) &
            (target_x >= 0) & (target_x < target_w) &
            (target_y >= 0) & (target_y < target_h)
        )
        if not valid.any():
            return radar_points.new_zeros((batch_size, 1, target_h, target_w))

        batch_idx = batch_idx[valid]
        target_x = target_x[valid]
        target_y = target_y[valid]
        score = score[valid]
        rcs_confidence = rcs_confidence[valid]
        flat_index = batch_idx * target_h * target_w + target_y * target_w + target_x
        flat_size = batch_size * target_h * target_w
        mask = self._scatter_points_to_bev(score, flat_index, flat_size, batch_size, target_h, target_w)
        doppler_mask = self._scatter_points_to_bev(doppler_score[valid], flat_index, flat_size, batch_size, target_h, target_w)
        rcs_mask = self._scatter_points_to_bev(rcs_confidence, flat_index, flat_size, batch_size, target_h, target_w)
        self.debug_maps['radar_doppler_score_mask'] = doppler_mask.detach()
        self.debug_maps['radar_rcs_confidence_mask'] = rcs_mask.detach()

        if self.evidence_rcs_aware_dilation:
            mid_score = score * (rcs_confidence >= self.evidence_rcs_mid_threshold).to(score.dtype)
            high_score = score * (rcs_confidence >= self.evidence_rcs_high_threshold).to(score.dtype)
            mid_mask = self._scatter_points_to_bev(mid_score, flat_index, flat_size, batch_size, target_h, target_w)
            high_mask = self._scatter_points_to_bev(high_score, flat_index, flat_size, batch_size, target_h, target_w)
            if self.evidence_kernel_size > 1:
                mask = self._max_pool_mask(mask, self.evidence_kernel_size)
            if self.evidence_rcs_mid_kernel_size > 1:
                mid_mask = self._max_pool_mask(mid_mask, self.evidence_rcs_mid_kernel_size)
                self.debug_maps['radar_rcs_mid_support_mask'] = mid_mask.detach()
                mask = torch.maximum(mask, mid_mask)
            if self.evidence_rcs_high_kernel_size > 1:
                high_mask = self._max_pool_mask(high_mask, self.evidence_rcs_high_kernel_size)
                self.debug_maps['radar_rcs_high_support_mask'] = high_mask.detach()
                mask = torch.maximum(mask, high_mask)
        elif self.evidence_kernel_size > 1:
            mask = self._max_pool_mask(mask, self.evidence_kernel_size)
        return mask.clamp_(0.0, 1.0)

    def _scatter_points_to_bev(self, score, flat_index, flat_size, batch_size, target_h, target_w):
        if self.evidence_reduce == 'mean':
            mask_flat = score.new_zeros(flat_size)
            count_flat = score.new_zeros(flat_size)
            mask_flat.index_add_(0, flat_index, score)
            count_flat.index_add_(0, flat_index, torch.ones_like(score))
            mask_flat = mask_flat / count_flat.clamp_min(1.0)
        else:
            mask_flat = score.new_full((flat_size,), float('-inf'))
            if hasattr(mask_flat, 'scatter_reduce_'):
                mask_flat.scatter_reduce_(0, flat_index, score, reduce='amax', include_self=True)
                mask_flat[mask_flat == float('-inf')] = 0.0
            else:
                mask_flat = score.new_zeros(flat_size)
                for idx, val in zip(flat_index.tolist(), score):
                    mask_flat[idx] = torch.maximum(mask_flat[idx], val)
        return mask_flat.view(batch_size, 1, target_h, target_w)

    @staticmethod
    def _max_pool_mask(mask, kernel_size):
        pad = kernel_size // 2
        return F.max_pool2d(mask, kernel_size, stride=1, padding=pad)

    @staticmethod
    def _feature_magnitude(feature):
        return torch.sqrt(torch.mean(feature.detach().float() ** 2, dim=1, keepdim=True) + 1e-8)

    def _build_lidar_intensity_mask(self, batch_dict, target_hw):
        if not self.evidence_use_lidar_intensity:
            return None
        points = batch_dict.get('points', None)
        if points is None:
            return None
        batch_size = int(batch_dict['batch_size'])
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if points.numel() == 0:
            return points.new_zeros((batch_size, 1, target_h, target_w))
        if points.shape[1] <= self.evidence_lidar_intensity_index:
            raise ValueError(
                'LiDAR intensity mask expects collated lidar points with '
                f'intensity at index {self.evidence_lidar_intensity_index}, '
                f'but got {points.shape[1]} columns.'
            )
        if self.point_cloud_range is None or self.grid_size is None:
            raise ValueError('LiDAR intensity mask needs point_cloud_range and grid_size in model config.')

        batch_idx = points[:, 0].long()
        xyz = points[:, 1:4]
        intensity = points[:, self.evidence_lidar_intensity_index].clamp_min(0.0)
        positive = intensity[intensity > 0]
        if positive.numel() > 0:
            scale = torch.quantile(
                positive.float(),
                min(max(self.evidence_lidar_intensity_percentile, 1e-3), 1.0),
            ).to(intensity.dtype).clamp_min(1e-6)
            score = (intensity / scale).clamp(0.0, 1.0)
        else:
            score = intensity.new_zeros(intensity.shape)

        pc_range = xyz.new_tensor(self.point_cloud_range)
        voxel_size = xyz.new_tensor(self.voxel_size)
        coords = torch.floor((xyz[:, [0, 1]] - pc_range[:2]) / voxel_size[:2]).long()
        grid_x, grid_y = int(self.grid_size[0]), int(self.grid_size[1])
        target_x = torch.floor(coords[:, 0].float() * target_w / float(grid_x)).long()
        target_y = torch.floor(coords[:, 1].float() * target_h / float(grid_y)).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < grid_x) &
            (coords[:, 1] >= 0) & (coords[:, 1] < grid_y) &
            (target_x >= 0) & (target_x < target_w) &
            (target_y >= 0) & (target_y < target_h)
        )
        if not valid.any():
            return points.new_zeros((batch_size, 1, target_h, target_w))

        batch_idx = batch_idx[valid]
        target_x = target_x[valid]
        target_y = target_y[valid]
        score = score[valid]
        flat_index = batch_idx * target_h * target_w + target_y * target_w + target_x
        flat_size = batch_size * target_h * target_w
        mask_flat = points.new_full((flat_size,), float('-inf'))
        if hasattr(mask_flat, 'scatter_reduce_'):
            mask_flat.scatter_reduce_(0, flat_index, score, reduce='amax', include_self=True)
            mask_flat[mask_flat == float('-inf')] = 0.0
        else:
            mask_flat = points.new_zeros(flat_size)
            for idx, val in zip(flat_index.tolist(), score):
                mask_flat[idx] = torch.maximum(mask_flat[idx], val)

        mask = mask_flat.view(batch_size, 1, target_h, target_w)
        if self.evidence_lidar_intensity_kernel_size > 1:
            pad = self.evidence_lidar_intensity_kernel_size // 2
            mask = F.max_pool2d(mask, self.evidence_lidar_intensity_kernel_size, stride=1, padding=pad)
        return mask.clamp_(0.0, 1.0)

    def low_loss(self, lidar_bev, radar_bev, debug_prefix='low'):
        # NOTE: This is the AFD part of RadarDistill. It builds activation-aware
        # masks from LiDAR and radar BEV features, then distills low-level radar
        # features toward LiDAR features with separate weighting for overlap and
        # radar-only regions.
        batch_size = radar_bev.shape[0]
        lidar_mask = (lidar_bev.sum(1, keepdim=True) > 0).float()
        radar_mask = radar_bev.sum(1, keepdim=True)
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5

        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float32)
        mask_radar_only = torch.zeros_like(activate_map, dtype=torch.float32)
        mask_radar_lidar[activate_map == 1.5] = 1
        mask_radar_only[activate_map == 1.0] = 1
        self.debug_maps[f'{debug_prefix}_afd_lidar_active_mask'] = lidar_mask.detach()
        self.debug_maps[f'{debug_prefix}_afd_radar_active_mask'] = (radar_mask > 0).float().detach()
        self.debug_maps[f'{debug_prefix}_afd_overlap_mask'] = mask_radar_lidar.detach()
        self.debug_maps[f'{debug_prefix}_afd_radar_only_mask'] = mask_radar_only.detach()

        if mask_radar_only.sum() > 0:
            mask_radar_only *= mask_radar_lidar.sum().clamp(min=1.0) / mask_radar_only.sum()

        loss_overlap = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_overlap = torch.sum(loss_overlap * mask_radar_lidar) / batch_size
        loss_radar_only = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_only = torch.sum(loss_radar_only * mask_radar_only) / batch_size

        feature_loss = 3e-4 * loss_overlap + 5e-5 * loss_radar_only
        mask_loss = nn.L1Loss()(radar_mask.sigmoid(), lidar_mask)
        return feature_loss, mask_loss

    def masked_low_loss(
            self,
            lidar_bev,
            radar_bev,
            heatmaps,
            radar_evidence_mask=None,
            lidar_intensity_mask=None,
            debug_prefix='masked_low'):
        # Normalized dense AFD variant. It keeps RadarDistill's active/inactive
        # region idea, but weights the dense feature loss with physical masks and
        # normalizes by the active weight sum instead of using a large raw sum.
        batch_size, channels = radar_bev.shape[:2]
        target_hw = radar_bev.shape[-2:]
        lidar_mask = (lidar_bev.sum(1, keepdim=True) > 0).float()
        radar_activation = radar_bev.sum(1, keepdim=True)
        radar_mask = (radar_activation > 0).float()
        activate_map = radar_mask + lidar_mask * 0.5

        overlap = torch.zeros_like(activate_map, dtype=torch.float32)
        radar_only = torch.zeros_like(activate_map, dtype=torch.float32)
        overlap[activate_map == 1.5] = 1.0
        radar_only[activate_map == 1.0] = 1.0
        if radar_only.sum() > 0:
            radar_only = radar_only * (overlap.sum().clamp_min(1.0) / radar_only.sum().clamp_min(1.0))
        base_weight = overlap + 0.25 * radar_only

        final_mask = None
        masks = []
        if radar_evidence_mask is not None:
            radar_evidence_mask = radar_evidence_mask.clamp(0.0, 1.0)
            if tuple(radar_evidence_mask.shape[-2:]) != tuple(target_hw):
                radar_evidence_mask = F.interpolate(
                    radar_evidence_mask,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            masks.append(radar_evidence_mask)
        if lidar_intensity_mask is not None:
            lidar_intensity_mask = lidar_intensity_mask.clamp(0.0, 1.0)
            if tuple(lidar_intensity_mask.shape[-2:]) != tuple(target_hw):
                lidar_intensity_mask = F.interpolate(
                    lidar_intensity_mask,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            masks.append(lidar_intensity_mask)
        if self.evidence_include_teacher:
            gt_batch_hm = torch.cat(heatmaps, dim=1)
            teacher_mask = torch.max(gt_batch_hm, dim=1, keepdim=True)[0].clamp(0.0, 1.0)
            if tuple(teacher_mask.shape[-2:]) != tuple(target_hw):
                teacher_mask = F.interpolate(
                    teacher_mask,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            masks.append(teacher_mask)
        if masks:
            final_mask = masks[0]
            for cur_mask in masks[1:]:
                final_mask = torch.maximum(final_mask, cur_mask)
            weight_before = base_weight.sum().detach()
            base_weight = base_weight * (self.evidence_mask_floor + self.evidence_weight_boost * final_mask)
            if self.evidence_preserve_weight_sum and weight_before > 0:
                base_weight = base_weight * (weight_before / base_weight.sum().clamp_min(1e-6))

        mse = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        feature_loss = (mse * base_weight).sum() / (base_weight.sum().clamp_min(1.0) * float(channels))
        mask_dist = F.l1_loss(radar_activation.sigmoid(), lidar_mask, reduction='none')
        mask_loss = (mask_dist * base_weight).sum() / base_weight.sum().clamp_min(1.0)

        self.debug_maps[f'{debug_prefix}_afd_lidar_active_mask'] = lidar_mask.detach()
        self.debug_maps[f'{debug_prefix}_afd_radar_active_mask'] = radar_mask.detach()
        self.debug_maps[f'{debug_prefix}_afd_overlap_mask'] = overlap.detach()
        self.debug_maps[f'{debug_prefix}_afd_radar_only_mask'] = radar_only.detach()
        self.debug_maps[f'{debug_prefix}_afd_weight_mask'] = base_weight.detach()
        if final_mask is not None:
            self.debug_maps[f'{debug_prefix}_physical_mask'] = final_mask.detach()
        return feature_loss, mask_loss

    def high_loss(
            self,
            radar_bev,
            radar_bev_8x,
            lidar_bev,
            lidar_bev_8x,
            heatmaps,
            radar_preds,
            teacher_preds,
            radar_points,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        # NOTE: This is the PFD part of RadarDistill. It uses proposal/heatmap
        # signals to emphasize teacher-student matching around likely objects
        # and hard proposal regions instead of treating all BEV locations equally.
        gt_batch_hm = torch.cat(heatmaps, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]
        radar_batch_hm = [clip_sigmoid(pred_dict['hm']) for pred_dict in radar_preds]
        radar_batch_hm = torch.cat(radar_batch_hm, dim=1)
        radar_batch_hm_max = torch.max(radar_batch_hm, dim=1, keepdim=True)[0]

        radar_fp_mask = torch.logical_and(gt_batch_hm_max < 0.1, radar_batch_hm_max > 0.1)
        radar_fn_mask = torch.logical_and(gt_batch_hm_max > 0.1, radar_batch_hm_max < 0.1)
        radar_tp_mask = torch.logical_and(gt_batch_hm_max > 0.1, radar_batch_hm_max > 0.1)
        weight = torch.zeros_like(radar_batch_hm_max)
        tp_fn = (radar_tp_mask + radar_fn_mask).sum()
        fp = radar_fp_mask.sum()
        if tp_fn > 0:
            weight[radar_tp_mask + radar_fn_mask] = 5 / tp_fn
        if fp > 0:
            weight[radar_fp_mask] = 1 / fp
        self.debug_maps['pfd_teacher_heatmap_mask'] = gt_batch_hm_max.detach()
        self.debug_maps['pfd_radar_heatmap_mask'] = radar_batch_hm_max.detach()
        self.debug_maps['pfd_tp_mask'] = radar_tp_mask.float().detach()
        self.debug_maps['pfd_fn_mask'] = radar_fn_mask.float().detach()
        self.debug_maps['pfd_fp_mask'] = radar_fp_mask.float().detach()
        self.debug_maps['pfd_base_weight'] = weight.detach()

        radar_evidence_mean = None
        lidar_intensity_mean = None
        final_mask_mean = None
        if (radar_evidence_mask is not None or lidar_intensity_mask is not None) and self.evidence_apply_to in {'high', 'both'}:
            if radar_evidence_mask is not None and tuple(radar_evidence_mask.shape[-2:]) != tuple(weight.shape[-2:]):
                radar_evidence_mask = F.interpolate(
                    radar_evidence_mask,
                    size=weight.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if lidar_intensity_mask is not None and tuple(lidar_intensity_mask.shape[-2:]) != tuple(weight.shape[-2:]):
                lidar_intensity_mask = F.interpolate(
                    lidar_intensity_mask,
                    size=weight.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            radar_mask = None if radar_evidence_mask is None else radar_evidence_mask.clamp(0.0, 1.0)
            lidar_mask = None if lidar_intensity_mask is None else lidar_intensity_mask.clamp(0.0, 1.0)
            teacher_mask = gt_batch_hm_max.clamp(0.0, 1.0)

            if self.evidence_fusion == 'teacher_intensity_or_radar':
                if lidar_mask is not None:
                    final_mask = teacher_mask * (0.5 + 0.5 * lidar_mask)
                else:
                    final_mask = teacher_mask
                if radar_mask is not None:
                    final_mask = torch.maximum(final_mask, radar_mask)
            else:
                masks = []
                if radar_mask is not None:
                    masks.append(radar_mask)
                if lidar_mask is not None:
                    masks.append(lidar_mask)
                if self.evidence_include_teacher:
                    masks.append(teacher_mask)
                final_mask = masks[0]
                for cur_mask in masks[1:]:
                    final_mask = torch.maximum(final_mask, cur_mask)

            weight_before = weight.sum().detach()
            multiplier = self.evidence_mask_floor + self.evidence_weight_boost * final_mask
            weight = weight * multiplier
            if self.evidence_preserve_weight_sum and weight_before > 0:
                weight = weight * (weight_before / weight.sum().clamp_min(1e-6))

            radar_evidence_mean = None if radar_mask is None else radar_mask.mean()
            lidar_intensity_mean = None if lidar_mask is None else lidar_mask.mean()
            final_mask_mean = final_mask.mean()
            self.debug_maps['distill_final_mask'] = final_mask.detach()
            self.debug_maps['pfd_weight_after_evidence'] = weight.detach()

        self.debug_maps['pfd_native_weight'] = weight.detach()
        guided_output = self.guided_pfd(
            weight, teacher_preds, radar_preds, radar_points, gt_boxes)
        self.guided_pfd_last_output = guided_output
        weight = guided_output['guided_weight']
        self.debug_maps['pfd_guided_weight'] = weight.detach()
        self.debug_maps['guided_car_object_mask'] = guided_output['car_object_mask']
        self.debug_maps['guided_teacher_reliability_map'] = guided_output['teacher_reliability_map']
        self.debug_maps['guided_radar_evidence_map'] = guided_output['radar_evidence_map']
        self.debug_maps['guided_combined_q_map'] = guided_output['combined_q_map']
        self.debug_maps['guided_pfd_difference'] = (
            weight - self.debug_maps['pfd_native_weight']).detach()

        scaled_radar = radar_bev.softmax(1)
        scaled_lidar = lidar_bev.softmax(1)
        scaled_radar_8x = radar_bev_8x.softmax(1)
        scaled_lidar_8x = lidar_bev_8x.softmax(1)

        high_loss = (F.l1_loss(scaled_radar, scaled_lidar, reduction='none') * weight).sum()
        high_loss_8x = (F.l1_loss(scaled_radar_8x, scaled_lidar_8x, reduction='none') * weight).sum()
        tb_dict = {}
        if radar_evidence_mean is not None:
            tb_dict['radar_evidence_mask_mean'] = radar_evidence_mean.item()
        if lidar_intensity_mean is not None:
            tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mean.item()
        if final_mask_mean is not None:
            tb_dict['distill_final_mask_mean'] = final_mask_mean.item()
        tb_dict['guided_pfd_total_budget_error'] = torch.max(torch.abs(
            guided_output['guided_total_mass'] - guided_output['native_total_mass'])).item()
        tb_dict['guided_pfd_car_budget_error'] = torch.max(torch.abs(
            guided_output['guided_car_mass'] - guided_output['native_car_mass'])).item()
        tb_dict['guided_pfd_rest_budget_error'] = torch.max(torch.abs(
            guided_output['guided_rest_mass'] - guided_output['native_rest_mass'])).item()
        if self.guided_pfd.enabled and self.guided_pfd.variant == 'opportunity':
            car_objects = (
                guided_output['object_class_index'] ==
                self.guided_pfd.car_class_index
            )
            opportunity = guided_output['opportunity'][car_objects]
            teacher_score = guided_output['teacher_score'][car_objects]
            student_score = guided_output['student_score'][car_objects]
            if opportunity.numel() > 0:
                quantiles = torch.quantile(
                    opportunity.float(),
                    opportunity.new_tensor([0.10, 0.90]).float(),
                )
                tb_dict.update({
                    'opportunity_mean': opportunity.mean().item(),
                    'opportunity_min': opportunity.min().item(),
                    'opportunity_max': opportunity.max().item(),
                    'opportunity_p10': quantiles[0].item(),
                    'opportunity_p90': quantiles[1].item(),
                    'opportunity_teacher_score_mean': teacher_score.mean().item(),
                    'opportunity_student_score_mean': student_score.mean().item(),
                    'opportunity_count': int(opportunity.numel()),
                })
            else:
                # Zero-Car batches remain finite and produce explicit neutral
                # telemetry instead of NaN TensorBoard values.
                tb_dict.update({
                    'opportunity_mean': 0.0,
                    'opportunity_min': 0.0,
                    'opportunity_max': 0.0,
                    'opportunity_p10': 0.0,
                    'opportunity_p90': 0.0,
                    'opportunity_teacher_score_mean': 0.0,
                    'opportunity_student_score_mean': 0.0,
                    'opportunity_count': 0,
                })
            tb_dict['opportunity_car_budget_error'] = tb_dict[
                'guided_pfd_car_budget_error']
        return 0.5 * (high_loss + high_loss_8x), tb_dict

    def object_kd_loss(
            self,
            radar_bev,
            radar_bev_8x,
            lidar_bev,
            lidar_bev_8x,
            heatmaps,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        if not self.use_object_kd:
            return radar_bev.new_tensor(0.0), {}

        gt_batch_hm = torch.cat(heatmaps, dim=1)
        heatmap_mask = torch.max(gt_batch_hm, dim=1, keepdim=True)[0].clamp(0.0, 1.0)
        if self.object_kd_foreground_source == 'box':
            base_object_mask = self._build_box_foreground_mask(
                gt_boxes,
                heatmap_mask.shape[-2:],
                heatmap_mask.device,
                heatmap_mask.dtype,
            )
        elif self.object_kd_foreground_source == 'box_or_heatmap':
            box_mask = self._build_box_foreground_mask(
                gt_boxes,
                heatmap_mask.shape[-2:],
                heatmap_mask.device,
                heatmap_mask.dtype,
            )
            base_object_mask = torch.maximum(heatmap_mask, box_mask)
        else:
            base_object_mask = heatmap_mask
        self.debug_maps['gt_box_mask'] = base_object_mask.detach()
        if base_object_mask.ndim == 4 and min(base_object_mask.shape[-2:]) > 2:
            eroded_box_mask = -F.max_pool2d(-base_object_mask, kernel_size=3, stride=1, padding=1)
            self.debug_maps['gt_box_outline_mask'] = (base_object_mask - eroded_box_mask).clamp(0.0, 1.0).detach()
        if base_object_mask.sum() < self.object_kd_min_mask_sum:
            return radar_bev.new_tensor(0.0), {'object_kd_mask_mean': base_object_mask.mean().item()}

        object_mask = base_object_mask

        if lidar_intensity_mask is not None and self.object_kd_use_lidar_intensity:
            if tuple(lidar_intensity_mask.shape[-2:]) != tuple(object_mask.shape[-2:]):
                lidar_intensity_mask = F.interpolate(
                    lidar_intensity_mask,
                    size=object_mask.shape[-2:],
                    mode='bilinear',
                align_corners=False,
                )
            lidar_intensity_mask = lidar_intensity_mask.clamp(0.0, 1.0)
            self.debug_maps['object_lidar_intensity_gate_mask'] = (base_object_mask * lidar_intensity_mask).detach()
            object_mask = base_object_mask * (
                self.object_kd_teacher_floor +
                (1.0 - self.object_kd_teacher_floor) * lidar_intensity_mask
            )

        if radar_evidence_mask is not None and self.object_kd_use_radar_support:
            if tuple(radar_evidence_mask.shape[-2:]) != tuple(object_mask.shape[-2:]):
                radar_evidence_mask = F.interpolate(
                    radar_evidence_mask,
                    size=object_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
            )
            radar_evidence_mask = radar_evidence_mask.clamp(0.0, 1.0)
            self.debug_maps['object_radar_evidence_gate_mask'] = (base_object_mask * radar_evidence_mask).detach()
            radar_supported_object = base_object_mask * (
                self.object_kd_radar_floor +
                (1.0 - self.object_kd_radar_floor) * radar_evidence_mask
            )
            object_mask = torch.maximum(object_mask, radar_supported_object)

        loss = 0.5 * (
            self._masked_feature_distance(radar_bev, lidar_bev, object_mask) +
            self._masked_feature_distance(radar_bev_8x, lidar_bev_8x, object_mask)
        )
        loss = loss * self.object_kd_weight
        self.debug_maps['object_kd_mask'] = object_mask.detach()
        return loss, {
            'object_kd_loss': loss.item(),
            'object_kd_mask_mean': object_mask.mean().item(),
        }

    def _build_box_foreground_mask(self, gt_boxes, target_hw, device, dtype):
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        batch_size = int(gt_boxes.shape[0])
        mask = torch.zeros((batch_size, 1, target_h, target_w), device=device, dtype=dtype)
        if gt_boxes.numel() == 0:
            return mask
        if self.point_cloud_range is None:
            raise ValueError('Object box KD mask needs point_cloud_range in model config.')

        pc_range = torch.as_tensor(self.point_cloud_range, device=device, dtype=dtype)
        x_min, y_min, _, x_max, y_max, _ = pc_range
        x_centers = torch.linspace(x_min, x_max, target_w, device=device, dtype=dtype)
        y_centers = torch.linspace(y_min, y_max, target_h, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')

        for batch_index in range(batch_size):
            cur_boxes = gt_boxes[batch_index]
            if cur_boxes.shape[-1] >= 8:
                cur_boxes = cur_boxes[cur_boxes[:, 7] > 0]
            if cur_boxes.numel() == 0:
                continue
            for box in cur_boxes:
                cx, cy = box[0], box[1]
                if self.object_kd_box_dim_order == 'hwl':
                    dx = box[5]
                    dy = box[4]
                elif self.object_kd_box_dim_order == 'lhw':
                    dx = box[3]
                    dy = box[4]
                else:
                    raise ValueError(f"Unsupported object_kd.box_dim_order '{self.object_kd_box_dim_order}'")
                dx = torch.clamp(dx.abs() * self.object_kd_box_scale, min=1e-3)
                dy = torch.clamp(dy.abs() * self.object_kd_box_scale, min=1e-3)
                yaw = box[6]
                rel_x = xx - cx
                rel_y = yy - cy
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                local_x = rel_x * cos_yaw + rel_y * sin_yaw
                local_y = -rel_x * sin_yaw + rel_y * cos_yaw
                norm_x = local_x / (0.5 * dx)
                norm_y = local_y / (0.5 * dy)
                inside = (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
                if self.object_kd_box_value == 'soft':
                    value = torch.exp(-0.5 * (norm_x * norm_x + norm_y * norm_y))
                    value = torch.clamp(value, min=self.object_kd_box_soft_floor) * inside.to(dtype)
                else:
                    value = inside.to(dtype)
                mask[batch_index, 0] = torch.maximum(mask[batch_index, 0], value)
        return mask.clamp(0.0, 1.0)

    def _build_single_box_mask(self, box, target_hw, device, dtype, box_scale=None):
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if self.point_cloud_range is None:
            raise ValueError('Object box KD mask needs point_cloud_range in model config.')

        pc_range = torch.as_tensor(self.point_cloud_range, device=device, dtype=dtype)
        x_min, y_min, _, x_max, y_max, _ = pc_range
        x_centers = torch.linspace(x_min, x_max, target_w, device=device, dtype=dtype)
        y_centers = torch.linspace(y_min, y_max, target_h, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')

        cx, cy = box[0], box[1]
        if self.object_kd_box_dim_order == 'hwl':
            dx = box[5]
            dy = box[4]
        elif self.object_kd_box_dim_order == 'lhw':
            dx = box[3]
            dy = box[4]
        else:
            raise ValueError(f"Unsupported object_kd.box_dim_order '{self.object_kd_box_dim_order}'")
        scale = self.object_kd_box_scale if box_scale is None else box_scale
        dx = torch.clamp(dx.abs() * scale, min=1e-3)
        dy = torch.clamp(dy.abs() * scale, min=1e-3)
        yaw = box[6]
        rel_x = xx - cx
        rel_y = yy - cy
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        norm_x = local_x / (0.5 * dx)
        norm_y = local_y / (0.5 * dy)
        inside = (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
        if self.object_kd_box_value == 'soft':
            value = torch.exp(-0.5 * (norm_x * norm_x + norm_y * norm_y))
            value = torch.clamp(value, min=self.object_kd_box_soft_floor) * inside.to(dtype)
        else:
            value = inside.to(dtype)
        return value.clamp(0.0, 1.0).view(1, 1, target_h, target_w)

    def _masked_feature_distance(self, radar_bev, lidar_bev, mask):
        if tuple(mask.shape[-2:]) != tuple(radar_bev.shape[-2:]):
            mask = F.interpolate(mask, size=radar_bev.shape[-2:], mode='bilinear', align_corners=False)
        if self.object_kd_metric == 'l1_softmax':
            radar_feat = radar_bev.softmax(1)
            lidar_feat = lidar_bev.softmax(1)
            dist = F.l1_loss(radar_feat, lidar_feat, reduction='none')
        else:
            radar_feat = F.normalize(radar_bev, p=2, dim=1)
            lidar_feat = F.normalize(lidar_bev, p=2, dim=1)
            dist = F.mse_loss(radar_feat, lidar_feat, reduction='none')
        mask = mask.to(dtype=dist.dtype)
        denom = mask.sum().clamp_min(1.0) * dist.shape[1]
        return (dist * mask).sum() / denom

    @staticmethod
    def _weighted_feature_pool(feature, weight):
        denom = weight.sum(dim=(2, 3)).clamp_min(1e-6)
        return (feature * weight).sum(dim=(2, 3)) / denom

    def object_proto_kd_loss(
            self,
            radar_bev,
            lidar_bev,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        if not self.use_object_proto_kd:
            return radar_bev.new_tensor(0.0), {}

        target_hw = radar_bev.shape[-2:]
        if tuple(lidar_bev.shape[-2:]) != tuple(target_hw):
            lidar_bev = F.interpolate(lidar_bev, size=target_hw, mode='bilinear', align_corners=False)
        if radar_evidence_mask is not None and tuple(radar_evidence_mask.shape[-2:]) != tuple(target_hw):
            radar_evidence_mask = F.interpolate(
                radar_evidence_mask,
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            )
        if lidar_intensity_mask is not None and tuple(lidar_intensity_mask.shape[-2:]) != tuple(target_hw):
            lidar_intensity_mask = F.interpolate(
                lidar_intensity_mask,
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            )

        radar_feat = F.normalize(radar_bev, p=2, dim=1)
        lidar_feat = F.normalize(lidar_bev, p=2, dim=1)
        losses = []
        observabilities = []
        proto_mask = radar_bev.new_zeros((radar_bev.shape[0], 1, target_hw[0], target_hw[1]))
        context_mask = torch.zeros_like(proto_mask)

        for batch_index in range(int(gt_boxes.shape[0])):
            cur_boxes = gt_boxes[batch_index]
            if cur_boxes.shape[-1] >= 8:
                cur_boxes = cur_boxes[cur_boxes[:, 7] > 0]
            if cur_boxes.numel() == 0:
                continue
            for box in cur_boxes:
                box_mask = self._build_single_box_mask(
                    box,
                    target_hw,
                    radar_bev.device,
                    radar_bev.dtype,
                    box_scale=self.object_kd_box_scale,
                )
                if box_mask.sum() < self.object_proto_min_box_weight:
                    continue
                context_box_mask = self._build_single_box_mask(
                    box,
                    target_hw,
                    radar_bev.device,
                    radar_bev.dtype,
                    box_scale=self.object_proto_context_scale,
                )
                radar_box = box_mask
                observability = radar_bev.new_tensor(0.0)
                if radar_evidence_mask is not None and self.object_proto_use_radar_support:
                    evidence_values = (radar_evidence_mask[batch_index:batch_index + 1] * box_mask)[box_mask > 0]
                    if evidence_values.numel() > 0:
                        top_k = max(1, int(evidence_values.numel() * self.object_proto_top_fraction))
                        observability = torch.topk(evidence_values, k=top_k).values.mean().clamp(0.0, 1.0)
                    radar_box = box_mask * (
                        self.object_proto_radar_floor +
                        (1.0 - self.object_proto_radar_floor) *
                        radar_evidence_mask[batch_index:batch_index + 1].clamp(0.0, 1.0)
                    )

                teacher_box = box_mask
                if lidar_intensity_mask is not None and self.object_proto_use_lidar_intensity:
                    teacher_box = box_mask * (
                        self.object_proto_teacher_floor +
                        (1.0 - self.object_proto_teacher_floor) *
                        lidar_intensity_mask[batch_index:batch_index + 1].clamp(0.0, 1.0)
                    )

                radar_evidence_proto = self._weighted_feature_pool(
                    radar_feat[batch_index:batch_index + 1],
                    radar_box,
                )
                radar_context_proto = self._weighted_feature_pool(
                    radar_feat[batch_index:batch_index + 1],
                    context_box_mask,
                )
                radar_proto = observability * radar_evidence_proto + (1.0 - observability) * radar_context_proto
                lidar_proto = self._weighted_feature_pool(
                    lidar_feat[batch_index:batch_index + 1],
                    teacher_box,
                )
                proto_loss = 1.0 - F.cosine_similarity(radar_proto, lidar_proto.detach(), dim=1)
                proto_weight = self.object_proto_weight_floor + (1.0 - self.object_proto_weight_floor) * observability
                losses.append(proto_loss * proto_weight)
                observabilities.append(observability.detach())
                proto_mask[batch_index:batch_index + 1] = torch.maximum(
                    proto_mask[batch_index:batch_index + 1],
                    radar_box.detach(),
                )
                context_mask[batch_index:batch_index + 1] = torch.maximum(
                    context_mask[batch_index:batch_index + 1],
                    context_box_mask.detach(),
                )

        self.debug_maps['object_proto_radar_support_mask'] = proto_mask.detach()
        self.debug_maps['object_proto_context_mask'] = context_mask.detach()
        if not losses:
            return radar_bev.new_tensor(0.0), {
                'object_proto_kd_loss': 0.0,
                'object_proto_count': 0.0,
                'object_proto_observability_mean': 0.0,
            }

        loss = torch.stack(losses).mean() * self.object_proto_kd_weight
        observability_mean = torch.stack(observabilities).mean() if observabilities else loss.new_tensor(0.0)
        return loss, {
            'object_proto_kd_loss': loss.item(),
            'object_proto_count': float(len(losses)),
            'object_proto_observability_mean': observability_mean.item(),
            'object_proto_mask_mean': proto_mask.mean().item(),
        }

    def semantic_heatmap_kd_loss(self, teacher_preds, radar_preds, heatmaps, radar_evidence_mask=None):
        if not self.use_semantic_heatmap_kd:
            return radar_preds[0]['hm'].new_tensor(0.0), {}

        losses = []
        weight_means = []
        fn_means = []
        teacher_maps = []
        radar_maps = []
        gt_maps = []
        weight_maps = []
        delta_maps = []
        fn_maps = []
        for teacher_pred, radar_pred, target_hm in zip(teacher_preds, radar_preds, heatmaps):
            teacher_logits = teacher_pred['hm'].detach()
            radar_logits = radar_pred['hm']
            teacher_hm = clip_sigmoid(teacher_logits)
            radar_hm = clip_sigmoid(radar_logits)
            gt_weight = target_hm.clamp(0.0, 1.0)
            cur_evidence = None
            if radar_evidence_mask is not None:
                cur_evidence = radar_evidence_mask
                if tuple(cur_evidence.shape[-2:]) != tuple(gt_weight.shape[-2:]):
                    cur_evidence = F.interpolate(
                        cur_evidence,
                        size=gt_weight.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )

            if self.semantic_heatmap_weight_source == 'teacher_confidence':
                weight = gt_weight * (
                    self.semantic_heatmap_teacher_floor +
                    (1.0 - self.semantic_heatmap_teacher_floor) * teacher_hm
                )
                if cur_evidence is not None and self.semantic_heatmap_radar_boost > 0:
                    weight = weight * (1.0 + self.semantic_heatmap_radar_boost * cur_evidence.clamp(0.0, 1.0))
            elif cur_evidence is not None and self.semantic_heatmap_use_radar_support:
                weight = gt_weight * (
                    self.semantic_heatmap_floor +
                    (1.0 - self.semantic_heatmap_floor) *
                    cur_evidence.clamp(0.0, 1.0)
                )
            else:
                weight = gt_weight * self.semantic_heatmap_floor

            if self.use_proposal_error_kd and self.proposal_fn_boost > 0:
                with torch.no_grad():
                    fn_mask = (
                        (gt_weight > self.proposal_gt_positive_threshold) &
                        (radar_hm.detach() < self.proposal_fn_student_threshold) &
                        (teacher_hm > self.proposal_fn_teacher_threshold)
                    ).to(weight.dtype)
                    if cur_evidence is not None:
                        fn_obs = (
                            self.proposal_observability_floor_fn +
                            (1.0 - self.proposal_observability_floor_fn) *
                            cur_evidence.clamp(0.0, 1.0)
                        )
                    else:
                        fn_obs = weight.new_tensor(1.0)
                weight = weight * (1.0 + self.proposal_fn_boost * fn_mask * fn_obs)
                fn_means.append(fn_mask.mean().detach())
                fn_maps.append(torch.max(fn_mask.detach(), dim=1, keepdim=True)[0])

            if weight.sum() <= 0:
                continue
            if self.semantic_heatmap_loss_type == 'bce_logits':
                temperature = self.semantic_heatmap_temperature
                teacher_target = torch.sigmoid(teacher_logits / temperature).clamp(1e-4, 1.0 - 1e-4)
                dist = F.binary_cross_entropy_with_logits(
                    radar_logits / temperature,
                    teacher_target,
                    reduction='none',
                ) * (temperature ** 2)
            else:
                dist = F.mse_loss(radar_hm, teacher_hm, reduction='none')
            losses.append((dist * weight).sum() / weight.sum().clamp_min(1.0))
            weight_means.append(weight.mean().detach())
            teacher_maps.append(torch.max(teacher_hm, dim=1, keepdim=True)[0])
            radar_maps.append(torch.max(radar_hm.detach(), dim=1, keepdim=True)[0])
            gt_maps.append(torch.max(target_hm.detach().clamp(0.0, 1.0), dim=1, keepdim=True)[0])
            weight_maps.append(torch.max(weight.detach(), dim=1, keepdim=True)[0])
            delta_maps.append(torch.max(torch.abs(radar_hm.detach() - teacher_hm), dim=1, keepdim=True)[0])

        if teacher_maps:
            self.debug_maps['semantic_teacher_heatmap_mask'] = torch.max(torch.cat(teacher_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['semantic_radar_heatmap_mask'] = torch.max(torch.cat(radar_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['semantic_gt_heatmap_mask'] = torch.max(torch.cat(gt_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['semantic_heatmap_weight_mask'] = torch.max(torch.cat(weight_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['semantic_heatmap_delta_mask'] = torch.max(torch.cat(delta_maps, dim=1), dim=1, keepdim=True)[0].detach()
            if fn_maps:
                self.debug_maps['proposal_fn_mask'] = torch.max(torch.cat(fn_maps, dim=1), dim=1, keepdim=True)[0].detach()

        if not losses:
            return radar_preds[0]['hm'].new_tensor(0.0), {
                'semantic_heatmap_kd_loss': 0.0,
                'semantic_heatmap_weight_mean': 0.0,
            }
        loss = torch.stack(losses).mean() * self.semantic_heatmap_kd_weight
        weight_mean = torch.stack(weight_means).mean() if weight_means else loss.new_tensor(0.0)
        tb_dict = {
            'semantic_heatmap_kd_loss': loss.item(),
            'semantic_heatmap_weight_mean': weight_mean.item(),
        }
        if fn_means:
            tb_dict['proposal_fn_cell_mean'] = torch.stack(fn_means).mean().item()
        return loss, tb_dict

    def task_dense_kd_loss(
            self,
            radar_bev,
            radar_bev_8x,
            lidar_bev,
            lidar_bev_8x,
            teacher_preds,
            heatmaps,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        if not self.use_task_dense_kd:
            return radar_bev.new_tensor(0.0), {}

        target_hw = radar_bev.shape[-2:]
        gt_heatmap_mask = torch.max(torch.cat(heatmaps, dim=1), dim=1, keepdim=True)[0].clamp(0.0, 1.0)
        if tuple(gt_heatmap_mask.shape[-2:]) != tuple(target_hw):
            gt_heatmap_mask = F.interpolate(
                gt_heatmap_mask,
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            )
        gt_box_mask = self._build_box_foreground_mask(
            gt_boxes,
            target_hw,
            radar_bev.device,
            radar_bev.dtype,
        )
        task_mask = torch.maximum(gt_box_mask, gt_heatmap_mask)

        teacher_conf_mask = None
        if self.task_dense_use_teacher_confidence:
            teacher_conf = []
            for teacher_pred in teacher_preds:
                teacher_conf.append(torch.max(clip_sigmoid(teacher_pred['hm'].detach()), dim=1, keepdim=True)[0])
            teacher_conf_mask = torch.max(torch.cat(teacher_conf, dim=1), dim=1, keepdim=True)[0]
            if tuple(teacher_conf_mask.shape[-2:]) != tuple(target_hw):
                teacher_conf_mask = F.interpolate(
                    teacher_conf_mask,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            threshold = self.task_dense_teacher_threshold
            teacher_task = ((teacher_conf_mask - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)
            task_mask = torch.maximum(task_mask, teacher_task)

        task_mask = task_mask.clamp(0.0, 1.0)
        base_task_mask = task_mask
        reliability_mask = None

        if lidar_intensity_mask is not None and self.task_dense_use_lidar_intensity:
            cur_lidar = lidar_intensity_mask.clamp(0.0, 1.0)
            if tuple(cur_lidar.shape[-2:]) != tuple(task_mask.shape[-2:]):
                cur_lidar = F.interpolate(
                    cur_lidar,
                    size=task_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            lidar_reliability = (
                self.task_dense_teacher_floor +
                (1.0 - self.task_dense_teacher_floor) * cur_lidar
            )
            reliability_mask = lidar_reliability if reliability_mask is None else torch.maximum(reliability_mask, lidar_reliability)

        if radar_evidence_mask is not None and self.task_dense_use_radar_support:
            cur_radar = radar_evidence_mask.clamp(0.0, 1.0)
            if tuple(cur_radar.shape[-2:]) != tuple(task_mask.shape[-2:]):
                cur_radar = F.interpolate(
                    cur_radar,
                    size=task_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            radar_reliability = (
                self.task_dense_radar_floor +
                (1.0 - self.task_dense_radar_floor) * cur_radar
            )
            reliability_mask = radar_reliability if reliability_mask is None else torch.maximum(reliability_mask, radar_reliability)

        if reliability_mask is not None:
            task_mask = task_mask * reliability_mask

        task_mask = task_mask.clamp(0.0, 1.0)
        if task_mask.sum() < self.task_dense_min_mask_sum:
            return radar_bev.new_tensor(0.0), {
                'task_dense_feature_kd_loss': 0.0,
                'task_dense_mask_mean': task_mask.mean().item(),
                'task_dense_teacher_conf_mean': 0.0 if teacher_conf_mask is None else teacher_conf_mask.mean().item(),
            }

        def cosine_map(student_feat, teacher_feat, mask):
            if tuple(mask.shape[-2:]) != tuple(student_feat.shape[-2:]):
                mask = F.interpolate(mask, size=student_feat.shape[-2:], mode='bilinear', align_corners=False)
            if tuple(teacher_feat.shape[-2:]) != tuple(student_feat.shape[-2:]):
                teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode='bilinear', align_corners=False)
            student_norm = F.normalize(student_feat, p=2, dim=1)
            teacher_norm = F.normalize(teacher_feat.detach(), p=2, dim=1)
            distance = 1.0 - torch.sum(student_norm * teacher_norm, dim=1, keepdim=True)
            return (distance * mask).sum() / mask.sum().clamp_min(1.0), distance

        high_loss, high_delta = cosine_map(radar_bev, lidar_bev, task_mask)
        de8x_loss, _de8x_delta = cosine_map(radar_bev_8x, lidar_bev_8x, task_mask)
        loss = 0.5 * (high_loss + de8x_loss) * self.task_dense_feature_weight

        self.debug_maps['task_dense_gt_mask'] = base_task_mask.detach()
        self.debug_maps['task_dense_gt_heatmap_mask'] = gt_heatmap_mask.detach()
        self.debug_maps['task_dense_gt_box_mask'] = gt_box_mask.detach()
        self.debug_maps['task_dense_weight_mask'] = task_mask.detach()
        self.debug_maps['task_dense_feature_delta_mask'] = high_delta.detach()
        if reliability_mask is not None:
            self.debug_maps['task_dense_reliability_mask'] = reliability_mask.detach()
        if teacher_conf_mask is not None:
            self.debug_maps['task_dense_teacher_conf_mask'] = teacher_conf_mask.detach()

        return loss, {
            'task_dense_feature_kd_loss': loss.item(),
            'task_dense_mask_mean': task_mask.mean().item(),
            'task_dense_teacher_conf_mean': 0.0 if teacher_conf_mask is None else teacher_conf_mask.mean().item(),
        }

    def _build_reliability_prior(
            self,
            radar_bev,
            teacher_preds,
            heatmaps,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        target_hw = radar_bev.shape[-2:]
        gt_heatmap_mask = torch.max(torch.cat(heatmaps, dim=1), dim=1, keepdim=True)[0].clamp(0.0, 1.0)
        if tuple(gt_heatmap_mask.shape[-2:]) != tuple(target_hw):
            gt_heatmap_mask = F.interpolate(gt_heatmap_mask, size=target_hw, mode='bilinear', align_corners=False)
        gt_box_mask = self._build_box_foreground_mask(gt_boxes, target_hw, radar_bev.device, radar_bev.dtype)

        teacher_conf = []
        for teacher_pred in teacher_preds:
            teacher_conf.append(torch.max(clip_sigmoid(teacher_pred['hm'].detach()), dim=1, keepdim=True)[0])
        teacher_conf_mask = torch.max(torch.cat(teacher_conf, dim=1), dim=1, keepdim=True)[0]
        if tuple(teacher_conf_mask.shape[-2:]) != tuple(target_hw):
            teacher_conf_mask = F.interpolate(teacher_conf_mask, size=target_hw, mode='bilinear', align_corners=False)
        teacher_task = (
            (teacher_conf_mask - self.learned_mask_teacher_threshold) /
            max(1.0 - self.learned_mask_teacher_threshold, 1e-6)
        ).clamp(0.0, 1.0)

        prior = torch.maximum(torch.maximum(gt_box_mask, gt_heatmap_mask), teacher_task)

        if radar_evidence_mask is not None and self.learned_mask_use_radar_prior:
            cur_radar = radar_evidence_mask.clamp(0.0, 1.0)
            if tuple(cur_radar.shape[-2:]) != tuple(target_hw):
                cur_radar = F.interpolate(cur_radar, size=target_hw, mode='bilinear', align_corners=False)
            prior = torch.maximum(prior, cur_radar)

        if lidar_intensity_mask is not None and self.learned_mask_use_lidar_prior:
            cur_lidar = lidar_intensity_mask.clamp(0.0, 1.0)
            if tuple(cur_lidar.shape[-2:]) != tuple(target_hw):
                cur_lidar = F.interpolate(cur_lidar, size=target_hw, mode='bilinear', align_corners=False)
            prior = torch.maximum(prior, teacher_conf_mask.clamp(0.0, 1.0) * cur_lidar)

        self.debug_maps['learned_mask_prior'] = prior.detach()
        self.debug_maps['learned_mask_prior_box'] = gt_box_mask.detach()
        self.debug_maps['learned_mask_prior_teacher'] = teacher_conf_mask.detach()
        return prior.clamp(0.0, 1.0)

    def learned_reliability_kd_loss(
            self,
            radar_bev,
            radar_bev_8x,
            lidar_bev,
            lidar_bev_8x,
            teacher_preds,
            heatmaps,
            gt_boxes,
            radar_evidence_mask=None,
            lidar_intensity_mask=None):
        if not self.use_learned_reliability_mask:
            zero = radar_bev.new_tensor(0.0)
            return zero, zero, None, {}

        prior = self._build_reliability_prior(
            radar_bev,
            teacher_preds,
            heatmaps,
            gt_boxes,
            radar_evidence_mask=radar_evidence_mask,
            lidar_intensity_mask=lidar_intensity_mask,
        )
        mask_logits = self.reliability_mask_head(radar_bev)
        mask_pred = torch.sigmoid(mask_logits)

        bce = F.binary_cross_entropy_with_logits(mask_logits, prior.detach(), reduction='none')
        pos_weight = prior.detach()
        neg_weight = (1.0 - prior.detach()) * self.learned_mask_neg_weight
        mask_weight = pos_weight + neg_weight
        mask_loss = (bce * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)
        mask_loss = mask_loss * self.learned_mask_loss_weight

        kd_source = mask_pred.detach() if self.learned_mask_detach_for_kd else mask_pred
        kd_weight = self.learned_mask_kd_floor + (1.0 - self.learned_mask_kd_floor) * kd_source

        def cosine_map(student_feat, teacher_feat, mask):
            if tuple(mask.shape[-2:]) != tuple(student_feat.shape[-2:]):
                mask = F.interpolate(mask, size=student_feat.shape[-2:], mode='bilinear', align_corners=False)
            if tuple(teacher_feat.shape[-2:]) != tuple(student_feat.shape[-2:]):
                teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode='bilinear', align_corners=False)
            student_norm = F.normalize(student_feat, p=2, dim=1)
            teacher_norm = F.normalize(teacher_feat.detach(), p=2, dim=1)
            distance = 1.0 - torch.sum(student_norm * teacher_norm, dim=1, keepdim=True)
            return (distance * mask).sum() / mask.sum().clamp_min(1.0), distance

        high_loss, high_delta = cosine_map(radar_bev, lidar_bev, kd_weight)
        de8x_loss, _de8x_delta = cosine_map(radar_bev_8x, lidar_bev_8x, kd_weight)
        feature_loss = 0.5 * (high_loss + de8x_loss) * self.task_dense_feature_weight

        self.debug_maps['learned_mask_pred'] = mask_pred.detach()
        self.debug_maps['learned_mask_kd_weight'] = kd_weight.detach()
        self.debug_maps['learned_mask_feature_delta'] = high_delta.detach()

        tb_dict = {
            'learned_mask_loss': mask_loss.item(),
            'learned_mask_prior_mean': prior.mean().item(),
            'learned_mask_pred_mean': mask_pred.mean().item(),
            'learned_mask_kd_mean': kd_weight.mean().item(),
            'learned_mask_feature_kd_loss': feature_loss.item(),
        }
        return feature_loss, mask_loss, kd_weight, tb_dict

    def proposal_error_kd_loss(self, teacher_preds, radar_preds, heatmaps, radar_evidence_mask=None):
        if not self.use_proposal_error_kd or self.proposal_fp_weight <= 0:
            return radar_preds[0]['hm'].new_tensor(0.0), {}

        losses = []
        fp_means = []
        fp_weight_means = []
        fp_maps = []
        for teacher_pred, radar_pred, target_hm in zip(teacher_preds, radar_preds, heatmaps):
            teacher_logits = teacher_pred['hm'].detach()
            radar_logits = radar_pred['hm']
            teacher_prob = torch.sigmoid(teacher_logits)
            radar_prob = torch.sigmoid(radar_logits)
            gt_max = target_hm.clamp(0.0, 1.0).max(dim=1, keepdim=True)[0]

            with torch.no_grad():
                radar_score, radar_class = radar_prob.detach().max(dim=1, keepdim=True)
                teacher_score = teacher_prob.gather(1, radar_class)
                fp_mask = (
                    (gt_max < self.proposal_gt_background_threshold) &
                    (radar_score > self.proposal_fp_student_threshold) &
                    (teacher_score < self.proposal_fp_teacher_threshold)
                )
                if self.proposal_max_fp_cells_per_sample > 0:
                    limited = torch.zeros_like(fp_mask)
                    flat_score = (radar_score * fp_mask.to(radar_score.dtype)).flatten(1)
                    flat_mask = fp_mask.flatten(1)
                    flat_limited = limited.flatten(1)
                    for batch_index in range(flat_score.shape[0]):
                        valid_count = int(flat_mask[batch_index].sum().item())
                        if valid_count == 0:
                            continue
                        top_k = min(valid_count, self.proposal_max_fp_cells_per_sample)
                        top_idx = torch.topk(flat_score[batch_index], k=top_k).indices
                        flat_limited[batch_index, top_idx] = True
                    fp_mask = limited

                if radar_evidence_mask is not None:
                    cur_evidence = radar_evidence_mask
                    if tuple(cur_evidence.shape[-2:]) != tuple(gt_max.shape[-2:]):
                        cur_evidence = F.interpolate(
                            cur_evidence,
                            size=gt_max.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    fp_weight = (
                        self.proposal_observability_floor_fp +
                        (1.0 - self.proposal_observability_floor_fp) *
                        cur_evidence.clamp(0.0, 1.0)
                    )
                else:
                    fp_weight = radar_score.new_ones(radar_score.shape)

            if fp_mask.sum() <= 0:
                fp_means.append(fp_mask.to(radar_logits.dtype).mean().detach())
                continue

            selected_logits = radar_logits.gather(1, radar_class)
            dist = F.binary_cross_entropy_with_logits(
                selected_logits,
                torch.zeros_like(selected_logits),
                reduction='none',
            )
            weight = fp_mask.to(dist.dtype) * fp_weight
            losses.append((dist * weight).sum() / weight.sum().clamp_min(1.0))
            fp_means.append(fp_mask.to(dist.dtype).mean().detach())
            fp_weight_means.append(weight.mean().detach())
            fp_maps.append(fp_mask.to(dist.dtype).detach())

        if fp_maps:
            self.debug_maps['proposal_fp_mask'] = torch.max(torch.cat(fp_maps, dim=1), dim=1, keepdim=True)[0].detach()
        if not losses:
            zero = radar_preds[0]['hm'].new_tensor(0.0)
            return zero, {
                'proposal_error_kd_loss': 0.0,
                'proposal_fp_cell_mean': torch.stack(fp_means).mean().item() if fp_means else 0.0,
                'proposal_fp_weight_mean': 0.0,
            }

        loss = torch.stack(losses).mean() * self.proposal_fp_weight
        return loss, {
            'proposal_error_kd_loss': loss.item(),
            'proposal_fp_cell_mean': torch.stack(fp_means).mean().item() if fp_means else 0.0,
            'proposal_fp_weight_mean': torch.stack(fp_weight_means).mean().item() if fp_weight_means else 0.0,
        }

    def response_kd_loss(self, teacher_preds, radar_preds, heatmaps, radar_evidence_mask=None, kd_weight_mask=None):
        if not self.use_response_kd:
            return radar_preds[0]['hm'].new_tensor(0.0), {}

        pos_losses = []
        neg_losses = []
        pos_means = []
        neg_means = []
        pos_maps = []
        neg_maps = []
        delta_maps = []
        for teacher_pred, radar_pred, target_hm in zip(teacher_preds, radar_preds, heatmaps):
            teacher_logits = teacher_pred['hm'].detach()
            radar_logits = radar_pred['hm']
            teacher_prob = torch.sigmoid(teacher_logits)
            radar_prob = torch.sigmoid(radar_logits)
            gt_weight = target_hm.clamp(0.0, 1.0)

            teacher_pos = (
                (teacher_prob - self.response_teacher_threshold) /
                max(1.0 - self.response_teacher_threshold, 1e-6)
            ).clamp(0.0, 1.0)
            pos_weight = torch.maximum(gt_weight, teacher_pos)
            pos_weight = pos_weight * (
                self.response_teacher_floor +
                (1.0 - self.response_teacher_floor) * teacher_prob
            )

            if pos_weight.sum() > 0:
                if kd_weight_mask is not None:
                    cur_kd_weight = kd_weight_mask
                    if tuple(cur_kd_weight.shape[-2:]) != tuple(pos_weight.shape[-2:]):
                        cur_kd_weight = F.interpolate(
                            cur_kd_weight,
                            size=pos_weight.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    pos_weight = pos_weight * cur_kd_weight.clamp(0.0, 1.0)
                temperature = self.response_temperature
                teacher_target = torch.sigmoid(teacher_logits / temperature).clamp(1e-4, 1.0 - 1e-4)
                pos_dist = F.binary_cross_entropy_with_logits(
                    radar_logits / temperature,
                    teacher_target,
                    reduction='none',
                ) * (temperature ** 2)
                pos_losses.append((pos_dist * pos_weight).sum() / pos_weight.sum().clamp_min(1.0))
                pos_means.append(pos_weight.mean().detach())

            with torch.no_grad():
                neg_mask = (
                    (gt_weight < self.response_gt_bg_threshold) &
                    (teacher_prob < self.response_teacher_bg_threshold) &
                    (radar_prob.detach() > self.response_student_fp_threshold)
                )
                if self.response_max_fp_cells_per_sample > 0:
                    limited = torch.zeros_like(neg_mask)
                    flat_score = (radar_prob.detach() * neg_mask.to(radar_prob.dtype)).flatten(1)
                    flat_mask = neg_mask.flatten(1)
                    flat_limited = limited.flatten(1)
                    for batch_index in range(flat_score.shape[0]):
                        valid_count = int(flat_mask[batch_index].sum().item())
                        if valid_count == 0:
                            continue
                        top_k = min(valid_count, self.response_max_fp_cells_per_sample)
                        top_idx = torch.topk(flat_score[batch_index], k=top_k).indices
                        flat_limited[batch_index, top_idx] = True
                    neg_mask = limited

                neg_weight = neg_mask.to(radar_logits.dtype)
                if radar_evidence_mask is not None:
                    cur_evidence = radar_evidence_mask
                    if tuple(cur_evidence.shape[-2:]) != tuple(neg_weight.shape[-2:]):
                        cur_evidence = F.interpolate(
                            cur_evidence,
                            size=neg_weight.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    cur_evidence = cur_evidence.clamp(0.0, 1.0)
                    neg_weight = neg_weight * (
                        self.response_neg_radar_floor +
                        (1.0 - self.response_neg_radar_floor) * cur_evidence
                    )

            if neg_weight.sum() > 0:
                if kd_weight_mask is not None:
                    cur_kd_weight = kd_weight_mask
                    if tuple(cur_kd_weight.shape[-2:]) != tuple(neg_weight.shape[-2:]):
                        cur_kd_weight = F.interpolate(
                            cur_kd_weight,
                            size=neg_weight.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    neg_weight = neg_weight * cur_kd_weight.clamp(0.0, 1.0)
                neg_dist = F.binary_cross_entropy_with_logits(
                    radar_logits,
                    torch.zeros_like(radar_logits),
                    reduction='none',
                )
                neg_losses.append((neg_dist * neg_weight).sum() / neg_weight.sum().clamp_min(1.0))
                neg_means.append(neg_mask.to(radar_logits.dtype).mean().detach())

            pos_maps.append(torch.max(pos_weight.detach(), dim=1, keepdim=True)[0])
            neg_maps.append(torch.max(neg_mask.to(radar_logits.dtype).detach(), dim=1, keepdim=True)[0])
            delta_maps.append(torch.max(torch.abs(radar_prob.detach() - teacher_prob), dim=1, keepdim=True)[0])

        zero = radar_preds[0]['hm'].new_tensor(0.0)
        pos_loss = torch.stack(pos_losses).mean() * self.response_pos_weight if pos_losses else zero
        neg_loss = torch.stack(neg_losses).mean() * self.response_neg_weight if neg_losses else zero
        loss = pos_loss + neg_loss

        if pos_maps:
            self.debug_maps['response_pos_weight_mask'] = torch.max(torch.cat(pos_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['response_neg_fp_mask'] = torch.max(torch.cat(neg_maps, dim=1), dim=1, keepdim=True)[0].detach()
            self.debug_maps['response_heatmap_delta_mask'] = torch.max(torch.cat(delta_maps, dim=1), dim=1, keepdim=True)[0].detach()

        return loss, {
            'response_kd_loss': loss.item(),
            'response_pos_kd_loss': pos_loss.item(),
            'response_neg_kd_loss': neg_loss.item(),
            'response_pos_weight_mean': torch.stack(pos_means).mean().item() if pos_means else 0.0,
            'response_neg_cell_mean': torch.stack(neg_means).mean().item() if neg_means else 0.0,
        }

    def get_loss(self, batch_dict):
        low_lidar_bev = batch_dict['multi_scale_2d_features']['x_conv4']
        low_radar_bev = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev = batch_dict['radar_spatial_features_2d']
        high_lidar_bev = batch_dict['spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        high_lidar_bev_8x = batch_dict['spatial_features_2d_8x']
        self.debug_maps = {}
        if 'radar_cma_before_xconv4' in batch_dict:
            cma_before = batch_dict['radar_cma_before_xconv4']
            cma_after = batch_dict.get('radar_cma_after_xconv4', low_radar_bev)
            cma_de8x = batch_dict.get('radar_cma_de8x', low_radar_de_8x)
            self.debug_maps['cma_radar_before_xconv4_mag'] = self._feature_magnitude(cma_before)
            self.debug_maps['cma_radar_after_xconv4_mag'] = self._feature_magnitude(cma_after)
            self.debug_maps['cma_radar_delta_xconv4_mag'] = self._feature_magnitude(cma_after - cma_before)
            self.debug_maps['cma_radar_de8x_mag'] = self._feature_magnitude(cma_de8x)
            self.debug_maps['cma_teacher_low_xconv4_mag'] = self._feature_magnitude(low_lidar_bev)
            self.debug_maps['cma_after_teacher_delta_xconv4_mag'] = self._feature_magnitude(cma_after - low_lidar_bev)
        radar_evidence_mask = self._build_radar_evidence_mask(batch_dict, high_radar_bev.shape[-2:])
        lidar_intensity_mask = self._build_lidar_intensity_mask(batch_dict, high_radar_bev.shape[-2:])
        gt_batch_hm = torch.cat(batch_dict['target_dicts']['heatmaps'], dim=1)
        self.debug_maps['teacher_heatmap_mask'] = torch.max(gt_batch_hm, dim=1, keepdim=True)[0].detach()
        if radar_evidence_mask is not None:
            self.debug_maps['radar_evidence_mask'] = radar_evidence_mask.detach()
        if lidar_intensity_mask is not None:
            self.debug_maps['lidar_intensity_mask'] = lidar_intensity_mask.detach()

        if self.kd_mode == 'masked_afd':
            feature_loss, mask_loss = self.masked_low_loss(
                low_lidar_bev,
                low_radar_bev,
                batch_dict['target_dicts']['heatmaps'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
                debug_prefix='masked_low_main',
            )
            de8x_feature_loss, de8x_mask_loss = self.masked_low_loss(
                low_lidar_bev,
                low_radar_de_8x,
                batch_dict['target_dicts']['heatmaps'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
                debug_prefix='masked_low_de8x',
            )
            low_distill_loss = (
                self.masked_afd_feature_weight * 0.5 * (feature_loss + de8x_feature_loss) +
                self.masked_afd_mask_weight * 0.5 * (mask_loss + de8x_mask_loss)
            ) * self.afd_loss_weight
            tb_dict = {
                'distill_loss': low_distill_loss.item(),
                'low_feature_loss': feature_loss.item(),
                'low_feature_loss_de8x': de8x_feature_loss.item(),
                'mask_loss': mask_loss.item(),
                'mask_loss_de8x': de8x_mask_loss.item(),
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            self.last_loss_terms = {
                'object_kd': low_distill_loss.new_tensor(0.0),
                'object_proto_kd': low_distill_loss.new_tensor(0.0),
                'semantic_heatmap_kd': low_distill_loss.new_tensor(0.0),
                'proposal_error_kd': low_distill_loss.new_tensor(0.0),
                'afd': low_distill_loss,
                'pfd': low_distill_loss.new_tensor(0.0),
                'distill_total': low_distill_loss,
            }
            return low_distill_loss, tb_dict

        if self.kd_mode == 'masked_feature_only':
            object_kd_loss, object_kd_tb = self.object_kd_loss(
                high_radar_bev,
                high_radar_bev_8x,
                high_lidar_bev,
                high_lidar_bev_8x,
                batch_dict['target_dicts']['heatmaps'],
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            tb_dict = {
                'distill_loss': object_kd_loss.item(),
                'low_feature_loss': 0.0,
                'low_feature_loss_de8x': 0.0,
                'mask_loss': 0.0,
                'mask_loss_de8x': 0.0,
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            tb_dict.update(object_kd_tb)
            self.last_loss_terms = {
                'object_kd': object_kd_loss,
                'object_proto_kd': object_kd_loss.new_tensor(0.0),
                'semantic_heatmap_kd': object_kd_loss.new_tensor(0.0),
                'proposal_error_kd': object_kd_loss.new_tensor(0.0),
                'afd': object_kd_loss.new_tensor(0.0),
                'pfd': object_kd_loss.new_tensor(0.0),
                'distill_total': object_kd_loss,
            }
            return object_kd_loss, tb_dict

        if self.kd_mode in {'task_dense_semantic', 'task_dense_response'}:
            task_dense_loss, task_dense_tb = self.task_dense_kd_loss(
                high_radar_bev,
                high_radar_bev_8x,
                high_lidar_bev,
                high_lidar_bev_8x,
                batch_dict['lidar_pred_dicts'],
                batch_dict['target_dicts']['heatmaps'],
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            if self.kd_mode == 'task_dense_response':
                response_loss, response_tb = self.response_kd_loss(
                    batch_dict['lidar_pred_dicts'],
                    batch_dict['radar_pred_dicts'],
                    batch_dict['target_dicts']['heatmaps'],
                    radar_evidence_mask=radar_evidence_mask,
                )
                semantic_loss = task_dense_loss.new_tensor(0.0)
                semantic_tb = {}
            else:
                semantic_loss, semantic_tb = self.semantic_heatmap_kd_loss(
                    batch_dict['lidar_pred_dicts'],
                    batch_dict['radar_pred_dicts'],
                    batch_dict['target_dicts']['heatmaps'],
                    radar_evidence_mask=radar_evidence_mask,
                )
                response_loss = task_dense_loss.new_tensor(0.0)
                response_tb = {}
            distill_loss = task_dense_loss + semantic_loss + response_loss
            tb_dict = {
                'distill_loss': distill_loss.item(),
                'low_feature_loss': 0.0,
                'low_feature_loss_de8x': 0.0,
                'mask_loss': 0.0,
                'mask_loss_de8x': 0.0,
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            tb_dict.update(task_dense_tb)
            tb_dict.update(semantic_tb)
            tb_dict.update(response_tb)
            self.last_loss_terms = {
                'object_kd': distill_loss.new_tensor(0.0),
                'object_proto_kd': distill_loss.new_tensor(0.0),
                'semantic_heatmap_kd': semantic_loss,
                'response_kd': response_loss,
                'proposal_error_kd': distill_loss.new_tensor(0.0),
                'task_dense_feature_kd': task_dense_loss,
                'afd': distill_loss.new_tensor(0.0),
                'pfd': distill_loss.new_tensor(0.0),
                'distill_total': distill_loss,
            }
            return distill_loss, tb_dict

        if self.kd_mode in {'learned_mask_response', 'learned_mask_feature_response'}:
            feature_loss, learned_mask_loss, kd_weight_mask, learned_mask_tb = self.learned_reliability_kd_loss(
                high_radar_bev,
                high_radar_bev_8x,
                high_lidar_bev,
                high_lidar_bev_8x,
                batch_dict['lidar_pred_dicts'],
                batch_dict['target_dicts']['heatmaps'],
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            response_loss, response_tb = self.response_kd_loss(
                batch_dict['lidar_pred_dicts'],
                batch_dict['radar_pred_dicts'],
                batch_dict['target_dicts']['heatmaps'],
                radar_evidence_mask=radar_evidence_mask,
                kd_weight_mask=kd_weight_mask if self.kd_mode == 'learned_mask_response' else None,
            )
            distill_loss = feature_loss + learned_mask_loss + response_loss
            tb_dict = {
                'distill_loss': distill_loss.item(),
                'low_feature_loss': 0.0,
                'low_feature_loss_de8x': 0.0,
                'mask_loss': learned_mask_loss.item(),
                'mask_loss_de8x': 0.0,
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            tb_dict.update(learned_mask_tb)
            tb_dict.update(response_tb)
            self.last_loss_terms = {
                'object_kd': distill_loss.new_tensor(0.0),
                'object_proto_kd': distill_loss.new_tensor(0.0),
                'semantic_heatmap_kd': distill_loss.new_tensor(0.0),
                'response_kd': response_loss,
                'proposal_error_kd': distill_loss.new_tensor(0.0),
                'task_dense_feature_kd': feature_loss,
                'learned_mask': learned_mask_loss,
                'afd': distill_loss.new_tensor(0.0),
                'pfd': distill_loss.new_tensor(0.0),
                'distill_total': distill_loss,
            }
            return distill_loss, tb_dict

        route_modes = {'prototype_only', 'proto_semantic', 'proto_semantic_error', 'afd_plus_proto', 'afd_proto_semantic', 'route_kd'}
        if self.kd_mode in route_modes:
            low_distill_loss = high_radar_bev.new_tensor(0.0)
            feature_loss = high_radar_bev.new_tensor(0.0)
            de8x_feature_loss = high_radar_bev.new_tensor(0.0)
            mask_loss = high_radar_bev.new_tensor(0.0)
            de8x_mask_loss = high_radar_bev.new_tensor(0.0)
            if self.kd_mode in {'afd_plus_proto', 'afd_proto_semantic', 'route_kd'}:
                feature_loss, mask_loss = self.low_loss(low_lidar_bev, low_radar_bev, debug_prefix='low_main')
                de8x_feature_loss, de8x_mask_loss = self.low_loss(low_lidar_bev, low_radar_de_8x, debug_prefix='low_de8x')
                low_distill_loss = (
                    0.5 * (feature_loss + de8x_feature_loss) +
                    0.5 * (mask_loss + de8x_mask_loss)
                ) * self.afd_loss_weight

            object_kd_loss, object_kd_tb = self.object_kd_loss(
                high_radar_bev,
                high_radar_bev_8x,
                high_lidar_bev,
                high_lidar_bev_8x,
                batch_dict['target_dicts']['heatmaps'],
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            object_proto_loss, object_proto_tb = self.object_proto_kd_loss(
                high_radar_bev,
                high_lidar_bev,
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            semantic_loss = high_radar_bev.new_tensor(0.0)
            semantic_tb = {}
            if self.kd_mode in {'proto_semantic', 'proto_semantic_error', 'afd_proto_semantic', 'route_kd'}:
                semantic_loss, semantic_tb = self.semantic_heatmap_kd_loss(
                    batch_dict['lidar_pred_dicts'],
                    batch_dict['radar_pred_dicts'],
                    batch_dict['target_dicts']['heatmaps'],
                    radar_evidence_mask=radar_evidence_mask,
                )
            proposal_error_loss = high_radar_bev.new_tensor(0.0)
            proposal_error_tb = {}
            if self.kd_mode in {'proto_semantic_error'}:
                proposal_error_loss, proposal_error_tb = self.proposal_error_kd_loss(
                    batch_dict['lidar_pred_dicts'],
                    batch_dict['radar_pred_dicts'],
                    batch_dict['target_dicts']['heatmaps'],
                    radar_evidence_mask=radar_evidence_mask,
                )

            distill_loss = low_distill_loss + object_kd_loss + object_proto_loss + semantic_loss + proposal_error_loss
            tb_dict = {
                'distill_loss': distill_loss.item(),
                'low_feature_loss': feature_loss.item(),
                'low_feature_loss_de8x': de8x_feature_loss.item(),
                'mask_loss': mask_loss.item(),
                'mask_loss_de8x': de8x_mask_loss.item(),
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            tb_dict.update(object_kd_tb)
            tb_dict.update(object_proto_tb)
            tb_dict.update(semantic_tb)
            tb_dict.update(proposal_error_tb)
            self.last_loss_terms = {
                'object_kd': object_kd_loss,
                'object_proto_kd': object_proto_loss,
                'semantic_heatmap_kd': semantic_loss,
                'proposal_error_kd': proposal_error_loss,
                'afd': low_distill_loss,
                'pfd': high_radar_bev.new_tensor(0.0),
                'distill_total': distill_loss,
            }
            return distill_loss, tb_dict

        if self.kd_mode not in {'radardistill', 'afd_plus_object'}:
            raise ValueError(f"Unsupported radar_distill.kd_mode '{self.kd_mode}'")

        feature_loss, mask_loss = self.low_loss(low_lidar_bev, low_radar_bev, debug_prefix='low_main')
        de8x_feature_loss, de8x_mask_loss = self.low_loss(low_lidar_bev, low_radar_de_8x, debug_prefix='low_de8x')
        low_distill_loss = (
            0.5 * (feature_loss + de8x_feature_loss) +
            0.5 * (mask_loss + de8x_mask_loss)
        ) * self.afd_loss_weight

        if self.kd_mode == 'afd_plus_object':
            object_kd_loss, object_kd_tb = self.object_kd_loss(
                high_radar_bev,
                high_radar_bev_8x,
                high_lidar_bev,
                high_lidar_bev_8x,
                batch_dict['target_dicts']['heatmaps'],
                batch_dict['gt_boxes'],
                radar_evidence_mask=radar_evidence_mask,
                lidar_intensity_mask=lidar_intensity_mask,
            )
            distill_loss = low_distill_loss + object_kd_loss
            tb_dict = {
                'distill_loss': distill_loss.item(),
                'low_feature_loss': feature_loss.item(),
                'low_feature_loss_de8x': de8x_feature_loss.item(),
                'mask_loss': mask_loss.item(),
                'mask_loss_de8x': de8x_mask_loss.item(),
                'high_distill_loss': 0.0,
            }
            if radar_evidence_mask is not None:
                tb_dict['radar_evidence_mask_mean'] = radar_evidence_mask.mean().item()
            if lidar_intensity_mask is not None:
                tb_dict['lidar_intensity_mask_mean'] = lidar_intensity_mask.mean().item()
            tb_dict.update(object_kd_tb)
            self.last_loss_terms = {
                'object_kd': object_kd_loss,
                'object_proto_kd': object_kd_loss.new_tensor(0.0),
                'semantic_heatmap_kd': object_kd_loss.new_tensor(0.0),
                'proposal_error_kd': object_kd_loss.new_tensor(0.0),
                'afd': low_distill_loss,
                'pfd': object_kd_loss.new_tensor(0.0),
                'distill_total': distill_loss,
            }
            return distill_loss, tb_dict

        high_distill_loss, evidence_tb = self.high_loss(
            high_radar_bev,
            high_radar_bev_8x,
            high_lidar_bev,
            high_lidar_bev_8x,
            batch_dict['target_dicts']['heatmaps'],
            batch_dict['radar_pred_dicts'],
            batch_dict['lidar_pred_dicts'],
            batch_dict['radar_points'],
            batch_dict['gt_boxes'],
            radar_evidence_mask=radar_evidence_mask,
            lidar_intensity_mask=lidar_intensity_mask,
        )
        high_distill_loss = high_distill_loss * self.pfd_loss_weight
        object_kd_loss, object_kd_tb = self.object_kd_loss(
            high_radar_bev,
            high_radar_bev_8x,
            high_lidar_bev,
            high_lidar_bev_8x,
            batch_dict['target_dicts']['heatmaps'],
            batch_dict['gt_boxes'],
            radar_evidence_mask=radar_evidence_mask,
            lidar_intensity_mask=lidar_intensity_mask,
        )
        # NOTE: The final RadarDistill loss is the sum of the low-level AFD term
        # and the proposal-guided PFD term, matching the original RadarDistill
        # training objective inside the distillation backbone.
        distill_loss = low_distill_loss + high_distill_loss + object_kd_loss

        tb_dict = {
            'distill_loss': distill_loss.item(),
            'low_feature_loss': feature_loss.item(),
            'low_feature_loss_de8x': de8x_feature_loss.item(),
            'mask_loss': mask_loss.item(),
            'mask_loss_de8x': de8x_mask_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
        }
        tb_dict.update(evidence_tb)
        tb_dict.update(object_kd_tb)
        self.last_loss_terms = {
            'object_kd': object_kd_loss,
            'object_proto_kd': object_kd_loss.new_tensor(0.0),
            'semantic_heatmap_kd': object_kd_loss.new_tensor(0.0),
            'proposal_error_kd': object_kd_loss.new_tensor(0.0),
            'afd': low_distill_loss,
            'pfd': high_distill_loss,
            'distill_total': distill_loss,
        }
        return distill_loss, tb_dict

    def forward(self, batch_dict):
        spatial_features = batch_dict['radar_multi_scale_2d_features']['x_conv4']
        batch_dict['radar_cma_before_xconv4'] = spatial_features
        en_16x = self.encoder_1(spatial_features)
        de_8x = self.agg_1(torch.cat((self.decoder_1(en_16x), spatial_features), dim=1))
        en_32x = self.encoder_2(en_16x)
        de_16x = self.agg_2(torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1))
        x_conv4 = self.agg_3(torch.cat((self.decoder_3(de_16x), de_8x), dim=1))
        batch_dict['radar_cma_de8x'] = de_8x
        batch_dict['radar_cma_after_xconv4'] = x_conv4

        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4
        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x

        x_conv5 = batch_dict['radar_multi_scale_2d_features']['x_conv5']
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        batch_dict['radar_spatial_features_2d_8x'] = ups[-1]
        batch_dict['radar_spatial_features_2d'] = self.blocks[0](torch.cat(ups, dim=1))
        return batch_dict
