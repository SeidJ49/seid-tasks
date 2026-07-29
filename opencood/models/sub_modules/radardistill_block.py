import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
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

    def high_loss(
            self,
            radar_bev,
            radar_bev_8x,
            lidar_bev,
            lidar_bev_8x,
            heatmaps,
            radar_preds,
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

    def get_loss(self, batch_dict):
        low_lidar_bev = batch_dict['multi_scale_2d_features']['x_conv4']
        low_radar_bev = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev = batch_dict['radar_spatial_features_2d']
        high_lidar_bev = batch_dict['spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        high_lidar_bev_8x = batch_dict['spatial_features_2d_8x']
        self.debug_maps = {}
        radar_evidence_mask = self._build_radar_evidence_mask(batch_dict, high_radar_bev.shape[-2:])
        lidar_intensity_mask = self._build_lidar_intensity_mask(batch_dict, high_radar_bev.shape[-2:])
        gt_batch_hm = torch.cat(batch_dict['target_dicts']['heatmaps'], dim=1)
        self.debug_maps['teacher_heatmap_mask'] = torch.max(gt_batch_hm, dim=1, keepdim=True)[0].detach()
        if radar_evidence_mask is not None:
            self.debug_maps['radar_evidence_mask'] = radar_evidence_mask.detach()
        if lidar_intensity_mask is not None:
            self.debug_maps['lidar_intensity_mask'] = lidar_intensity_mask.detach()

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
                'afd': object_kd_loss.new_tensor(0.0),
                'pfd': object_kd_loss.new_tensor(0.0),
                'distill_total': object_kd_loss,
            }
            return object_kd_loss, tb_dict

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
            'afd': low_distill_loss,
            'pfd': high_distill_loss,
            'distill_total': distill_loss,
        }
        return distill_loss, tb_dict

    def forward(self, batch_dict):
        spatial_features = batch_dict['radar_multi_scale_2d_features']['x_conv4']
        en_16x = self.encoder_1(spatial_features)
        de_8x = self.agg_1(torch.cat((self.decoder_1(en_16x), spatial_features), dim=1))
        en_32x = self.encoder_2(en_16x)
        de_16x = self.agg_2(torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1))
        x_conv4 = self.agg_3(torch.cat((self.decoder_3(de_16x), de_8x), dim=1))

        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4
        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x

        x_conv5 = batch_dict['radar_multi_scale_2d_features']['x_conv5']
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        batch_dict['radar_spatial_features_2d_8x'] = ups[-1]
        batch_dict['radar_spatial_features_2d'] = self.blocks[0](torch.cat(ups, dim=1))
        return batch_dict
