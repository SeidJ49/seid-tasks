import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import RadarCenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import RadarPillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import RadarDynamicPillarVFESimple2D


class PillarnetRadarStudentKdRcs(nn.Module):
    """
    WP3 custom PillarNet radar student with motion-aware KD support.

    Expected inputs from the dataset:
    - raw `radar_points` with batch-index column
    - `object_bbx_center` / `object_bbx_mask` with TruckScenes class ids for CenterHead

    Key outputs:
    - loss / tb_dict from CenterHead when training or compute_loss=True
    - feature: radar BEV feature before CenterHead
    - kd_motion_mask: radar Doppler/motion prior in BEV feature space
    - rcs_confidence_mask: optional RCS prior in BEV feature space
    - final_box_dict for inference/post-processing
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = args['grid_size']
        self.class_names = args['class_names']
        self.radar_num_point_features = args.get('radar_num_point_features', 4)
        self.loss_weight = float(args.get('radar_loss_weight', 1.0))
        self.feature_key = args.get('feature_key', 'feature')
        self.use_kd_feature_adapter = bool(args.get('use_kd_feature_adapter', False))
        self.adapted_feature_key = args.get('adapted_feature_key', 'adapted_feature')
        self.use_adapted_feature_for_head = bool(args.get('use_adapted_feature_for_head', False))
        self.return_kd_motion_mask = args.get('return_kd_motion_mask', True)
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 7))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))
        self.radar_velocity_index = int(args.get('radar_velocity_index', 4))
        self.min_motion_velocity = float(args.get('min_motion_velocity', 0.2))

        rcs_cfg = args.get('rcs_mask', {})
        self.use_rcs_branch = bool(args.get('use_rcs_branch', False))
        self.return_rcs_confidence_mask = bool(args.get('return_rcs_confidence_mask', self.use_rcs_branch))
        # radar_points are collated as [batch_idx, x, y, z, radial_velocity, rcs].
        # Therefore the default RCS index is 5 here, while voxelized WP1 configs
        # use index 4 before the batch column is prepended.
        self.rcs_index = int(rcs_cfg.get('index', 5))
        self.rcs_thresh = float(rcs_cfg.get('thresh', 0.0))
        self.rcs_mode = rcs_cfg.get('mode', 'soft')
        self.rcs_temp = float(rcs_cfg.get('temp', 1.0))
        self.rcs_reduce = rcs_cfg.get('reduce', 'max')
        self.rcs_mask_kernel_size = int(rcs_cfg.get('kernel_size', 5))
        self.rcs_mask_scale = float(rcs_cfg.get('scale', 1.0))
        self.rcs_boost = float(args.get('rcs_boost', 0.0))

        evidence_cfg = args.get('radar_evidence_mask', {})
        self.use_radar_evidence_mask = bool(evidence_cfg.get('enabled', False))
        self.radar_evidence_mask_key = evidence_cfg.get('key', 'radar_evidence_mask')
        self.evidence_rcs_center = float(evidence_cfg.get('rcs_center', -12.0))
        self.evidence_rcs_scale = max(float(evidence_cfg.get('rcs_scale', 4.0)), 1e-6)
        self.evidence_doppler_center = float(evidence_cfg.get('doppler_center', 2.0))
        self.evidence_doppler_scale = max(float(evidence_cfg.get('doppler_scale', 4.0)), 1e-6)
        self.evidence_rcs_alpha = float(evidence_cfg.get('rcs_alpha', 0.35))
        self.evidence_combine = evidence_cfg.get('combine', 'max')
        self.evidence_reduce = evidence_cfg.get('reduce', 'max')
        self.evidence_kernel_size = int(evidence_cfg.get('kernel_size', 5))
        self.evidence_scale = float(evidence_cfg.get('scale', 1.0))

        gt_mask_cfg = args.get('gt_foreground_mask', {})
        self.use_gt_foreground_mask = bool(gt_mask_cfg.get('enabled', False))
        self.gt_foreground_mask_key = gt_mask_cfg.get('key', 'gt_foreground_mask')
        self.gt_foreground_mask_kernel_size = int(gt_mask_cfg.get('kernel_size', 1))
        self.gt_foreground_mask_scale = float(gt_mask_cfg.get('scale', 1.0))

        self.radar_vfe = RadarDynamicPillarVFESimple2D(
            args['radar_vfe'],
            num_point_features=self.radar_num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.backbone = RadarPillarRes18BackBone8x(self.grid_size)
        self.bev_backbone = BaseBEVBackboneV2(args['radar_bev_backbone'])
        adapter_channels = int(args.get('kd_feature_adapter_channels', args['radar_head']['input_channels']))
        self.kd_feature_adapter = nn.Sequential(
            nn.Conv2d(args['radar_head']['input_channels'], adapter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_channels, args['radar_head']['input_channels'], kernel_size=1, bias=True),
        ) if self.use_kd_feature_adapter else None
        self.head = RadarCenterHead(
            args['radar_head'],
            input_channels=args['radar_head']['input_channels'],
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            feature_key='spatial_features_2d',
            pred_dicts_key='radar_pred_dicts',
            target_dicts_key='target_dicts',
            final_box_dict_key='final_box_dict',
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

    def _build_motion_mask(self, radar_points, batch_size, target_hw):
        if radar_points.numel() == 0 or radar_points.shape[1] <= self.radar_velocity_index:
            return None

        batch_idx = radar_points[:, 0].long()
        xyz = radar_points[:, 1:4]
        velocity = radar_points[:, self.radar_velocity_index].abs()
        moving = velocity >= self.min_motion_velocity
        coords = torch.floor(
            (xyz[:, [0, 1]] - xyz.new_tensor(self.point_cloud_range[:2])) /
            xyz.new_tensor(self.voxel_size[:2])
        ).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < int(self.grid_size[0])) &
            (coords[:, 1] >= 0) & (coords[:, 1] < int(self.grid_size[1])) &
            moving
        )
        if not valid.any():
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))

        coords = coords[valid]
        batch_idx = batch_idx[valid]
        mask = radar_points.new_zeros((batch_size, 1, int(self.grid_size[1]), int(self.grid_size[0])))
        mask[batch_idx, 0, coords[:, 1], coords[:, 0]] = 1.0

        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='nearest')
        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.motion_mask_kernel_size, stride=1, padding=pad)
        mask = mask.clamp_(0.0, 1.0)
        if self.motion_mask_scale != 1.0:
            mask = (mask * self.motion_mask_scale).clamp_(0.0, 1.0)
        return mask

    def _build_rcs_mask(self, radar_points, batch_size, target_hw):
        if not self.use_rcs_branch:
            return None
        if radar_points.numel() == 0:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
        if radar_points.shape[1] <= self.rcs_index:
            raise ValueError(
                f'RCS branch expects radar_points[:, {self.rcs_index}], '
                f'but got only {radar_points.shape[1]} columns. Set '
                'preprocess.args.radar_num_point_features and model.args.radar_num_point_features to 5.'
            )

        batch_idx = radar_points[:, 0].long()
        xyz = radar_points[:, 1:4]
        rcs_val = radar_points[:, self.rcs_index]
        if self.rcs_mode == 'soft':
            score = torch.sigmoid(self.rcs_temp * (rcs_val - self.rcs_thresh))
        else:
            score = (rcs_val > self.rcs_thresh).to(radar_points.dtype)

        coords = torch.floor(
            (xyz[:, [0, 1]] - xyz.new_tensor(self.point_cloud_range[:2])) /
            xyz.new_tensor(self.voxel_size[:2])
        ).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < int(self.grid_size[0])) &
            (coords[:, 1] >= 0) & (coords[:, 1] < int(self.grid_size[1]))
        )
        if not valid.any():
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))

        batch_idx = batch_idx[valid]
        coords = coords[valid]
        score = score[valid]
        flat_index = batch_idx * int(self.grid_size[1]) * int(self.grid_size[0]) + coords[:, 1] * int(self.grid_size[0]) + coords[:, 0]
        flat_size = batch_size * int(self.grid_size[1]) * int(self.grid_size[0])
        if self.rcs_reduce == 'mean':
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
        mask = mask_flat.view(batch_size, 1, int(self.grid_size[1]), int(self.grid_size[0]))

        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)
        if self.rcs_mask_kernel_size > 1:
            pad = self.rcs_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.rcs_mask_kernel_size, stride=1, padding=pad)
        mask = mask.clamp_(0.0, 1.0)
        if self.rcs_mask_scale != 1.0:
            mask = (mask * self.rcs_mask_scale).clamp_(0.0, 1.0)
        return mask

    def _build_radar_evidence_mask(self, radar_points, batch_size, target_hw):
        if not self.use_radar_evidence_mask:
            return None
        if radar_points.numel() == 0:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
        if radar_points.shape[1] <= max(self.radar_velocity_index, self.rcs_index):
            raise ValueError(
                'Radar evidence mask expects collated radar_points with '
                f'Doppler index {self.radar_velocity_index} and RCS index {self.rcs_index}, '
                f'but got {radar_points.shape[1]} columns.'
            )

        batch_idx = radar_points[:, 0].long()
        xyz = radar_points[:, 1:4]
        doppler = radar_points[:, self.radar_velocity_index].abs()
        rcs_val = radar_points[:, self.rcs_index]

        doppler_score = torch.sigmoid(
            (doppler - self.evidence_doppler_center) / self.evidence_doppler_scale
        )
        rcs_score = torch.sigmoid(
            (rcs_val - self.evidence_rcs_center) / self.evidence_rcs_scale
        ) * self.evidence_rcs_alpha

        if self.evidence_combine == 'sum':
            score = (doppler_score + rcs_score).clamp(0.0, 1.0)
        elif self.evidence_combine == 'mean':
            score = ((doppler_score + rcs_score) * 0.5).clamp(0.0, 1.0)
        else:
            score = torch.maximum(doppler_score, rcs_score).clamp(0.0, 1.0)

        coords = torch.floor(
            (xyz[:, [0, 1]] - xyz.new_tensor(self.point_cloud_range[:2])) /
            xyz.new_tensor(self.voxel_size[:2])
        ).long()
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        target_x = torch.floor(coords[:, 0].float() * target_w / float(self.grid_size[0])).long()
        target_y = torch.floor(coords[:, 1].float() * target_h / float(self.grid_size[1])).long()
        valid = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (coords[:, 0] >= 0) & (coords[:, 0] < int(self.grid_size[0])) &
            (coords[:, 1] >= 0) & (coords[:, 1] < int(self.grid_size[1])) &
            (target_x >= 0) & (target_x < target_w) &
            (target_y >= 0) & (target_y < target_h)
        )
        if not valid.any():
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))

        batch_idx = batch_idx[valid]
        target_x = target_x[valid]
        target_y = target_y[valid]
        score = score[valid]
        flat_index = batch_idx * target_h * target_w + target_y * target_w + target_x
        flat_size = batch_size * target_h * target_w
        if self.evidence_reduce == 'mean':
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
        mask = mask_flat.view(batch_size, 1, target_h, target_w)

        if self.evidence_kernel_size > 1:
            pad = self.evidence_kernel_size // 2
            mask = F.max_pool2d(mask, self.evidence_kernel_size, stride=1, padding=pad)
        mask = mask.clamp_(0.0, 1.0)
        if self.evidence_scale != 1.0:
            mask = (mask * self.evidence_scale).clamp_(0.0, 1.0)
        return mask

    def _build_gt_foreground_mask(self, gt_boxes, target_hw):
        if not self.use_gt_foreground_mask:
            return None

        batch_size = gt_boxes.shape[0]
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        device = gt_boxes.device
        dtype = gt_boxes.dtype
        mask = gt_boxes.new_zeros((batch_size, 1, target_h, target_w))

        xs = (
            torch.arange(target_w, device=device, dtype=dtype) + 0.5
        ) * (self.point_cloud_range[3] - self.point_cloud_range[0]) / target_w + self.point_cloud_range[0]
        ys = (
            torch.arange(target_h, device=device, dtype=dtype) + 0.5
        ) * (self.point_cloud_range[4] - self.point_cloud_range[1]) / target_h + self.point_cloud_range[1]
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        for batch_idx in range(batch_size):
            valid = gt_boxes[batch_idx, :, -1] > 0
            cur_boxes = gt_boxes[batch_idx, valid]
            for box in cur_boxes:
                cx, cy = box[0], box[1]
                dx = box[3].clamp_min(1e-3)
                dy = box[4].clamp_min(1e-3)
                yaw = box[6]
                rel_x = xx - cx
                rel_y = yy - cy
                cos_yaw = torch.cos(-yaw)
                sin_yaw = torch.sin(-yaw)
                local_x = rel_x * cos_yaw - rel_y * sin_yaw
                local_y = rel_x * sin_yaw + rel_y * cos_yaw
                inside = (local_x.abs() <= dx * 0.5) & (local_y.abs() <= dy * 0.5)
                mask[batch_idx, 0] = torch.maximum(mask[batch_idx, 0], inside.to(dtype))

        if self.gt_foreground_mask_kernel_size > 1:
            pad = self.gt_foreground_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.gt_foreground_mask_kernel_size, stride=1, padding=pad)
        mask = mask.clamp_(0.0, 1.0)
        if self.gt_foreground_mask_scale != 1.0:
            mask = (mask * self.gt_foreground_mask_scale).clamp_(0.0, 1.0)
        return mask

    def _encode_radar_to_bev(self, batch_dict):
        batch_dict = self.radar_vfe(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # Use the standard BEV neck without the full RadarDistill fusion block.
        batch_dict['multi_scale_2d_features'] = batch_dict['radar_multi_scale_2d_features']
        batch_dict['multi_scale_2d_strides'] = batch_dict['radar_multi_scale_2d_strides']
        batch_dict = self.bev_backbone(batch_dict)
        return batch_dict

    def forward(self, data_dict):
        batch_size = int(data_dict['object_bbx_center'].shape[0])
        compute_loss = bool(data_dict.get('compute_loss', False))
        batch_dict = {
            'batch_size': batch_size,
            'radar_points': data_dict['radar_points'],
            'gt_boxes': self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask']),
            'compute_loss': compute_loss,
        }

        batch_dict = self._encode_radar_to_bev(batch_dict)
        feature = batch_dict['spatial_features_2d']
        rcs_confidence_mask = self._build_rcs_mask(data_dict['radar_points'], batch_size, feature.shape[-2:])
        if rcs_confidence_mask is not None and self.rcs_boost != 0.0:
            feature = feature * (1.0 + self.rcs_boost * rcs_confidence_mask)
            batch_dict['spatial_features_2d'] = feature
        adapted_feature = None
        if self.kd_feature_adapter is not None:
            adapted_feature = self.kd_feature_adapter(feature)
            if self.use_adapted_feature_for_head:
                batch_dict['spatial_features_2d'] = adapted_feature
        batch_dict = self.head(batch_dict)
        raw_pred_dicts = [
            {key: value for key, value in pred_dict.items()}
            for pred_dict in batch_dict.get('radar_pred_dicts', [])
        ]

        output_dict = {
            self.feature_key: feature,
            'radar_pred_dicts': raw_pred_dicts,
            'gt_boxes_for_kd': batch_dict['gt_boxes'],
            'final_box_dict': batch_dict.get('final_box_dict', []),
        }
        if adapted_feature is not None:
            output_dict[self.adapted_feature_key] = adapted_feature
            output_dict['detector_feature_key'] = self.adapted_feature_key if self.use_adapted_feature_for_head else self.feature_key
        if self.return_rcs_confidence_mask and rcs_confidence_mask is not None:
            output_dict['rcs_confidence_mask'] = rcs_confidence_mask
        radar_evidence_mask = self._build_radar_evidence_mask(data_dict['radar_points'], batch_size, feature.shape[-2:])
        if radar_evidence_mask is not None:
            output_dict[self.radar_evidence_mask_key] = radar_evidence_mask
        gt_foreground_mask = self._build_gt_foreground_mask(batch_dict['gt_boxes'], feature.shape[-2:])
        if gt_foreground_mask is not None:
            output_dict[self.gt_foreground_mask_key] = gt_foreground_mask
        if self.return_kd_motion_mask:
            kd_motion_mask = self._build_motion_mask(data_dict['radar_points'], batch_size, feature.shape[-2:])
            if kd_motion_mask is not None:
                output_dict['kd_motion_mask'] = kd_motion_mask

        if self.training or compute_loss:
            head_loss, head_tb = self.head.get_loss()
            total_loss = self.loss_weight * head_loss
            output_dict.update({
                'loss': total_loss,
                'tb_dict': {
                    'total_loss': total_loss.item(),
                    'radar_head_loss': head_loss.item(),
                    **head_tb,
                },
                'radar_head_loss': head_loss,
            })

        return output_dict
