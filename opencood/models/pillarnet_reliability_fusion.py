import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.models.sub_modules.radardistill_head import CenterHead
from opencood.models.sub_modules.radardistill_spconv_backbone import PillarRes18BackBone8x, RadarPillarRes18BackBone8x
from opencood.models.sub_modules.radardistill_vfe import DynamicPillarVFESimple2D, RadarDynamicPillarVFESimple2D


class ReliabilityGuidedPillarFusion(nn.Module):
    """Small WP5 BEV-local fusion gate.

    ``lidar_unreliability`` is U_L from WP4: high means LiDAR should be trusted
    less.  The learned gate predicts a LiDAR weight and is biased by
    ``1 - U_L`` so the final map stays interpretable and coupled to WP4.
    """

    def __init__(self, feature_channels, cfg=None):
        super().__init__()
        cfg = cfg or {}
        hidden_channels = int(cfg.get('hidden_channels', max(32, feature_channels // 2)))
        self.use_radar_cues = bool(cfg.get('use_radar_cues', True))
        self.reliability_bias = float(cfg.get('reliability_bias', 0.5))
        self.min_lidar_gate = float(cfg.get('min_lidar_gate', 0.05))
        self.max_lidar_gate = float(cfg.get('max_lidar_gate', 0.95))
        cue_channels = 2 if self.use_radar_cues else 0
        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 1 + cue_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.post_fusion = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_feature, radar_feature, lidar_unreliability, radar_motion_mask=None, rcs_confidence_mask=None):
        if lidar_unreliability.shape[-2:] != lidar_feature.shape[-2:]:
            lidar_unreliability = F.interpolate(
                lidar_unreliability.float(), size=lidar_feature.shape[-2:], mode='bilinear', align_corners=False
            )
        lidar_unreliability = lidar_unreliability.clamp(0.0, 1.0)

        inputs = [lidar_feature, radar_feature, lidar_unreliability]
        if self.use_radar_cues:
            if radar_motion_mask is None:
                radar_motion_mask = lidar_unreliability.new_zeros(lidar_unreliability.shape)
            if rcs_confidence_mask is None:
                rcs_confidence_mask = lidar_unreliability.new_zeros(lidar_unreliability.shape)
            if radar_motion_mask.shape[-2:] != lidar_feature.shape[-2:]:
                radar_motion_mask = F.interpolate(
                    radar_motion_mask.float(), size=lidar_feature.shape[-2:], mode='bilinear', align_corners=False
                )
            if rcs_confidence_mask.shape[-2:] != lidar_feature.shape[-2:]:
                rcs_confidence_mask = F.interpolate(
                    rcs_confidence_mask.float(), size=lidar_feature.shape[-2:], mode='bilinear', align_corners=False
                )
            inputs.extend([radar_motion_mask.clamp(0.0, 1.0), rcs_confidence_mask.clamp(0.0, 1.0)])

        learned_lidar_gate = torch.sigmoid(self.gate_net(torch.cat(inputs, dim=1)))
        reliability_lidar_gate = 1.0 - lidar_unreliability
        lidar_gate = (
            (1.0 - self.reliability_bias) * learned_lidar_gate
            + self.reliability_bias * reliability_lidar_gate
        ).clamp(self.min_lidar_gate, self.max_lidar_gate)
        radar_gate = 1.0 - lidar_gate
        fused_feature = lidar_gate * lidar_feature + radar_gate * radar_feature
        fused_feature = self.post_fusion(fused_feature)
        return fused_feature, lidar_gate, radar_gate, learned_lidar_gate


class PillarnetReliabilityFusion(nn.Module):
    """WP5 true PillarNet reliability-guided LiDAR-radar fusion.

    This is the PillarNet/CenterHead counterpart to the older PointPillar WP5
    file. It keeps the WP5 requirements explicit:

    - two independent BEV branches: LiDAR and radar,
    - WP4-style LiDAR unreliability map U_L from point count + occupancy,
    - a small spatial gate in BEV,
    - interpretable outputs: U_L, lidar/radar gates, motion/RCS cue masks,
    - no transformer or heavy routing stack.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_cloud_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.grid_size = args['grid_size']
        self.class_names = args['class_names']
        self.lidar_num_point_features = int(args.get('lidar_num_point_features', args.get('num_point_features', 5)))
        self.radar_num_point_features = int(args.get('radar_num_point_features', 5))
        self.loss_weight = float(args.get('fusion_loss_weight', 1.0))
        self.feature_key = args.get('feature_key', 'feature')

        rel_cfg = args.get('unreliability', {})
        self.reliability_pool_kernel = int(rel_cfg.get('pool_kernel', 7))
        self.expected_points_per_cell = float(rel_cfg.get('expected_points_per_cell', 6.0))
        self.point_count_weight = float(rel_cfg.get('point_count_weight', 0.6))
        self.occupancy_sparsity_weight = float(rel_cfg.get('occupancy_sparsity_weight', 0.4))
        self.unreliability_eps = float(rel_cfg.get('eps', 1e-6))

        self.radar_velocity_index = int(args.get('radar_velocity_index', 4))
        self.min_motion_velocity = float(args.get('min_motion_velocity', 0.2))
        self.motion_mask_kernel_size = int(args.get('motion_mask_kernel_size', 7))
        self.motion_mask_scale = float(args.get('motion_mask_scale', 1.0))
        rcs_cfg = args.get('rcs_mask', {})
        self.use_rcs_branch = bool(args.get('use_rcs_branch', True))
        self.rcs_index = int(rcs_cfg.get('index', 5))
        self.rcs_thresh = float(rcs_cfg.get('thresh', 0.0))
        self.rcs_temp = float(rcs_cfg.get('temp', 1.0))
        self.rcs_mode = rcs_cfg.get('mode', 'soft')
        self.rcs_reduce = rcs_cfg.get('reduce', 'max')
        self.rcs_mask_kernel_size = int(rcs_cfg.get('kernel_size', 5))
        self.rcs_mask_scale = float(rcs_cfg.get('scale', 1.0))

        self.lidar_vfe = DynamicPillarVFESimple2D(
            args['lidar_vfe'],
            num_point_features=self.lidar_num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.radar_vfe = RadarDynamicPillarVFESimple2D(
            args['radar_vfe'],
            num_point_features=self.radar_num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.lidar_backbone = PillarRes18BackBone8x(self.grid_size)
        self.radar_backbone = RadarPillarRes18BackBone8x(self.grid_size)
        self.lidar_bev_backbone = BaseBEVBackboneV2(args['lidar_bev_backbone'])
        self.radar_bev_backbone = BaseBEVBackboneV2(args['radar_bev_backbone'])

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
            self.point_count_weight * point_count_unreliability
            + self.occupancy_sparsity_weight * occupancy_sparsity
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

    def _build_motion_mask(self, radar_points, batch_size, target_hw):
        if radar_points.numel() == 0 or radar_points.shape[1] <= self.radar_velocity_index:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
        velocity = radar_points[:, self.radar_velocity_index].abs()
        moving_score = (velocity >= self.min_motion_velocity).to(radar_points.dtype)
        mask = self._points_to_bev_counts(radar_points, batch_size, value=moving_score, reduce='max')
        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask.float(), size=target_hw, mode='nearest')
        if self.motion_mask_kernel_size > 1:
            pad = self.motion_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.motion_mask_kernel_size, stride=1, padding=pad)
        return (mask * self.motion_mask_scale).clamp(0.0, 1.0)

    def _build_rcs_mask(self, radar_points, batch_size, target_hw):
        if not self.use_rcs_branch:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
        if radar_points.numel() == 0:
            return radar_points.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
        if radar_points.shape[1] <= self.rcs_index:
            raise ValueError(
                f'WP5 RCS gate expects radar_points[:, {self.rcs_index}], but got {radar_points.shape[1]} columns. '
                'Use radar_num_point_features: 5 so raw radar_points are [batch_idx, x, y, z, radial_velocity, rcs].'
            )
        rcs = radar_points[:, self.rcs_index]
        if self.rcs_mode == 'soft':
            rcs_score = torch.sigmoid(self.rcs_temp * (rcs - self.rcs_thresh))
        else:
            rcs_score = (rcs > self.rcs_thresh).to(radar_points.dtype)
        mask = self._points_to_bev_counts(radar_points, batch_size, value=rcs_score, reduce=self.rcs_reduce)
        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask.float(), size=target_hw, mode='bilinear', align_corners=False)
        if self.rcs_mask_kernel_size > 1:
            pad = self.rcs_mask_kernel_size // 2
            mask = F.max_pool2d(mask, self.rcs_mask_kernel_size, stride=1, padding=pad)
        return (mask * self.rcs_mask_scale).clamp(0.0, 1.0)

    def _encode_lidar_to_bev(self, data_dict, batch_size):
        batch_dict = {'batch_size': batch_size, 'points': data_dict['lidar_points']}
        batch_dict = self.lidar_vfe(batch_dict)
        batch_dict = self.lidar_backbone(batch_dict)
        batch_dict = self.lidar_bev_backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def _encode_radar_to_bev(self, data_dict, batch_size):
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
        gt_boxes = self._masked_gt_boxes(data_dict['object_bbx_center'], data_dict['object_bbx_mask'])

        lidar_feature = self._encode_lidar_to_bev(data_dict, batch_size)
        radar_feature = self._encode_radar_to_bev(data_dict, batch_size)
        target_hw = lidar_feature.shape[-2:]
        if radar_feature.shape[-2:] != target_hw:
            radar_feature = F.interpolate(radar_feature, size=target_hw, mode='bilinear', align_corners=False)

        reliability_maps = self._build_lidar_unreliability(data_dict['lidar_points'], batch_size, target_hw)
        radar_motion_mask = self._build_motion_mask(data_dict['radar_points'], batch_size, target_hw)
        rcs_confidence_mask = self._build_rcs_mask(data_dict['radar_points'], batch_size, target_hw)
        fused_feature, lidar_gate, radar_gate, learned_lidar_gate = self.fusion_gate(
            lidar_feature,
            radar_feature,
            reliability_maps['U_L'],
            radar_motion_mask=radar_motion_mask,
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
            'kd_motion_mask': radar_motion_mask,
            'rcs_confidence_mask': rcs_confidence_mask,
            'final_box_dict': self._select_final_box_dict(head_dict),
        }
        if 'batch_cls_preds' in head_dict:
            output_dict['cls_preds'] = head_dict['batch_cls_preds']
        if 'batch_box_preds' in head_dict:
            output_dict['reg_preds'] = head_dict['batch_box_preds']

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