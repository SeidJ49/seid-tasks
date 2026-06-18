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

    This is Seid's own lightweight PillarNet student, not the BM2CP
    RadarDistill model. It uses PillarNet-style dynamic radar pillarization,
    sparse ResNet BEV encoding, BEV neck, and CenterHead. It does not use
    PointPillar VFE/scatter/anchor heads.

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

        self.radar_vfe = RadarDynamicPillarVFESimple2D(
            args['radar_vfe'],
            num_point_features=self.radar_num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.backbone = RadarPillarRes18BackBone8x(self.grid_size)
        self.bev_backbone = BaseBEVBackboneV2(args['radar_bev_backbone'])
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
        batch_dict = self.head(batch_dict)

        output_dict = {
            self.feature_key: feature,
            'final_box_dict': batch_dict.get('final_box_dict', []),
        }
        if self.return_rcs_confidence_mask and rcs_confidence_mask is not None:
            output_dict['rcs_confidence_mask'] = rcs_confidence_mask
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