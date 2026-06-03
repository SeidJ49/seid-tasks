import torch.nn as nn

from opencood.models.sub_modules.thesis_modules import LidarUnreliabilityEstimator


class LidarReliabilityStage3(nn.Module):
    """
    WP4 / Stage 3: LiDAR unreliability estimation only.

    This module does not fuse radar and does not introduce a new detector. It
    emits U_L in [0, 1] over BEV cells, where high means LiDAR is unreliable.
    U_L is heuristic: point-count degradation + occupancy sparsity.
    """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args['batch_size']
        self.unreliability_estimator = LidarUnreliabilityEstimator(
            grid_size=args['grid_size'],
            cfg=args.get('unreliability', {}),
        )

    def _infer_batch_size(self, data_dict):
        if 'label_dict' in data_dict and 'pos_equal_one' in data_dict['label_dict']:
            return int(data_dict['label_dict']['pos_equal_one'].shape[0])
        if 'object_bbx_center' in data_dict:
            return int(data_dict['object_bbx_center'].shape[0])
        if 'record_len' in data_dict:
            return int(data_dict['record_len'].sum().item())
        processed_lidar = data_dict['processed_lidar']
        voxel_coords = processed_lidar['voxel_coords']
        if voxel_coords.numel() == 0:
            return self.batch_size
        return int(voxel_coords[:, 0].max().item()) + 1

    def forward(self, data_dict):
        processed_lidar = data_dict['processed_lidar']
        batch_size = self._infer_batch_size(data_dict)
        maps = self.unreliability_estimator(
            voxel_coords=processed_lidar['voxel_coords'],
            voxel_num_points=processed_lidar['voxel_num_points'],
            batch_size=batch_size,
            device=processed_lidar['voxel_num_points'].device,
        )
        maps['batch_size'] = batch_size
        # Backwards-friendly aliases, but semantics are explicit: high = unreliable.
        maps['U_L'] = maps['lidar_unreliability']
        return maps