"""WP4-only LiDAR reliability stage.

This model deliberately does not perform detection or LiDAR-radar fusion. It is a
lightweight diagnostic stage that returns BEV reliability/unreliability maps from
``processed_lidar`` so WP4 can be validated before WP5 adds a fusion gate.
"""

import torch.nn as nn

from opencood.models.sub_modules.lidar_unreliability_estimator import \
    LidarUnreliabilityEstimator


class LidarReliabilityStage3(nn.Module):
    """OpenCOOD model wrapper for WP4 LiDAR unreliability estimation."""

    def __init__(self, args):
        super(LidarReliabilityStage3, self).__init__()
        args = args or {}
        self.processed_lidar_key = args.get('processed_lidar_key', 'processed_lidar')
        estimator_args = args.get('unreliability', args)

        # Convenience: allow a normal point-pillar config to pass grid size under
        # point_pillar_scatter, while keeping the estimator config compact.
        if ('grid_size' not in estimator_args and
                'point_pillar_scatter' in args and
                'grid_size' in args['point_pillar_scatter']):
            estimator_args = dict(estimator_args)
            estimator_args['grid_size'] = args['point_pillar_scatter']['grid_size']

        self.unreliability_estimator = LidarUnreliabilityEstimator(estimator_args)

    def forward(self, data_dict):
        if self.processed_lidar_key not in data_dict:
            raise KeyError('%s missing from data_dict' % self.processed_lidar_key)

        processed_lidar = data_dict[self.processed_lidar_key]
        # Optional externally prepared masks let analysis code evaluate WP4 on
        # detection range / valid cells / GT object regions without changing the
        # estimator itself. The estimator accepts [B,H,W] or [B,1,H,W].
        if ('valid_region_mask' in data_dict or 'object_region_mask' in data_dict):
            processed_lidar = dict(processed_lidar)
            if 'valid_region_mask' in data_dict:
                processed_lidar['valid_region_mask'] = data_dict['valid_region_mask']
            if 'object_region_mask' in data_dict:
                processed_lidar['object_region_mask'] = data_dict['object_region_mask']

        return self.unreliability_estimator(processed_lidar)