"""BEV LiDAR unreliability estimation for WP4.

This module intentionally stays fusion-free: it consumes LiDAR voxel evidence and
emits diagnostic BEV maps that can later be used by WP5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarUnreliabilityEstimator(nn.Module):
    """Compute a heuristic LiDAR unreliability map ``U_L`` in BEV.

    ``U_L`` semantics:
      * 0.0 -> LiDAR evidence looks reliable/dense.
      * 1.0 -> LiDAR evidence looks unreliable/sparse/degraded.

    The estimator combines two interpretable signals:
      1. point-count unreliability, with robust percentile scaling rather than
         max scaling to avoid saturation by outliers;
      2. local occupancy sparsity, measured by average pooling a binary BEV
         occupancy map.

    Args are plain YAML-friendly values. Important options:
      - grid_size: OpenCOOD grid size [W, H, D] or [D, H, W]. Only H/W matter.
      - map_size: explicit [H, W] override.
      - point_count_percentile: e.g. 0.95 for robust scaling over non-empty cells.
      - valid_region_mode: ``all`` or ``occupancy_neighborhood``.
    """

    def __init__(self, args):
        super(LidarUnreliabilityEstimator, self).__init__()
        args = args or {}
        self.map_size = self._resolve_map_size(args)
        self.pool_kernel = int(args.get('pool_kernel', 7))
        self.point_count_weight = float(args.get('point_count_weight', 0.6))
        self.occupancy_sparsity_weight = float(
            args.get('occupancy_sparsity_weight', 0.4))
        self.point_count_percentile = float(
            args.get('point_count_percentile', 0.95))
        self.min_point_count_scale = float(args.get('min_point_count_scale', 1.0))
        self.eps = float(args.get('eps', 1e-6))
        self.valid_region_mode = args.get('valid_region_mode', 'occupancy_neighborhood')

        if self.pool_kernel < 1 or self.pool_kernel % 2 == 0:
            raise ValueError('pool_kernel must be a positive odd integer')
        if not 0.0 < self.point_count_percentile <= 1.0:
            raise ValueError('point_count_percentile must be in (0, 1]')
        if self.point_count_weight < 0 or self.occupancy_sparsity_weight < 0:
            raise ValueError('unreliability weights must be non-negative')
        if self.point_count_weight + self.occupancy_sparsity_weight <= 0:
            raise ValueError('at least one unreliability weight must be positive')

    @staticmethod
    def _resolve_map_size(args):
        if 'map_size' in args and args['map_size'] is not None:
            h, w = args['map_size']
            return int(h), int(w)

        grid_size = args.get('grid_size', None)
        if grid_size is None and 'point_pillar_scatter' in args:
            grid_size = args['point_pillar_scatter'].get('grid_size', None)

        if grid_size is None:
            return None

        # OpenCOOD commonly stores grid_size as [W, H, D].
        if len(grid_size) >= 2:
            w = int(grid_size[0])
            h = int(grid_size[1])
            return h, w
        raise ValueError('grid_size must contain at least W and H')

    @staticmethod
    def _infer_batch_size(processed_lidar, voxel_coords):
        if 'record_len' in processed_lidar:
            record_len = processed_lidar['record_len']
            if torch.is_tensor(record_len):
                return int(record_len.sum().item())
        if voxel_coords.numel() == 0:
            return 1
        return int(voxel_coords[:, 0].max().item()) + 1

    def _infer_map_size_from_coords(self, voxel_coords):
        if self.map_size is not None:
            return self.map_size
        if voxel_coords.numel() == 0:
            raise ValueError('map_size/grid_size is required for empty voxel_coords')
        h = int(voxel_coords[:, -2].max().item()) + 1
        w = int(voxel_coords[:, -1].max().item()) + 1
        return h, w

    def _build_count_and_occupancy(self, processed_lidar):
        voxel_coords = processed_lidar['voxel_coords'].long()
        voxel_num_points = processed_lidar['voxel_num_points'].float()
        device = voxel_coords.device
        batch_size = self._infer_batch_size(processed_lidar, voxel_coords)
        height, width = self._infer_map_size_from_coords(voxel_coords)

        point_count = torch.zeros(batch_size, 1, height, width, device=device)
        occupancy_hits = torch.zeros_like(point_count)

        if voxel_coords.numel() == 0:
            occupancy = occupancy_hits
            return point_count, occupancy

        batch_idx = voxel_coords[:, 0]
        y_idx = voxel_coords[:, -2]
        x_idx = voxel_coords[:, -1]
        valid = ((batch_idx >= 0) & (batch_idx < batch_size) &
                 (y_idx >= 0) & (y_idx < height) &
                 (x_idx >= 0) & (x_idx < width))

        if valid.any():
            batch_idx = batch_idx[valid]
            y_idx = y_idx[valid]
            x_idx = x_idx[valid]
            points = voxel_num_points[valid]
            flat_idx = batch_idx * (height * width) + y_idx * width + x_idx
            point_count.view(-1).scatter_add_(0, flat_idx, points)
            occupancy_hits.view(-1).scatter_add_(
                0, flat_idx, torch.ones_like(points, dtype=point_count.dtype))

        occupancy = (occupancy_hits > 0).float()
        return point_count, occupancy

    def _robust_point_count_norm(self, point_count):
        """Percentile-normalize non-empty cells independently per batch item."""
        normalized = torch.zeros_like(point_count)
        batch_size = point_count.shape[0]
        for b in range(batch_size):
            pc = point_count[b]
            positive = pc[pc > 0]
            if positive.numel() == 0:
                scale = torch.tensor(self.min_point_count_scale,
                                     device=point_count.device,
                                     dtype=point_count.dtype)
            else:
                scale = torch.quantile(positive, self.point_count_percentile)
                scale = torch.clamp(scale, min=self.min_point_count_scale)
            normalized[b] = torch.clamp(pc / (scale + self.eps), 0.0, 1.0)
        return normalized

    def _masked_mean_per_batch(self, values, mask):
        dims = (1, 2, 3)
        numerator = (values * mask.float()).sum(dim=dims)
        denominator = mask.float().sum(dim=dims).clamp(min=1.0)
        return numerator / denominator

    def _coerce_mask(self, mask, like):
        """Convert optional [B,H,W] or [B,1,H,W] masks to ``like`` shape."""
        if mask is None:
            return None
        mask = mask.to(device=like.device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != like.shape[-2:]:
            raise ValueError('mask spatial shape must match U_L')
        return mask.float()

    def forward(self, processed_lidar):
        point_count, occupancy = self._build_count_and_occupancy(processed_lidar)
        pc_norm = self._robust_point_count_norm(point_count)
        point_count_unreliability = 1.0 - pc_norm

        pad = self.pool_kernel // 2
        local_occupancy_ratio = F.avg_pool2d(
            occupancy, kernel_size=self.pool_kernel, stride=1, padding=pad)
        occupancy_sparsity = 1.0 - local_occupancy_ratio

        weight_sum = self.point_count_weight + self.occupancy_sparsity_weight
        U_L = ((self.point_count_weight * point_count_unreliability +
                self.occupancy_sparsity_weight * occupancy_sparsity) /
               weight_sum)
        U_L = torch.clamp(U_L, 0.0, 1.0)

        occupancy_neighborhood_mask = (local_occupancy_ratio > 0).float()
        external_valid_mask = self._coerce_mask(
            processed_lidar.get('valid_region_mask', None), U_L)
        object_region_mask = self._coerce_mask(
            processed_lidar.get('object_region_mask', None), U_L)

        if external_valid_mask is not None:
            valid_region_mask = external_valid_mask
        elif self.valid_region_mode == 'all':
            valid_region_mask = torch.ones_like(U_L)
        elif self.valid_region_mode == 'occupancy_neighborhood':
            valid_region_mask = occupancy_neighborhood_mask
        else:
            raise ValueError('unknown valid_region_mode: %s' % self.valid_region_mode)

        output = {
            'U_L': U_L,
            'lidar_unreliability': U_L,
            'lidar_point_count_map': point_count,
            'lidar_point_count_norm': pc_norm,
            'lidar_point_count_unreliability': point_count_unreliability,
            'lidar_occupancy_map': occupancy,
            'lidar_local_occupancy_ratio': local_occupancy_ratio,
            'lidar_occupancy_sparsity': occupancy_sparsity,
            'lidar_valid_region_mask': valid_region_mask,
            'lidar_occupancy_neighborhood_mask': occupancy_neighborhood_mask,
            'mean_U_L_global': U_L.flatten(1).mean(dim=1),
            'mean_U_L_valid': self._masked_mean_per_batch(U_L, valid_region_mask),
            'mean_U_count_valid': self._masked_mean_per_batch(
                point_count_unreliability, valid_region_mask),
            'mean_U_sparsity_valid': self._masked_mean_per_batch(
                occupancy_sparsity, valid_region_mask),
        }
        if object_region_mask is not None:
            output['lidar_object_region_mask'] = object_region_mask
            output['mean_U_L_object'] = self._masked_mean_per_batch(
                U_L, object_region_mask)
            output['mean_U_count_object'] = self._masked_mean_per_batch(
                point_count_unreliability, object_region_mask)
            output['mean_U_sparsity_object'] = self._masked_mean_per_batch(
                occupancy_sparsity, object_region_mask)
        return output