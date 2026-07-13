"""BEV LiDAR unreliability estimation for WP4.

This module intentionally stays fusion-free: it consumes LiDAR voxel evidence and
emits diagnostic BEV maps that can later be used by WP5.  ``U_L`` is an
unreliability map, so high values mean the LiDAR evidence in that BEV cell should
be trusted less.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarUnreliabilityEstimator(nn.Module):
    """Compute a LiDAR unreliability map ``U_L`` in BEV.

    ``U_L`` semantics:
      * 0.0 -> LiDAR evidence looks reliable/dense.
      * 1.0 -> LiDAR evidence looks unreliable/sparse/degraded.

    Two WP4 variants are supported:
      * ``mode: heuristic`` combines interpretable LiDAR-only cues.
      * ``mode: learned`` feeds the same cue maps into a small CNN head.

    The estimator never consumes radar features and never fuses modalities.  It
    only returns ``U_L`` plus diagnostic cue maps for later WP5 use.
    """

    def __init__(self, args):
        super(LidarUnreliabilityEstimator, self).__init__()
        args = args or {}
        self.map_size = self._resolve_map_size(args)
        self.mode = args.get('mode', 'heuristic')
        self.pool_kernel = int(args.get('pool_kernel', 7))
        self.point_count_weight = float(args.get('point_count_weight', 0.55))
        self.occupancy_sparsity_weight = float(
            args.get('occupancy_sparsity_weight', 0.35))
        self.distance_weight = float(args.get('distance_weight', 0.10))
        self.intensity_weight = float(args.get('intensity_weight', 0.0))
        self.use_intensity = bool(args.get('use_intensity', self.intensity_weight > 0))
        self.intensity_index = int(args.get('intensity_index', 3))
        self.point_count_percentile = float(
            args.get('point_count_percentile', 0.95))
        self.fixed_point_count_scale = args.get('fixed_point_count_scale', None)
        if self.fixed_point_count_scale is not None:
            self.fixed_point_count_scale = float(self.fixed_point_count_scale)
        self.intensity_percentile = float(args.get('intensity_percentile', 0.95))
        self.min_point_count_scale = float(args.get('min_point_count_scale', 1.0))
        self.min_intensity_scale = float(args.get('min_intensity_scale', 1.0))
        self.eps = float(args.get('eps', 1e-6))
        self.valid_region_mode = args.get('valid_region_mode', 'occupancy_neighborhood')

        if self.mode not in ('heuristic', 'learned'):
            raise ValueError('mode must be either heuristic or learned')
        if self.pool_kernel < 1 or self.pool_kernel % 2 == 0:
            raise ValueError('pool_kernel must be a positive odd integer')
        if not 0.0 < self.point_count_percentile <= 1.0:
            raise ValueError('point_count_percentile must be in (0, 1]')
        if not 0.0 < self.intensity_percentile <= 1.0:
            raise ValueError('intensity_percentile must be in (0, 1]')
        for name, weight in (
                ('point_count_weight', self.point_count_weight),
                ('occupancy_sparsity_weight', self.occupancy_sparsity_weight),
                ('distance_weight', self.distance_weight),
                ('intensity_weight', self.intensity_weight)):
            if weight < 0:
                raise ValueError('%s must be non-negative' % name)
        if (self.point_count_weight + self.occupancy_sparsity_weight +
                self.distance_weight + self.intensity_weight) <= 0:
            raise ValueError('at least one unreliability weight must be positive')

        signal_channels = 5 + int(self.use_intensity)
        hidden_channels = int(args.get('hidden_channels', 16))
        if self.mode == 'learned':
            self.learned_head = nn.Sequential(
                nn.Conv2d(signal_channels, hidden_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.learned_head = None

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

    def _voxel_intensity(self, processed_lidar, valid):
        if not self.use_intensity or 'voxel_features' not in processed_lidar:
            return None
        voxel_features = processed_lidar['voxel_features']
        if voxel_features.dim() != 3 or voxel_features.shape[-1] <= self.intensity_index:
            return None

        voxel_num_points = processed_lidar['voxel_num_points'].long()
        max_points = voxel_features.shape[1]
        point_ids = torch.arange(max_points, device=voxel_features.device)
        point_mask = point_ids.view(1, -1) < voxel_num_points.view(-1, 1)
        intensity = voxel_features[:, :, self.intensity_index].float()
        denom = point_mask.float().sum(dim=1).clamp(min=1.0)
        voxel_intensity = (intensity * point_mask.float()).sum(dim=1) / denom
        return voxel_intensity[valid]

    def _build_count_occupancy_intensity(self, processed_lidar):
        voxel_coords = processed_lidar['voxel_coords'].long()
        voxel_num_points = processed_lidar['voxel_num_points'].float()
        device = voxel_coords.device
        batch_size = self._infer_batch_size(processed_lidar, voxel_coords)
        height, width = self._infer_map_size_from_coords(voxel_coords)

        point_count = torch.zeros(batch_size, 1, height, width, device=device)
        occupancy_hits = torch.zeros_like(point_count)
        intensity_sum = torch.zeros_like(point_count)
        intensity_hits = torch.zeros_like(point_count)

        if voxel_coords.numel() == 0:
            return point_count, occupancy_hits, intensity_sum, False

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

            voxel_intensity = self._voxel_intensity(processed_lidar, valid)
            if voxel_intensity is not None:
                intensity_sum.view(-1).scatter_add_(
                    0, flat_idx, voxel_intensity.to(point_count.dtype))
                intensity_hits.view(-1).scatter_add_(
                    0, flat_idx, torch.ones_like(points, dtype=point_count.dtype))

        occupancy = (occupancy_hits > 0).float()
        has_intensity = bool((intensity_hits > 0).any().item())
        intensity_map = intensity_sum / intensity_hits.clamp(min=1.0)
        return point_count, occupancy, intensity_map, has_intensity

    def _robust_positive_norm(self, values, percentile, min_scale, fixed_scale=None):
        """Normalize positive cells with either a fixed or per-item percentile scale."""
        normalized = torch.zeros_like(values)
        batch_size = values.shape[0]
        for b in range(batch_size):
            vals = values[b]
            positive = vals[vals > 0]
            if fixed_scale is not None:
                scale = torch.tensor(fixed_scale, device=values.device,
                                     dtype=values.dtype)
                scale = torch.clamp(scale, min=min_scale)
            elif positive.numel() == 0:
                scale = torch.tensor(min_scale, device=values.device,
                                     dtype=values.dtype)
            else:
                scale = torch.quantile(positive, percentile)
                scale = torch.clamp(scale, min=min_scale)
            normalized[b] = torch.clamp(vals / (scale + self.eps), 0.0, 1.0)
        return normalized

    def _distance_unreliability(self, like):
        batch_size, _, height, width = like.shape
        ys = torch.linspace(-1.0, 1.0, height, device=like.device,
                            dtype=like.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=like.device,
                            dtype=like.dtype)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(ys, xs)
        distance = torch.sqrt(xx.pow(2) + yy.pow(2)) / (2.0 ** 0.5)
        distance = distance.clamp(0.0, 1.0)
        return distance.view(1, 1, height, width).expand(batch_size, -1, -1, -1)

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

    def _heuristic_unreliability(self, point_count_unreliability,
                                 occupancy_sparsity, distance_unreliability,
                                 intensity_unreliability, has_intensity):
        total = (
            self.point_count_weight * point_count_unreliability +
            self.occupancy_sparsity_weight * occupancy_sparsity +
            self.distance_weight * distance_unreliability
        )
        weight_sum = (
            self.point_count_weight +
            self.occupancy_sparsity_weight +
            self.distance_weight
        )
        if self.use_intensity and has_intensity and self.intensity_weight > 0:
            total = total + self.intensity_weight * intensity_unreliability
            weight_sum = weight_sum + self.intensity_weight
        return torch.clamp(total / max(weight_sum, self.eps), 0.0, 1.0)

    def forward(self, processed_lidar):
        (point_count, occupancy, intensity_map,
         has_intensity) = self._build_count_occupancy_intensity(processed_lidar)
        pc_norm = self._robust_positive_norm(
            point_count,
            self.point_count_percentile,
            self.min_point_count_scale,
            fixed_scale=self.fixed_point_count_scale,
        )
        point_count_unreliability = 1.0 - pc_norm

        pad = self.pool_kernel // 2
        local_occupancy_ratio = F.avg_pool2d(
            occupancy,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        occupancy_sparsity = 1.0 - local_occupancy_ratio
        distance_unreliability = self._distance_unreliability(point_count)
        intensity_norm = self._robust_positive_norm(
            intensity_map, self.intensity_percentile, self.min_intensity_scale)
        intensity_unreliability = 1.0 - intensity_norm

        heuristic_U_L = self._heuristic_unreliability(
            point_count_unreliability,
            occupancy_sparsity,
            distance_unreliability,
            intensity_unreliability,
            has_intensity,
        )

        signal_maps = [
            pc_norm,
            point_count_unreliability,
            local_occupancy_ratio,
            occupancy_sparsity,
            distance_unreliability,
        ]
        if self.use_intensity:
            signal_maps.append(intensity_norm)
        lidar_reliability_signals = torch.cat(signal_maps, dim=1)

        if self.learned_head is not None:
            U_L = torch.sigmoid(self.learned_head(lidar_reliability_signals))
        else:
            U_L = heuristic_U_L
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
            'lidar_heuristic_unreliability': heuristic_U_L,
            'lidar_reliability_signals': lidar_reliability_signals,
            'lidar_evidence_confidence': pc_norm,
            'lidar_degradation_proxy': point_count_unreliability,
            'lidar_point_count_map': point_count,
            'lidar_point_count_norm': pc_norm,
            'lidar_point_count_unreliability': point_count_unreliability,
            'lidar_occupancy_map': occupancy,
            'lidar_local_occupancy_ratio': local_occupancy_ratio,
            'lidar_occupancy_sparsity': occupancy_sparsity,
            'lidar_distance_unreliability': distance_unreliability,
            'lidar_intensity_map': intensity_map,
            'lidar_intensity_norm': intensity_norm,
            'lidar_intensity_available': has_intensity,
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
