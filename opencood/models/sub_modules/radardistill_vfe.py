import torch
import torch.nn as nn

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


def scatter_mean(src, index):
    if torch_scatter is not None:
        return torch_scatter.scatter_mean(src, index, dim=0)
    num_segments = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if num_segments == 0:
        return src.new_zeros((0, src.shape[-1]))
    out = src.new_zeros((num_segments, src.shape[-1]))
    counts = src.new_zeros((num_segments, 1))
    out.index_add_(0, index, src)
    counts.index_add_(0, index, src.new_ones((src.shape[0], 1)))
    return out / counts.clamp_min(1.0)


def scatter_max(src, index):
    if torch_scatter is not None:
        return torch_scatter.scatter_max(src, index, dim=0)[0]
    num_segments = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if num_segments == 0:
        return src.new_zeros((0, src.shape[-1]))
    out = src.new_full((num_segments, src.shape[-1]), float('-inf'))
    for segment_idx in range(num_segments):
        mask = index == segment_idx
        if mask.any():
            out[segment_idx] = src[mask].max(dim=0)[0]
    out[out == float('-inf')] = 0
    return out


def build_mlp(in_channels, hidden_channels, out_channels, use_norm=True):
    layers = [nn.Linear(in_channels, hidden_channels, bias=not use_norm)]
    if use_norm:
        layers.append(nn.BatchNorm1d(hidden_channels, eps=1e-3, momentum=0.01))
    layers.extend([nn.ReLU(), nn.Linear(hidden_channels, out_channels), nn.ReLU()])
    return nn.Sequential(*layers)


class PFNLayerV2(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=not use_norm)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01) if use_norm else None
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        features = self.linear(inputs)
        if self.norm is not None:
            features = self.norm(features)
        features = self.relu(features)
        features_max = scatter_max(features, unq_inv)

        if self.last_vfe:
            return features_max

        return torch.cat([features, features_max[unq_inv, :]], dim=1)


class DynamicPillarVFESimple2D(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, points_key='points'):
        super().__init__()
        self.model_cfg = model_cfg
        self.points_key = points_key
        self.use_norm = model_cfg.get('use_norm', True)
        self.with_distance = model_cfg.get('with_distance', False)
        self.use_absolute_xyz = model_cfg.get('use_absolute_xyz', True)
        self.use_cluster_xyz = model_cfg.get('use_cluster_xyz', True)
        self.use_relative_xyz = model_cfg.get('use_relative_xyz', True)

        if self.use_absolute_xyz:
            num_point_features += 3
        if self.use_cluster_xyz:
            num_point_features += 3
        if self.use_relative_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        num_filters = [num_point_features] + list(model_cfg['num_filters'])
        self.pfn_layers = nn.ModuleList([
            PFNLayerV2(
                num_filters[index],
                num_filters[index + 1],
                use_norm=self.use_norm,
                last_layer=index >= len(num_filters) - 2,
            )
            for index in range(len(num_filters) - 1)
        ])

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xy = int(grid_size[0] * grid_size[1])
        self.scale_y = int(grid_size[1])

        self.register_buffer('grid_size', torch.tensor(grid_size[:2], dtype=torch.int32), persistent=False)
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)

    def get_output_feature_dim(self):
        return self.pfn_layers[-1].linear.out_features

    def _forward_impl(self, batch_dict, input_key, feature_key, coord_key):
        points = batch_dict[input_key]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + points_coords[:, 0] * self.scale_y + points_coords[:, 1]
        _, unq_inv = torch.unique(merge_coords, return_inverse=True)
        unq_coords = torch.unique(merge_coords)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        if self.use_cluster_xyz:
            points_mean = scatter_mean(points_xyz, unq_inv)
            features.append(points_xyz - points_mean[unq_inv, :])

        if self.with_distance:
            features.append(torch.norm(points[:, 1:4], 2, dim=1, keepdim=True))

        if self.use_relative_xyz:
            features.append(points_xyz - self.point_cloud_range[:3])

        features = torch.cat(features, dim=-1)
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        unq_coords = unq_coords.int()
        pillar_coords = torch.stack(
            (
                torch.div(unq_coords, self.scale_xy, rounding_mode='trunc'),
                torch.div(unq_coords % self.scale_xy, self.scale_y, rounding_mode='trunc'),
                unq_coords % self.scale_y,
            ),
            dim=1,
        )
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict[feature_key] = features
        batch_dict[coord_key] = pillar_coords
        return batch_dict

    def forward(self, batch_dict):
        return self._forward_impl(batch_dict, self.points_key, 'pillar_features', 'pillar_coords')


class RadarDynamicPillarVFESimple2D(DynamicPillarVFESimple2D):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range):
        super().__init__(model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, points_key='radar_points')

    def forward(self, batch_dict):
        return self._forward_impl(batch_dict, 'radar_points', 'radar_pillar_features', 'radar_pillar_coords')


class FactorizedRadarDynamicPillarVFE(nn.Module):
    """Radar VFE with separate geometry, RCS, and Doppler branches.

    Input radar points are expected as [batch_idx, x, y, z, doppler, rcs].
    No timestamp feature is used.
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_norm = model_cfg.get('use_norm', True)
        self.use_absolute_xyz = model_cfg.get('use_absolute_xyz', True)
        self.use_cluster_xyz = model_cfg.get('use_cluster_xyz', True)
        self.use_relative_xyz = model_cfg.get('use_relative_xyz', True)
        self.with_distance = model_cfg.get('with_distance', False)
        self.doppler_index = int(model_cfg.get('doppler_index', 4))
        self.rcs_index = int(model_cfg.get('rcs_index', 5))
        self.branch_channels = int(model_cfg.get('branch_channels', 16))
        self.hidden_channels = int(model_cfg.get('hidden_channels', 32))
        self.use_pillar_stats = bool(model_cfg.get('use_pillar_stats', True))

        geo_in_channels = 3
        if self.use_absolute_xyz:
            geo_in_channels += 3
        if self.use_cluster_xyz:
            geo_in_channels += 3
        if self.use_relative_xyz:
            geo_in_channels += 3
        if self.with_distance:
            geo_in_channels += 1

        self.geo_mlp = build_mlp(geo_in_channels, self.hidden_channels, self.branch_channels, self.use_norm)
        self.rcs_mlp = build_mlp(4, self.hidden_channels, self.branch_channels, self.use_norm)
        self.doppler_mlp = build_mlp(6, self.hidden_channels, self.branch_channels, self.use_norm)
        self.rv_mlp = build_mlp(self.branch_channels * 3, self.hidden_channels, self.branch_channels, self.use_norm)
        self.rcs_gate = nn.Sequential(nn.Linear(self.branch_channels * 2, self.branch_channels), nn.Sigmoid())
        self.doppler_gate = nn.Sequential(nn.Linear(self.branch_channels * 2, self.branch_channels), nn.Sigmoid())

        pfn_input_channels = self.branch_channels * 4
        num_filters = [pfn_input_channels] + list(model_cfg['num_filters'])
        self.pfn_layers = nn.ModuleList([
            PFNLayerV2(
                num_filters[index],
                num_filters[index + 1],
                use_norm=self.use_norm,
                last_layer=index >= len(num_filters) - 2,
            )
            for index in range(len(num_filters) - 1)
        ])
        out_channels = num_filters[-1]
        if self.use_pillar_stats:
            self.stats_fusion = nn.Sequential(
                nn.Linear(out_channels + 7, out_channels, bias=not self.use_norm),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01) if self.use_norm else nn.Identity(),
                nn.ReLU(),
            )
        else:
            self.stats_fusion = None

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xy = int(grid_size[0] * grid_size[1])
        self.scale_y = int(grid_size[1])

        self.register_buffer('grid_size', torch.tensor(grid_size[:2], dtype=torch.int32), persistent=False)
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)

    def get_output_feature_dim(self):
        return self.pfn_layers[-1].linear.out_features

    def _pillar_std(self, values, unq_inv, mean_values):
        values_sq_mean = scatter_mean(values * values, unq_inv)
        var = (values_sq_mean - mean_values * mean_values).clamp_min(0.0)
        return torch.sqrt(var + 1e-6)

    def forward(self, batch_dict):
        points = batch_dict['radar_points']
        if points.shape[1] <= max(self.doppler_index, self.rcs_index):
            raise ValueError(
                'FactorizedRadarDynamicPillarVFE expects radar_points as '
                f'[batch_idx, x, y, z, doppler, rcs], but got {points.shape[1]} columns.'
            )

        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + points_coords[:, 0] * self.scale_y + points_coords[:, 1]
        _, unq_inv = torch.unique(merge_coords, return_inverse=True)
        unq_coords = torch.unique(merge_coords)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        geo_features = [f_center]
        if self.use_absolute_xyz:
            geo_features.append(points_xyz)
        if self.use_cluster_xyz:
            points_mean = scatter_mean(points_xyz, unq_inv)
            geo_features.append(points_xyz - points_mean[unq_inv, :])
        if self.use_relative_xyz:
            geo_features.append(points_xyz - self.point_cloud_range[:3])
        if self.with_distance:
            geo_features.append(torch.norm(points_xyz, 2, dim=1, keepdim=True))
        geo_features = torch.cat(geo_features, dim=-1)

        xy_range = torch.norm(points[:, 1:3], 2, dim=1, keepdim=True)
        theta = torch.atan2(points[:, 2:3], points[:, 1:2])
        doppler = points[:, self.doppler_index:self.doppler_index + 1]
        rcs = points[:, self.rcs_index:self.rcs_index + 1]
        vx = doppler * torch.cos(theta)
        vy = doppler * torch.sin(theta)

        rcs_features = torch.cat([rcs, xy_range, theta, points[:, 3:4]], dim=-1)
        doppler_features = torch.cat([doppler, doppler.abs(), doppler.sign(), vx, vy, xy_range], dim=-1)

        geo_feat = self.geo_mlp(geo_features)
        rcs_feat = self.rcs_mlp(rcs_features)
        doppler_feat = self.doppler_mlp(doppler_features)
        rv_feat = self.rv_mlp(torch.cat([rcs_feat, doppler_feat, rcs_feat * doppler_feat], dim=-1))
        gated_rcs = self.rcs_gate(torch.cat([rcs_feat, doppler_feat], dim=-1)) * rcs_feat
        gated_doppler = self.doppler_gate(torch.cat([doppler_feat, rcs_feat], dim=-1)) * doppler_feat
        features = torch.cat([geo_feat, gated_rcs, gated_doppler, rv_feat], dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        if self.stats_fusion is not None:
            num_pillars = features.shape[0]
            counts = features.new_zeros((num_pillars, 1))
            counts.index_add_(0, unq_inv, features.new_ones((unq_inv.shape[0], 1)))
            rcs_mean = scatter_mean(rcs, unq_inv)
            rcs_max = scatter_max(rcs, unq_inv)
            rcs_std = self._pillar_std(rcs, unq_inv, rcs_mean)
            doppler_mean = scatter_mean(doppler, unq_inv)
            abs_doppler_max = scatter_max(doppler.abs(), unq_inv)
            doppler_std = self._pillar_std(doppler, unq_inv, doppler_mean)
            stats = torch.cat([
                torch.log1p(counts),
                rcs_mean,
                rcs_max,
                rcs_std,
                doppler_mean,
                abs_doppler_max,
                doppler_std,
            ], dim=-1)
            features = self.stats_fusion(torch.cat([features, stats], dim=-1))

        unq_coords = unq_coords.int()
        pillar_coords = torch.stack(
            (
                torch.div(unq_coords, self.scale_xy, rounding_mode='trunc'),
                torch.div(unq_coords % self.scale_xy, self.scale_y, rounding_mode='trunc'),
                unq_coords % self.scale_y,
            ),
            dim=1,
        )
        batch_dict['radar_pillar_features'] = features
        batch_dict['radar_pillar_coords'] = pillar_coords[:, [0, 2, 1]]
        return batch_dict
