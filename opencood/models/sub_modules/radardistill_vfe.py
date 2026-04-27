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