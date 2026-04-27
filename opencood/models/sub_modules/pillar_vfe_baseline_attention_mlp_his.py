import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------
class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm

        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part]) for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x

        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
# ----------------------------------------------------------------------------------------------------------------------

class PillarVFEBaselineAttentionMlpHis(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']
        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']

        # ----------------------------------------- VERSION X ----------------------------------------------------------
        self.use_velocity_branch = self.model_cfg.get('use_velocity_branch', True) # (x,y,z, v_rel_norm, v_r_norm, v_r_x_norm, v_r_y_norm)

        vm = self.model_cfg.get('velocity_mask', {})
        self.vm_channel = vm.get('channel', 'v_r')          # 'v_rel'|'v_r'|'v_rx'|'v_ry'
        self.vm_abs     = vm.get('abs', True)
        self.vm_thresh  = float(vm.get('thresh', 0.15))     # in normalized units
        self.vm_reduce  = vm.get('reduce', 'max')           # 'max'|'mean'
        self.vm_mode    = vm.get('mode', 'soft')            # 'hard'|'soft'
        self.vm_temp    = float(vm.get('temp', 12.0))       # only for soft
        # ----------------------------------------- VERSION X ----------------------------------------------------------

        if self.use_velocity_branch:
            # spatial: x,y,z and extra channels processed via center/cluster -> 6 channels
            # velocity: 4 channels from the extra features
            spatial_channels = 3        # x,y,z used for f_cluster and f_center
            f_cluster_channels = 3      # f_cluster features (offset from mean)
            f_center_channels = 3       # f_center features (offset from voxel center)

            # ----------------------------------------- VERSION X ------------------------------------------------------
            velocity_channels = 4       # extra channels for velocity
            hidden_dim = velocity_channels * 2  # hidden dimension for velocity branch MLP

            self.velocity_mlp = nn.Sequential(
                nn.Linear(velocity_channels, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim, eps=1e-3, momentum=0.01),
                nn.PReLU(),
                nn.Linear(hidden_dim, velocity_channels, bias=False),
                nn.BatchNorm1d(velocity_channels, eps=1e-3, momentum=0.01),
                nn.PReLU()
            )

            num_point_features = spatial_channels + velocity_channels  + f_cluster_channels + f_center_channels
            # ----------------------------------------- VERSION X ------------------------------------------------------
        else:
            # Optionally include absolute (x,y,z) positions
            num_point_features += 6 if self.use_absolute_xyz else 3

        # Optionally include Euclidean distance feature.
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0

        # Create the list of filter dimensions.
        num_filters = [num_point_features] + list(self.num_filters)

        # Build PFN layers as defined by the configuration.
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            # Last layer's PFN does not perform feature concatenation.
            pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Pre-compute voxel center offsets for later point feature center alignment.
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        """Return the output feature dimension of the last PFN layer."""
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):

        voxel_features = batch_dict['voxel_features']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean # Compute Offset from each point to the voxel's mean.

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # ----------------------------------------- VERSION X ----------------------------------------------------------
        if self.use_velocity_branch:
            velocity_feats = voxel_features[:, :, 3:7]

            B, N, C = velocity_feats.shape
            velocity_input = velocity_feats.reshape(-1, C)
            processed_velocity = self.velocity_mlp(velocity_input).reshape(B, N, C)

            chan_map = {'v_rel':0, 'v_r':1, 'v_rx':2, 'v_ry':3}
            sel_idx  = chan_map.get(self.vm_channel, 1)  # default v_r
            sel = velocity_feats[:, :, sel_idx]          # (M, num_points)

            if self.vm_abs:
                sel = torch.abs(sel)

            if self.vm_mode == 'soft':
                # mask score in [0,1], steepness via temperature
                score = torch.sigmoid(self.vm_temp * (sel - self.vm_thresh))
            else:
                score = (sel > self.vm_thresh).float()

            if self.vm_reduce == 'mean':
                voxel_score = score.mean(dim=1, keepdim=True)  # (M,1)
            else:
                voxel_score = score.max(dim=1, keepdim=True)[0]  # (M,1)

            batch_dict['velocity_confidence'] = voxel_score

            features = [voxel_features[:, :, :3], f_cluster, f_center, processed_velocity]
        # ----------------------------------------- VERSION X ----------------------------------------------------------
        else:
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()

        batch_dict['pillar_features'] = features

        return batch_dict
