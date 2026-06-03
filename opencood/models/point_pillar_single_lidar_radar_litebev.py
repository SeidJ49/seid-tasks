import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.visualization.visualization_debug import save_heatmaps


class PointPillarSingleLidarRadarLitebev(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleLidarRadarLitebev, self).__init__()

        self.lidar_pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'])
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.radar_pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'])
        self.radar_scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(64 * 2 + 1, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_number'],
                kernel_size=1)
        else:
            self.use_dir = False

    @staticmethod
    def _build_density_map(voxel_coords, voxel_num_points, spatial_size, device, dtype):
        batch_size = int(voxel_coords[:, 0].max().item()) + 1 if voxel_coords.numel() > 0 else 1
        height, width = spatial_size
        density_maps = []

        if voxel_num_points.numel() > 0:
            norm_factor = torch.clamp(voxel_num_points.float().max(), min=1.0)
        else:
            norm_factor = torch.tensor(1.0, device=device)

        for batch_idx in range(batch_size):
            density = torch.zeros(height * width, device=device, dtype=dtype)
            if voxel_coords.numel() > 0:
                batch_mask = voxel_coords[:, 0] == batch_idx
                this_coords = voxel_coords[batch_mask, :]
                this_points = voxel_num_points[batch_mask].float() / norm_factor

                if this_coords.numel() > 0:
                    ny = height
                    nx = width
                    flat_indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
                    flat_indices = flat_indices.long()
                    density[flat_indices] = this_points.to(device=device, dtype=dtype)

            density_maps.append(density.view(1, height, width))

        return torch.stack(density_maps, dim=0)

    @staticmethod
    def _ensure_2d(features):
        if features.dim() == 1:
            return features.unsqueeze(0)
        return features

    def forward(self, data_dict):

        # --- LiDAR ----------------------------------------------------------------------------------------------------
        lidar_voxel_features = data_dict['processed_lidar']['voxel_features']
        lidar_voxel_coords = data_dict['processed_lidar']['voxel_coords']
        lidar_voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        lidar_batch_dict = {
            'voxel_features': lidar_voxel_features,
            'voxel_coords': lidar_voxel_coords,
            'voxel_num_points': lidar_voxel_num_points,
        }
        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict['pillar_features'] = self._ensure_2d(lidar_batch_dict['pillar_features'])
        lidar_batch_dict = self.lidar_scatter(lidar_batch_dict)
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        radar_voxel_features = data_dict['processed_radar']['voxel_features']
        radar_voxel_coords = data_dict['processed_radar']['voxel_coords']
        radar_voxel_num_points = data_dict['processed_radar']['voxel_num_points']

        radar_batch_dict = {
            'voxel_features': radar_voxel_features,
            'voxel_coords': radar_voxel_coords,
            'voxel_num_points': radar_voxel_num_points,
        }
        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict['pillar_features'] = self._ensure_2d(radar_batch_dict['pillar_features'])
        radar_batch_dict = self.radar_scatter(radar_batch_dict)

        lidar_spatial_features = lidar_batch_dict['spatial_features']
        radar_spatial_features = radar_batch_dict['spatial_features']

        density_map = self._build_density_map(
            lidar_voxel_coords,
            lidar_voxel_num_points,
            lidar_spatial_features.shape[-2:],
            lidar_spatial_features.device,
            lidar_spatial_features.dtype)

        reliability_map = 1.0 - density_map.clamp(0.0, 1.0)

        gate_input = torch.cat([
            lidar_spatial_features,
            radar_spatial_features,
            reliability_map,
        ], dim=1)
        gate_map = torch.sigmoid(self.fusion_gate(gate_input))

        fused_spatial_features = (
            (1.0 - gate_map) * lidar_spatial_features
            + gate_map * radar_spatial_features
        )

        batch_dict = {
            'spatial_features': fused_spatial_features,
            'fusion_gate': gate_map,
            'reliability_map': reliability_map,
        }
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {
            'cls_preds': psm,
            'reg_preds': rm,
            'gate_map': gate_map,
            'reliability_map': reliability_map,
        }

        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict