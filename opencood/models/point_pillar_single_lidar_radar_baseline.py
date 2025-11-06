import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.visualization.visualization_debug import save_heatmaps


class PointPillarSingleLidarRadarBaseline(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleLidarRadarBaseline, self).__init__()

        # --- LiDAR ----------------------------------------------------------------------------------------------------
        self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        self.radar_pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.radar_scatter = PointPillarScatter(args['point_pillar_scatter'])
        # --------------------------------------------------------------------------------------------------------------

        # --- BEV Backbone ---------------------------------------------------------------------------------------------
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 128)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        # --------------------------------------------------------------------------------------------------------------

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1) # 384
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],kernel_size=1) # 384

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False

    def forward(self, data_dict):

        # --- LiDAR ----------------------------------------------------------------------------------------------------
        lidar_voxel_features = data_dict['processed_lidar']['voxel_features']
        lidar_voxel_coords = data_dict['processed_lidar']['voxel_coords']
        lidar_voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        lidar_batch_dict = {'voxel_features': lidar_voxel_features,
                            'voxel_coords': lidar_voxel_coords,
                            'voxel_num_points': lidar_voxel_num_points}

        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict = self.lidar_scatter(lidar_batch_dict)

        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        radar_voxel_features = data_dict['processed_radar']['voxel_features']
        radar_voxel_coords = data_dict['processed_radar']['voxel_coords']
        radar_voxel_num_points = data_dict['processed_radar']['voxel_num_points']

        radar_batch_dict = {'voxel_features': radar_voxel_features,
                            'voxel_coords': radar_voxel_coords,
                            'voxel_num_points': radar_voxel_num_points}

        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.radar_scatter(radar_batch_dict)

        # --------------------------------------------------------------------------------------------------------------

        save_heatmaps(radar_batch_dict['spatial_features'], 'radar_spatial_features', 'frame')
        save_heatmaps(lidar_batch_dict['spatial_features'], 'lidar_spatial_features', 'frame')

        # --- BOTH -----------------------------------------------------------------------------------------------------
        batch_dict = {'spatial_features': torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']], dim=1),}
        # --------------------------------------------------------------------------------------------------------------

        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        save_heatmaps(spatial_features_2d, 'spatial_features_2d', 'frame')

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        save_heatmaps(psm, 'psm', 'frame')
        save_heatmaps(rm, 'rm', 'frame')

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict