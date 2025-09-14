import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

from opencood.models.sub_modules.pillar_vfe_baseline_attention_mlp_his import PillarVFEBaselineAttentionMlpHis
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention_mlp_his import PointPillarScatterBaselineAttentionMlpHis

# ------------------------------------------------------------VERSION X ------------------------------------------------

class MaskGuidedSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_conv = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=True)
        self.bn_attn = nn.InstanceNorm2d(1, affine=True, track_running_stats=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat: torch.Tensor, dyn_mask: torch.Tensor):
        # CBAM-Pooling
        avg_pool = feat.mean(dim=1, keepdim=True)  # [B,1,H,W]
        max_pool, _ = feat.max(dim=1, keepdim=True)  # [B,1,H,W]

        # concat + conv -> spatial attention map
        x = torch.cat([avg_pool, max_pool, dyn_mask], dim=1)  # [B,3,H,W]
        attn_map = torch.sigmoid(self.bn_attn(self.spatial_conv(x)))  # [B,1,H,W]

        attn_map = attn_map / (attn_map.amax(dim=[2, 3], keepdim=True) + 1e-6)
        gamma = self.gamma.clamp_min(0.0)
        return feat * (1 + gamma * attn_map)  # Residual original CBAM

class ChannelGate(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(C, C // r, bias=False), nn.ReLU(True), nn.Linear(C // r, C, bias=False))

    def forward(self, feat):
        b, c, _, _ = feat.shape
        avg = F.adaptive_avg_pool2d(feat, 1).view(b, c)
        mx = F.adaptive_max_pool2d(feat, 1).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return feat * att

# -----------------------------------------------------------VERSION X -------------------------------------------------

class PointPillarSingleBaselineAttentionMlpHis(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleBaselineAttentionMlpHis, self).__init__()

        # --- LiDAR ----------------------------------------------------------------------------------------------------
        self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        self.radar_pillar_vfe = PillarVFEBaselineAttentionMlpHis(args['pillar_vfe'], num_point_features=7, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.radar_scatter = PointPillarScatterBaselineAttentionMlpHis(args['point_pillar_scatter'])
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

        # -------------------------------------------- VERSION X -------------------------------------------------------
        self.chan_gate = ChannelGate(args['shrink_header']['input_dim'], r=16)
        self.attn_mod = MaskGuidedSpatialAttention()
        self.gamma_pre = nn.Parameter(torch.tensor(0.1))
        # -------------------------------------------- VERSION X -------------------------------------------------------
        
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

        # --- BOTH -----------------------------------------------------------------------------------------------------
        batch_dict = {'spatial_features': torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']], dim=1),}
        # --------------------------------------------------------------------------------------------------------------

        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = self.chan_gate(spatial_features_2d)

        # -------------------------------------------- VERSION X -------------------------------------------------------
        original_dynamic_mask = radar_batch_dict['velocity_confidence_mask']

        kernel_size = 7
        padding = kernel_size // 2

        dilated = F.max_pool2d(original_dynamic_mask, kernel_size, stride=1, padding=padding)


        sf = batch_dict['spatial_features']
        batch_dict['spatial_features'] = sf * (1.0 + self.gamma_pre * dilated)

        dyn_mask = F.interpolate(dilated.float(), size=spatial_features_2d.shape[-2:], mode='bilinear', align_corners=False).clamp_(0.0, 1.0)


        assert dyn_mask.shape[0] == spatial_features_2d.shape[0], f"Batch size mismatch: dyn_mask {dyn_mask.shape[0]} vs spatial_features_2d {spatial_features_2d.shape[0]}"

        spatial_features_2d = self.attn_mod(spatial_features_2d, dyn_mask)
        # -------------------------------------------- VERSION X -------------------------------------------------------

         # optional shrinkage
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict