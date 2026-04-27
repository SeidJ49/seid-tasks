import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

from opencood.models.sub_modules.pillar_vfe_baseline_attention_mlp_his import PillarVFEBaselineAttentionMlpHis
from opencood.models.sub_modules.point_pillar_scatter_baseline_attention_mlp_his import \
    PointPillarScatterBaselineAttentionMlpHis

# DEBUG
from opencood.visualization.visualization_debug import save_heatmaps

# --- VERSION CLUSTER --------------------------------------------------------------------------------------------------
import numpy as np
from sklearn.cluster import DBSCAN


# --- VERSION CLUSTER --------------------------------------------------------------------------------------------------


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


def _dbscan_rect_mask_from_binary(
        bin_mask: torch.Tensor,
        eps_px: int = 7,
        min_samples: int = 5,
        thr: float = 0.99,
        min_area: int = 6,
        max_area: int = None,
        max_area_frac: float = 0.01,  # 1% der Maske als Default-Grenze
        max_w: int = None,
        max_h: int = None,
        aspect_min: float = 0.25,  # min. Seitenverhältnis (w/h)
        aspect_max: float = 4.0  # max. Seitenverhältnis
) -> torch.Tensor:
    """
    Binäre/float Maske in [0,1]; nur Werte > thr werden geclustert.
    Es werden NUR Cluster akzeptiert, deren Bounding-Box "auto-groß" ist.
    Filter: min_area, max_area (oder max_area_frac), max_w, max_h, Seitenverhältnis.
    """
    if bin_mask.dim() == 3:
        bin_mask = bin_mask[0]
    H, W = bin_mask.shape

    # Punktmenge für DBSCAN (nur wirklich aktive Pixel)
    mask_np = (bin_mask.detach().cpu().numpy() > thr).astype(np.uint8)
    ys, xs = np.where(mask_np == 1)

    out = np.zeros((H, W), dtype=np.float32)
    if xs.size == 0 or DBSCAN is None:
        return torch.from_numpy(out).unsqueeze(0)

    # Default-Grenzen ableiten
    if max_area is None:
        max_area = int(max_area_frac * H * W)  # z.B. 1% der Gesamtfläche
    pts = np.stack([xs, ys], axis=1)  # (x,y)

    clustering = DBSCAN(eps=eps_px, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    unique = [l for l in np.unique(labels) if l != -1]

    for lbl in unique:
        idx = np.where(labels == lbl)[0]
        if idx.size == 0:
            continue
        c = pts[idx]
        x0, y0 = np.maximum(c.min(axis=0) - 1, 0)
        x1, y1 = np.minimum(c.max(axis=0) + 1, [W - 1, H - 1])
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

        w_box = x1 - x0 + 1
        h_box = y1 - y0 + 1
        area = w_box * h_box
        aspect = (w_box / max(h_box, 1e-6))

        # --- Größen-/Formfilter: nur "Auto-ähnliche" Boxen behalten ---
        if area < min_area:  # sehr kleine Fragmente verwerfen
            continue
        if area > max_area:  # zu große Flächen (wie dein Beispiel unten rechts) verwerfen
            continue
        if max_w is not None and w_box > max_w:
            continue
        if max_h is not None and h_box > max_h:
            continue
        if not (aspect_min <= aspect <= aspect_max):
            continue

        out[y0:y1 + 1, x0:x1 + 1] = 1.0  # akzeptierter Cluster → weiße gefüllte Box

    return torch.from_numpy(out).unsqueeze(0)  # [1,H,W]


# -----------------------------------------------------------VERSION X -------------------------------------------------

class PointPillarSingleLidarRadarBaselineAttentionMlpHisCluster(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleLidarRadarBaselineAttentionMlpHisCluster, self).__init__()

        # --- LiDAR ----------------------------------------------------------------------------------------------------
        self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'],
                                          point_cloud_range=args['lidar_range'])
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        self.radar_pillar_vfe = PillarVFEBaselineAttentionMlpHis(args['pillar_vfe'], num_point_features=7,
                                                                 voxel_size=args['voxel_size'],
                                                                 point_cloud_range=args['lidar_range'])
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

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)  # 384
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], kernel_size=1)  # 384

        # -------------------------------------------- VERSION X -------------------------------------------------------
        self.chan_gate = ChannelGate(args['shrink_header']['input_dim'], r=16)
        self.attn_mod = MaskGuidedSpatialAttention()
        self.gamma_pre = nn.Parameter(torch.tensor(0.1))
        # -------------------------------------------- VERSION X -------------------------------------------------------

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                      kernel_size=1)  # BIN_NUM = 2， # 384
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
        batch_dict = {
            'spatial_features': torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']],
                                          dim=1), }
        # --------------------------------------------------------------------------------------------------------------

        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = self.chan_gate(spatial_features_2d)

        # ----------------------------------------- dbscan_eps7_min4 ---------------------------------------------------
        # 1) Originale dynamische Maske holen (erwartet [B,1,H,W])
        original_dynamic_mask = radar_batch_dict['velocity_confidence_mask']  # [B,1,H,W]

        save_heatmaps(original_dynamic_mask, 'original_dynamic_mask', 'original_dynamic_mask')

        B, _, Hm, Wm = original_dynamic_mask.shape

        # 2) Für jedes Batch-Element: DBSCAN-Cluster finden und als gefüllte Rechtecke schreiben
        rect_masks = []
        for b in range(B):
            rect_mask_b = _dbscan_rect_mask_from_binary(
                original_dynamic_mask[b],  # [1,H,W]
                eps_px=7,
                min_samples=5,
                thr=0.99,  # nimm 0.6–0.7 je nach Rauschlevel
                min_area=9,  # < 3x3 verwerfen
                max_area=None,  # None → automatisch über max_area_frac
                max_area_frac=0.005,  # 0.5% der Gesamtmaske als Obergrenze (schärfer als 1%)
                max_w=None,  # optional zusätzlich deckeln, z.B. max_w=50
                max_h=None,  # optional zusätzlich deckeln
                aspect_min=0.3,  # sehr dünne Streifen weg
                aspect_max=3.5  # extrem lange Bänder weg
            )

            rect_masks.append(rect_mask_b)

        cluster_rect_mask = torch.stack(rect_masks, dim=0).to(original_dynamic_mask.device).float()  # [B,1,H,W]

        # 3) Diese (rechteckige) Maske als Verstärkung auf die "spatial_features" anwenden
        sf = batch_dict['spatial_features']  # [B,C,Hs,Ws]
        _, _, Hs, Ws = sf.shape
        mask_for_sf = cluster_rect_mask
        if (Hm, Wm) != (Hs, Ws):
            mask_for_sf = F.interpolate(mask_for_sf, size=(Hs, Ws), mode='nearest')  # keine Glättung für Rechtecke
        batch_dict['spatial_features'] = sf * (1.0 + self.gamma_pre * mask_for_sf)

        # 4) Für die 2D-Features (Backbone-Ausgabe) die Maske bilinear anpassen
        dyn_mask = F.interpolate(cluster_rect_mask, size=spatial_features_2d.shape[-2:], mode='bilinear',
                                 align_corners=False).clamp_(0.0, 1.0)

        # 5) Masken-gesteuerte räumliche Attention
        spatial_features_2d = self.attn_mod(spatial_features_2d, dyn_mask)
        # --------------------------------------- dbscan_eps7_min4 -----------------------------------------------------

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