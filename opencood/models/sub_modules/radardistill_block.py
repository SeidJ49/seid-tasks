import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.radardistill_bev_backbone import BaseBEVBackboneV2
from opencood.pcdet_utils.basicblock.modules.Basicblock_convn import ConvNeXtBlock


def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


class RadarDistill(BaseBEVBackboneV2):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.encoder_1 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())

        self.encoder_2 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_2 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())

        self.encoder_3 = nn.Sequential(ConvNeXtBlock(dim=256, downsample=True), ConvNeXtBlock(dim=256, downsample=False))
        self.decoder_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU())
        self.agg_3 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())

    def low_loss(self, lidar_bev, radar_bev):
        batch_size = radar_bev.shape[0]
        lidar_mask = (lidar_bev.sum(1, keepdim=True) > 0).float()
        radar_mask = radar_bev.sum(1, keepdim=True)
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5

        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float32)
        mask_radar_only = torch.zeros_like(activate_map, dtype=torch.float32)
        mask_radar_lidar[activate_map == 1.5] = 1
        mask_radar_only[activate_map == 1.0] = 1

        if mask_radar_only.sum() > 0:
            mask_radar_only *= mask_radar_lidar.sum().clamp(min=1.0) / mask_radar_only.sum()

        loss_overlap = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_overlap = torch.sum(loss_overlap * mask_radar_lidar) / batch_size
        loss_radar_only = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_only = torch.sum(loss_radar_only * mask_radar_only) / batch_size

        feature_loss = 3e-4 * loss_overlap + 5e-5 * loss_radar_only
        mask_loss = nn.L1Loss()(radar_mask.sigmoid(), lidar_mask)
        return feature_loss, mask_loss

    def high_loss(self, radar_bev, radar_bev_8x, lidar_bev, lidar_bev_8x, heatmaps, radar_preds):
        gt_batch_hm = torch.cat(heatmaps, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]
        radar_batch_hm = [clip_sigmoid(pred_dict['hm']) for pred_dict in radar_preds]
        radar_batch_hm = torch.cat(radar_batch_hm, dim=1)
        radar_batch_hm_max = torch.max(radar_batch_hm, dim=1, keepdim=True)[0]

        radar_fp_mask = torch.logical_and(gt_batch_hm_max < 0.1, radar_batch_hm_max > 0.1)
        radar_fn_mask = torch.logical_and(gt_batch_hm_max > 0.1, radar_batch_hm_max < 0.1)
        radar_tp_mask = torch.logical_and(gt_batch_hm_max > 0.1, radar_batch_hm_max > 0.1)
        weight = torch.zeros_like(radar_batch_hm_max)
        tp_fn = (radar_tp_mask + radar_fn_mask).sum()
        fp = radar_fp_mask.sum()
        if tp_fn > 0:
            weight[radar_tp_mask + radar_fn_mask] = 5 / tp_fn
        if fp > 0:
            weight[radar_fp_mask] = 1 / fp

        scaled_radar = radar_bev.softmax(1)
        scaled_lidar = lidar_bev.softmax(1)
        scaled_radar_8x = radar_bev_8x.softmax(1)
        scaled_lidar_8x = lidar_bev_8x.softmax(1)

        high_loss = (F.l1_loss(scaled_radar, scaled_lidar, reduction='none') * weight).sum()
        high_loss_8x = (F.l1_loss(scaled_radar_8x, scaled_lidar_8x, reduction='none') * weight).sum()
        return 0.5 * (high_loss + high_loss_8x)

    def get_loss(self, batch_dict):
        low_lidar_bev = batch_dict['multi_scale_2d_features']['x_conv4']
        low_radar_bev = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev = batch_dict['radar_spatial_features_2d']
        high_lidar_bev = batch_dict['spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        high_lidar_bev_8x = batch_dict['spatial_features_2d_8x']

        feature_loss, mask_loss = self.low_loss(low_lidar_bev, low_radar_bev)
        de8x_feature_loss, de8x_mask_loss = self.low_loss(low_lidar_bev, low_radar_de_8x)
        high_distill_loss = self.high_loss(
            high_radar_bev,
            high_radar_bev_8x,
            high_lidar_bev,
            high_lidar_bev_8x,
            batch_dict['target_dicts']['heatmaps'],
            batch_dict['radar_pred_dicts'],
        ) * 25
        low_distill_loss = (0.5 * (feature_loss + de8x_feature_loss) + 0.5 * (mask_loss + de8x_mask_loss)) * 5
        distill_loss = low_distill_loss + high_distill_loss

        return distill_loss, {
            'distill_loss': distill_loss.item(),
            'low_feature_loss': feature_loss.item(),
            'low_feature_loss_de8x': de8x_feature_loss.item(),
            'mask_loss': mask_loss.item(),
            'mask_loss_de8x': de8x_mask_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
        }

    def forward(self, batch_dict):
        spatial_features = batch_dict['radar_multi_scale_2d_features']['x_conv4']
        en_16x = self.encoder_1(spatial_features)
        de_8x = self.agg_1(torch.cat((self.decoder_1(en_16x), spatial_features), dim=1))
        en_32x = self.encoder_2(en_16x)
        de_16x = self.agg_2(torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1))
        x_conv4 = self.agg_3(torch.cat((self.decoder_3(de_16x), de_8x), dim=1))

        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4
        batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x

        x_conv5 = batch_dict['radar_multi_scale_2d_features']['x_conv5']
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        batch_dict['radar_spatial_features_2d_8x'] = ups[-1]
        batch_dict['radar_spatial_features_2d'] = self.blocks[0](torch.cat(ups, dim=1))
        return batch_dict