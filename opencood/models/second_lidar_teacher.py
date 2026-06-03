import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class SecondLidarTeacher(nn.Module):
    """
    LiDAR-only SECOND teacher for the TruckScenes thesis setup.

    This mirrors the existing PointPillarLidarTeacher contract but uses the
    sparse 3D SECOND encoder already available through OpenCOOD submodules.
    The feature tap is the final BEV feature map immediately before the heads.
    """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args['batch_size']
        self.prediction_key_prefix = args.get('prediction_key_prefix', 'teacher_')
        self.feature_key = args.get('feature_key', 'teacher_feature')

        num_point_features = args.get('num_point_features', 4)
        self.mean_vfe = MeanVFE(args.get('mean_vfe', {}), num_point_features)
        self.backbone_3d = VoxelBackBone8x(
            args.get('backbone_3d', {}),
            num_point_features,
            args['grid_size']
        )
        self.height_compression = HeightCompression(args['height_compression'])
        bev_backbone_type = args.get('bev_backbone_type', 'base').lower()
        if bev_backbone_type in ('resnet', 'resnet_bev'):
            bev_cfg = dict(args['base_bev_backbone'])
            bev_cfg.setdefault('inplanes', args['height_compression']['feature_num'])
            self.backbone_2d = ResNetBEVBackbone(
                bev_cfg,
                args['height_compression']['feature_num']
            )
        else:
            self.backbone_2d = BaseBEVBackbone(
                args['base_bev_backbone'],
                args['height_compression']['feature_num']
            )
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        anchor_number = args['anchor_number']
        self.cls_head = nn.Conv2d(self.out_channel, anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * anchor_number, kernel_size=1)

        self.use_dir = 'dir_args' in args
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * anchor_number,
                kernel_size=1,
            )

    def _infer_batch_size(self, processed_lidar, data_dict=None):
        if data_dict is not None:
            if 'label_dict' in data_dict and 'pos_equal_one' in data_dict['label_dict']:
                return int(data_dict['label_dict']['pos_equal_one'].shape[0])
            if 'object_bbx_center' in data_dict:
                return int(data_dict['object_bbx_center'].shape[0])
            if 'record_len' in data_dict:
                return int(data_dict['record_len'].sum().item())
        voxel_coords = processed_lidar['voxel_coords']
        if voxel_coords.numel() == 0:
            return self.batch_size
        return int(voxel_coords[:, 0].max().item()) + 1

    def _encode_to_bev(self, processed_lidar, batch_size):
        batch_dict = {
            'voxel_features': processed_lidar['voxel_features'],
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
            'batch_size': batch_size,
        }
        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        bev_features = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)
        return bev_features

    def forward(self, data_dict):
        batch_size = self._infer_batch_size(data_dict['processed_lidar'], data_dict)
        bev_features = self._encode_to_bev(data_dict['processed_lidar'], batch_size)
        prefix = self.prediction_key_prefix
        # With prefix='teacher_' these become teacher_cls_preds,
        # teacher_reg_preds, and teacher_dir_preds for KD training.
        cls_key = f'{prefix}cls_preds'
        reg_key = f'{prefix}reg_preds'
        dir_key = f'{prefix}dir_preds'
        output_dict = {
            cls_key: self.cls_head(bev_features),
            reg_key: self.reg_head(bev_features),
            self.feature_key: bev_features,
            'batch_size': batch_size,
        }
        if self.use_dir:
            output_dict[dir_key] = self.dir_head(bev_features)
        return output_dict
