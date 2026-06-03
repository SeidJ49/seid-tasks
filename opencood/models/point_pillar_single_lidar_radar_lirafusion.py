import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarSingleLidarRadarLirafusion(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleLidarRadarLirafusion, self).__init__()

        # LiRaFusion repo path can be overridden from YAML for portability.
        lira_repo_root = args.get('lira_repo_root', '/home/ws-ids-es3-01/repo/LiRaFusion')
        lira_repo_root = str(Path(lira_repo_root).expanduser().resolve())
        if lira_repo_root not in sys.path:
            sys.path.append(lira_repo_root)

        try:
            from plugin.lirafusion.models.backbones.opencood_lirafusion_gate import OpenCoodLiRaFusionGate
        except Exception as exc:
            gate_file = Path(lira_repo_root) / 'plugin' / 'lirafusion' / 'models' / 'backbones' / 'opencood_lirafusion_gate.py'
            if not gate_file.exists():
                raise ImportError(
                    'Could not import OpenCoodLiRaFusionGate from LiRaFusion package and fallback file is missing. '
                    f'Expected file: {gate_file}'
                ) from exc

            spec = importlib.util.spec_from_file_location('opencood_lirafusion_gate', str(gate_file))
            if spec is None or spec.loader is None:
                raise ImportError(
                    f'Could not create import spec for fallback gate file: {gate_file}'
                ) from exc

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            OpenCoodLiRaFusionGate = getattr(module, 'OpenCoodLiRaFusionGate', None)
            if OpenCoodLiRaFusionGate is None:
                raise ImportError(
                    f'Fallback gate file does not define OpenCoodLiRaFusionGate: {gate_file}'
                ) from exc

        self.lidar_pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.lidar_scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.radar_pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.radar_scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.lira_fusion_gate = OpenCoodLiRaFusionGate(channels=64)
        self.gate_zero_init = bool(args.get('gate_zero_init', False))
        if self.gate_zero_init:
            for gate_branch in [self.lira_fusion_gate.lidar_gate, self.lira_fusion_gate.radar_gate]:
                if hasattr(gate_branch[0], 'weight') and gate_branch[0].weight is not None:
                    nn.init.constant_(gate_branch[0].weight, 0.0)
                if hasattr(gate_branch[0], 'bias') and gate_branch[0].bias is not None:
                    nn.init.constant_(gate_branch[0].bias, 0.0)
                if hasattr(gate_branch[1], 'weight') and gate_branch[1].weight is not None:
                    nn.init.constant_(gate_branch[1].weight, 1.0)
                if hasattr(gate_branch[1], 'bias') and gate_branch[1].bias is not None:
                    nn.init.constant_(gate_branch[1].bias, 0.0)

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 128)
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
                kernel_size=1,
            )
        else:
            self.use_dir = False

    def forward(self, data_dict):
        lidar_batch_dict = {
            'voxel_features': data_dict['processed_lidar']['voxel_features'],
            'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
            'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
        }
        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict = self.lidar_scatter(lidar_batch_dict)

        radar_batch_dict = {
            'voxel_features': data_dict['processed_radar']['voxel_features'],
            'voxel_coords': data_dict['processed_radar']['voxel_coords'],
            'voxel_num_points': data_dict['processed_radar']['voxel_num_points'],
        }
        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.radar_scatter(radar_batch_dict)

        lidar_feat = lidar_batch_dict['spatial_features']
        radar_feat = radar_batch_dict['spatial_features']

        fused_lidar, fused_radar, gate_lidar, gate_radar = self.lira_fusion_gate(lidar_feat, radar_feat)

        batch_dict = {
            'spatial_features': torch.cat([fused_lidar, fused_radar], dim=1),
            'gate_lidar': gate_lidar,
            'gate_radar': gate_radar,
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
            'gate_lidar': gate_lidar,
            'gate_radar': gate_radar,
        }

        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict
