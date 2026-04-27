from functools import partial

import torch.nn as nn

try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv as spconv


def replace_feature(out, new_features):
    if hasattr(out, 'replace_feature'):
        return out.replace_feature(new_features)
    out.features = new_features
    return out


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    else:
        raise NotImplementedError(conv_type)

    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super().__init__()
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample

    def forward(self, sparse_tensor):
        identity = sparse_tensor
        out = self.conv1(sparse_tensor)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(sparse_tensor)
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super().__init__()
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample

    def forward(self, features):
        identity = features
        out = self.relu(self.bn1(self.conv1(features)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(features)
        return self.relu(out + identity)


class _BasePillarRes18BackBone8x(nn.Module):
    def __init__(self, grid_size, feature_key, coord_key, output_prefix=None):
        super().__init__()
        self.feature_key = feature_key
        self.coord_key = coord_key
        self.output_prefix = output_prefix
        self.sparse_shape = grid_size[[1, 0]]
        sparse_norm = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        dense_norm = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=sparse_norm, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=sparse_norm, indice_key='res1'),
        )
        self.conv2 = spconv.SparseSequential(
            post_act_block(32, 64, 3, norm_fn=sparse_norm, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=sparse_norm, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=sparse_norm, indice_key='res2'),
        )
        self.conv3 = spconv.SparseSequential(
            post_act_block(64, 128, 3, norm_fn=sparse_norm, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=sparse_norm, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=sparse_norm, indice_key='res3'),
        )
        self.conv4 = spconv.SparseSequential(
            post_act_block(128, 256, 3, norm_fn=sparse_norm, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=sparse_norm, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=sparse_norm, indice_key='res4'),
        )
        self.conv5 = nn.Sequential(
            post_act_block_dense(256, 256, 3, norm_fn=dense_norm, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=dense_norm),
            BasicBlock(256, 256, norm_fn=dense_norm),
        )

    def forward(self, batch_dict):
        pillar_features = batch_dict[self.feature_key]
        pillar_coords = batch_dict[self.coord_key]
        batch_size = batch_dict['batch_size']
        sparse_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x_conv1 = self.conv1(sparse_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3).dense()
        x_conv5 = self.conv5(x_conv4)

        prefix = '' if self.output_prefix is None else f'{self.output_prefix}_'
        batch_dict[f'{prefix}multi_scale_2d_features'] = {
            'x_conv1': x_conv1,
            'x_conv2': x_conv2,
            'x_conv3': x_conv3,
            'x_conv4': x_conv4,
            'x_conv5': x_conv5,
        }
        batch_dict[f'{prefix}multi_scale_2d_strides'] = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }
        return batch_dict


class PillarRes18BackBone8x(_BasePillarRes18BackBone8x):
    def __init__(self, grid_size):
        super().__init__(grid_size, feature_key='pillar_features', coord_key='pillar_coords', output_prefix=None)


class RadarPillarRes18BackBone8x(_BasePillarRes18BackBone8x):
    def __init__(self, grid_size):
        super().__init__(grid_size, feature_key='radar_pillar_features', coord_key='radar_pillar_coords', output_prefix='radar')