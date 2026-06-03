import torch
import torch.nn as nn


class BaseBEVBackboneV2(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        layer_nums = model_cfg['layer_nums']
        num_filters = model_cfg['num_filters']
        num_upsample_filters = model_cfg['num_upsample_filters']
        upsample_strides = model_cfg['upsample_strides']

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for index in range(len(layer_nums)):
            input_channels = num_filters[index] * 2 if index == 0 else num_filters[index]
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(input_channels, num_filters[index], kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_filters[index], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[index]):
                layers.extend([
                    nn.Conv2d(num_filters[index], num_filters[index], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[index], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*layers))
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[index],
                    num_upsample_filters[index] * 2,
                    upsample_strides[index],
                    stride=upsample_strides[index],
                    bias=False,
                ),
                nn.BatchNorm2d(num_upsample_filters[index] * 2, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.deblocks = self.deblocks[1:]

    def forward(self, batch_dict):
        spatial_features = batch_dict['multi_scale_2d_features']
        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        batch_dict['spatial_features_2d_8x'] = ups[-1]

        x = torch.cat(ups, dim=1)
        batch_dict['spatial_features_2d'] = self.blocks[0](x)
        return batch_dict