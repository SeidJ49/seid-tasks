import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1]

        assert self.nz == 1

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)

            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :]  # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :]  # (batch_idx_voxel,64)
            pillars = pillars.t()  # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_dict['spatial_features_3d'] = batch_spatial_features.view(batch_size, self.num_bev_features, self.nz, self.ny, self.nx)

        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)  # It put y axis(in lidar frame) as image height. [..., 200, 704]
        batch_dict['spatial_features'] = batch_spatial_features

        # --------------------------------------------NEW 15.04.2025 ---------------------------------------------------
        # scatter velocity_confidence to generate a confidence mask with the same spatial dimensions
        velocity_confidence = batch_dict['velocity_confidence']  # assume shape (M,)
        batch_confidence_masks = []
        for batch_idx in range(batch_size):
            confidence_feature = torch.zeros(
                1,  # one channel for confidence
                self.nz * self.nx * self.ny,
                dtype=velocity_confidence.dtype,
                device=velocity_confidence.device)
            # select indices that belong to the current batch
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]  # (num_voxels_in_batch, 4)
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.long()
            # get corresponding confidence values (assumed shape: (num_voxels_in_batch,))
            conf_vals = velocity_confidence[batch_mask]
            confidence_feature[:, indices] = conf_vals.view(1, -1)
            batch_confidence_masks.append(confidence_feature)

        batch_confidence_masks = torch.stack(batch_confidence_masks, 0)
        batch_confidence_masks = batch_confidence_masks.view(batch_size, 1 * self.nz, self.ny, self.nx)
        batch_dict['velocity_confidence_mask'] = batch_confidence_masks

        #save_tensor_images(batch_confidence_masks, "./saved_images", prefix="confidence_mask")

        # --------------------------------------------NEW 15.04.2025 ---------------------------------------------------

        return batch_dict


counter = 0

import os
import torch
import numpy as np
from PIL import Image


def save_tensor_images(tensor, folder, prefix="img"):
    """
    Saves each image in a batch tensor as an image file in the specified folder,
    with non-zero values highlighted in bright red in RGB mode.

    Args:
        tensor (torch.Tensor): A tensor of shape (B, C, H, W) containing mask data.
        folder (str): Path to the folder where images will be saved.
        prefix (str): Prefix for the saved image filenames.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    tensor = tensor.detach().cpu()

    batch_size, channels, height, width = tensor.shape
    global counter  # Global counter for unique file names

    for i in range(batch_size):
        img_tensor = tensor[i]  # shape: (C, H, W)

        # Ensure tensor is in (H, W) format for a single-channel mask
        if channels == 1:
            img_array = img_tensor.squeeze(0).numpy()  # shape: (H, W)
            # Create an RGB image array initialized with zeros
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

            # Apply bright red: Set red channel to 255 where values are not zero
            non_zero_mask = img_array > 0
            rgb_array[..., 0] = non_zero_mask * 255  # Red channel
        else:
            raise ValueError("This function only supports single-channel tensors (C=1) for brightness masking.")

        # Convert the array to a PIL Image
        img = Image.fromarray(rgb_array, mode='RGB')

        # Save the image
        file_path = os.path.join(folder, f"{prefix}_{counter:03d}.png")
        img.save(file_path)
        print(f"Saved image: {file_path}")
        counter += 1

# Example usage:
# Suppose your tensor has shape (3, 1, 192, 704) or (3, 3, 192, 704)
# tensor = torch.rand(3, 1, 192, 704)  # or torch.rand(3, 3, 192, 704)
# save_tensor_images(tensor, "./saved_images", prefix="example")