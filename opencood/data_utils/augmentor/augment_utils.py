# -*- coding: utf-8 -*-
# Author: OpenPCDet

import numpy as np

from opencood.utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0],
                                       rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = \
        common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                           np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation]))[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points

# ------------------------------------------------------------------------------------------------------------------------------------------
# region -- DATA-AUGMENT-RARAR-(NOT-USED)---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

'''
    # -*- coding: utf-8 -*-
    # Author: OpenPCDet (Modified for Radar)
    
    import numpy as np
    from opencood.utils import common_utils
    
    # Define indices for clarity (adjust if your format differs)
    IDX_X, IDX_Y, IDX_Z = 0, 1, 2
    IDX_VX, IDX_VY, IDX_VZ = 3, 4, 5 # For points array
    GT_IDX_VX, GT_IDX_VY, GT_IDX_VZ = 7, 8, 9 # For gt_boxes array
    
    def random_flip_along_x(gt_boxes, points):
        """
        Flips data along the X-axis (y coordinate and vy velocity change sign).
        Args:
            gt_boxes: (N, 10), [x, y, z, dx, dy, dz, heading, vx, vy, vz]
            points: (M, 6 + C), [x, y, z, vx, vy, vz, ...]
        Returns:
            gt_boxes, points: Flipped data.
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            # Flip y coordinate
            gt_boxes[:, IDX_Y] = -gt_boxes[:, IDX_Y]
            points[:, IDX_Y] = -points[:, IDX_Y]
    
            # Flip heading
            gt_boxes[:, 6] = -gt_boxes[:, 6]
    
            # Flip vy velocity
            if gt_boxes.shape[1] > GT_IDX_VY: # Check if vy exists in gt_boxes
                gt_boxes[:, GT_IDX_VY] = -gt_boxes[:, GT_IDX_VY]
            if points.shape[1] > IDX_VY:    # Check if vy exists in points
                 points[:, IDX_VY] = -points[:, IDX_VY]
    
        return gt_boxes, points
    
    
    def random_flip_along_y(gt_boxes, points):
        """
        Flips data along the Y-axis (x coordinate and vx velocity change sign).
        Args:
            gt_boxes: (N, 10), [x, y, z, dx, dy, dz, heading, vx, vy, vz]
            points: (M, 6 + C), [x, y, z, vx, vy, vz, ...]
        Returns:
            gt_boxes, points: Flipped data.
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            # Flip x coordinate
            gt_boxes[:, IDX_X] = -gt_boxes[:, IDX_X]
            points[:, IDX_X] = -points[:, IDX_X]
    
            # Flip heading (adjusting angle)
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
    
            # Flip vx velocity
            if gt_boxes.shape[1] > GT_IDX_VX: # Check if vx exists in gt_boxes
                gt_boxes[:, GT_IDX_VX] = -gt_boxes[:, GT_IDX_VX]
            if points.shape[1] > IDX_VX:    # Check if vx exists in points
                points[:, IDX_VX] = -points[:, IDX_VX]
    
        return gt_boxes, points
    
    
    def global_rotation(gt_boxes, points, rot_range):
        """
        Rotates data globally around the Z-axis.
        Args:
            gt_boxes: (N, 10), [x, y, z, dx, dy, dz, heading, vx, vy, vz]
            points: (M, 6 + C), [x, y, z, vx, vy, vz, ...]
            rot_range: [min, max] rotation angle in radians.
        Returns:
            gt_boxes, points: Rotated data.
        """
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        rotation_matrix = np.array([[np.cos(noise_rotation), -np.sin(noise_rotation), 0],
                                    [np.sin(noise_rotation),  np.cos(noise_rotation), 0],
                                    [0,                             0,                1]])
    
        # Rotate points' positions
        points[:, IDX_X:IDX_Z+1] = points[:, IDX_X:IDX_Z+1] @ rotation_matrix.T
        # Rotate points' velocities (vx, vy rotate, vz stays the same)
        if points.shape[1] > IDX_VZ:
            points[:, IDX_VX:IDX_VZ+1] = points[:, IDX_VX:IDX_VZ+1] @ rotation_matrix.T # Vz rotation is identity * vz = vz
    
    
        # Rotate gt_boxes' positions
        gt_boxes[:, IDX_X:IDX_Z+1] = \
            common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, IDX_X:IDX_Z+1],
                                               np.array([noise_rotation]))[0]
        # Adjust heading
        gt_boxes[:, 6] += noise_rotation
        gt_boxes[:, 6] = common_utils.limit_period(gt_boxes[:, 6], period=2*np.pi) # Keep heading within [-pi, pi) or [0, 2pi) depending on common_utils
    
        # Rotate gt_boxes' velocities (vx, vy rotate, vz stays the same)
        if gt_boxes.shape[1] > GT_IDX_VZ:
            # Use the common_utils function if it handles 3D points, otherwise apply matrix manually
            # Assuming common_utils.rotate_points_along_z handles Nx3 arrays
            velocities = gt_boxes[:, GT_IDX_VX:GT_IDX_VZ+1] # Shape (N, 3)
            rotated_velocities = common_utils.rotate_points_along_z(velocities[np.newaxis, :, :],
                                                                     np.array([noise_rotation]))[0]
            gt_boxes[:, GT_IDX_VX:GT_IDX_VZ+1] = rotated_velocities
            # Manual rotation if common_utils doesn't work directly on velocities:
            # gt_boxes[:, GT_IDX_VX:GT_IDX_VZ+1] = gt_boxes[:, GT_IDX_VX:GT_IDX_VZ+1] @ rotation_matrix.T
    
    
        return gt_boxes, points
    
    
    def global_scaling(gt_boxes, points, scale_range):
        """
        Scales data globally.
        Args:
            gt_boxes: (N, 10), [x, y, z, dx, dy, dz, heading, vx, vy, vz]
            points: (M, 6 + C), [x, y, z, vx, vy, vz, ...]
            scale_range: [min, max] scaling factor.
        Returns:
            gt_boxes, points: Scaled data.
        """
        if scale_range[1] - scale_range[0] < 1e-3:
            return gt_boxes, points
    
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    
        # Scale points' positions
        points[:, IDX_X:IDX_Z+1] *= noise_scale
        # Scale points' velocities
        if points.shape[1] > IDX_VZ:
            points[:, IDX_VX:IDX_VZ+1] *= noise_scale
    
        # Scale gt_boxes' positions and dimensions
        gt_boxes[:, :6] *= noise_scale
        # Scale gt_boxes' velocities
        if gt_boxes.shape[1] > GT_IDX_VZ:
            gt_boxes[:, GT_IDX_VX:GT_IDX_VZ+1] *= noise_scale
    
        return gt_boxes, points
'''

# endregion
# ------------------------------------------------------------------------------------------------------------------------------------------
