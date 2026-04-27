# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

import numpy as np

from opencood.data_utils.augmentor import augment_utils


class DataAugmentor(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue = []
        self.train = train

        for cur_cfg in augment_config:
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points,
            )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        gt_boxes_valid, points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range=rot_range
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(
            gt_boxes_valid, points, config['WORLD_SCALE_RANGE']
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict

# ------------------------------------------------------------------------------------------------------------------------------------------
# region DATA-AUGMENT-RADAR-(NOT-USED) -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

'''
    # -*- coding: utf-8 -*-
    """
    Class for data augmentation (Modified for Radar)
    """
    
    from functools import partial
    import numpy as np
    
    # Make sure to import the MODIFIED augment_utils
    from . import augment_utils # Or adjust the import path as needed
    
    class DataAugmentor(object):
        """
        Data Augmentor for Radar and potentially other modalities.
    
        Parameters
        ----------
        augment_config : list
            A list of augmentation configuration.
        radar_key : str, optional
            The key in the data_dict holding the radar point cloud numpy array.
            Defaults to 'radar_np'.
        """
    
        def __init__(self, augment_config, train=True, radar_key='radar_np'):
            self.data_augmentor_queue = []
            self.train = train
            self.radar_key = radar_key # Store the key for radar points
    
            # Use partial to pass the radar key to the augmentation methods
            aug_func_mapping = {
                'random_world_flip': self.random_world_flip,
                'random_world_rotation': self.random_world_rotation,
                'random_world_scaling': self.random_world_scaling
            }
    
            for cur_cfg in augment_config:
                if cur_cfg['NAME'] in aug_func_mapping:
                    # Pass the radar_key using partial when creating the augmentor instance
                    cur_augmentor = partial(aug_func_mapping[cur_cfg['NAME']],
                                            config=cur_cfg,
                                            radar_key=self.radar_key)
                    self.data_augmentor_queue.append(cur_augmentor)
                else:
                    print(f"Warning: Augmentation '{cur_cfg['NAME']}' not recognized or implemented.")
                    # Optionally: Implement other augmentations or raise an error
    
        def random_world_flip(self, data_dict=None, config=None, radar_key='radar_np'):
            # This function now gets radar_key passed via partial
            if data_dict is None:
                 # Return a partial function with config and radar_key already set
                return partial(self.random_world_flip, config=config, radar_key=radar_key)
    
            # Use the provided radar_key to access points
            if radar_key not in data_dict:
                print(f"Warning: Radar key '{radar_key}' not found in data_dict for random_world_flip.")
                return data_dict
    
            gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                        data_dict['object_bbx_mask'], \
                                        data_dict[radar_key] # Use the key here
            gt_boxes_valid = gt_boxes[gt_mask == 1]
    
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['x', 'y']
                # Call the modified augment_utils function
                gt_boxes_valid, points = getattr(augment_utils,
                                                 f'random_flip_along_{cur_axis}')(
                                                    gt_boxes_valid, points)
    
            gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
    
            data_dict['object_bbx_center'] = gt_boxes
            data_dict['object_bbx_mask'] = gt_mask
            data_dict[radar_key] = points # Update the dict with the key
    
            return data_dict
    
        def random_world_rotation(self, data_dict=None, config=None, radar_key='radar_np'):
            # This function now gets radar_key passed via partial
            if data_dict is None:
                # Return a partial function with config and radar_key already set
                return partial(self.random_world_rotation, config=config, radar_key=radar_key)
    
            # Use the provided radar_key to access points
            if radar_key not in data_dict:
                print(f"Warning: Radar key '{radar_key}' not found in data_dict for random_world_rotation.")
                return data_dict
    
            rot_range = config['WORLD_ROT_ANGLE']
            if not isinstance(rot_range, list):
                rot_range = [-rot_range, rot_range]
    
            gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                        data_dict['object_bbx_mask'], \
                                        data_dict[radar_key] # Use the key here
            gt_boxes_valid = gt_boxes[gt_mask == 1]
    
            # Call the modified augment_utils function
            gt_boxes_valid, points = augment_utils.global_rotation(
                gt_boxes_valid, points, rot_range=rot_range
            )
            gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
    
            data_dict['object_bbx_center'] = gt_boxes
            data_dict['object_bbx_mask'] = gt_mask
            data_dict[radar_key] = points # Update the dict with the key
    
            return data_dict
    
        def random_world_scaling(self, data_dict=None, config=None, radar_key='radar_np'):
             # This function now gets radar_key passed via partial
            if data_dict is None:
                # Return a partial function with config and radar_key already set
                return partial(self.random_world_scaling, config=config, radar_key=radar_key)
    
            # Use the provided radar_key to access points
            if radar_key not in data_dict:
                print(f"Warning: Radar key '{radar_key}' not found in data_dict for random_world_scaling.")
                return data_dict
    
            gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                        data_dict['object_bbx_mask'], \
                                        data_dict[radar_key] # Use the key here
            gt_boxes_valid = gt_boxes[gt_mask == 1]
    
            # Call the modified augment_utils function
            gt_boxes_valid, points = augment_utils.global_scaling(
                gt_boxes_valid, points, config['WORLD_SCALE_RANGE']
            )
            gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
    
            data_dict['object_bbx_center'] = gt_boxes
            data_dict['object_bbx_mask'] = gt_mask
            data_dict[radar_key] = points # Update the dict with the key
    
            return data_dict
    
        def forward(self, data_dict):
            """
            Applies the augmentation sequence to the data dictionary.
    
            Args:
                data_dict: Dictionary containing sensor data and ground truth.
                           Expected to have keys like 'object_bbx_center',
                           'object_bbx_mask', and the radar key (e.g., 'radar_np').
                           Points format: (M, 6 + C) [x, y, z, vx, vy, vz, ...]
                           GT boxes format: (N, 10) [x, y, z, dx, dy, dz, heading, vx, vy, vz]
    
            Returns:
                data_dict: The augmented data dictionary.
            """
            if self.train:
                # data_dict is passed to each function in the queue
                # The partial function already has config and radar_key baked in
                for cur_augmentor in self.data_augmentor_queue:
                    data_dict = cur_augmentor(data_dict=data_dict)
    
            # Optionally handle gt_boxes shape consistency if needed after augmentation
            # gt_boxes = data_dict['object_bbx_center']
            # if gt_boxes.shape[1] == 9: # Example: If vz was sometimes missing and added
            #     # Pad or handle appropriately
            #     pass
    
            return data_dict
'''

# endregion
# ------------------------------------------------------------------------------------------------------------------------------------------