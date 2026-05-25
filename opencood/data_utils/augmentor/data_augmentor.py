# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial
from pathlib import Path

import numpy as np

from opencood.data_utils.augmentor import augment_utils
from opencood.data_utils.augmentor.database_sampler import DataBaseSampler


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

    def __init__(self, augment_config, train=True, data_root=None, class_names=None):
        self.data_augmentor_queue = []
        self.train = train
        self.data_root = Path(data_root) if data_root is not None else None
        self.class_names = class_names or []
        self.gt_sampler = None

        for cur_cfg in augment_config:
            if cur_cfg['NAME'] == 'gt_sampling' and self.data_root is not None and self.class_names:
                self.gt_sampler = DataBaseSampler(self.data_root, cur_cfg, self.class_names)
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            if cur_augmentor is not None:
                self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, data_dict=None, config=None):
        if self.gt_sampler is None:
            return None
        if data_dict is None:
            return partial(self.gt_sampling, config=config)

        gt_boxes, gt_mask, box_key, mask_key = self._get_boxes_and_mask(data_dict)
        sampled_dict = self.gt_sampler.sample_all(
            gt_boxes=gt_boxes[gt_mask == 1],
            gt_names=data_dict['gt_names'],
            lidar_points=data_dict['lidar_np'],
            radar_points=data_dict['radar_np'],
            scene_gt_names=data_dict.get('raw_gt_names'),
        )
        if box_key == 'object_bbx_center':
            merged_boxes = sampled_dict['gt_boxes']
            updated_boxes = np.zeros_like(gt_boxes)
            updated_mask = np.zeros_like(gt_mask)
            valid_count = min(updated_boxes.shape[0], merged_boxes.shape[0])
            updated_boxes[:valid_count, :merged_boxes.shape[1]] = merged_boxes[:valid_count]
            updated_mask[:valid_count] = 1
            data_dict['object_bbx_center'] = updated_boxes
            data_dict['object_bbx_mask'] = updated_mask
        else:
            data_dict['gt_boxes'] = sampled_dict['gt_boxes']
        data_dict['gt_names'] = sampled_dict['gt_names']
        data_dict['lidar_np'] = sampled_dict['lidar_points']
        data_dict['radar_np'] = sampled_dict['radar_points']
        data_dict['sampled_object_ids'] = sampled_dict['sampled_object_ids']
        return data_dict

    @staticmethod
    def _get_boxes_and_mask(data_dict):
        if 'object_bbx_center' in data_dict and 'object_bbx_mask' in data_dict:
            gt_boxes = data_dict['object_bbx_center']
            gt_mask = data_dict['object_bbx_mask']
            box_key = 'object_bbx_center'
            mask_key = 'object_bbx_mask'
            return gt_boxes, gt_mask, box_key, mask_key

        gt_boxes = data_dict['gt_boxes']
        gt_mask = np.ones(gt_boxes.shape[0], dtype=np.float32)
        return gt_boxes, gt_mask, 'gt_boxes', None

    @staticmethod
    def _update_boxes(data_dict, box_key, mask_key, gt_boxes, gt_mask, gt_boxes_valid):
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
        data_dict[box_key] = gt_boxes
        if mask_key is not None:
            data_dict[mask_key] = gt_mask
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, box_key, mask_key = self._get_boxes_and_mask(data_dict)
        points = data_dict['lidar_np']
        radar_points = data_dict.get('radar_np', None)
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            gt_boxes_valid, points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points,
                enable=enable,
            )
            if radar_points is not None:
                _, radar_points = getattr(augment_utils,
                                          'random_flip_along_%s' % cur_axis)(
                    gt_boxes_valid.copy(), radar_points,
                    enable=enable,
                )

        data_dict = self._update_boxes(data_dict, box_key, mask_key, gt_boxes, gt_mask, gt_boxes_valid)
        data_dict['lidar_np'] = points
        if radar_points is not None:
            data_dict['radar_np'] = radar_points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, box_key, mask_key = self._get_boxes_and_mask(data_dict)
        points = data_dict['lidar_np']
        radar_points = data_dict.get('radar_np', None)
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        gt_boxes_valid, points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range=rot_range, noise_rotation=noise_rotation
        )
        if radar_points is not None:
            _, radar_points = augment_utils.global_rotation(
                gt_boxes_valid.copy(), radar_points, rot_range=rot_range, noise_rotation=noise_rotation
            )
        data_dict = self._update_boxes(data_dict, box_key, mask_key, gt_boxes, gt_mask, gt_boxes_valid)
        data_dict['lidar_np'] = points
        if radar_points is not None:
            data_dict['radar_np'] = radar_points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, box_key, mask_key = self._get_boxes_and_mask(data_dict)
        points = data_dict['lidar_np']
        radar_points = data_dict.get('radar_np', None)
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        noise_scale = None
        scale_range = config['WORLD_SCALE_RANGE']
        if scale_range[1] - scale_range[0] >= 1e-3:
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])

        gt_boxes_valid, points = augment_utils.global_scaling(
            gt_boxes_valid, points, scale_range, noise_scale=noise_scale
        )
        if radar_points is not None:
            _, radar_points = augment_utils.global_scaling(
                gt_boxes_valid.copy(), radar_points, scale_range, noise_scale=noise_scale
            )
        data_dict = self._update_boxes(data_dict, box_key, mask_key, gt_boxes, gt_mask, gt_boxes_valid)
        data_dict['lidar_np'] = points
        if radar_points is not None:
            data_dict['radar_np'] = radar_points

        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)

        gt_boxes, gt_mask, box_key, mask_key = self._get_boxes_and_mask(data_dict)
        points = data_dict['lidar_np']
        radar_points = data_dict.get('radar_np', None)
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        translate_std = np.asarray(config['NOISE_TRANSLATE_STD'], dtype=np.float32)
        noise_translate = None
        if not np.all(translate_std < 1e-6):
            noise_translate = np.random.normal(loc=0.0, scale=translate_std, size=3).astype(np.float32)

        gt_boxes_valid, points = augment_utils.global_translation(
            gt_boxes_valid, points, translate_std, noise_translate=noise_translate
        )
        if radar_points is not None:
            _, radar_points = augment_utils.global_translation(
                gt_boxes_valid.copy(), radar_points, translate_std, noise_translate=noise_translate
            )
        data_dict = self._update_boxes(data_dict, box_key, mask_key, gt_boxes, gt_mask, gt_boxes_valid)
        data_dict['lidar_np'] = points
        if radar_points is not None:
            data_dict['radar_np'] = radar_points

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
