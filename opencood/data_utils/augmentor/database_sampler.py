import copy
import pickle
from pathlib import Path

import numpy as np

from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_utils
from opencood.data_utils.datasets.truckscenes_class_utils import TRUCKSCENES_NAME_TO_DETECTION
from opencood.utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names):
        self.root_path = Path(root_path)
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.db_infos = {class_name: [] for class_name in class_names}

        for db_info_path in sampler_cfg.get('DB_INFO_PATH', []):
            info_path = self.root_path / db_info_path
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            for class_name in class_names:
                self.db_infos[class_name].extend(infos.get(class_name, []))

        for func_name, value in sampler_cfg.get('PREPARE', {}).items():
            self.db_infos = getattr(self, func_name)(self.db_infos, value)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for group in sampler_cfg.get('SAMPLE_GROUPS', []):
            class_name, sample_num = group.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = int(sample_num)
            self.sample_groups[class_name] = {
                'sample_num': int(sample_num),
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

        self.num_point_features = sampler_cfg.get('NUM_POINT_FEATURES', 4)
        self.remove_extra_width = np.asarray(sampler_cfg.get('REMOVE_EXTRA_WIDTH', [0.0, 0.0, 0.0]), dtype=np.float32)

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num <= 0 or name not in db_infos:
                continue
            db_infos[name] = [info for info in db_infos[name] if info.get('num_points_in_gt', 0) >= min_num]
        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        sample_num = int(sample_group['sample_num'])
        pointer = sample_group['pointer']
        indices = sample_group['indices']
        if len(self.db_infos[class_name]) == 0 or sample_num <= 0:
            return []

        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer:pointer + sample_num]]
        sample_group['pointer'] = pointer + sample_num
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def _coarse_name(name):
        if name is None:
            return None
        name = str(name)
        return TRUCKSCENES_NAME_TO_DETECTION.get(name, name)

    @staticmethod
    def _db_box_to_hwl(boxes):
        boxes = boxes.copy()
        boxes[:, 3:6] = boxes[:, [5, 4, 3]]
        return boxes

    @staticmethod
    def _boxes_hwl_to_lwh(boxes):
        if boxes.shape[0] == 0:
            return boxes.astype(np.float32)
        converted = boxes.copy().astype(np.float32)
        converted[:, 3:6] = converted[:, [5, 4, 3]]
        return converted

    @staticmethod
    def _points_in_boxes_mask(points, boxes3d):
        if points.shape[0] == 0 or boxes3d.shape[0] == 0:
            return np.zeros(points.shape[0], dtype=bool)
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], DataBaseSampler._boxes_hwl_to_lwh(boxes3d[:, :7]))
        return point_indices.sum(axis=0) > 0

    def _load_points(self, info, key, count_key, feature_dim):
        rel_path = info.get(key)
        if rel_path is None:
            return np.zeros((0, feature_dim), dtype=np.float32)
        file_path = self.root_path / rel_path
        if not file_path.exists():
            return np.zeros((0, feature_dim), dtype=np.float32)

        points = np.fromfile(str(file_path), dtype=np.float32)
        if points.size == 0:
            return np.zeros((0, feature_dim), dtype=np.float32)

        expected = int(info.get(count_key, points.shape[0]))
        stored_feature_dim = feature_dim
        if expected > 0 and points.size % expected == 0:
            stored_feature_dim = points.size // expected
        points = points.reshape(-1, stored_feature_dim)
        if expected != points.shape[0]:
            points = points[:expected]

        if key == 'radar_path' and points.shape[1] > feature_dim:
            if points.shape[1] >= 6:
                radius = np.linalg.norm(points[:, :3], axis=1)
                radius = np.where(radius < 1e-6, 1e-6, radius)
                unit_vec = points[:, :3] / radius[:, None]
                v_rel = np.sum(points[:, 3:6] * unit_vec, axis=1, keepdims=True)
                points = np.concatenate([points[:, :3], v_rel], axis=1)
            else:
                points = points[:, :feature_dim]

        points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)
        return points

    def sample_all(self, gt_boxes, gt_names, lidar_points, radar_points, scene_gt_names=None):
        existed_boxes = gt_boxes.astype(np.float32)
        limit_gt_names = gt_names if scene_gt_names is None else scene_gt_names
        coarse_gt_names = np.array([self._coarse_name(name) for name in limit_gt_names])
        total_valid_sampled = []

        for class_name, sample_group in self.sample_groups.items():
            sample_num = self.sample_class_num[class_name]
            if self.limit_whole_scene:
                sample_num = sample_num - int(np.sum(coarse_gt_names == class_name))
            if sample_num <= 0:
                continue

            original_sample_num = sample_group['sample_num']
            sample_group['sample_num'] = sample_num
            sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
            sample_group['sample_num'] = original_sample_num
            if len(sampled_dict) == 0:
                continue

            sampled_boxes = np.stack([item['box3d_lidar'][:7] for item in sampled_dict], axis=0).astype(np.float32)
            sampled_boxes = self._db_box_to_hwl(sampled_boxes)
            sampled_boxes_lwh = self._boxes_hwl_to_lwh(sampled_boxes[:, :7])
            existed_boxes_lwh = self._boxes_hwl_to_lwh(existed_boxes[:, :7]) if existed_boxes.shape[0] > 0 else np.zeros((0, 7), dtype=np.float32)
            iou_existing = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes_lwh, existed_boxes_lwh) if existed_boxes_lwh.shape[0] > 0 else np.zeros((sampled_boxes_lwh.shape[0], 0), dtype=np.float32)
            iou_sampled = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes_lwh, sampled_boxes_lwh)
            np.fill_diagonal(iou_sampled, 0.0)
            max_existing = iou_existing.max(axis=1) if iou_existing.shape[1] > 0 else np.zeros(sampled_boxes_lwh.shape[0], dtype=np.float32)
            max_sampled = iou_sampled.max(axis=1) if iou_sampled.shape[1] > 0 else np.zeros(sampled_boxes_lwh.shape[0], dtype=np.float32)
            valid_mask = (max_existing + max_sampled) == 0

            if not np.any(valid_mask):
                continue

            valid_indices = np.nonzero(valid_mask)[0]
            valid_sampled = [sampled_dict[idx] for idx in valid_indices]
            valid_boxes = sampled_boxes[valid_indices]
            existed_boxes = np.concatenate([existed_boxes, valid_boxes], axis=0)
            total_valid_sampled.extend(valid_sampled)

        if len(total_valid_sampled) == 0:
            return {
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'lidar_points': lidar_points,
                'radar_points': radar_points,
                'sampled_object_ids': [],
            }

        sampled_boxes = existed_boxes[gt_boxes.shape[0]:]
        sampled_names = np.array([item['name'] for item in total_valid_sampled])

        lidar_objects = []
        radar_objects = []
        for info in total_valid_sampled:
            lidar_objects.append(self._load_points(info, 'path', 'num_points_in_gt', self.num_point_features))
            radar_objects.append(self._load_points(info, 'radar_path', 'num_radar_points_in_gt', 4))

        sampled_lidar = np.concatenate(lidar_objects, axis=0) if lidar_objects else np.zeros((0, lidar_points.shape[1]), dtype=np.float32)
        sampled_radar = np.concatenate(radar_objects, axis=0) if radar_objects else np.zeros((0, radar_points.shape[1]), dtype=np.float32)

        if sampled_boxes.shape[0] > 0:
            enlarged_boxes = sampled_boxes.copy()
            if self.remove_extra_width.size == 3:
                enlarged_boxes[:, 3:6] += self.remove_extra_width[None, [2, 1, 0]] * 2.0
            lidar_keep = ~self._points_in_boxes_mask(lidar_points, enlarged_boxes)
            radar_keep = ~self._points_in_boxes_mask(radar_points, enlarged_boxes)
            lidar_points = lidar_points[lidar_keep]
            radar_points = radar_points[radar_keep]

        lidar_points = np.concatenate([sampled_lidar[:, :lidar_points.shape[1]], lidar_points], axis=0)
        radar_points = np.concatenate([sampled_radar[:, :radar_points.shape[1]], radar_points], axis=0)

        sampled_object_ids = [-(idx + 1) for idx, _ in enumerate(total_valid_sampled)]
        gt_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
        gt_names = np.concatenate([gt_names, sampled_names], axis=0)

        return {
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'lidar_points': lidar_points,
            'radar_points': radar_points,
            'sampled_object_ids': sampled_object_ids,
        }