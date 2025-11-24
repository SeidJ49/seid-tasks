import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from pypcd import pypcd
import torch
from torch.utils.data import Dataset
from opencood.utils.pcd_utils import shuffle_points, mask_ego_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.pcd_utils import mask_points_by_range
from pathlib import Path


class SingleDatasetLidarRadarBaselineAttentionMlpHis(Dataset):

    def __init__(self,params: Dict, visualize: bool = False, train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.ref_frame = params.get('ref_frame', 'ego_pose')

        # Build preprocessors
        # --- LIDAR ----------------------------------------------------------------------------------------------------
        lidar_feature_count = params['preprocess']['args']['lidar_num_point_features']
        lidar_args = {**params['preprocess']['args'],'num_point_features': lidar_feature_count}
        self.lidar_pre_processor = build_preprocessor({**params['preprocess'], 'args': lidar_args}, train)
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        radar_feature_count = params['preprocess']['args']['radar_num_point_features']
        radar_args = {**params['preprocess']['args'],
                      'max_points_per_voxel': params['preprocess']['args']['radar_max_points_per_voxel'],
                      'max_voxel_train': params['preprocess']['args']['radar_max_voxel_train'],
                      'max_voxel_test': params['preprocess']['args']['radar_max_voxel_test'],
                      'num_point_features': radar_feature_count}
        self.radar_pre_processor = build_preprocessor({**params['preprocess'], 'args': radar_args}, train)
        # --------------------------------------------------------------------------------------------------------------

        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.root_dir = params["root_dir"] if train else params["validate_dir"]
        print("Dataset dir:", self.root_dir)
        self.dataset_dir = os.path.dirname(self.root_dir)
        print("Dataset parent dir:", self.dataset_dir)

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False

        self.label_type = params['label_type']
        assert self.label_type in ['lidar']

        self.generate_object_center = self.generate_object_center_lidar
        self.generate_object_center_single = self.generate_object_center

        self.ego_mode = 'one'

        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        self.reinitialize()

        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)

        self.his_frames = params['train_params']['his_frames'] if 'his_frames' in params['train_params'] else 2
        self.fps = params['train_params']['fps'] if 'fps' in params['train_params'] else 2
        self.v_threshold = params['train_params']['v_threshold'] if 'v_threshold' in params['train_params'] else 2.5

    def reinitialize(self):
        self.scene_database = OrderedDict()
        self.len_record = []

        scene_counter = 0
        total = 0
        for scene_token, samples in self.dataset_info_pkl.items():
            self.scene_database[scene_counter] = OrderedDict()
            for sample_idx, sample in enumerate(samples):
                cav_id = 1
                self.scene_database[scene_counter].setdefault(cav_id, OrderedDict())
                self.scene_database[scene_counter][cav_id][sample_idx] = OrderedDict()

                cav_entry = self.scene_database[scene_counter][cav_id][sample_idx]
                cav_entry['ego'] = True
                cav_entry['sensors'] = sample['agents']['1']['sensors']

                radar_list = [k for k, v in cav_entry['sensors'].items() if v.get('sensor_type') == 'radar']
                lidar_list = [k for k, v in cav_entry['sensors'].items() if v.get('sensor_type') == 'lidar']
                cav_entry['radar_sensors'] = radar_list
                cav_entry['lidar_sensors'] = lidar_list

                cav_entry['params'] = OrderedDict()
                cav_entry['params']['vehicles'] = sample['labels']['gt_boxes_global']
                cav_entry['params']['object_ids'] = sample['labels']['gt_object_ids'].tolist()
                cav_entry['params']['ego_pose'] = sample['agents']['1']['ego_pose']['transform']
                cav_entry['params']['lidar_top_front_pose'] = sample['agents']['1']['lidar_top_front_pose']['transform']
                cav_entry['params']['ego_speed'] = sample['agents']['1']['ego_motion_cabin']

                total += 1

            scene_counter += 1
            self.len_record.append(total)

    def get_ref_pose(self, cav_params):
        if self.ref_frame == 'lidar_top_front_pose':
            return cav_params['lidar_top_front_pose']
        return cav_params['ego_pose']

    def T_sensor_to_ref(self, sensor_entry, cav_params):
        T_sego = np.asarray(sensor_entry['T_sensor_to_ego'], dtype=np.float64)
        if self.ref_frame == 'ego_pose':
            return T_sego

        ego_pose = np.asarray(cav_params['ego_pose'], dtype=np.float64)
        lidar_pose = np.asarray(cav_params['lidar_top_front_pose'], dtype=np.float64)
        T_ego_to_lidar = x1_to_x2(ego_pose, lidar_pose)
        return T_ego_to_lidar @ T_sego

    def __len__(self):
        return self.len_record[-1]

    @abstractmethod
    def __getitem__(self, idx):
        base_data_dict = self.retrieve_his_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict, idx)

        return reformat_data_dict


    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        ego_base = None
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_base = cav_content
                break
        assert ego_base is not None

        selected_cav_processed = self.get_item_single_car(ego_base)

        # No Cars
        if selected_cav_processed is None:
            return None

        processed_data_dict.update({"ego": selected_cav_processed})
        return processed_data_dict


    def get_item_test(self, base_data_dict, idx):
        processed_data_dict = OrderedDict()

        ego_id = -1
        ref_pose = []
        ego_base = None

        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ref_pose = self.get_ref_pose(cav_content['params'])
                ego_base = cav_content
                break

        assert ego_id != -1
        assert len(ref_pose) > 0

        transformation_matrix = x1_to_x2(ref_pose, ref_pose)

        ego_processed = self.get_item_single_car(ego_base)

        # No Cars
        if ego_processed is None:
            return None

        ego_processed.update({
            'transformation_matrix': transformation_matrix,
            'idx': idx,
            'cav_list': ['ego']
        })

        processed_data_dict['ego'] = ego_processed
        return processed_data_dict


    def retrieve_his_data(self, idx):

        scene_index = 0
        for i, end_idx in enumerate(self.len_record):
            if idx < end_idx:
                scene_index = i
                break

        sample_index = idx if scene_index == 0 else idx - self.len_record[scene_index - 1]

        ego_cav_id = 1
        cav_content = self.scene_database[scene_index][ego_cav_id][sample_index]

        data = OrderedDict()
        data[f'{ego_cav_id}'] = OrderedDict()
        data[f'{ego_cav_id}']['ego'] = True
        data[f'{ego_cav_id}']['params'] = cav_content['params']

        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        radar_points = []
        for sensor_name in cav_content['radar_sensors']:
            s = cav_content['sensors'][sensor_name]

            old_sensor_path = s['sensor_path']
            sensor_path = self.find_sensor_path(old_sensor_path)

            # --- VELOCITY PROCESSING ----------------------------------------------------------------------------------
            radar_np = self.pcd_to_npy_array(sensor_path) # local coordinate
            radar_transform = np.asarray(s['sensor_pose'])
            lidar_velocity_xyz = np.array([cav_content['params']['ego_speed']['vx'],cav_content['params']['ego_speed']['vy'],cav_content['params']['ego_speed']['vz']])
            ref_vehicle_pose = np.array(self.get_ref_pose(cav_content['params']))

            pts = self.process_all_radar_velocity(radar_np, radar_transform, lidar_velocity_xyz, ref_vehicle_pose)

            # --- HIS --------------------------------------------------------------------------------------------------
            all_his_pts = []
            for i in range(self.his_frames):
                if sample_index - (i + 1) < 0:
                    break
                his_cav_content = self.scene_database[scene_index][ego_cav_id][sample_index - (i + 1)]
                his_s = his_cav_content['sensors'][sensor_name]
                his_old_sensor_path = his_s['sensor_path']
                his_sensor_path = self.find_sensor_path(his_old_sensor_path)
                his_radar_np = self.pcd_to_npy_array(his_sensor_path)  # local coordinate
                his_radar_transform = np.asarray(his_s['sensor_pose'])
                his_lidar_velocity_xyz = np.array([his_cav_content['params']['ego_speed']['vx'], his_cav_content['params']['ego_speed']['vy'], his_cav_content['params']['ego_speed']['vz']])
                his_ref_vehicle_pose = np.array(self.get_ref_pose(his_cav_content['params']))

                his_pts = self.process_his_radar_velocity(his_radar_np, his_radar_transform, his_lidar_velocity_xyz,his_ref_vehicle_pose, radar_transform, i + 1)
                all_his_pts.append(his_pts)
            # ----------------------------------------------------------------------------------------------------------

            pts = np.concatenate([pts] + all_his_pts, axis=0)

            # Sensor -> Ref
            T = self.T_sensor_to_ref(s, cav_content['params'])
            xyz = pts[:, :3]
            xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
            xyz_ref = (T @ xyz_h.T).T[:, :3]
            pts[:, :3] = xyz_ref

            radar_points.append(pts)
        # --------------------------------------------------------------------------------------------------------------

        # --- LIDAR ----------------------------------------------------------------------------------------------------
        lidar_points = []
        for sensor_name in cav_content['lidar_sensors']:
            s = cav_content['sensors'][sensor_name]
            old_sensor_path = s['sensor_path']
            root = Path(self.dataset_dir).name
            p = Path(old_sensor_path.strip())
            i = p.parts.index(root)
            sensor_path = Path(self.dataset_dir) / Path(*p.parts[i + 1:])
            pts = self.pcd_to_npy_array_lidar(sensor_path)
            T = self.T_sensor_to_ref(s, cav_content['params'])

            # Sensor -> Ego
            xyz = pts[:, :3]
            xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
            xyz_ref= (T @ xyz_h.T).T[:, :3]
            pts[:, :3] = xyz_ref

            lidar_points.append(pts)
        # --------------------------------------------------------------------------------------------------------------
        data[f'{ego_cav_id}']['lidar_np'] = np.vstack(lidar_points)
        data[f'{ego_cav_id}']['radar_np'] = np.vstack(radar_points)

        return data


    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        return self.post_processor.generate_object_center_v2x(cav_contents, reference_lidar_pose)


    def get_item_single_car(self, selected_cav_base):
        selected_cav_processed = {}

        ref_pose = self.get_ref_pose(selected_cav_base['params'])
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single([selected_cav_base], ref_pose)

        # No Cars
        if len(object_ids) == 0:
            print("No Cars in the scene")
            return None

        # --- LIDAR ----------------------------------------------------------------------------------------------------
        if self.load_lidar_file or self.visualize:
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            lidar_np = mask_ego_points(lidar_np) # FIXME: check it

            lidar_dict = self.lidar_pre_processor.preprocess(lidar_np)
            selected_cav_processed.update({'processed_lidar': lidar_dict})
        # --------------------------------------------------------------------------------------------------------------

        # --- RADAR ----------------------------------------------------------------------------------------------------
        if self.load_lidar_file or self.visualize:
            radar_np = selected_cav_base['radar_np']
            radar_np = shuffle_points(radar_np)
            radar_np = mask_points_by_range(radar_np, self.params['preprocess']['cav_lidar_range'])
            radar_np = mask_ego_points(radar_np) # FIXME: check it

            radar_dict = self.radar_pre_processor.preprocess(radar_np)
            selected_cav_processed.update({'processed_radar': radar_dict})



        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
            }
        )

        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask)
        selected_cav_processed.update({"label_dict": label_dict})

        return selected_cav_processed


    def collate_batch_train(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        processed_radar_list = []
        label_dict_list = []
        origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        # for centerpoint
        label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                 'object_bbx_mask': object_bbx_mask})

        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'anchor_box': torch.from_numpy(self.anchor_box),
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})


        if self.load_lidar_file:
            for i in range(len(batch)):
                processed_lidar_list.append(batch[i]['ego']['processed_lidar'])
            processed_lidar_torch_dict = self.lidar_pre_processor.collate_batch(processed_lidar_list)
            output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

        # --- RADAR ----------------------------------------------------------------------------------------------------
        if self.load_lidar_file:
            for i in range(len(batch)):
                processed_radar_list.append(batch[i]['ego']['processed_radar'])
            processed_radar_torch_dict = self.radar_pre_processor.collate_batch(processed_radar_list)
            output_dict['ego'].update({'processed_radar': processed_radar_torch_dict})

        return output_dict


    def collate_batch_test(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        cav_id = 'ego'
        cav_content = batch[cav_id]

        output_dict = {cav_id: {}}

        object_bbx_center = torch.from_numpy(np.array([cav_content['object_bbx_center']]))
        object_bbx_mask = torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
        object_ids = cav_content['object_ids']

        output_dict[cav_id].update({"anchor_box": self.anchor_box_torch})

        if self.load_lidar_file:
            processed_lidar_torch_dict = self.lidar_pre_processor.collate_batch([cav_content['processed_lidar']])
            output_dict[cav_id].update({'processed_lidar': processed_lidar_torch_dict})

        # --- RADAR ----------------------------------------------------------------------------------------------------
        if self.load_lidar_file:
            processed_radar_torch_dict = self.radar_pre_processor.collate_batch([cav_content['processed_radar']])
            output_dict[cav_id].update({'processed_radar': processed_radar_torch_dict})


        label_torch_dict = self.post_processor.collate_batch([cav_content['label_dict']])
        label_torch_dict.update({
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask
        })

        tm = torch.from_numpy(np.array(cav_content['transformation_matrix'])).float()

        output_dict[cav_id].update({
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'label_dict': label_torch_dict,
            'object_ids': object_ids,
            'transformation_matrix': tm,
        })

        if self.visualize:
            projected_lidar_list = [cav_content['origin_lidar']]
            projected_lidar_stack = [torch.from_numpy(np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return output_dict


    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion(self, data_dict, output_dict_ego):
        data_dict_ego = OrderedDict()
        data_dict_ego["ego"] = data_dict["ego"]
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict_ego, output_dict_ego)
        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion_uncertainty(self, data_dict, output_dict_ego):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        pred_box_tensor, pred_score, uncertainty = self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
        return pred_box_tensor, pred_score, gt_box_tensor, uncertainty

    # --- HIS ----------------------------------------------------------------------------------------------------------
    def find_sensor_path(self, old_sensor_path):
        root = Path(self.dataset_dir).name
        p = Path(old_sensor_path.strip())
        i = p.parts.index(root)
        sensor_path = Path(self.dataset_dir) / Path(*p.parts[i+1:])
        return sensor_path
    # ------------------------------------------------------------------------------------------------------------------

    # ---LIDAR ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def pcd_to_npy_array_lidar(pcd_path):
        lidar = pypcd.PointCloud.from_path(pcd_path)
        pc = lidar.pc_data
        points = np.array([
            pc["x"],
            pc["y"],
            pc["z"],
            pc["intensity"]
        ], dtype=np.float64).T
        return points

    # ------------------------------------------------------------------------------------------------------------------

    # --- RADAR --------------------------------------------------------------------------------------------------------
    @staticmethod
    def pcd_to_npy_array(pcd_path):
        radar = pypcd.PointCloud.from_path(pcd_path)
        radar_data = radar.pc_data
        points = np.array([
            radar_data["x"],
            radar_data["y"],
            radar_data["z"],
            radar_data["vrel_x"],
            radar_data["vrel_y"],
            radar_data["vrel_z"],
            # radar_data["rcs"]
        ], dtype=np.float64).T
        return points

    def process_all_radar_velocity(self, radar_np, radar_transform, lidar_velocity_xyz, lidar_transform):
        l2r_transform = x1_to_x2(lidar_transform, radar_transform)
        l2r_rotation_matrix = l2r_transform[:3, :3]
        radar_velocity_xyz = np.dot(l2r_rotation_matrix, lidar_velocity_xyz)
        radar_np = radar_np.copy()

        r = np.linalg.norm(radar_np[:, :3], axis=1)
        r_safe = np.where(r == 0, 1e-6, r)
        unit_vec = radar_np[:, :3] / r_safe[:, None]

        v_rel_vec = radar_np[:, 3:6]
        v_rel = np.matmul(v_rel_vec, unit_vec.T).diagonal()

        # Compute radial speed from ego motion and sum with relative velocity
        v_ego_radial = unit_vec @ radar_velocity_xyz
        v_r = v_rel + v_ego_radial

        # Decompose radial speed into x and y components
        beta = np.arctan2(radar_np[:, 1], radar_np[:, 0])
        v_r_x = np.cos(beta) * v_r
        v_r_y = np.sin(beta) * v_r

        # Normalize the computed velocities (clip to [-12.5, 12.5] and scale to [-1, 1])
        v_rel_norm = self.normalize_velocity(v_rel)
        v_r_norm = self.normalize_velocity(v_r)
        v_r_x_norm = self.normalize_velocity(v_r_x)
        v_r_y_norm = self.normalize_velocity(v_r_y)

        result_np = np.column_stack((radar_np[:, 0], radar_np[:, 1], radar_np[:, 2], v_rel_norm, v_r_norm, v_r_x_norm, v_r_y_norm))
        return result_np

    def process_his_radar_velocity(self, his_radar_np, his_radar_transform, his_lidar_velocity_xyz, his_lidar_transform, radar_transform, d_idx):
        l2r_transform = x1_to_x2(his_lidar_transform, his_radar_transform)
        l2r_rotation_matrix = l2r_transform[:3, :3]
        radar_velocity_xyz = np.dot(l2r_rotation_matrix, his_lidar_velocity_xyz)

        his_radar_np = his_radar_np.copy()

        r = np.linalg.norm(his_radar_np[:, :3], axis=1)
        r_safe = np.where(r == 0, 1e-6, r)
        unit_vec = his_radar_np[:, :3] / r_safe[:, None]

        v_rel_vec = his_radar_np[:, 3:6]
        v_rel  = np.matmul(v_rel_vec, unit_vec.T).diagonal()

        # Compute radial speed from ego motion and sum with relative velocity
        v_ego_radial = unit_vec @ radar_velocity_xyz
        v_r = v_rel + v_ego_radial

        # Decompose radial speed into x and y components
        beta = np.arctan2(his_radar_np[:, 1], his_radar_np[:, 0])
        v_r_x = np.cos(beta) * v_r
        v_r_y = np.sin(beta) * v_r

        # Normalize the computed velocities (clip to [-12.5, 12.5] and scale to [-1, 1])
        v_rel_norm = self.normalize_velocity(v_rel)
        v_r_norm = self.normalize_velocity(v_r)
        v_r_x_norm = self.normalize_velocity(v_r_x)
        v_r_y_norm = self.normalize_velocity(v_r_y)

        # doppler compensation for dynamic points
        d_t = d_idx * (1/self.fps)
        dynamic_mask = np.abs(v_r_norm) > self.v_threshold
        if np.any(dynamic_mask):
            delta_pos = (v_r[dynamic_mask] * d_t)[:, None] * unit_vec[dynamic_mask]
            his_radar_np[dynamic_mask, :3] += delta_pos

        # Transform historical radar points to current frame
        his2cur = x1_to_x2(his_radar_transform, radar_transform)
        his_radar_np[:, :3] = (his2cur[:3, :3] @ his_radar_np[:, :3].T).T + his2cur[:3, 3]

        result_np = np.column_stack((his_radar_np[:, 0], his_radar_np[:, 1], his_radar_np[:, 2], v_rel_norm, v_r_norm, v_r_x_norm, v_r_y_norm))
        return result_np

    @staticmethod
    def normalize_velocity(v: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize 1D velocity array.

        Parameters
        ----------
        v : np.ndarray
            Input velocity array of shape (N,).
        method : str, optional
            Normalization method, one of:
            - "zscore"  : (v - mean) / std
            - "robust"  : (v - median) / IQR (interquartile range)

        Returns
        -------
        v_norm : np.ndarray
            Normalized velocity array of shape (N,).
        """

        if not isinstance(v, np.ndarray):
            v = np.asarray(v)

        if method == "zscore":
            mu, sigma = v.mean(), v.std()
            v_norm = (v - mu) / (sigma + 1e-6)

        elif method == "robust":
            median = np.median(v)
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            v_norm = (v - median) / (iqr + 1e-6)

        else:
            raise ValueError(f"Unknown method: {method}")

        return v_norm

    # ------------------------------------------------------------------------------------------------------------------