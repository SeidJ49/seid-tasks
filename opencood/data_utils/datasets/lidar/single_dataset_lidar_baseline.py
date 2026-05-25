import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
from pypcd import pypcd
import torch
from torch.utils.data import Dataset
from opencood.data_utils.datasets.truckscenes_class_utils import attach_class_ids, build_class_id_lookup
from opencood.utils.pcd_utils import shuffle_points, mask_ego_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range


class SingleDatasetLidarBaseline(Dataset):

    def __init__(self,params: Dict, visualize: bool = False, train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.ref_frame = params.get('ref_frame', 'ego_pose')

        # Build preprocessors
        # --- LIDAR ----------------------------------------------------------------------------------------------------
        lidar_feature_count = params['preprocess']['args']['lidar_num_point_features']
        lidar_args = {**params['preprocess']['args'], 'num_point_features': lidar_feature_count}
        self.lidar_pre_processor = build_preprocessor({**params['preprocess'], 'args': lidar_args}, train)
        # --------------------------------------------------------------------------------------------------------------
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.root_dir = params["root_dir"] if train else params["validate_dir"]
        print("Dataset dir:", self.root_dir)

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False

        self.label_type = params['label_type']
        assert self.label_type in ['lidar']

        self.class_names = params.get('model', {}).get('args', {}).get('class_names', [])

        self.generate_object_center = self.generate_object_center_lidar
        self.generate_object_center_single = self.generate_object_center

        self.ego_mode = 'one'

        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        self.reinitialize()

        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)

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

                lidar_list = [k for k, v in cav_entry['sensors'].items() if v.get('sensor_type') == 'lidar']
                cav_entry['lidar_sensors'] = lidar_list

                cav_entry['params'] = OrderedDict()
                cav_entry['params']['vehicles'] = sample['labels']['gt_boxes_global']
                cav_entry['params']['object_ids'] = sample['labels']['gt_object_ids'].tolist()
                cav_entry['params']['gt_names'] = sample['labels']['gt_names'].tolist()
                cav_entry['params']['sample_token'] = sample.get('sample_token', sample.get('token'))
                cav_entry['params']['scene_token'] = sample.get('scene_token')
                cav_entry['params']['timestamp'] = sample.get('timestamp')
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
        base_data_dict = self.retrieve_base_data(idx)
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
            'cav_list': ['ego'],
            'sample_token': ego_base['params'].get('sample_token'),
            'scene_token': ego_base['params'].get('scene_token'),
            'timestamp': ego_base['params'].get('timestamp'),
            'ref_pose': ref_pose,
            'ref_frame': self.ref_frame,
        })

        processed_data_dict['ego'] = ego_processed
        return processed_data_dict


    def retrieve_base_data(self, idx):

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
        # --- LIDAR ----------------------------------------------------------------------------------------------------
        lidar_points = []
        for sensor_name in cav_content['lidar_sensors']:
            s = cav_content['sensors'][sensor_name]
            pts = self.pcd_to_npy_array_lidar(s['sensor_path'])
            T = self.T_sensor_to_ref(s, cav_content['params'])

            # Sensor -> Ego
            xyz = pts[:, :3]
            xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
            xyz_ref= (T @ xyz_h.T).T[:, :3]
            pts[:, :3] = xyz_ref
            lidar_points.append(pts)
        # --------------------------------------------------------------------------------------------------------------
        data[f'{ego_cav_id}']['lidar_np'] = np.vstack(lidar_points)

        return data


    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        return self.post_processor.generate_object_center_v2x(cav_contents, reference_lidar_pose)


    def get_item_single_car(self, selected_cav_base):
        selected_cav_processed = {}

        ref_pose = self.get_ref_pose(selected_cav_base['params'])
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single([selected_cav_base], ref_pose)
        class_id_lookup = build_class_id_lookup(
            selected_cav_base['params']['gt_names'],
            selected_cav_base['params']['object_ids'],
            self.class_names,
        )
        object_bbx_center, object_bbx_mask, object_ids = attach_class_ids(
            object_bbx_center,
            object_bbx_mask,
            object_ids,
            class_id_lookup,
        )

        # No target classes in the scene.
        if len(object_ids) == 0:
            print("No target classes in the scene")
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

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
            }
        )

        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center[:, :7], anchors=self.anchor_box, mask=object_bbx_mask)
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
            'sample_token': cav_content.get('sample_token'),
            'scene_token': cav_content.get('scene_token'),
            'timestamp': cav_content.get('timestamp'),
            'ref_pose': cav_content.get('ref_pose'),
            'ref_frame': cav_content.get('ref_frame'),
        })

        if self.visualize:
            projected_lidar_list = [cav_content['origin_lidar']]
            projected_lidar_stack = [torch.from_numpy(np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return output_dict


    def post_process(self, data_dict, output_dict):
        ego_output = output_dict.get('ego') if isinstance(output_dict, dict) else None
        if isinstance(ego_output, dict) and 'final_box_dict' in ego_output:
            gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx_with_labels(data_dict)
            final_dict = ego_output['final_box_dict'][0]
            pred_box_tensor = box_utils.boxes_to_corners_3d(final_dict['pred_boxes'], order=self.post_processor.params['order'])
            return pred_box_tensor, final_dict['pred_scores'], gt_box_tensor, final_dict.get('pred_labels'), gt_label_tensor

        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion(self, data_dict, output_dict_ego):
        ego_output = output_dict_ego.get('ego') if isinstance(output_dict_ego, dict) else None
        if isinstance(ego_output, dict) and 'final_box_dict' in ego_output:
            gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx_with_labels(data_dict)
            final_dict = ego_output['final_box_dict'][0]
            pred_box_tensor = box_utils.boxes_to_corners_3d(final_dict['pred_boxes'], order=self.post_processor.params['order'])
            return pred_box_tensor, final_dict['pred_scores'], gt_box_tensor, final_dict.get('pred_labels'), gt_label_tensor

        data_dict_ego = OrderedDict()
        data_dict_ego["ego"] = data_dict["ego"]
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict_ego, output_dict_ego)
        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion_uncertainty(self, data_dict, output_dict_ego):
        ego_output = output_dict_ego.get('ego') if isinstance(output_dict_ego, dict) else None
        if isinstance(ego_output, dict) and 'final_box_dict' in ego_output:
            gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx_with_labels(data_dict)
            final_dict = ego_output['final_box_dict'][0]
            pred_box_tensor = box_utils.boxes_to_corners_3d(final_dict['pred_boxes'], order=self.post_processor.params['order'])
            return pred_box_tensor, final_dict['pred_scores'], gt_box_tensor, final_dict.get('pred_labels'), gt_label_tensor, None

        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        pred_box_tensor, pred_score, uncertainty = self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
        return pred_box_tensor, pred_score, gt_box_tensor, uncertainty


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
