import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.pcd_utils import shuffle_points, mask_ego_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.pcd_utils import mask_points_by_range


class SingleVehicleDatasetRadar(Dataset):

    def __init__(self,params: Dict, visualize: bool = False, train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        self.root_dir = params["root_dir"] if train else params["validate_dir"]
        print("Dataset dir:", self.root_dir)

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False

        self.label_type = params['label_type']  # 'lidar'
        assert self.label_type in ['lidar']

        self.generate_object_center = self.generate_object_center_lidar
        self.generate_object_center_single = self.generate_object_center

        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        self.ego_mode = 'one'

        self.reinitialize()

        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)


    def reinitialize(self):
        self.scene_database = OrderedDict()
        if self.ego_mode == 'one':
            self.len_record = len(self.dataset_info_pkl)
        else:
            raise NotImplementedError(self.ego_mode)

        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.scene_database.update({i: OrderedDict()})
            cav_num = scene_info['agent_num']
            assert cav_num > 0

            cav_ids = list(range(1, cav_num + 1))

            for j, cav_id in enumerate(cav_ids):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break

                self.scene_database[i][cav_id] = OrderedDict()
                self.scene_database[i][cav_id]['ego'] = (cav_id == 1)
                self.scene_database[i][cav_id]['lidar'] = scene_info[f'lidar_path_{cav_id}']

                self.scene_database[i][cav_id]['params'] = OrderedDict()
                self.scene_database[i][cav_id]['params']['vehicles'] = scene_info[f'labels_{cav_id}']['gt_boxes_global']
                self.scene_database[i][cav_id]['params']['object_ids'] = scene_info[f'labels_{cav_id}']['gt_object_ids'].tolist()

                pose = tfm_to_pose(scene_info[f"lidar_pose_{cav_id}"]) # [x, y, z, roll, pitch, yaw]
                self.scene_database[i][cav_id]['params']['lidar_pose'] = pose # [x, y, z, roll, pitch, yaw]



    def __len__(self) -> int:
        return self.len_record


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
        processed_data_dict.update({"ego": selected_cav_processed})
        return processed_data_dict


    def get_item_test(self, base_data_dict, idx):
        processed_data_dict = OrderedDict()

        ego_id = -1
        ego_lidar_pose = []
        ego_base = None

        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_base = cav_content
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        transformation_matrix = x1_to_x2(ego_lidar_pose, ego_lidar_pose)

        ego_processed = self.get_item_single_car(ego_base)
        ego_processed.update({
            'transformation_matrix': transformation_matrix,
            'idx': idx,
            'cav_list': ['ego']
        })

        processed_data_dict['ego'] = ego_processed
        return processed_data_dict


    def retrieve_base_data(self, idx):
        data = OrderedDict()
        scene = self.scene_database[idx]

        ego_cav_id = None
        for cav_id, cav_content in scene.items():
            if cav_content['ego']:
                ego_cav_id = cav_id
                break
        assert ego_cav_id is not None

        cav_content = scene[ego_cav_id]
        data[f'{ego_cav_id}'] = OrderedDict()
        data[f'{ego_cav_id}']['ego'] = True
        data[f'{ego_cav_id}']['params'] = cav_content['params']

        nbr_dims = 4  # x,y,z,intensity
        scan = np.fromfile(cav_content['lidar'], dtype='float32')
        points = scan.reshape((-1, 5))[:, :nbr_dims]
        data[f'{ego_cav_id}']['lidar_np'] = points

        return data


    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        return self.post_processor.generate_object_center_v2x(cav_contents, reference_lidar_pose)


    def get_item_single_car(self, selected_cav_base):
        selected_cav_processed = {}

        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single([selected_cav_base], selected_cav_base["params"]["lidar_pose"])

        if self.load_lidar_file or self.visualize:
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            lidar_np = mask_ego_points(lidar_np)

            lidar_dict = self.pre_processor.preprocess(lidar_np)
            selected_cav_processed.update({'processed_lidar': lidar_dict})

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
            processed_lidar_torch_dict = self.pre_processor.collate_batch(processed_lidar_list)
            output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

        return output_dict


    def collate_batch_test(self, batch):
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
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content['processed_lidar']])
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
