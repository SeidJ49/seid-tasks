# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu>
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


# single_radar_np = pcd_utils.shuffle_points(single_radar_np)  # TODO: change after radar is fused
# single_radar_np = pcd_utils.mask_ego_points(single_radar_np)  # TODO: change after radar is fused
"""
Dataset class for intermediate fusion with lidar-camera
"""
import bisect
import math
import os
from collections import OrderedDict

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, cm

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils, pcd_utils, transformation_utils, camera_utils, sensor_transformation_utils
from opencood.utils.transformation_utils import x1_to_x2

matplotlib.use('Agg')

import logging

# Creating a logger specific for this module
logger = logging.getLogger(__name__)


def merge_features_to_dict(processed_feature_list, merge=None):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():

            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature)

    # stack them
    # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    # cat them
    # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    if merge == 'stack':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge == 'cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    return merged_feature_dict


def extract_timestamps(yaml_files):
    """
    Given the list of the yaml files, extract the mocked timestamps.

    Parameters
    ----------
    yaml_files : list
        The full path of all yaml files of ego vehicle

    Returns
    -------
    timestamps : list
        The list containing timestamps only.
    """
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]

        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps


def load_camera_files(cav_path, timestamp):
    """
    Retrieve the paths to all camera files.

    Parameters
    ----------
    cav_path : str
        The full file path of current cav.

    timestamp : str
        Current timestamp

    Returns
    -------
    camera_files : list
        The list containing all camera png file paths.
    """
    camera0_file = os.path.join(cav_path, timestamp + '_camera0.png')
    camera1_file = os.path.join(cav_path, timestamp + '_camera1.png')
    camera2_file = os.path.join(cav_path, timestamp + '_camera2.png')
    camera3_file = os.path.join(cav_path, timestamp + '_camera3.png')

    return [camera0_file, camera1_file, camera2_file, camera3_file]


def load_radar_files(cav_path, timestamp):
    """
    Retrieve the paths to all radar files.

    Parameters
    ----------
    cav_path : str
        The full file path of current cav.

    timestamp : str
        Current timestamp

    Returns
    -------
    radar_files : list
        The list containing all radar npy file paths.
    """
    radar0_file = os.path.join(cav_path, timestamp + '_radar0.npy')
    radar1_file = os.path.join(cav_path, timestamp + '_radar1.npy')
    radar2_file = os.path.join(cav_path, timestamp + '_radar2.npy')
    radar3_file = os.path.join(cav_path, timestamp + '_radar3.npy')
    radar4_file = os.path.join(cav_path, timestamp + '_radar4.npy')
    radar5_file = os.path.join(cav_path, timestamp + '_radar5.npy')

    return [radar0_file, radar1_file, radar2_file, radar3_file, radar4_file, radar5_file]


def return_timestamp_key(scenario_database, timestamp_index):
    """
    Given the timestamp index, return the correct timestamp key, e.g.
    2 --> '000078'.

    Parameters
    ----------
    scenario_database : OrderedDict
        The dictionary contains all contents in the current scenario.

    timestamp_index : int
        The index for timestamp.

    Returns
    -------
    timestamp_key : str
        The timestamp key saved in the cav dictionary.
    """
    # get all timestamp keys
    timestamp_keys = list(scenario_database.items())[0][1]
    # retrieve the correct index
    timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

    return timestamp_key


class RadarCameraIntermediateFusionDataset(torch.utils.data.Dataset):
    """
    This class is for intermediate fusion where each vehicle transmits the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True):

        # Initializing the core parameters
        self.params = params
        self.visualize = visualize
        self.train = train

        # Build preprocessors
        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], train=train, dataset="opv2v" )

        # Default maximum number of CAVs
        self.max_cav = 7 if 'train_params' not in params or 'max_cav' not in params['train_params'] else params['train_params']['max_cav']

        # Project first flag: defines whether to project CAV's lidar to ego's coordinate frame first
        self.proj_first = params['fusion']['args'].get('proj_first', False)
        logger.info(f"proj_first: {self.proj_first}")
        self.cur_ego_pose_flag = params['fusion']['args'].get('cur_ego_pose_flag', True)
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.depth_discre = camera_utils.depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode'])
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # Wild setting: handles noisy settings and transmission overhead parameters
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = 'sim' if 'async_mode' not in params['wild_setting'] else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # Localization error configuration
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # Transmission data size and speed
            self.data_size = params['wild_setting'].get('data_size', 0)
            self.transmission_speed = params['wild_setting'].get('transmission_speed', 27)
            self.backbone_delay = params['wild_setting'].get('backbone_delay', 0)
        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        # Dataset-Modus from config
        self.duplicate_ego = params.get('dataset', {}).get('duplicate_ego', False)
        self.velocity_encoding_flag = params.get('dataset', {}).get('velocity_encoding', False)

        # Set the root directory based on train or validate mode
        root_dir = params['root_dir'] if self.train else params['validate_dir']
        scenario_folders = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])

        self.scenario_database = OrderedDict()  # Structure: {scenario_id : {cav_id : {timestamp: {yaml: path, radars: path, camera: list, ...}, ...}, ...}}
        self.len_record = []
        scenario_counter = 0

        for scenario_folder in scenario_folders:
            # List CAVs in the scenario folder
            cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0  # Ensure at least one CAV exists

            # Move roadside unit data to the end of the list (it has negative IDs)
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # Determine ego indices: single (0) vs. duplicate all
            ego_indices = (range(min(len(cav_list), self.max_cav)) if self.duplicate_ego else [0])

            for ego_idx in ego_indices:
                self.scenario_database[scenario_counter] = OrderedDict()
                for (j, cav_id) in enumerate(cav_list):
                    if j >= self.max_cav:
                        logger.info('Too many CAVs')
                        break

                    self.scenario_database[scenario_counter][cav_id] = OrderedDict()

                    # Get paths for YAML files (In Directory)
                    cav_path = os.path.join(scenario_folder, cav_id)
                    yaml_files = sorted([os.path.join(cav_path, x) for x in os.listdir(cav_path) if x.endswith('.yaml') and 'additional' not in x])
                    timestamps = extract_timestamps(yaml_files)

                    if train:
                        cav_path_additional = cav_path.replace("/Dataset_OPV2V/train/", "/Dataset_OPV2V/train_additional/")
                    else:
                        cav_path_additional = cav_path.replace("/Dataset_OPV2V/validate/", "/Dataset_OPV2V/validate_additional/")

                    # Store data for each timestamp
                    for timestamp in timestamps:
                        self.scenario_database[scenario_counter][cav_id][timestamp] = OrderedDict()
                        self.scenario_database[scenario_counter][cav_id][timestamp]['yaml'] = os.path.join(cav_path, timestamp + '.yaml')
                        self.scenario_database[scenario_counter][cav_id][timestamp]['yaml_additional'] = os.path.join(cav_path_additional, timestamp + '.yaml')
                        self.scenario_database[scenario_counter][cav_id][timestamp]['radars'] = load_radar_files(cav_path_additional, timestamp)
                        self.scenario_database[scenario_counter][cav_id][timestamp]['camera'] = load_camera_files(cav_path, timestamp)

                    # Mark ego vehicle based on current ego_idx
                    if j == ego_idx:
                        self.scenario_database[scenario_counter][cav_id]['ego'] = True
                        if not self.len_record:
                            self.len_record.append(len(timestamps))
                        else:
                            self.len_record.append(self.len_record[-1] + len(timestamps))
                    else:
                        self.scenario_database[scenario_counter][cav_id]['ego'] = False

                scenario_counter += 1
        logger.info(f"dataset length: {self.len_record[-1]}")

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        # Find the ego vehicle in base_data_dict by checking for the 'ego' flag.
        ego_vehicle_key = next((key for key, value in base_data_dict.items() if value.get('ego', False)), None)
        assert ego_vehicle_key is not None, "Ego vehicle not found in the base_data_dict"
        assert cav_id == ego_vehicle_key, "The provided cav_id does not match the ego vehicle"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix, img_pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)

        agents_image_inputs = []
        processed_radar_features = []
        object_stack = []
        object_id_stack = []
        # prior knowledge for time delay correction and indicating data type (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        if self.visualize:
            projected_radar_stack = []

        lidar_pose = []
        ego_flag = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                                 (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose, cav_id)

            # SELECTED_CAV_PROCESSED
            object_id_stack += selected_cav_processed['object_ids']
            object_stack.append(selected_cav_processed['object_bbx_center'])
            processed_radar_features.append(selected_cav_processed['processed_radar_features'])
            agents_image_inputs.append(selected_cav_processed['image_inputs'])

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            # this is only useful when proj_first = True, and communication delay is considered. Right now only V2X-ViT utilizes the
            # spatial_correction. There is a time delay when the cavs project their lidar to ego and when the ego receives the feature, and
            # this variable is used to correct such pose difference (ego_t-1 to ego_t)
            spatial_correction_matrix.append(selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

            lidar_pose.append(selected_cav_base['params']['lidar_pose'])
            if ego_id == cav_id:
                ego_flag.append(True)
            else:
                ego_flag.append(False)

            if self.visualize:
                projected_radar_stack.append(selected_cav_processed['projected_radar'])

        # merge preprocessed features from different cavs into the same dict
        merged_radar_feature_dict = merge_features_to_dict(processed_radar_features)
        merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')

        # exclude all repetitive objects
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {
            'cav_num': len(processed_radar_features),
            'lidar_pose': lidar_pose,
            'img_pairwise_t_matrix': img_pairwise_t_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
            'spatial_correction_matrix': spatial_correction_matrix,
            'image_inputs': merged_image_inputs_dict,
            'processed_radar': merged_radar_feature_dict,
            'label_dict': label_dict,
            'anchor_box': anchor_box,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': mask,

            'velocity': velocity,
            'time_delay': time_delay,
            'infra': infra,
            'ego_flag': ego_flag
        }

        if self.visualize:
            processed_data_dict['ego'].update({'origin_radar': np.vstack(projected_radar_stack)})
        return processed_data_dict

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = return_timestamp_key(scenario_database, timestamp_index_delay)

            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)

            # RADAR
            data[cav_id]['radars_np'] = []
            for radar_file in cav_content[timestamp_key_delay]['radars']:
                data[cav_id]['radars_np'].append(np.load(radar_file))

            # CAMERA
            img_src = []
            for idx in range(self.data_aug_conf['Ncams']):
                image_path = cav_content[timestamp_key_delay]['camera'][idx]
                img_src.append(Image.open(image_path))
            data[cav_id]['camera_data'] = img_src
        return data

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break
        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = math.sqrt((cur_lidar_pose[0] - ego_lidar_pose[0]) ** 2 + (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})
        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur, timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """

        # -- YAML --------------------------------------------------------------------------------------------------------------------------

        # EGO
        # Ego Parameters with Merged Additional YAML
        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        cur_ego_params_additional = load_yaml(ego_content[timestamp_cur]['yaml_additional'])
        cur_ego_params.update(cur_ego_params_additional)  # Merge additional into normal

        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])
        delay_ego_params_additional = load_yaml(ego_content[timestamp_delay]['yaml_additional'])
        delay_ego_params.update(delay_ego_params_additional)  # Merge additional into normal

        # CAV Parameters with Merged Additional YAML
        cur_cav_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        cur_cav_params_additional = load_yaml(cav_content[timestamp_cur]['yaml_additional'])
        cur_cav_params.update(cur_cav_params_additional)  # Merge additional into normal

        delay_cav_params = load_yaml(cav_content[timestamp_delay]['yaml'])
        delay_cav_params_additional = load_yaml(cav_content[timestamp_delay]['yaml_additional'])
        delay_cav_params.update(delay_cav_params_additional)  # Merge additional into normal

        # -- LIDAR -------------------------------------------------------------------------------------------------------------------------

        # we need to calculate the transformation matrix from cav to ego at the delayed timestamp

        # EGO
        delay_ego_lidar_pose = delay_ego_params['lidar_pose']
        cur_ego_lidar_pose = cur_ego_params['lidar_pose']

        # CAV
        delay_cav_lidar_pose = delay_cav_params['lidar_pose']
        cur_cav_lidar_pose = cur_cav_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = transformation_utils.x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose)

        # This is only used for late fusion, as it did the transformation in the postprocess, so we want the gt object transformation use the correct one
        gt_transformation_matrix = transformation_utils.x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_cav_params['vehicles'] = cur_cav_params['vehicles']

        delay_cav_params['transformation_matrix'] = transformation_matrix
        delay_cav_params['gt_transformation_matrix'] = gt_transformation_matrix
        delay_cav_params['spatial_correction_matrix'] = spatial_correction_matrix

        # -- CAMERA ------------------------------------------------------------------------------------------------------------------------
        camera_to_lidar_matrix = []
        camera_intrinsic = []
        for idx in range(self.data_aug_conf['Ncams']):
            camera_id = self.data_aug_conf['cams'][idx]

            extrinsic = delay_cav_params[camera_id]['extrinsic']
            camera_to_lidar_matrix.append(extrinsic)
            camera_intrinsic.append(delay_cav_params[camera_id]['intrinsic'])

        delay_cav_params['camera2lidar_matrix'] = camera_to_lidar_matrix
        delay_cav_params['camera_intrinsic'] = camera_intrinsic

        # -- RADAR -------------------------------------------------------------------------------------------------------------------------

        # EGO
        delay_ego_radar_poses = [delay_ego_params[f'radar{i}'] for i in range(6)] # TODO: normally it should be radar cords but i made a mistake in logreplay
        cur_ego_radar_poses = [cur_ego_params[f'radar{i}'] for i in range(6)]

        # CAV
        delay_cav_radar_poses = [delay_cav_params[f'radar{i}'] for i in range(6)]
        cur_cav_radar_poses = [cur_cav_params[f'radar{i}'] for i in range(6)]

        transformation_matrix_radar = []
        spatial_correction_matrix_radar = []
        gt_transformation_matrix_radar = []

        for i in range(len(cur_ego_radar_poses)):

            if not cav_content['ego'] and self.loc_err_flag:
                delay_cav_radar_poses[i] = self.add_loc_noise(delay_cav_radar_poses[i], self.xyz_noise_std, self.ryp_noise_std)
                cur_cav_radar_poses[i] = self.add_loc_noise(cur_cav_radar_poses[i], self.xyz_noise_std, self.ryp_noise_std)

            if cur_ego_pose_flag:
                transformation_matrix_radar.append(transformation_utils.x1_to_x2(delay_cav_radar_poses[i], cur_ego_lidar_pose))
                spatial_correction_matrix_radar.append(np.eye(4))
            else:
                transformation_matrix_radar.append(transformation_utils.x1_to_x2(delay_cav_radar_poses[i], delay_ego_lidar_pose))
                spatial_correction_matrix_radar.append(transformation_utils.x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose))

            gt_transformation_matrix_radar.append(transformation_utils.x1_to_x2(cur_cav_radar_poses[i], cur_ego_lidar_pose))

        delay_cav_params['transformation_matrix_radar'] = transformation_matrix_radar
        delay_cav_params['gt_transformation_matrix_radar'] = gt_transformation_matrix_radar
        delay_cav_params['spatial_correction_matrix_radar'] = spatial_correction_matrix_radar

        # ----------------------------------------------------------------------------------------------------------------------------------

        return delay_cav_params

    def get_item_single_car(self, selected_cav_base, ego_pose, cav_id=None):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # ----------------------------------------------------------------------------------------------------------------------------------
        # region --  CAMERA SETUP ----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------

        # Extract camera calibration parameters
        camera_to_lidar_matrix = np.array(selected_cav_base['params']['camera2lidar_matrix']).reshape(4, 4, 4).astype(np.float32)
        camera_intrinsic = np.array(selected_cav_base['params']["camera_intrinsic"]).reshape(4, 3, 3).astype(np.float32)

        # -- DATA AUGMENTATION PLACEHOLDER -------------------------------------------------------------------------------------------------
        # Identity (or zero) matrices for augmentation:
        post_translations = torch.zeros(4, 3)
        post_rotations = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)


        # -- IMAGE RESIZING AND NORMALIZATION ----------------------------------------------------------------------------------------------
        final_height, final_width = self.data_aug_conf['final_dim']
        normalized_images = []

        for img in selected_cav_base["camera_data"]:
            # Resize image to defined final dimensions
            resized_img = img.resize((final_width, final_height))
            # Normalize the resized image
            normalized_images.append(camera_utils.normalize_img(resized_img))

            imgH, imgW = img.height, img.width  # needed later

        # -- PACKAGE OUTPUT INTO A DICTIONARY ----------------------------------------------------------------------------------------------
        selected_cav_processed = {
            "image_inputs": {
                "imgs": torch.stack(normalized_images, dim=0),  # [N(cam), 3, H, W]
                "intrins": torch.from_numpy(camera_intrinsic),  # 4*3*3
                "extrins": torch.from_numpy(camera_to_lidar_matrix),  # 4*3*3
                "rots": torch.from_numpy(camera_to_lidar_matrix[:, :3, :3]),  # R_wc, we consider world-coord is the lidar-coord
                "trans": torch.from_numpy(camera_to_lidar_matrix[:, :3, 3]),  # T_wc
                "post_rots": post_rotations,  # 4*3
                "post_trans": post_translations,  # 4*3*3
            }
        }
        # endregion
        # ----------------------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------------------
        # region -- VELOCITY ENCODING ------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------

        if self.velocity_encoding_flag:
            # Lidar_pose
            lidar_transform = selected_cav_base['params']['lidar_pose']  # [x, y, z, roll, yaw, pitch]
            # lidar_velocity_xyz = np.array(selected_cav_base['params']['ego_speed_x_y_z'])  # [vx, vy, vz] in m/
            lidar_velocity_xyz = np.array([np.array(selected_cav_base['params']['ego_speed']) / 3.6, 0, 0])

            # Velcocity Encoding for one Radar
            selected_cav_base['radars_np'][0] = self.process_all_radar_velocity(selected_cav_base['radars_np'][0],
                                                                                selected_cav_base['params']['radar0'],
                                                                                lidar_velocity_xyz, lidar_transform)
            selected_cav_base['radars_np'][1] = self.process_all_radar_velocity(selected_cav_base['radars_np'][1],
                                                                                selected_cav_base['params']['radar1'],
                                                                                lidar_velocity_xyz, lidar_transform)
            selected_cav_base['radars_np'][2] = self.process_all_radar_velocity(selected_cav_base['radars_np'][2],
                                                                                selected_cav_base['params']['radar2'],
                                                                                lidar_velocity_xyz, lidar_transform)
            selected_cav_base['radars_np'][3] = self.process_all_radar_velocity(selected_cav_base['radars_np'][3],
                                                                                selected_cav_base['params']['radar3'],
                                                                                lidar_velocity_xyz, lidar_transform)
            selected_cav_base['radars_np'][4] = self.process_all_radar_velocity(selected_cav_base['radars_np'][4],
                                                                                selected_cav_base['params']['radar4'],
                                                                                lidar_velocity_xyz, lidar_transform)
            selected_cav_base['radars_np'][5] = self.process_all_radar_velocity(selected_cav_base['radars_np'][5],
                                                                                selected_cav_base['params']['radar5'],
                                                                                lidar_velocity_xyz, lidar_transform)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------------------
        # region -- RADAR PREPROCESSING -----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------

        # 1. Get radar transformation matrices.
        transformation_matrix_radar = selected_cav_base['params']['transformation_matrix_radar']

        # 2. Get object detection info.
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        # 3. Prepare lists for raw and projected radar points.
        radar_np_list = []
        projected_radar_np_list = []

        # 4. Process each radar sensor's data.
        for i, single_radar_np in enumerate(selected_cav_base['radars_np']):

            # Create a copy for projection visualization.
            projected_single_radar_np = single_radar_np.copy()
            projected_single_radar_np[:, :3] = box_utils.project_points_by_matrix_torch(single_radar_np[:, :3], transformation_matrix_radar[i])

            projected_radar_np_list.append(projected_single_radar_np)

            # Mask points based on projection flag.
            if self.proj_first:
                masked_single_radar_np = pcd_utils.mask_points_by_range(projected_single_radar_np, self.params['preprocess']['cav_lidar_range'])
            else:
                masked_single_radar_np = pcd_utils.mask_points_by_range(single_radar_np, self.params['preprocess']['cav_lidar_range'])

            radar_np_list.append(masked_single_radar_np)

        # endregion
        # ----------------------------------------------------------------------------------------------------------------------------------

        processed_radar_np_combined = self.pre_processor.preprocess(np.vstack(radar_np_list))
        projected_radar_np_combined = np.vstack(projected_radar_np_list)

        # endregion
        # ----------------------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------------------
        # region -- MAP GENERATION ---------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------

        # Depth map generation
        depth_maps = []
        draws_boolean = False
        for idx, camera_data in enumerate(selected_cav_base["camera_data"]):
            if draws_boolean:
                file_path = camera_data.filename
                desired_part = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path).replace('.png', ''))
                modified_path = desired_part.replace('/', '_')

            depth_maps.append(self.generate_depth_map(idx, np.array(camera_data), projected_radar_np_combined, camera_intrinsic[idx],
                                      camera_to_lidar_matrix[idx], imgH, imgW, draws=draws_boolean, cav_id=cav_id))

        depth_maps = torch.concat(depth_maps, dim=0)
        selected_cav_processed["image_inputs"].update({"depth_map": depth_maps})

        # Velocity map generation
        #velocity_map = []
        #draws_boolean_velo = True
        #for idx, camera_data in enumerate(selected_cav_base["camera_data"]):
        #    if idx == 3:
        #        if draws_boolean_velo:
        #            file_path = camera_data.filename
        #            desired_part = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path).replace('.png', ''))
        #            modified_path = desired_part.replace('/', '_')
        #        velocity_map.append(self.generate_velocity_map(idx, np.array(camera_data), projected_radar_np_list[3], camera_intrinsic[idx],
        #                                  camera_to_lidar_matrix[idx], imgH, imgW, draws=draws_boolean_velo, cav_id=cav_id, namename = modified_path))

        # endregion
        # ----------------------------------------------------------------------------------------------------------------------------------



        # Normalize ego speed and update processed dict
        velocity = (selected_cav_base['params']['ego_speed'] / 3.6 ) / 12.15 # ego_speed is in Km/h. now here in m/s
        velocity = np.clip(velocity, 0, 1)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_radar': projected_radar_np_combined,
             'processed_radar_features': processed_radar_np_combined,
             'velocity': velocity})

        return selected_cav_processed

    @staticmethod
    def velocity_vector(ego_speed_kmh: float, true_ego_pos: list) -> np.ndarray:
        """
        Compute the 2D velocity vector [vx, vy, vz] from ego speed and pose.

        Parameters:
        - ego_speed_kmh: float
            Speed of the ego-vehicle in kilometers per hour.
        - true_ego_pos: list or array-like
            Pose of the ego-vehicle; must have at least 5 elements,
            with yaw (radians) at index 4.

        Returns:
        - np.ndarray of shape (3,)
            Velocity vector [vx, vy, vz] in meters per second.
        """
        # extract yaw, defaulting to 0 if not provided
        yaw = float(true_ego_pos[4]) if len(true_ego_pos) > 4 else 0.0

        # convert speed to m/s
        speed_mps = ego_speed_kmh / 3.6

        # compute components
        vx = speed_mps * math.cos(yaw)
        vy = speed_mps * math.sin(yaw)
        vz = 0.0  # assume planar motion

        return np.array([vx, vy, vz])

    @staticmethod
    def process_all_radar_velocity(radar_np, radar_transform, lidar_velocity_xyz, lidar_transform):

        # Get the velocity of the radar sensor
        l2r_transform = x1_to_x2(lidar_transform, radar_transform)

        # Extract the roatation matrix (R) from 4 x 4 transform
        l2r_rotation_matrix = l2r_transform[:3, :3]

        radar_velocity_xyz = np.dot(l2r_rotation_matrix, lidar_velocity_xyz)

        radar_np = radar_np.copy()  # [x, y, z, relative radial velocity (m/s]
        v_rel = radar_np[:, 3]  # [v_rel]

        # Calculate unit vectors for radar points
        r = np.sqrt(radar_np[:, 0] ** 2 + radar_np[:, 1] ** 2)
        r_safe = np.where(r == 0, 1, r)
        ux = radar_np[:, 0] / r_safe
        uy = radar_np[:, 1] / r_safe
        uz = radar_np[:, 2] / r_safe

        # Compute radial speed from ego motion and sum with relative velocity
        v_ego_radial = radar_velocity_xyz[0] * ux + radar_velocity_xyz[1] * uy + radar_velocity_xyz[2] * uz
        v_r = v_rel + v_ego_radial

        # Decompose radial speed into x and y components
        beta = np.arctan2(radar_np[:, 1], radar_np[:, 0])
        v_r_x = np.cos(beta) * v_r
        v_r_y = np.sin(beta) * v_r

        # Normalize the computed velocities (clip to [-12.5, 12.5] and scale to [-1, 1])
        v_rel_norm = np.clip(v_rel, -12.5, 12.5) / 12.5
        v_r_norm = np.clip(v_r, -12.5, 12.5) / 12.5
        v_r_x_norm = np.clip(v_r_x, -12.5, 12.5) / 12.5
        v_r_y_norm = np.clip(v_r_y, -12.5, 12.5) / 12.5

        result_np = np.column_stack((radar_np[:, 0], radar_np[:, 1], radar_np[:, 2], v_rel_norm, v_r_norm, v_r_x_norm, v_r_y_norm))

        return result_np

    @staticmethod
    def process_front_or_rear(radar_np, vx_sensor, vy_sensor):
        radar_np = radar_np.copy()  # [x, y, z, relative radial velocity (m/s]
        v_rel = radar_np[:, 3]  # [v_rel]

        # Calculate unit vectors for radar points
        r = np.sqrt(radar_np[:, 0] ** 2 + radar_np[:, 1] ** 2)
        r_safe = np.where(r == 0, 1, r)
        ux = radar_np[:, 0] / r_safe
        uy = radar_np[:, 1] / r_safe

        # Compute radial speed from ego motion and sum with relative velocity
        v_ego_radial = vx_sensor * ux + vy_sensor * uy
        v_r = v_rel + v_ego_radial

        # Decompose radial speed into x and y components
        beta = np.arctan2(radar_np[:, 1], radar_np[:, 0])
        v_r_x = np.cos(beta) * v_r
        v_r_y = np.sin(beta) * v_r

        # Normalize the computed velocities (clip to [-12.5, 12.5] and scale to [-1, 1])
        v_rel_norm = np.clip(v_rel, -12.5, 12.5) / 12.5
        v_r_norm = np.clip(v_r, -12.5, 12.5) / 12.5
        v_r_x_norm = np.clip(v_r_x, -12.5, 12.5) / 12.5
        v_r_y_norm = np.clip(v_r_y, -12.5, 12.5) / 12.5

        result_np = np.column_stack((radar_np[:, 0], radar_np[:, 1], radar_np[:, 2], v_rel_norm, v_r_norm, v_r_x_norm, v_r_y_norm))

        return result_np


    def visualize_radar_point_cloud(self, projected_radar_np_combined):
        import open3d as o3d
        # Define the lidar range as given
        cav_lidar_range = np.array([-140.8, -38.4, -3, 140.8, 38.4, 1])
        x_min, y_min, z_min, x_max, y_max, z_max = cav_lidar_range

        # Extract the first three columns as the point coordinates (x, y, z)
        points = projected_radar_np_combined[:, :3]

        # Apply filtering to only keep points within the lidar range
        mask = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        filtered_points = points[mask]

        # Create an Open3D point cloud and set its points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # Set a uniform color for all points (red)
        single_color = np.array([1, 0, 0])
        colors = np.tile(single_color, (filtered_points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

    def visualize_voxel(self, processed_radar_np_combined):
        import numpy as np
        import open3d as o3d

        # Assume these variables come from your preprocessor output
        voxel_features = processed_radar_np_combined['voxel_features']
        voxel_coords = processed_radar_np_combined['voxel_coords']
        voxel_num_points = processed_radar_np_combined['voxel_num_points']

        # Example parameters; adjust these to your configuration.
        lidar_range = np.array([-140.8, -38.4, -3, 140.8, 38.4, 1])
        voxel_size = np.array([0.4, 0.4, 0.4])
        coords = voxel_coords

        # Compute voxel centers
        voxel_centers = coords * voxel_size + lidar_range[:3] + voxel_size / 2.0

        # Create a point cloud to visualize voxel centers.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)

        # Set all points to the same color (e.g., red)
        single_color = np.array([1, 0, 0])
        colors = np.tile(single_color, (len(voxel_centers), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Display the voxel centers.
        o3d.visualization.draw_geometries([pcd])

    def generate_velocity_map(self, index, image, xyz, int_matrix, ext_matrix, imgH, imgW, draws=True, scenario_id=0, cav_id=0, namename=''):
        # Create a directory for the scenario if it doesn't exist
        scenario_dir = f'scenario_{scenario_id}'
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)

        blank_image = np.zeros(image.shape, dtype=np.float32)
        rgb_image, points_2d = sensor_transformation_utils.project_lidar_to_camera(
            index, rgb_image=blank_image, point_cloud=xyz,
            camera_intrinsic=int_matrix, image_size=(imgH, imgW)
        )

        # Instead of using the projected depth, use the velocity column (fifth column, index 4)
        velocity_vals = xyz[:, 4]

        uv = points_2d[:, :2]
        uv_int = (np.ceil(uv) - ((uv - np.floor(uv)) < 0.5).astype(np.int32)).astype(np.int32)
        uv_int = uv_int[:, ::-1]

        # Filter points based solely on image boundaries
        valid_mask = ((uv_int[:, 0] >= 0) & (uv_int[:, 0] < imgH) &
                      (uv_int[:, 1] >= 0) & (uv_int[:, 1] < imgW))
        valid_uvint = uv_int[valid_mask]
        valid_velocity = velocity_vals[valid_mask]

        # Create a velocity map image
        velocity_map = -1 * np.ones((imgH, imgW), dtype=np.float32)
        for idx, valid_coord in enumerate(valid_uvint):
            u, v = valid_coord[0], valid_coord[1]
            vel_val = valid_velocity[idx]
            velocity_map[u, v] = vel_val if velocity_map[u, v] < 0 else min(velocity_map[u, v], vel_val)

        # Resize according to configuration
        assert imgH % self.data_aug_conf['final_dim'][0] == 0
        assert imgW % self.data_aug_conf['final_dim'][1] == 0
        scaleH = imgH // self.data_aug_conf['final_dim'][0]
        scaleW = imgW // self.data_aug_conf['final_dim'][1]

        max_vel = np.max(velocity_map)
        velocity_map[velocity_map < 0] = max_vel + 1
        velocity_map = torch.FloatTensor(-1 * velocity_map).unsqueeze(0)
        pool_layer = torch.nn.MaxPool2d(kernel_size=(scaleH, scaleW), stride=(scaleH, scaleW))
        velocity_map = -1 * pool_layer(velocity_map)
        velocity_map[velocity_map > max_vel] = -1

        # Create overlay image using the resized image and drawing valid circles with rescaled coordinates.
        if draws and valid_velocity.size > 0 and index == 3 and int(cav_id) == 229:
            dpi = 100
            final_h, final_w = self.data_aug_conf['final_dim']
            overlay_image = cv2.resize(image, (final_w, final_h))
            # Draw circles with three specific colors based on velocity thresholds
            for idx, valid_coord in enumerate(valid_uvint):
                u, v = valid_coord[0], valid_coord[1]
                new_u = int(u // scaleH)
                new_v = int(v // scaleW)
                vel_val = valid_velocity[idx]
                if vel_val < -4:
                    color = (255, 0, 0)  # red in BGR
                elif vel_val <= 4:
                    color = (255, 255, 0)  # yellow in BGR
                else:
                    color = (0, 255, 0)  # green in BGR
                cv2.circle(overlay_image, (new_v, new_u), 2, color, -1)

            fig = plt.figure(figsize=(final_w / dpi, final_h / dpi), dpi=dpi)
            plt.imshow(overlay_image)
            plt.axis('off')
            fig.savefig(os.path.join(scenario_dir, f'{namename}overlay_image.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        return velocity_map


    def generate_depth_map(self, index, image, xyz, int_matrix, ext_matrix, imgH, imgW, draws=False, scenario_id=0, cav_id=0, namename=''):

        blank_image = np.zeros(image.shape, dtype=np.float32)
        rgb_image, points_2d = sensor_transformation_utils.project_lidar_to_camera(index, rgb_image=blank_image, point_cloud=xyz,
                                                                                   camera_intrinsic=int_matrix, image_size=(imgH, imgW))

        depth = points_2d[:, 2]
        uv = points_2d[:, :2]
        uv_int = (np.ceil(uv) - ((uv - np.floor(uv)) < 0.5).astype(np.int32)).astype(np.int32) #FIXME: INVALID VALUE ENCOUNTERED IN SUBTRACTION
        uv_int = uv_int[:, ::-1]

        valid_mask = ((depth >= self.grid_conf['ddiscr'][0]) &
                      (uv_int[:, 0] >= 0) & (uv_int[:, 0] < imgH) &
                      (uv_int[:, 1] >= 0) & (uv_int[:, 1] < imgW))
        valid_uvint, valid_depth = uv_int[valid_mask], depth[valid_mask]

        depth_map = -1 * np.ones((imgH, imgW), dtype=np.float32)
        for idx, valid_coord in enumerate(valid_uvint):
            u, v = valid_coord[0], valid_coord[1]
            depth_level = bisect.bisect_left(self.depth_discre, valid_depth[idx])
            if depth_level == 0:
                depth_level == 1
            depth_map[u, v] = depth_level - 1 if depth_map[u, v] < 0 else min(depth_map[u, v], depth_level - 1)

        # resize as config requires
        assert imgH % self.data_aug_conf['final_dim'][0] == 0
        assert imgW % self.data_aug_conf['final_dim'][1] == 0
        scaleH= imgH // self.data_aug_conf['final_dim'][0]
        scaleW = imgW // self.data_aug_conf['final_dim'][1]

        max_depth_level = np.max(depth_map)
        depth_map[depth_map < 0] = max_depth_level + 1
        depth_map = torch.FloatTensor(-1 * depth_map).unsqueeze(0)
        pool_layer = torch.nn.MaxPool2d(kernel_size=(scaleH, scaleW), stride=(scaleH, scaleW))
        depth_map = -1 * pool_layer(depth_map)
        depth_map[depth_map > max_depth_level] = -1

        if draws:
            # Create a directory for the scenario if it doesn't exist
            scenario_dir = f'scenario_{scenario_id}'
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)

        # Create overlay image using the resized image and drawing valid circles with rescaled coordinates.
        if draws and valid_depth.size > 0 and index==0:
            dpi = 100
            final_h, final_w = self.data_aug_conf['final_dim']
            overlay_image = cv2.resize(image, (final_w, final_h))
            # Using depth_map for normalization range
            norm = plt.Normalize(vmin=depth_map.min().item(), vmax=depth_map.max().item())
            cmap = cm.get_cmap('plasma')
            # Rescale valid_uvint positions for the resized image.
            for idx, valid_coord in enumerate(valid_uvint):
                u, v = valid_coord[0], valid_coord[1]
                new_u = int(u // scaleH)
                new_v = int(v // scaleW)
                depth_level = valid_depth[idx]
                color = cmap(norm(depth_level))[:3]
                color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                cv2.circle(overlay_image, (new_v, new_u), 2, color, -1)

            fig = plt.figure(figsize=(final_w / dpi, final_h / dpi), dpi=dpi)
            plt.imshow(overlay_image)
            plt.axis('off')
            fig.savefig(os.path.join(scenario_dir, f'{namename}overlay_image.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        return depth_map


    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two

        record_len = []  # used to record different scenario
        lidar_poses = []

        img_pairwise_t_matrix_list = []  # pairwise transformation matrix for image
        pairwise_t_matrix_list = []  # pairwise transformation matrix
        processed_radar_list = []
        image_inputs_list = []
        label_dict_list = []

        object_ids = []
        object_bbx_center = []
        object_bbx_mask = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []
        ego_flag = []

        if self.visualize:
            origin_radar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            record_len.append(ego_dict['cav_num'])
            lidar_poses.append(ego_dict['lidar_pose'])
            img_pairwise_t_matrix_list.append(ego_dict['img_pairwise_t_matrix'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            processed_radar_list.append(ego_dict['processed_radar'])
            image_inputs_list.append(ego_dict['image_inputs'])  # different cav_num, ego_dict['image_inputs'] is dict.
            label_dict_list.append(ego_dict['label_dict'])
            object_ids.append(ego_dict['object_ids'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])

            ego_flag.append(ego_dict['ego_flag'])

            if self.visualize:
                origin_radar.append(ego_dict['origin_radar'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_radar_feature_dict = merge_features_to_dict(processed_radar_list)
        processed_radar_torch_dict = self.pre_processor.collate_batch(merged_radar_feature_dict)
        merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        img_pairwise_t_matrix = torch.from_numpy(np.array(img_pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict = {
            'ego': {
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'object_ids': object_ids[0],
                'label_dict': label_torch_dict,

                'processed_radar': processed_radar_torch_dict,
                'image_inputs': merged_image_inputs_dict,
                'record_len': record_len,
                'prior_encoding': prior_encoding,
                'spatial_correction_matrix': spatial_correction_matrix_list,
                'pairwise_t_matrix': pairwise_t_matrix,
                'img_pairwise_t_matrix': img_pairwise_t_matrix,
                'lidar_pose': lidar_poses,
                'ego_flag': ego_flag
            }
        }

        if self.visualize:  # TODO: CHECK KOLABOR HABEN ANDERES
            origin_radar = np.array(pcd_utils.downsample_lidar_minimum(pcd_np_list=origin_radar))
            origin_radar = torch.from_numpy(origin_radar)
            output_dict['ego'].update({'origin_radar': origin_radar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box': torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix': transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        # pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        preds = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        # return pred_box_tensor, pred_score, gt_box_tensor
        return preds + (gt_box_tensor,)

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        identity_pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        # if lidar projected to ego first, then the pairwise matrix becomes identity
        identity_pairwise_t_matrix[:, :] = np.identity(4)

        # save all transformation matrix in a list in order first.
        t_list = []
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    t_matrix = np.eye(4)
                    pairwise_t_matrix[i, j] = t_matrix
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        if self.proj_first:
            return identity_pairwise_t_matrix, pairwise_t_matrix
        else:
            return pairwise_t_matrix, pairwise_t_matrix
