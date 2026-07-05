import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from opencood.utils import box_utils
from opencood.utils.transformation_utils import x_to_world


DEFAULT_TRUCKSCENES_DEVKIT_SRC = Path('/home/nj644/dev/Studis/truckscenes-devkit/src')
OPENCOOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIDAR_CHANNELS = (
    'LIDAR_LEFT',
    'LIDAR_RIGHT',
    'LIDAR_REAR',
    'LIDAR_TOP_FRONT',
    'LIDAR_TOP_LEFT',
    'LIDAR_TOP_RIGHT',
    'LIDAR_TOP',
)

INTERNAL_TO_TRUCKSCENES_NAME = {
    'construction_vehicle': 'other_vehicle',
}

TRUCKSCENES_ATTR_DEFAULTS = {
    'car': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.parked',
    'trailer': 'vehicle.parked',
    'other_vehicle': 'vehicle.parked',
    'pedestrian': 'pedestrian.standing',
    'motorcycle': 'cycle.without_rider',
    'bicycle': 'cycle.without_rider',
    'traffic_cone': '',
    'barrier': '',
    'animal': '',
    'traffic_sign': 'traffic_sign.pole_mounted',
}


def _log(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger(msg)


def _ensure_truckscenes_import(devkit_src):
    if str(OPENCOOD_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENCOOD_ROOT))
    if str(devkit_src) not in sys.path:
        sys.path.append(str(devkit_src))


def _resolve_dataroot(dataset, explicit_root=None):
    if explicit_root:
        return Path(explicit_root)

    sample = next(iter(dataset.dataset_info_pkl.values()))[0]
    sensors = sample['agents']['1']['sensors']
    first_sensor = next(iter(sensors.values()))
    sensor_path = Path(first_sensor['sensor_path'])
    return sensor_path.parents[2]


def _resolve_attribute(name, velocity_xy):
    speed = float(np.linalg.norm(np.asarray(velocity_xy[:2], dtype=np.float32)))
    if speed > 0.2:
        if name in {'car', 'truck', 'bus', 'trailer', 'other_vehicle'}:
            return 'vehicle.moving'
        if name in {'bicycle', 'motorcycle'}:
            return 'cycle.with_rider'
        if name == 'pedestrian':
            return 'pedestrian.moving'
    return TRUCKSCENES_ATTR_DEFAULTS.get(name, '')


def _map_pred_name(name):
    return INTERNAL_TO_TRUCKSCENES_NAME.get(name, name)


def _resolve_ref_lidar_channel(sample_record):
    sample_data = sample_record['data']
    lidar_channels = [channel for channel in DEFAULT_LIDAR_CHANNELS if channel in sample_data]
    if not lidar_channels:
        lidar_channels = sorted(key for key in sample_data if key.startswith('LIDAR'))
    if not lidar_channels:
        raise KeyError(f'No lidar channel found for sample {sample_record.get("token", "<unknown>")}')
    return lidar_channels[0]


def _pose_to_quaternion(pose):
    world_tfm = x_to_world(pose)
    return Quaternion(matrix=world_tfm[:3, :3])


def _transform_local_box_to_global(box, ref_pose, box_order='lwh'):
    local_center = np.asarray([box[0], box[1], box[2], 1.0], dtype=np.float64)
    world_tfm = x_to_world(ref_pose)
    global_center = (world_tfm @ local_center)[:3]

    pose_q = Quaternion(matrix=world_tfm[:3, :3])
    local_q = Quaternion(axis=[0, 0, 1], angle=float(box[6]))
    global_q = pose_q * local_q

    if box_order == 'hwl':
        height, width, length = map(float, box[3:6])
    else:
        length, width, height = map(float, box[3:6])
    size_wlh = [width, length, height]
    return global_center.tolist(), size_wlh, global_q


def build_prediction_record(batch_data, infer_result, class_names, box_order='lwh'):
    sample_token = batch_data['ego'].get('sample_token')
    ref_pose = batch_data['ego'].get('ref_pose')
    pred_box_tensor = infer_result.get('pred_box_tensor')
    pred_score = infer_result.get('pred_score')
    pred_label_tensor = infer_result.get('pred_label_tensor')

    if sample_token is None or ref_pose is None:
        raise KeyError('TruckScenes official evaluation requires sample_token and ref_pose in batch_data["ego"].')

    annos = []
    if pred_box_tensor is None:
        return sample_token, annos

    pred_boxes = pred_box_tensor.detach().cpu().numpy()
    pred_scores = pred_score.detach().cpu().numpy()
    pred_boxes = box_utils.corner_to_center(pred_boxes, order=box_order)

    if pred_label_tensor is None:
        if len(class_names) != 1:
            raise ValueError('pred_label_tensor is required for multi-class official TruckScenes evaluation.')
        pred_labels = np.ones((pred_boxes.shape[0],), dtype=np.int64)
    else:
        pred_labels = pred_label_tensor.detach().cpu().numpy().astype(np.int64)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        class_index = int(label) - 1
        if class_index < 0 or class_index >= len(class_names):
            continue
        name = _map_pred_name(class_names[class_index])
        translation, size, rotation = _transform_local_box_to_global(box, ref_pose, box_order=box_order)
        velocity = [0.0, 0.0]
        annos.append({
            'sample_token': sample_token,
            'translation': translation,
            'size': size,
            'rotation': rotation.elements.tolist(),
            'velocity': velocity,
            'detection_name': name,
            'detection_score': float(score),
            'attribute_name': _resolve_attribute(name, velocity),
        })

    return sample_token, annos


def _load_gt_from_infos(trucksc, sample_tokens, detection_box_cls, category_to_detection_name):
    attribute_map = {a['token']: a['name'] for a in trucksc.attribute}
    gt_boxes = {}

    for sample_token in sample_tokens:
        sample = trucksc.get('sample', sample_token)
        sample_boxes = []
        for ann_token in sample['anns']:
            ann = trucksc.get('sample_annotation', ann_token)
            detection_name = category_to_detection_name(ann['category_name'])
            if detection_name is None:
                continue

            attr_tokens = ann['attribute_tokens']
            if len(attr_tokens) == 0:
                attribute_name = ''
            elif len(attr_tokens) == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise RuntimeError('GT annotations must not have more than one attribute')

            num_pts = int(ann['num_lidar_pts'] + ann['num_radar_pts'])
            velocity = trucksc.box_velocity(ann['token'])[:2]
            if np.any(np.isnan(velocity)):
                velocity = np.nan_to_num(velocity, nan=0.0)

            sample_boxes.append(
                detection_box_cls(
                    sample_token=sample_token,
                    translation=tuple(ann['translation']),
                    size=tuple(ann['size']),
                    rotation=tuple(ann['rotation']),
                    velocity=tuple(velocity),
                    num_pts=num_pts,
                    detection_name=detection_name,
                    detection_score=-1.0,
                    attribute_name=attribute_name,
                )
            )
        gt_boxes[sample_token] = sample_boxes

    return gt_boxes


def _add_center_dist_dynamic(trucksc, eval_boxes):
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = trucksc.get('sample', sample_token)
        lidar_channel = _resolve_ref_lidar_channel(sample_rec)
        sd_record = trucksc.get('sample_data', sample_rec['data'][lidar_channel])
        pose_record = trucksc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            box.ego_translation = (
                box.translation[0] - pose_record['translation'][0],
                box.translation[1] - pose_record['translation'][1],
                box.translation[2] - pose_record['translation'][2],
            )
    return eval_boxes


def _format_results(metrics_summary):
    metrics = metrics_summary['all']
    lines = ['----------------TruckScenes official results-----------------']
    for name, ap in metrics['mean_dist_aps'].items():
        tp = metrics['label_tp_errors'][name]
        lines.append(
            f'{name}: AP={ap:.4f}, ATE={tp["trans_err"]:.4f}, ASE={tp["scale_err"]:.4f}, '
            f'AOE={tp["orient_err"]:.4f}, AVE={tp["vel_err"]:.4f}, AAE={tp["attr_err"]:.4f}'
        )
    lines.append('--------------average performance-------------')
    for key, val in metrics['tp_errors'].items():
        lines.append(f'{key}:       {val:.4f}')
    lines.append(f'mAP:     {metrics["mean_ap"]:.4f}')
    lines.append(f'NDS:     {metrics["nd_score"]:.4f}')
    return '\n'.join(lines), {
        **metrics['tp_errors'],
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    }


def run_official_truckscenes_eval(
    prediction_records,
    dataset,
    output_dir,
    version='v1.1-mini',
    dataroot=None,
    devkit_src=None,
    logger=None,
):
    devkit_src = Path(devkit_src) if devkit_src else DEFAULT_TRUCKSCENES_DEVKIT_SRC
    _ensure_truckscenes_import(devkit_src)

    from truckscenes import TruckScenes
    from truckscenes.eval.common.data_classes import EvalBoxes
    from truckscenes.eval.common.loaders import filter_eval_boxes, get_scene_tag_masks, load_prediction
    from truckscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from truckscenes.eval.detection.config import config_factory
    from truckscenes.eval.detection.constants import TP_METRICS
    from truckscenes.eval.detection.data_classes import (
        DetectionBox, DetectionMetricDataList, DetectionMetrics, DetectionMetricsList
    )
    from truckscenes.eval.detection.utils import category_to_detection_name

    dataroot = _resolve_dataroot(dataset, dataroot)
    trucksc = TruckScenes(version=version, dataroot=str(dataroot), verbose=True)

    results = {}
    for sample_token, annos in prediction_records:
        results[sample_token] = annos

    result_json = {
        'results': results,
        'meta': {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'results_truckscenes.json'
    with open(result_path, 'w') as f:
        json.dump(result_json, f)
    _log(logger, f'The predictions of TruckScenes have been saved to {result_path}')

    cfg = config_factory('detection_cvpr_2024')
    pred_boxes, meta = load_prediction(str(result_path), cfg.max_boxes_per_sample, DetectionBox, verbose=True)

    gt_eval_boxes = EvalBoxes()
    sample_tokens = [sample_token for sample_token, _ in prediction_records]
    for sample_token, boxes in _load_gt_from_infos(trucksc, sample_tokens, DetectionBox, category_to_detection_name).items():
        gt_eval_boxes.add_boxes(sample_token, boxes)

    assert set(pred_boxes.sample_tokens) == set(gt_eval_boxes.sample_tokens), \
        'Samples in split does not match samples in predictions.'

    pred_boxes = _add_center_dist_dynamic(trucksc, pred_boxes)
    gt_eval_boxes = _add_center_dist_dynamic(trucksc, gt_eval_boxes)

    has_any_pred_boxes = any(len(pred_boxes[sample_token]) > 0 for sample_token in pred_boxes.sample_tokens)
    if has_any_pred_boxes:
        pred_boxes = filter_eval_boxes(trucksc, pred_boxes, cfg.class_range, verbose=True)
    else:
        _log(logger, 'TruckScenes evaluation: all prediction lists are empty, skipping prediction filtering.')
    gt_eval_boxes = filter_eval_boxes(trucksc, gt_eval_boxes, cfg.class_range, verbose=True)

    pred_masks = get_scene_tag_masks(trucksc, pred_boxes)
    gt_masks = get_scene_tag_masks(trucksc, gt_eval_boxes)

    start_time = time.time()
    metric_data_list = DetectionMetricDataList()
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(
                gt_eval_boxes, pred_boxes, class_name,
                cfg.dist_fcn_callable, dist_th,
                gt_masks.get('all'), pred_masks.get('all')
            )
            metric_data_list.set('all', class_name, dist_th, md)
            metrics.add_label_ap(class_name, dist_th, calc_ap(md, cfg.min_recall, cfg.min_precision))

        metric_data = metric_data_list[('all', class_name, cfg.dist_th_tp)]
        for metric_name in TP_METRICS:
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            elif class_name in ['animal'] and metric_name in ['attr_err']:
                tp = np.nan
            elif class_name in ['traffic_sign'] and metric_name in ['vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    metrics_list = DetectionMetricsList()
    metrics.add_runtime(time.time() - start_time)
    metrics_list.add_detection_metrics('all', metrics)
    metrics_list.add_runtime(metrics.eval_time)

    metrics_summary = metrics_list.serialize()
    metrics_summary['meta'] = meta.copy()

    summary_path = output_dir / 'metrics_summary.json'
    details_path = output_dir / 'metrics_details.json'
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(details_path, 'w') as f:
        json.dump(metric_data_list.serialize(), f, indent=2)

    return _format_results(metrics_summary)
