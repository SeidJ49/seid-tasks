import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.datasets.truckscenes_class_utils import TRUCKSCENES_NAME_TO_DETECTION
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils
from opencood.utils.transformation_utils import x_to_world


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare CenterHead predictions against attached GT and raw projected GT.'
    )
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--max_batches', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--score_thresholds', default='0.03,0.10,0.20,0.30,0.50')
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--post_score_thresh', type=float, default=None)
    parser.add_argument('--post_max_obj', type=int, default=None)
    parser.add_argument('--post_nms_thresh', type=float, default=None)
    return parser.parse_args()


def override_paths(hypes, args):
    if args.root_dir:
        hypes['root_dir'] = args.root_dir
    if args.validate_dir:
        hypes['validate_dir'] = args.validate_dir
    if args.test_dir:
        hypes['test_dir'] = args.test_dir
    if args.split == 'val':
        hypes['validate_dir'] = hypes.get('validate_dir', hypes.get('root_dir'))
    elif args.split == 'test':
        hypes['validate_dir'] = hypes.get('test_dir', hypes.get('validate_dir'))


def override_postprocess(hypes, args):
    radar_head = hypes.get('model', {}).get('args', {}).get('radar_head', {})
    post_cfg = radar_head.get('post_processing', {})
    if args.post_score_thresh is not None:
        post_cfg['score_thresh'] = args.post_score_thresh
    if args.post_max_obj is not None:
        post_cfg['max_obj_per_sample'] = args.post_max_obj
    if args.post_nms_thresh is not None:
        post_cfg.setdefault('nms_config', {})['nms_thresh'] = args.post_nms_thresh


def load_state_dict_compatible(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    state_dict = {
        (key[7:] if isinstance(key, str) and key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    print(f'[model] loaded {len(compatible)} tensors from {checkpoint_path}')


def raw_sample_at(dataset, idx):
    scene_index = 0
    for i, end_idx in enumerate(dataset.len_record):
        if idx < end_idx:
            scene_index = i
            break
    sample_index = idx if scene_index == 0 else idx - dataset.len_record[scene_index - 1]
    scene_key = list(dataset.dataset_info_pkl.keys())[scene_index]
    return dataset.dataset_info_pkl[scene_key][sample_index]


def canonical_name(name):
    return TRUCKSCENES_NAME_TO_DETECTION.get(str(name), str(name))


def project_raw_boxes_to_ref(raw_sample, reference_pose, order, class_names):
    gt_boxes = raw_sample['labels']['gt_boxes_global']
    object_ids = raw_sample['labels']['gt_object_ids']
    gt_names = raw_sample['labels'].get('gt_names', [])
    if hasattr(object_ids, 'tolist'):
        object_ids = object_ids.tolist()
    if hasattr(gt_names, 'tolist'):
        gt_names = gt_names.tolist()

    centers = []
    labels = []
    names = []
    t_world_ref = x_to_world(reference_pose)

    for box, gt_name in zip(gt_boxes, gt_names):
        canonical = canonical_name(gt_name)
        if canonical not in class_names:
            continue
        x, y, z, dx, dy, dz, w, a, b, c = box
        quat = Quaternion([w, a, b, c])
        t_world_object = quat.transformation_matrix
        t_world_object[:3, 3] = [x, y, z]
        object_to_ref = np.linalg.solve(t_world_ref, t_world_object)

        x_corners = dx / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        y_corners = dy / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        z_corners = dz / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.r_[corners, [np.ones(corners.shape[1])]]
        corners_ref = np.dot(object_to_ref, corners).T[:, :3][None]
        center = box_utils.corner_to_center(corners_ref, order=order)[0]
        centers.append(center)
        labels.append(class_names.index(canonical) + 1)
        names.append(canonical)

    if not centers:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.asarray(centers, dtype=np.float32), np.asarray(labels, dtype=np.int64), names


def corners_from_centers(centers, order):
    if centers.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return box_utils.boxes_to_corners_3d(centers[:, :7], order)


def best_iou(pred_corners, gt_corners, pred_label=None, gt_labels=None):
    if pred_corners.shape[0] == 0 or gt_corners.shape[0] == 0:
        return np.zeros((pred_corners.shape[0],), dtype=np.float32)
    pred_polys = common_utils.convert_format(pred_corners)
    gt_polys = common_utils.convert_format(gt_corners)
    output = []
    for pred_idx, poly in enumerate(pred_polys):
        candidate_polys = gt_polys
        if pred_label is not None and gt_labels is not None:
            label_mask = gt_labels == pred_label[pred_idx]
            candidate_polys = gt_polys[label_mask]
        if len(candidate_polys) == 0:
            output.append(0.0)
            continue
        output.append(float(np.max(common_utils.compute_iou(poly, candidate_polys))))
    return np.asarray(output, dtype=np.float32)


def summarize_matches(scores, labels, attached_iou, raw_iou, thresholds, iou_threshold, class_names):
    row = {'pred_count': int(scores.size)}
    for threshold in thresholds:
        score_mask = scores >= threshold
        row[f'pred_ge_{threshold:.2f}'] = int(score_mask.sum())
        row[f'attached_match_ge_{threshold:.2f}'] = int((score_mask & (attached_iou >= iou_threshold)).sum())
        row[f'raw_match_ge_{threshold:.2f}'] = int((score_mask & (raw_iou >= iou_threshold)).sum())
        row[f'raw_only_match_ge_{threshold:.2f}'] = int(
            (score_mask & (raw_iou >= iou_threshold) & (attached_iou < iou_threshold)).sum()
        )
    for label in sorted(set(labels.tolist())):
        name = class_names[int(label) - 1] if 0 < int(label) <= len(class_names) else f'class_{int(label)}'
        label_mask = labels == label
        row[f'pred_{name}'] = int(label_mask.sum())
        row[f'attached_match_{name}'] = int((label_mask & (attached_iou >= iou_threshold)).sum())
        row[f'raw_match_{name}'] = int((label_mask & (raw_iou >= iou_threshold)).sum())
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.score_thresholds.split(',') if x.strip()]
    output_dir = Path(args.output_dir)

    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)
    override_postprocess(hypes, args)
    dataset = build_dataset(hypes, visualize=False, train=args.split == 'train')
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(args.device)
    model = train_utils.create_model(hypes).to(device)
    load_state_dict_compatible(model, args.checkpoint)
    model.eval()

    class_names = hypes['model']['args'].get('class_names', [])
    order = hypes['postprocess']['order']
    sample_rows = []
    prediction_rows = []
    total = Counter()
    valid_count = 0
    skipped_empty = 0

    for batch_idx, batch_data in enumerate(loader):
        if valid_count >= args.max_batches:
            break
        if batch_data is None:
            skipped_empty += 1
            print(f'[sample {batch_idx}] skipped empty/no-target batch')
            continue

        raw = raw_sample_at(dataset, batch_idx)
        base_data = dataset.retrieve_base_data(batch_idx)
        ego_base = next(cav for cav in base_data.values() if cav['ego'])
        ref_pose = dataset.get_ref_pose(ego_base['params'])
        raw_centers, raw_labels, _ = project_raw_boxes_to_ref(raw, ref_pose, order, class_names)
        raw_corners = corners_from_centers(raw_centers, order)

        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            output = model(batch_data['ego'])

        final_dict = output.get('final_box_dict', [{}])[0]
        pred_boxes = final_dict.get('pred_boxes', torch.zeros((0, 7), device=device)).detach().float().cpu().numpy()
        pred_scores = final_dict.get('pred_scores', torch.zeros((0,), device=device)).detach().float().cpu().numpy()
        pred_labels = final_dict.get('pred_labels', torch.zeros((0,), dtype=torch.long, device=device)).detach().long().cpu().numpy()

        attached_boxes = batch_data['ego']['object_bbx_center'][0].detach().float().cpu().numpy()
        attached_mask = batch_data['ego']['object_bbx_mask'][0].detach().cpu().numpy() == 1
        attached_boxes = attached_boxes[attached_mask]
        attached_labels = attached_boxes[:, 7].astype(np.int64) if attached_boxes.shape[1] > 7 else np.ones((attached_boxes.shape[0],), dtype=np.int64)
        attached_corners = corners_from_centers(attached_boxes[:, :7], order)
        pred_corners = corners_from_centers(pred_boxes, order)

        attached_iou = best_iou(pred_corners, attached_corners, pred_labels, attached_labels)
        raw_iou = best_iou(pred_corners, raw_corners, pred_labels, raw_labels)

        row = {
            'sample': batch_idx,
            'valid_index': valid_count,
            'attached_gt_count': int(attached_boxes.shape[0]),
            'raw_gt_count': int(raw_centers.shape[0]),
        }
        row.update(summarize_matches(
            pred_scores, pred_labels, attached_iou, raw_iou,
            thresholds, args.iou_threshold, class_names,
        ))
        sample_rows.append(row)
        for pred_index, (score, label, a_iou, r_iou) in enumerate(zip(pred_scores, pred_labels, attached_iou, raw_iou)):
            label_int = int(label)
            class_name = class_names[label_int - 1] if 0 < label_int <= len(class_names) else f'class_{label_int}'
            prediction_rows.append({
                'sample': batch_idx,
                'valid_index': valid_count,
                'pred_index': pred_index,
                'score': float(score),
                'label': label_int,
                'class_name': class_name,
                'attached_iou': float(a_iou),
                'raw_iou': float(r_iou),
                'attached_match': int(a_iou >= args.iou_threshold),
                'raw_match': int(r_iou >= args.iou_threshold),
                'raw_only_match': int(r_iou >= args.iou_threshold and a_iou < args.iou_threshold),
            })
        for key, value in row.items():
            if key not in {'sample', 'valid_index'}:
                total[key] += float(value)

        print(
            f"[sample {batch_idx}] preds={row['pred_count']} attached_gt={row['attached_gt_count']} "
            f"raw_gt={row['raw_gt_count']} raw_only@0.30={row.get('raw_only_match_ge_0.30', 0)}"
        )
        valid_count += 1

    summary_rows = [{'metric': key, 'sum': value, 'per_valid_sample': value / max(valid_count, 1)} for key, value in sorted(total.items())]
    write_csv(output_dir / 'prediction_gt_iou_samples.csv', sample_rows)
    write_csv(output_dir / 'prediction_gt_iou_summary.csv', summary_rows)
    write_csv(output_dir / 'prediction_gt_iou_predictions.csv', prediction_rows)
    print(f'wrote {output_dir / "prediction_gt_iou_samples.csv"}')
    print(f'wrote {output_dir / "prediction_gt_iou_summary.csv"}')
    print(f'wrote {output_dir / "prediction_gt_iou_predictions.csv"}')
    if skipped_empty:
        print(f'skipped {skipped_empty} empty/no-target batches')


if __name__ == '__main__':
    main()
