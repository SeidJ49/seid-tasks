import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect WP3/RadarDistill GT class IDs and CenterHead target assignment.'
    )
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--max_samples', type=int, default=40)
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


def canonical_gt_name(name):
    mapping = {
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.bus.rigid': 'bus',
        'vehicle.bus.bendy': 'bus',
        'vehicle.trailer': 'trailer',
        'vehicle.ego_trailer': 'trailer',
        'vehicle.construction': 'other_vehicle',
        'vehicle.other': 'other_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.personal_mobility': 'pedestrian',
        'human.pedestrian.stroller': 'pedestrian',
    }
    return mapping.get(str(name), str(name))


def masked_gt_boxes(object_bbx_center, object_bbx_mask):
    if object_bbx_center.shape[-1] == 8:
        gt_boxes = object_bbx_center.float().clone()
        gt_boxes[:, :, 7] = gt_boxes[:, :, 7] * object_bbx_mask.float()
        return gt_boxes
    gt_boxes = object_bbx_center.new_zeros(object_bbx_center.shape[0], object_bbx_center.shape[1], 8)
    gt_boxes[:, :, :7] = object_bbx_center.float()
    gt_boxes[:, :, 7] = object_bbx_mask.float()
    return gt_boxes


def find_center_head(model):
    if hasattr(model, 'head'):
        return model.head
    if hasattr(model, 'radar_head'):
        return model.radar_head
    if hasattr(model, 'teacher_head'):
        return model.teacher_head
    raise RuntimeError('Could not find CenterHead-like module as model.head/radar_head/teacher_head.')


def raw_sample_at(dataset, idx):
    scene_index = 0
    for i, end_idx in enumerate(dataset.len_record):
        if idx < end_idx:
            scene_index = i
            break
    sample_index = idx if scene_index == 0 else idx - dataset.len_record[scene_index - 1]
    scene_key = list(dataset.dataset_info_pkl.keys())[scene_index]
    return dataset.dataset_info_pkl[scene_key][sample_index]


def count_target_masks(target_dicts, head):
    counts = {}
    total = 0
    for head_idx, mask in enumerate(target_dicts['masks']):
        count = int(mask.sum().item())
        counts[f'head{head_idx}_target_count'] = count
        counts[f'head{head_idx}_classes'] = '|'.join(head.class_names_each_head[head_idx])
        total += count
    counts['total_target_count'] = total
    return counts


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
    output_dir = Path(args.output_dir)
    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)

    dataset = build_dataset(hypes, visualize=False, train=args.split == 'train')
    model = train_utils.create_model(hypes)
    head = find_center_head(model)

    class_names = hypes['model']['args'].get('class_names', [])
    class_id_to_name = {idx + 1: name for idx, name in enumerate(class_names)}
    print('class id mapping:', class_id_to_name)
    print('heads:', {idx: names for idx, names in enumerate(head.class_names_each_head)})

    rows = []
    summary = Counter()
    valid_count = 0
    skipped = 0

    for idx in range(len(dataset)):
        if valid_count >= args.max_samples:
            break
        processed = dataset[idx]
        if processed is None:
            skipped += 1
            continue

        ego = processed['ego']
        raw = raw_sample_at(dataset, idx)
        raw_names = [canonical_gt_name(name) for name in raw.get('labels', {}).get('gt_names', [])]
        raw_counts = Counter(raw_names)

        object_bbx_center = torch.from_numpy(np.asarray(ego['object_bbx_center']))[None]
        object_bbx_mask = torch.from_numpy(np.asarray(ego['object_bbx_mask']))[None]
        gt_boxes = masked_gt_boxes(object_bbx_center, object_bbx_mask)
        valid = gt_boxes[0, :, 7] > 0
        class_ids = gt_boxes[0, valid, 7].long().cpu().numpy().tolist()
        attached_names = [class_id_to_name.get(int(class_id), f'class_{int(class_id)}') for class_id in class_ids]
        attached_counts = Counter(attached_names)

        target_dicts = head.assign_targets(gt_boxes, feature_map_size=(180, 180))
        row = {
            'sample': idx,
            'valid_index': valid_count,
            'raw_gt_names': '|'.join(raw_names),
            'attached_class_ids': '|'.join(str(x) for x in class_ids),
            'attached_class_names': '|'.join(attached_names),
            'raw_gt_count': sum(raw_counts.values()),
            'attached_gt_count': len(class_ids),
        }
        for name, count in raw_counts.items():
            row[f'raw_count_{name}'] = count
        for name, count in attached_counts.items():
            row[f'attached_count_{name}'] = count
        row.update(count_target_masks(target_dicts, head))
        rows.append(row)

        summary.update({f'raw_{name}': count for name, count in raw_counts.items()})
        summary.update({f'attached_{name}': count for name, count in attached_counts.items()})
        for key, value in row.items():
            if key.endswith('_target_count'):
                summary[key] += int(value)

        target_text = ', '.join(
            'h%d:%d' % (head_idx, row.get('head%d_target_count' % head_idx, 0))
            for head_idx in range(len(head.class_names_each_head))
        )
        print(
            f"[sample {idx}] raw={dict(raw_counts)} attached={dict(attached_counts)} "
            f"targets={{{target_text}}}"
        )
        valid_count += 1

    write_csv(output_dir / 'centerhead_target_assignment.csv', rows)
    summary_rows = [{'metric': key, 'count': value} for key, value in sorted(summary.items())]
    write_csv(output_dir / 'centerhead_target_assignment_summary.csv', summary_rows)
    print(f'wrote {output_dir / "centerhead_target_assignment.csv"}')
    print(f'wrote {output_dir / "centerhead_target_assignment_summary.csv"}')
    if skipped:
        print(f'skipped {skipped} empty/no-target samples')


if __name__ == '__main__':
    main()
