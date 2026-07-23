import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze radar RCS/Doppler values inside and outside GT boxes.'
    )
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--hist_bins', type=int, default=80)
    parser.add_argument('--save_point_csv', action='store_true')
    parser.add_argument('--max_point_rows', type=int, default=2000000)
    return parser.parse_args()


def override_paths(hypes, args):
    if args.root_dir:
        hypes['root_dir'] = args.root_dir
    if args.validate_dir:
        hypes['validate_dir'] = args.validate_dir
    if args.test_dir:
        hypes['test_dir'] = args.test_dir
    if args.split == 'test':
        hypes['validate_dir'] = hypes.get('test_dir', hypes.get('validate_dir'))


def boxes_hwl_to_lwh(boxes, order):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 7)
    boxes = boxes[:, :7].copy()
    if order == 'hwl':
        boxes[:, 3:6] = boxes[:, [5, 4, 3]]
    return boxes


def points_in_single_box(points_xyz, box_lwh):
    center = box_lwh[:3]
    length, width, height = box_lwh[3:6]
    yaw = box_lwh[6]
    shifted = points_xyz - center[None, :]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_x = shifted[:, 0] * cos_yaw + shifted[:, 1] * sin_yaw
    local_y = -shifted[:, 0] * sin_yaw + shifted[:, 1] * cos_yaw
    local_z = shifted[:, 2]
    return (
        (np.abs(local_x) <= length / 2.0) &
        (np.abs(local_y) <= width / 2.0) &
        (np.abs(local_z) <= height / 2.0)
    )


def assign_points_to_boxes(points_xyz, boxes, class_ids):
    assigned_class = np.zeros(points_xyz.shape[0], dtype=np.int64)
    assigned_box = np.full(points_xyz.shape[0], -1, dtype=np.int64)
    for box_index, box in enumerate(boxes):
        inside = points_in_single_box(points_xyz, box) & (assigned_box < 0)
        assigned_box[inside] = box_index
        assigned_class[inside] = int(class_ids[box_index])
    return assigned_class, assigned_box


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'p05': np.nan,
            'p25': np.nan,
            'median': np.nan,
            'p75': np.nan,
            'p95': np.nan,
            'max': np.nan,
        }
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'p05': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'median': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values)),
    }


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'group', 'class_id', 'class_name', 'variable',
        'count', 'mean', 'std', 'min', 'p05', 'p25', 'median', 'p75', 'p95', 'max',
    ]
    with path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_inside_outside(output_dir, name, inside, outside, bins):
    inside = np.asarray(inside, dtype=np.float64)
    outside = np.asarray(outside, dtype=np.float64)
    if inside.size == 0 and outside.size == 0:
        return
    plt.figure(figsize=(8, 5))
    if outside.size:
        plt.hist(outside, bins=bins, density=True, alpha=0.55, label='outside GT')
    if inside.size:
        plt.hist(inside, bins=bins, density=True, alpha=0.55, label='inside GT')
    plt.xlabel(name)
    plt.ylabel('density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_inside_vs_outside.png', dpi=160)
    plt.close()


def plot_per_class(output_dir, name, values_by_class, class_names, bins):
    plt.figure(figsize=(9, 5))
    plotted = False
    for class_id, values in sorted(values_by_class.items()):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            continue
        label = class_names[class_id - 1] if 1 <= class_id <= len(class_names) else f'class_{class_id}'
        plt.hist(values, bins=bins, density=True, histtype='step', linewidth=1.5, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(name)
    plt.ylabel('density')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_inside_per_class.png', dpi=160)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)

    dataset = build_dataset(hypes, visualize=False, train=args.split == 'train')
    class_names = hypes.get('model', {}).get('args', {}).get('class_names', [])
    order = hypes.get('postprocess', {}).get('order', 'hwl')
    max_samples = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))

    inside_rcs = []
    outside_rcs = []
    inside_doppler = []
    outside_doppler = []
    inside_abs_doppler = []
    outside_abs_doppler = []
    rcs_by_class = defaultdict(list)
    doppler_by_class = defaultdict(list)
    abs_doppler_by_class = defaultdict(list)
    point_rows = []
    skipped = 0

    for sample_index in range(max_samples):
        item = dataset[sample_index]
        if item is None or 'ego' not in item:
            skipped += 1
            continue
        ego = item['ego']
        radar_points = np.asarray(ego.get('radar_points', np.empty((0, 5))), dtype=np.float64)
        if radar_points.shape[0] == 0 or radar_points.shape[1] < 5:
            skipped += 1
            continue

        object_mask = np.asarray(ego['object_bbx_mask']).astype(bool)
        object_boxes = np.asarray(ego['object_bbx_center'])[object_mask]
        if object_boxes.size == 0:
            skipped += 1
            continue
        class_ids = object_boxes[:, 7].astype(np.int64) if object_boxes.shape[1] > 7 else np.ones(object_boxes.shape[0], dtype=np.int64)
        boxes_lwh = boxes_hwl_to_lwh(object_boxes[:, :7], order)

        assigned_class, assigned_box = assign_points_to_boxes(radar_points[:, :3], boxes_lwh, class_ids)
        inside = assigned_box >= 0

        doppler = radar_points[:, 3]
        rcs = radar_points[:, 4]
        abs_doppler = np.abs(doppler)

        inside_rcs.extend(rcs[inside].tolist())
        outside_rcs.extend(rcs[~inside].tolist())
        inside_doppler.extend(doppler[inside].tolist())
        outside_doppler.extend(doppler[~inside].tolist())
        inside_abs_doppler.extend(abs_doppler[inside].tolist())
        outside_abs_doppler.extend(abs_doppler[~inside].tolist())

        for class_id in np.unique(assigned_class[inside]):
            class_mask = assigned_class == class_id
            rcs_by_class[int(class_id)].extend(rcs[class_mask].tolist())
            doppler_by_class[int(class_id)].extend(doppler[class_mask].tolist())
            abs_doppler_by_class[int(class_id)].extend(abs_doppler[class_mask].tolist())

        if args.save_point_csv and len(point_rows) < args.max_point_rows:
            remaining = args.max_point_rows - len(point_rows)
            count = min(remaining, radar_points.shape[0])
            for idx in range(count):
                class_id = int(assigned_class[idx])
                class_name = 'background'
                if class_id > 0 and class_id <= len(class_names):
                    class_name = class_names[class_id - 1]
                point_rows.append({
                    'sample_index': sample_index,
                    'point_index': idx,
                    'inside_gt': int(assigned_box[idx] >= 0),
                    'box_index': int(assigned_box[idx]),
                    'class_id': class_id,
                    'class_name': class_name,
                    'x': float(radar_points[idx, 0]),
                    'y': float(radar_points[idx, 1]),
                    'z': float(radar_points[idx, 2]),
                    'doppler': float(doppler[idx]),
                    'abs_doppler': float(abs_doppler[idx]),
                    'rcs': float(rcs[idx]),
                })

        if (sample_index + 1) % 100 == 0:
            print(f'processed {sample_index + 1}/{max_samples}')

    summary_rows = []
    groups = [
        ('inside_gt', 0, 'all_objects', 'rcs', inside_rcs),
        ('outside_gt', 0, 'background', 'rcs', outside_rcs),
        ('inside_gt', 0, 'all_objects', 'doppler', inside_doppler),
        ('outside_gt', 0, 'background', 'doppler', outside_doppler),
        ('inside_gt', 0, 'all_objects', 'abs_doppler', inside_abs_doppler),
        ('outside_gt', 0, 'background', 'abs_doppler', outside_abs_doppler),
    ]
    for group, class_id, class_name, variable, values in groups:
        row = {'group': group, 'class_id': class_id, 'class_name': class_name, 'variable': variable}
        row.update(describe(values))
        summary_rows.append(row)

    for class_id, values in sorted(rcs_by_class.items()):
        class_name = class_names[class_id - 1] if 1 <= class_id <= len(class_names) else f'class_{class_id}'
        for variable, class_values in [
            ('rcs', values),
            ('doppler', doppler_by_class[class_id]),
            ('abs_doppler', abs_doppler_by_class[class_id]),
        ]:
            row = {'group': 'inside_gt_by_class', 'class_id': class_id, 'class_name': class_name, 'variable': variable}
            row.update(describe(class_values))
            summary_rows.append(row)

    write_summary_csv(output_dir / 'rcs_doppler_gt_summary.csv', summary_rows)
    plot_inside_outside(output_dir, 'rcs', inside_rcs, outside_rcs, args.hist_bins)
    plot_inside_outside(output_dir, 'doppler', inside_doppler, outside_doppler, args.hist_bins)
    plot_inside_outside(output_dir, 'abs_doppler', inside_abs_doppler, outside_abs_doppler, args.hist_bins)
    plot_per_class(output_dir, 'rcs', rcs_by_class, class_names, args.hist_bins)
    plot_per_class(output_dir, 'doppler', doppler_by_class, class_names, args.hist_bins)
    plot_per_class(output_dir, 'abs_doppler', abs_doppler_by_class, class_names, args.hist_bins)

    if args.save_point_csv:
        point_csv = output_dir / 'rcs_doppler_points.csv'
        with point_csv.open('w', newline='') as csv_file:
            fieldnames = [
                'sample_index', 'point_index', 'inside_gt', 'box_index',
                'class_id', 'class_name', 'x', 'y', 'z',
                'doppler', 'abs_doppler', 'rcs',
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(point_rows)

    print(f'wrote {output_dir / "rcs_doppler_gt_summary.csv"}')
    print(f'skipped samples: {skipped}')
    print(f'inside points: {len(inside_rcs)}')
    print(f'outside points: {len(outside_rcs)}')


if __name__ == '__main__':
    main()
