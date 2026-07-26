import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.datasets.truckscenes_class_utils import (
    TRUCKSCENES_NAME_TO_DETECTION,
    attach_class_ids,
    build_class_id_lookup,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Explain raw TruckScenes GT -> CenterHead target filtering for WP3/RadarDistill configs.'
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


def canonical_name(name):
    return TRUCKSCENES_NAME_TO_DETECTION.get(str(name), str(name))


def raw_sample_at(dataset, idx):
    scene_index = 0
    for i, end_idx in enumerate(dataset.len_record):
        if idx < end_idx:
            scene_index = i
            break
    sample_index = idx if scene_index == 0 else idx - dataset.len_record[scene_index - 1]
    scene_key = list(dataset.dataset_info_pkl.keys())[scene_index]
    return dataset.dataset_info_pkl[scene_key][sample_index]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_by_reason(rows):
    summary = Counter()
    for row in rows:
        summary['raw_total'] += 1
        summary[f"reason_{row['reason']}"] += 1
        summary[f"raw_{row['canonical_name']}"] += 1
        if row['reason'] == 'kept':
            summary['attached_total'] += 1
            summary[f"attached_{row['canonical_name']}"] += 1
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)
    dataset = build_dataset(hypes, visualize=False, train=args.split == 'train')

    class_names = hypes.get('model', {}).get('args', {}).get('class_names', [])
    print('class names:', class_names)

    object_rows = []
    sample_rows = []
    valid_samples = 0

    for idx in range(len(dataset)):
        if valid_samples >= args.max_samples:
            break

        raw = raw_sample_at(dataset, idx)
        raw_labels = raw.get('labels', {})
        raw_ids = raw_labels.get('gt_object_ids', [])
        raw_names = raw_labels.get('gt_names', [])
        if hasattr(raw_ids, 'tolist'):
            raw_ids = raw_ids.tolist()
        if hasattr(raw_names, 'tolist'):
            raw_names = raw_names.tolist()

        base_data = dataset.retrieve_base_data(idx)
        ego_base = next(cav for cav in base_data.values() if cav['ego'])
        ref_pose = dataset.get_ref_pose(ego_base['params'])

        ranged_boxes, ranged_mask, ranged_ids = dataset.generate_object_center_single([ego_base], ref_pose)
        id_to_name = {int(object_id): gt_name for object_id, gt_name in zip(raw_ids, raw_names)}
        id_to_canonical = {object_id: canonical_name(name) for object_id, name in id_to_name.items()}

        class_id_lookup = build_class_id_lookup(
            [id_to_name.get(int(object_id), '') for object_id in ranged_ids],
            ranged_ids,
            class_names,
        )
        _, _, attached_ids = attach_class_ids(ranged_boxes, ranged_mask, ranged_ids, class_id_lookup)

        ranged_id_set = {int(object_id) for object_id in ranged_ids}
        attached_id_set = {int(object_id) for object_id in attached_ids}
        raw_id_set = {int(object_id) for object_id in raw_ids}

        per_sample = Counter()
        for raw_pos, object_id in enumerate(raw_ids):
            object_id = int(object_id)
            raw_name = str(raw_names[raw_pos]) if raw_pos < len(raw_names) else ''
            canonical = id_to_canonical.get(object_id, canonical_name(raw_name))

            if object_id in attached_id_set:
                reason = 'kept'
            elif object_id not in ranged_id_set:
                reason = 'range_or_corner_filter'
            elif canonical not in class_names:
                reason = 'class_not_in_model'
            else:
                reason = 'id_class_attach_mismatch'

            row = {
                'sample': idx,
                'object_id': object_id,
                'raw_index': raw_pos,
                'raw_name': raw_name,
                'canonical_name': canonical,
                'reason': reason,
            }
            object_rows.append(row)
            per_sample[f'raw_{canonical}'] += 1
            per_sample[f'reason_{reason}'] += 1
            if reason == 'kept':
                per_sample[f'attached_{canonical}'] += 1

        generated_extra_ids = ranged_id_set - raw_id_set
        sample_row = {
            'sample': idx,
            'raw_count': len(raw_ids),
            'range_kept_count': len(ranged_ids),
            'attached_count': len(attached_ids),
            'generated_extra_id_count': len(generated_extra_ids),
        }
        sample_row.update(per_sample)
        sample_rows.append(sample_row)

        print(
            '[sample %d] raw=%d range_kept=%d attached=%d reasons=%s'
            % (idx, len(raw_ids), len(ranged_ids), len(attached_ids), dict(per_sample))
        )
        valid_samples += 1

    summary = count_by_reason(object_rows)
    summary_rows = [{'metric': key, 'count': value} for key, value in sorted(summary.items())]

    write_csv(output_dir / 'gt_filter_objects.csv', object_rows)
    write_csv(output_dir / 'gt_filter_samples.csv', sample_rows)
    write_csv(output_dir / 'gt_filter_summary.csv', summary_rows)

    print(f'wrote {output_dir / "gt_filter_objects.csv"}')
    print(f'wrote {output_dir / "gt_filter_samples.csv"}')
    print(f'wrote {output_dir / "gt_filter_summary.csv"}')


if __name__ == '__main__':
    main()
