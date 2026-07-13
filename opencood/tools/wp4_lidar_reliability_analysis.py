#!/usr/bin/env python3
"""WP4 LiDAR evidence-unreliability analysis.

The main WP4 statistic is ``mean_U_L_valid``: the mean LiDAR evidence
insufficiency in cells that are supported by LiDAR occupancy neighborhoods or an
externally supplied valid-region mask.  The full-grid mean is still written as
``mean_U_L_global`` for comparison, but it should not be used as the primary
weather metric because empty BEV cells dominate it.
"""

import argparse
import copy
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


DEFAULT_VRU_CLASSES = ('pedestrian', 'motorcycle', 'bicycle')
DEFAULT_PERSON_CLASSES = ('pedestrian',)
DEFAULT_DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, None))


def parse_condition(value):
    if ':' not in value:
        raise argparse.ArgumentTypeError("condition must be NAME:CONFIG.yaml")
    name, path = value.split(':', 1)
    if not name:
        raise argparse.ArgumentTypeError("condition name is empty")
    return name, path


def parse_distance_bins(value):
    bins = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' not in item:
            raise argparse.ArgumentTypeError(
                "distance bins must look like 0-20,20-40,60-inf")
        lo, hi = item.split('-', 1)
        hi_value = None if hi.lower() in ('inf', 'infinity', 'none') else float(hi)
        bins.append((float(lo), hi_value))
    if not bins:
        raise argparse.ArgumentTypeError("at least one distance bin is required")
    return tuple(bins)


def distance_bin_label(distance_bin):
    lo, hi = distance_bin
    lo_text = str(int(lo)) if float(lo).is_integer() else str(lo).replace('.', 'p')
    if hi is None:
        hi_text = 'inf'
    else:
        hi_text = str(int(hi)) if float(hi).is_integer() else str(hi).replace('.', 'p')
    return f'distance_{lo_text}_{hi_text}'


def save_heatmap(array, path, title, cmap='magma', vmin=0.0, vmax=1.0,
                 colorbar_label='U_L: LiDAR evidence insufficiency'):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(arr)
    plot_arr = np.ma.masked_invalid(arr)
    cmap_obj = copy.copy(plt.get_cmap(cmap))
    cmap_obj.set_bad(color='0.75')

    if vmin is None and finite.any():
        vmin = float(np.nanmin(arr))
    if vmax is None and finite.any():
        vmax = float(np.nanmax(arr))
    if finite.any() and vmin == vmax:
        vmax = vmin + 1.0

    plt.figure(figsize=(7, 6))
    plt.imshow(plot_arr, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel('BEV x cell')
    plt.ylabel('BEV y cell')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_delta_heatmap(array, path, title):
    arr = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(arr)
    limit = float(np.nanmax(np.abs(arr))) if finite.any() else 1.0
    limit = max(limit, 1e-6)
    save_heatmap(
        arr,
        path,
        title,
        cmap='coolwarm',
        vmin=-limit,
        vmax=limit,
        colorbar_label='Delta U_L',
    )


def nan_stat(func, values, axis=None):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        result = func(values, axis=axis)
    return result


def build_object_class_lookup(dataset):
    lookup = {}
    name_map = {
        'vehicle.car': 'car',
        'car': 'car',
        'vehicle.truck': 'truck',
        'truck': 'truck',
        'vehicle.bus.rigid': 'bus',
        'vehicle.bus.bendy': 'bus',
        'bus': 'bus',
        'vehicle.trailer': 'trailer',
        'vehicle.ego_trailer': 'trailer',
        'trailer': 'trailer',
        'vehicle.construction': 'other_vehicle',
        'vehicle.other': 'other_vehicle',
        'other_vehicle': 'other_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'bicycle': 'bicycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.personal_mobility': 'pedestrian',
        'human.pedestrian.stroller': 'pedestrian',
        'pedestrian': 'pedestrian',
    }
    for samples in dataset.dataset_info_pkl.values():
        for sample in samples:
            labels = sample.get('labels', {})
            object_ids = labels.get('gt_object_ids', [])
            gt_names = labels.get('gt_names', [])
            if hasattr(object_ids, 'tolist'):
                object_ids = object_ids.tolist()
            if hasattr(gt_names, 'tolist'):
                gt_names = gt_names.tolist()
            for object_id, gt_name in zip(object_ids, gt_names):
                canonical = name_map.get(str(gt_name), str(gt_name))
                lookup[int(object_id)] = canonical
    return lookup


def sample_id_for_index(dataset, dataset_idx):
    scene_tokens = list(dataset.dataset_info_pkl.keys())
    scene_index = 0
    for idx, end_idx in enumerate(dataset.len_record):
        if dataset_idx < end_idx:
            scene_index = idx
            break
    sample_index = dataset_idx if scene_index == 0 else dataset_idx - dataset.len_record[scene_index - 1]
    scene_token = scene_tokens[scene_index]
    sample = dataset.dataset_info_pkl[scene_token][sample_index]
    for key in ('sample_token', 'token'):
        value = sample.get(key)
        if value:
            return str(value)
    return f'{scene_token}:{sample_index:06d}'


def bev_cell_centers(map_shape, lidar_range):
    height, width = map_shape
    x_min, y_min, _, x_max, y_max, _ = lidar_range
    xs = x_min + (np.arange(width, dtype=np.float32) + 0.5) * ((x_max - x_min) / width)
    ys = y_min + (np.arange(height, dtype=np.float32) + 0.5) * ((y_max - y_min) / height)
    return np.meshgrid(xs, ys)


def boxes_to_bev_mask(boxes, classes, target_classes, grid_x, grid_y, order):
    mask = np.zeros(grid_x.shape, dtype=np.float32)
    if boxes.size == 0:
        return mask

    target_classes = set(target_classes)
    for box, cls_name in zip(boxes, classes):
        if cls_name not in target_classes:
            continue
        cx, cy = float(box[0]), float(box[1])
        yaw = float(box[6])
        if order == 'hwl':
            box_w = float(box[4])
            box_l = float(box[5])
        else:
            box_l = float(box[3])
            box_w = float(box[4])
        if box_l <= 0 or box_w <= 0:
            continue

        dx = grid_x - cx
        dy = grid_y - cy
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        inside = (
            (np.abs(local_x) <= box_l / 2.0) &
            (np.abs(local_y) <= box_w / 2.0)
        )
        mask[inside] = 1.0
    return mask


def masked_mean(values, mask):
    valid = mask > 0
    if not np.any(valid):
        return np.nan
    return float(np.nanmean(values[valid]))


def occupancy_rate(occupancy, mask):
    valid = mask > 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(occupancy[valid] > 0))


def count_objects_in_distance(boxes, distance_bin):
    if boxes.size == 0:
        return 0
    lo, hi = distance_bin
    distance = np.sqrt(np.square(boxes[:, 0]) + np.square(boxes[:, 1]))
    in_bin = distance >= lo
    if hi is not None:
        in_bin = in_bin & (distance < hi)
    return int(in_bin.sum())


def write_rows(path, rows):
    if not rows:
        return path
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_condition(name, config_path, output_dir, max_batches, max_samples,
                        device, vru_classes, person_classes, distance_bins):
    hypes = yaml_utils.load_yaml(config_path)
    dataset = build_dataset(hypes, visualize=False, train=False)
    object_class_lookup = build_object_class_lookup(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes).to(device)
    model.eval()

    maps = []
    valid_maps = []
    valid_masks = []
    point_counts = []
    occupancy_maps = []
    sparsities = []
    object_values = []
    vru_values = []
    person_values = []
    background_values = []
    object_counts = []
    vru_counts = []
    person_counts = []
    per_sample_rows = []
    region_accumulators = {}
    sample_count = 0
    grid_x = None
    grid_y = None
    radial_distance = None

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            if max_batches is not None and batch_idx >= max_batches:
                break
            if max_samples is not None and sample_count >= max_samples:
                break

            batch_data = train_utils.to_device(batch_data, device)
            output = model(batch_data['ego'])
            u_l = output['U_L'].detach().float().cpu().numpy()[:, 0]
            pc = output['lidar_point_count_map'].detach().float().cpu().numpy()[:, 0]
            sp = output['lidar_occupancy_sparsity'].detach().float().cpu().numpy()[:, 0]
            occupancy = output['lidar_occupancy_map'].detach().float().cpu().numpy()[:, 0]
            valid_region = output['lidar_valid_region_mask'].detach().float().cpu().numpy()[:, 0]

            for sample_offset in range(u_l.shape[0]):
                if max_samples is not None and sample_count >= max_samples:
                    break

                u = u_l[sample_offset]
                points = pc[sample_offset]
                sparse = sp[sample_offset]
                occ = occupancy[sample_offset]
                valid_mask = valid_region[sample_offset] > 0
                valid_float = valid_mask.astype(np.float32)

                map_shape = u.shape
                if grid_x is None:
                    lidar_range = hypes['preprocess']['cav_lidar_range']
                    grid_x, grid_y = bev_cell_centers(map_shape, lidar_range)
                    radial_distance = np.sqrt(np.square(grid_x) + np.square(grid_y))

                boxes = batch_data['ego']['object_bbx_center'].detach().float().cpu().numpy()[sample_offset]
                box_mask = batch_data['ego']['object_bbx_mask'].detach().cpu().numpy()[sample_offset] > 0
                boxes = boxes[box_mask]
                object_ids = batch_data['ego'].get('object_ids', [])
                object_ids = list(object_ids)[:len(boxes)]
                classes = [
                    object_class_lookup.get(int(object_id), 'unknown')
                    for object_id in object_ids
                ]

                order = hypes['postprocess'].get('order', 'hwl')
                object_mask = boxes_to_bev_mask(
                    boxes, classes, set(classes), grid_x, grid_y, order)
                vru_mask = boxes_to_bev_mask(
                    boxes, classes, vru_classes, grid_x, grid_y, order)
                person_mask = boxes_to_bev_mask(
                    boxes, classes, person_classes, grid_x, grid_y, order)
                background_mask = valid_float * (1.0 - np.clip(object_mask, 0.0, 1.0))

                for prefix, region_mask in (
                        ('object', object_mask),
                        ('vru', vru_mask),
                        ('person', person_mask)):
                    if prefix not in region_accumulators:
                        region_accumulators[prefix] = {
                            'sum': np.zeros(map_shape, dtype=np.float64),
                            'count': np.zeros(map_shape, dtype=np.float64),
                        }
                    region_accumulators[prefix]['sum'] += u * region_mask
                    region_accumulators[prefix]['count'] += region_mask

                maps.append(u)
                valid_maps.append(np.where(valid_mask, u, np.nan))
                valid_masks.append(valid_float)
                point_counts.append(points)
                occupancy_maps.append(occ)
                sparsities.append(sparse)

                object_mean = masked_mean(u, object_mask)
                vru_mean = masked_mean(u, vru_mask)
                person_mean = masked_mean(u, person_mask)
                background_mean = masked_mean(u, background_mask)
                object_values.append(object_mean)
                vru_values.append(vru_mean)
                person_values.append(person_mean)
                background_values.append(background_mean)
                object_counts.append(len(classes))
                vru_counts.append(sum(cls in set(vru_classes) for cls in classes))
                person_counts.append(sum(cls in set(person_classes) for cls in classes))

                row = {
                    'condition': name,
                    'sample_id': sample_id_for_index(dataset, batch_idx),
                    'sample_index': sample_count,
                    'mean_U_global': float(np.mean(u)),
                    'mean_U_valid': masked_mean(u, valid_float),
                    'mean_U_occupied': masked_mean(u, occ),
                    'mean_U_object': object_mean,
                    'mean_U_background': background_mean,
                    'mean_U_vru': vru_mean,
                    'mean_U_person': person_mean,
                    'occupancy_rate': occupancy_rate(occ, valid_float),
                    'mean_points_occupied': masked_mean(points, occ),
                    'object_count': int(object_counts[-1]),
                    'vru_count': int(vru_counts[-1]),
                    'person_count': int(person_counts[-1]),
                }

                for distance_bin in distance_bins:
                    label = distance_bin_label(distance_bin)
                    lo, hi = distance_bin
                    distance_mask = radial_distance >= lo
                    if hi is not None:
                        distance_mask = distance_mask & (radial_distance < hi)
                    distance_valid = valid_float * distance_mask.astype(np.float32)
                    distance_occupied = distance_valid * (occ > 0).astype(np.float32)
                    distance_object = distance_valid * object_mask
                    row[f'{label}_U'] = masked_mean(u, distance_valid)
                    row[f'{label}_occupancy_rate'] = occupancy_rate(occ, distance_valid)
                    row[f'{label}_mean_points_occupied'] = masked_mean(points, distance_occupied)
                    row[f'{label}_object_U'] = masked_mean(u, distance_object)
                    row[f'{label}_object_count'] = count_objects_in_distance(boxes, distance_bin)

                per_sample_rows.append(row)
                sample_count += 1

    if not maps:
        raise RuntimeError(f"No valid samples produced for condition {name}")

    maps = np.stack(maps, axis=0)
    valid_maps = np.stack(valid_maps, axis=0)
    valid_masks = np.stack(valid_masks, axis=0)
    point_counts = np.stack(point_counts, axis=0)
    occupancy_maps = np.stack(occupancy_maps, axis=0)
    sparsities = np.stack(sparsities, axis=0)
    object_values = np.asarray(object_values, dtype=np.float64)
    vru_values = np.asarray(vru_values, dtype=np.float64)
    person_values = np.asarray(person_values, dtype=np.float64)
    background_values = np.asarray(background_values, dtype=np.float64)
    object_counts = np.asarray(object_counts, dtype=np.float64)
    vru_counts = np.asarray(vru_counts, dtype=np.float64)
    person_counts = np.asarray(person_counts, dtype=np.float64)

    mean_u_global_map = maps.mean(axis=0)
    mean_u_valid_map = nan_stat(np.nanmean, valid_maps, axis=0)
    std_u_valid_map = nan_stat(np.nanstd, valid_maps, axis=0)
    median_u_valid_map = nan_stat(np.nanmedian, valid_maps, axis=0)
    support_count_map = np.isfinite(valid_maps).sum(axis=0).astype(np.float32)
    mean_point_count = point_counts.mean(axis=0)
    mean_sparsity = sparsities.mean(axis=0)

    cond_dir = output_dir / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    np.save(cond_dir / 'mean_U_L.npy', mean_u_valid_map)
    np.save(cond_dir / 'mean_U_L_global.npy', mean_u_global_map)
    np.save(cond_dir / 'std_U_L.npy', std_u_valid_map)
    np.save(cond_dir / 'median_U_L.npy', median_u_valid_map)
    np.save(cond_dir / 'support_count.npy', support_count_map)
    np.save(cond_dir / 'mean_point_count.npy', mean_point_count)
    np.save(cond_dir / 'mean_occupancy_sparsity.npy', mean_sparsity)
    save_heatmap(mean_u_valid_map, cond_dir / 'mean_U_L_heatmap.png', f'{name}: mean U_L valid')
    save_heatmap(std_u_valid_map, cond_dir / 'std_U_L_heatmap.png', f'{name}: std U_L valid',
                 cmap='viridis', vmin=None, vmax=None, colorbar_label='std U_L')
    save_heatmap(median_u_valid_map, cond_dir / 'median_U_L_heatmap.png', f'{name}: median U_L valid')
    save_heatmap(support_count_map, cond_dir / 'support_count_heatmap.png', f'{name}: valid support',
                 cmap='viridis', vmin=0.0, vmax=None, colorbar_label='valid sample count')
    save_heatmap(np.clip(mean_sparsity, 0, 1), cond_dir / 'mean_occupancy_sparsity.png',
                 f'{name}: occupancy sparsity')

    for prefix, accum in region_accumulators.items():
        map_sum = accum['sum']
        map_count = accum['count']
        mean_region_map = np.full_like(map_sum, np.nan, dtype=np.float64)
        np.divide(map_sum, map_count, out=mean_region_map, where=map_count > 0)
        np.save(cond_dir / f'mean_U_L_{prefix}_regions.npy', mean_region_map)
        np.save(cond_dir / f'{prefix}_support_count.npy', map_count)
        save_heatmap(
            mean_region_map,
            cond_dir / f'mean_U_L_{prefix}_regions_heatmap.png',
            f'{name}: mean U_L in {prefix} regions',
        )
        save_heatmap(
            map_count,
            cond_dir / f'{prefix}_support_count_heatmap.png',
            f'{name}: {prefix} support',
            cmap='viridis',
            vmin=0.0,
            vmax=None,
            colorbar_label='region support count',
        )

    occupied_mask = occupancy_maps > 0
    valid_not_object_mean = (
        float(np.nanmean(background_values))
        if np.isfinite(background_values).any() else np.nan
    )
    mean_valid = float(np.nanmean(valid_maps))
    summary = {
        'condition': name,
        'samples': sample_count,
        'mean_U_L': mean_valid,
        'mean_U_L_global': float(maps.mean()),
        'mean_U_L_valid': mean_valid,
        'mean_U_L_occupied': masked_mean(maps, occupied_mask),
        'mean_U_L_object': float(np.nanmean(object_values)) if np.isfinite(object_values).any() else np.nan,
        'mean_U_L_background': valid_not_object_mean,
        'median_U_L_valid': float(np.nanmedian(valid_maps)),
        'p90_U_L_valid': float(np.nanpercentile(valid_maps, 90)),
        'mean_point_count': float(point_counts.mean()),
        'mean_points_occupied': masked_mean(point_counts, occupied_mask),
        'occupancy_rate_valid': float((occupancy_maps[valid_masks > 0] > 0).mean()),
        'mean_occupancy_sparsity': float(sparsities.mean()),
        'object_count': int(object_counts.sum()),
        'vru_count': int(vru_counts.sum()),
        'person_count': int(person_counts.sum()),
        'mean_U_L_vru': float(np.nanmean(vru_values)) if np.isfinite(vru_values).any() else np.nan,
        'mean_U_L_person': float(np.nanmean(person_values)) if np.isfinite(person_values).any() else np.nan,
    }
    for distance_bin in distance_bins:
        label = distance_bin_label(distance_bin)
        values = [row[f'{label}_U'] for row in per_sample_rows]
        summary[f'{label}_U'] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        values = [row[f'{label}_occupancy_rate'] for row in per_sample_rows]
        summary[f'{label}_occupancy_rate'] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        values = [row[f'{label}_mean_points_occupied'] for row in per_sample_rows]
        summary[f'{label}_mean_points_occupied'] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        values = [row[f'{label}_object_U'] for row in per_sample_rows]
        summary[f'{label}_object_U'] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        summary[f'{label}_object_count'] = int(sum(row[f'{label}_object_count'] for row in per_sample_rows))

    return {
        'summary': summary,
        'samples': per_sample_rows,
        'mean_valid_map': mean_u_valid_map,
    }


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlate(rows, metrics_csv, output_dir):
    if not metrics_csv:
        return None
    metrics = {}
    metric_name = None
    with open(metrics_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row['condition']
            if 'map_drop' in row and row['map_drop'] != '':
                metrics[cond] = float(row['map_drop'])
                metric_name = 'map_drop'
            elif 'mAP' in row and row['mAP'] != '':
                metrics[cond] = float(row['mAP'])
                metric_name = 'mAP'
            elif 'map' in row and row['map'] != '':
                metrics[cond] = float(row['map'])
                metric_name = 'map'

    xs, ys, names = [], [], []
    for row in rows:
        cond = row['condition']
        if cond in metrics:
            xs.append(row['mean_U_L_valid'])
            ys.append(metrics[cond])
            names.append(cond)
    if len(xs) < 2:
        return None

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    pearson = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else np.nan
    spearman = float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1]) if len(xs) >= 2 else np.nan
    expected = 'positive' if metric_name == 'map_drop' else 'negative'
    out = output_dir / 'wp4_reliability_correlation.csv'
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'mean_U_L_valid', metric_name or 'metric'])
        for name, x, y in zip(names, xs, ys):
            writer.writerow([name, x, y])
        writer.writerow([])
        writer.writerow(['n_conditions', len(xs)])
        writer.writerow(['expected_sign', expected])
        writer.writerow(['pearson_corr_descriptive', pearson])
        writer.writerow(['spearman_corr_descriptive', spearman])
        if len(xs) <= 3:
            writer.writerow(['note', 'condition-level correlation is descriptive only'])
    return out, pearson, spearman


def write_paired_deltas(sample_rows, output_dir):
    by_condition = {}
    for row in sample_rows:
        by_condition.setdefault(row['condition'], {})[row['sample_id']] = row
    if 'clear' not in by_condition:
        return None

    rows = []
    clear_rows = by_condition['clear']
    for condition, condition_rows in by_condition.items():
        if condition == 'clear':
            continue
        for sample_id, row in condition_rows.items():
            clear = clear_rows.get(sample_id)
            if clear is None:
                continue
            rows.append({
                'condition': condition,
                'sample_id': sample_id,
                'delta_U_valid': row['mean_U_valid'] - clear['mean_U_valid'],
                'delta_U_object': row['mean_U_object'] - clear['mean_U_object']
                if np.isfinite(row['mean_U_object']) and np.isfinite(clear['mean_U_object']) else np.nan,
                'delta_U_occupied': row['mean_U_occupied'] - clear['mean_U_occupied'],
                'delta_occupancy_rate': row['occupancy_rate'] - clear['occupancy_rate'],
            })
    if not rows:
        return None
    return write_rows(output_dir / 'wp4_paired_deltas.csv', rows)


def write_delta_heatmaps(results, output_dir):
    if 'clear' not in results:
        return []
    paths = []
    clear_map = results['clear']['mean_valid_map']
    for condition in ('fog', 'rain'):
        if condition not in results:
            continue
        delta = results[condition]['mean_valid_map'] - clear_map
        path = output_dir / f'delta_{condition}_clear_heatmap.png'
        np.save(output_dir / f'delta_{condition}_clear.npy', delta)
        save_delta_heatmap(delta, path, f'{condition} - clear: mean U_L valid')
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', action='append', type=parse_condition, required=True,
                        help='Weather condition and config as NAME:CONFIG.yaml. Repeat for clear/fog/rain.')
    parser.add_argument('--output_dir', default='outputs/wp4_reliability')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--metrics_csv', default=None,
                        help='Optional CSV with condition,mAP or condition,map_drop for descriptive correlation.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--vru_classes', default=','.join(DEFAULT_VRU_CLASSES),
                        help='Comma-separated classes for vulnerable road-user object-region analysis.')
    parser.add_argument('--person_classes', default=','.join(DEFAULT_PERSON_CLASSES),
                        help='Comma-separated classes for vulnerable-person object-region analysis.')
    parser.add_argument('--distance_bins', type=parse_distance_bins,
                        default=DEFAULT_DISTANCE_BINS,
                        help='Comma-separated bins, e.g. 0-20,20-40,40-60,60-inf.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vru_classes = tuple(cls.strip() for cls in args.vru_classes.split(',') if cls.strip())
    person_classes = tuple(cls.strip() for cls in args.person_classes.split(',') if cls.strip())

    rows = []
    sample_rows = []
    results = {}
    for name, cfg in args.condition:
        result = summarize_condition(
            name,
            cfg,
            output_dir,
            args.max_batches,
            args.max_samples,
            args.device,
            vru_classes,
            person_classes,
            args.distance_bins,
        )
        rows.append(result['summary'])
        sample_rows.extend(result['samples'])
        results[name] = result

    summary_path = write_rows(output_dir / 'wp4_reliability_summary.csv', rows)
    per_sample_path = write_rows(output_dir / 'wp4_per_sample_metrics.csv', sample_rows)
    paired_path = write_paired_deltas(sample_rows, output_dir)
    delta_paths = write_delta_heatmaps(results, output_dir)
    corr_result = correlate(rows, args.metrics_csv, output_dir)

    print(f'Wrote summary: {summary_path}')
    print(f'Wrote per-sample metrics: {per_sample_path}')
    if paired_path:
        print(f'Wrote paired deltas: {paired_path}')
    for path in delta_paths:
        print(f'Wrote delta heatmap: {path}')
    if corr_result:
        corr_path, pearson, spearman = corr_result
        print(f'Wrote descriptive correlation: {corr_path} pearson={pearson:.4f} spearman={spearman:.4f}')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
