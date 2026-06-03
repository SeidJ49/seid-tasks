#!/usr/bin/env python3
"""WP4 LiDAR unreliability heatmaps and degradation correlation.

Example:
python opencood/tools/wp4_lidar_reliability_analysis.py \
  --condition clear:opencood/hypes_yaml/truckscences_lidar/lidar_reliability_stage3_clear.yaml \
  --condition fog:opencood/hypes_yaml/truckscences_lidar/lidar_reliability_stage3_fog.yaml \
  --condition rain:opencood/hypes_yaml/truckscences_lidar/lidar_reliability_stage3_rain.yaml \
  --output_dir outputs/wp4_reliability \
  --max_batches 100 \
  --metrics_csv outputs/wp4_reliability/metrics.csv

metrics_csv columns for correlation: condition,mAP or condition,map_drop.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def parse_condition(value):
    if ':' not in value:
        raise argparse.ArgumentTypeError("condition must be NAME:CONFIG.yaml")
    name, path = value.split(':', 1)
    if not name:
        raise argparse.ArgumentTypeError("condition name is empty")
    return name, path


def save_heatmap(array, path, title):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(array, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
    plt.colorbar(label='U_L: LiDAR unreliability (high = unreliable)')
    plt.title(title)
    plt.xlabel('BEV x cell')
    plt.ylabel('BEV y cell')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def summarize_condition(name, config_path, output_dir, max_batches, device):
    hypes = yaml_utils.load_yaml(config_path)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes).to(device)
    model.eval()

    maps = []
    point_counts = []
    sparsities = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_data = train_utils.to_device(batch_data, device)
            output = model(batch_data['ego'])
            u_l = output['U_L'].detach().float().cpu().numpy()[:, 0]
            pc = output['lidar_point_count_map'].detach().float().cpu().numpy()[:, 0]
            sp = output['lidar_occupancy_sparsity'].detach().float().cpu().numpy()[:, 0]
            maps.append(u_l)
            point_counts.append(pc)
            sparsities.append(sp)
            sample_count += u_l.shape[0]

    if not maps:
        raise RuntimeError(f"No valid batches produced for condition {name}")

    maps = np.concatenate(maps, axis=0)
    point_counts = np.concatenate(point_counts, axis=0)
    sparsities = np.concatenate(sparsities, axis=0)

    mean_u = maps.mean(axis=0)
    mean_point_count = point_counts.mean(axis=0)
    mean_sparsity = sparsities.mean(axis=0)

    cond_dir = output_dir / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    np.save(cond_dir / 'mean_U_L.npy', mean_u)
    np.save(cond_dir / 'mean_point_count.npy', mean_point_count)
    np.save(cond_dir / 'mean_occupancy_sparsity.npy', mean_sparsity)
    save_heatmap(mean_u, cond_dir / 'mean_U_L_heatmap.png', f'{name}: mean U_L')
    save_heatmap(np.clip(mean_sparsity, 0, 1), cond_dir / 'mean_occupancy_sparsity.png', f'{name}: occupancy sparsity')

    return {
        'condition': name,
        'samples': sample_count,
        'mean_U_L': float(maps.mean()),
        'median_U_L': float(np.median(maps)),
        'p90_U_L': float(np.percentile(maps, 90)),
        'mean_point_count': float(point_counts.mean()),
        'mean_occupancy_sparsity': float(sparsities.mean()),
    }


def write_summary(rows, output_dir):
    path = output_dir / 'wp4_reliability_summary.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def correlate(rows, metrics_csv, output_dir):
    if not metrics_csv:
        return None
    metrics = {}
    with open(metrics_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row['condition']
            if 'map_drop' in row and row['map_drop'] != '':
                metrics[cond] = float(row['map_drop'])
            elif 'mAP' in row and row['mAP'] != '':
                metrics[cond] = float(row['mAP'])
            elif 'map' in row and row['map'] != '':
                metrics[cond] = float(row['map'])

    xs, ys, names = [], [], []
    for row in rows:
        cond = row['condition']
        if cond in metrics:
            xs.append(row['mean_U_L'])
            ys.append(metrics[cond])
            names.append(cond)
    if len(xs) < 2:
        return None

    corr = float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
    out = output_dir / 'wp4_reliability_correlation.csv'
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'mean_U_L', 'metric'])
        for name, x, y in zip(names, xs, ys):
            writer.writerow([name, x, y])
        writer.writerow([])
        writer.writerow(['pearson_corr', corr])
    return out, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', action='append', type=parse_condition, required=True,
                        help='Weather condition and config as NAME:CONFIG.yaml. Repeat for clear/fog/rain.')
    parser.add_argument('--output_dir', default='outputs/wp4_reliability')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--metrics_csv', default=None,
                        help='Optional CSV with condition,mAP or condition,map_drop for correlation analysis.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, cfg in args.condition:
        rows.append(summarize_condition(name, cfg, output_dir, args.max_batches, args.device))

    summary_path = write_summary(rows, output_dir)
    corr_result = correlate(rows, args.metrics_csv, output_dir)
    print(f'Wrote summary: {summary_path}')
    if corr_result:
        corr_path, corr = corr_result
        print(f'Wrote correlation: {corr_path} pearson={corr:.4f}')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()