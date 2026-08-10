import argparse
import csv
from pathlib import Path
import sys
from pathlib import Path
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize WP3 RadarDistill object-KD masks.')
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--color_percentile', type=float, default=99.0)
    parser.add_argument('--no_gt_overlay', action='store_true')
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


def load_state_dict_compatible(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state_dict, dict):
        state_dict = state_dict.get('state_dict', state_dict.get('model_state_dict', state_dict))
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


def extract_gt_boxes(ego_data):
    boxes = ego_data['object_bbx_center'][0].detach().cpu().numpy()
    mask = ego_data['object_bbx_mask'][0].detach().cpu().numpy().astype(bool)
    boxes = boxes[mask]
    if boxes.size == 0:
        return None
    return boxes[:, :7]


def draw_gt_overlay(ax, gt_boxes, array_shape, pc_range):
    if gt_boxes is None or len(gt_boxes) == 0 or pc_range is None:
        return
    corners = box_utils.boxes_to_corners_3d(gt_boxes, order='hwl')[:, :4, :2]
    height, width = array_shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_den = max(float(x_max - x_min), 1e-6)
    y_den = max(float(y_max - y_min), 1e-6)
    px = (corners[:, :, 0] - x_min) / x_den * (width - 1)
    py = (corners[:, :, 1] - y_min) / y_den * (height - 1)
    centers_x = (gt_boxes[:, 0] - x_min) / x_den * (width - 1)
    centers_y = (gt_boxes[:, 1] - y_min) / y_den * (height - 1)
    yaw = gt_boxes[:, 6]
    lengths = gt_boxes[:, 5] if gt_boxes.shape[1] > 5 else np.ones_like(yaw)
    head_x = (gt_boxes[:, 0] + 0.5 * lengths * np.cos(yaw) - x_min) / x_den * (width - 1)
    head_y = (gt_boxes[:, 1] + 0.5 * lengths * np.sin(yaw) - y_min) / y_den * (height - 1)
    for box_px, box_py, cx, cy, hx, hy in zip(px, py, centers_x, centers_y, head_x, head_y):
        ax.plot(np.r_[box_px, box_px[0]], np.r_[box_py, box_py[0]], color='cyan', linewidth=1.2)
        ax.plot([cx, hx], [cy, hy], color='cyan', linewidth=1.0)


def save_map(array, path, title, gt_boxes=None, pc_range=None, color_percentile=99.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), array)
    vmax = None
    finite = array[np.isfinite(array)]
    if finite.size > 0 and color_percentile < 100.0:
        vmax = float(np.percentile(finite, color_percentile))
        if vmax <= float(np.min(finite)):
            vmax = None
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    image = ax.imshow(array, origin='lower', cmap='magma', vmin=0.0, vmax=vmax)
    draw_gt_overlay(ax, gt_boxes, array.shape, pc_range)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_rgb_map(maps, path, title, gt_boxes=None, pc_range=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*next(iter(maps.values())).shape, 3), dtype=np.float32)
    channel_map = {'radar': 0, 'object': 1, 'lidar': 2}
    for key, channel in channel_map.items():
        if key not in maps:
            continue
        arr = maps[key]
        rgb[..., channel] = arr / max(float(np.max(arr)), 1e-8)
    np.save(path.with_suffix('.npy'), rgb)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.imshow(np.clip(rgb, 0.0, 1.0), origin='lower')
    draw_gt_overlay(ax, gt_boxes, rgb.shape[:2], pc_range)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_custom_rgb_map(red, green, blue, path, title, gt_boxes=None, pc_range=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*red.shape, 3), dtype=np.float32)
    for channel, arr in enumerate((red, green, blue)):
        rgb[..., channel] = arr / max(float(np.max(arr)), 1e-8)
    np.save(path.with_suffix('.npy'), rgb)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.imshow(np.clip(rgb, 0.0, 1.0), origin='lower')
    draw_gt_overlay(ax, gt_boxes, rgb.shape[:2], pc_range)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def tensor_to_map(tensor):
    arr = tensor[0, 0].detach().float().cpu().numpy()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def summarize(array, prefix):
    return {
        f'{prefix}_mean': float(np.mean(array)),
        f'{prefix}_max': float(np.max(array)),
        f'{prefix}_nonzero': float(np.mean(array > 0.0)),
        f'{prefix}_gt025': float(np.mean(array > 0.25)),
        f'{prefix}_gt050': float(np.mean(array > 0.50)),
    }


def write_summary(output_dir, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / 'summary.csv').open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)

    dataset = build_dataset(hypes, visualize=False, train=args.split == 'train')
    collate_fn = dataset.collate_batch_train if args.split == 'train' else dataset.collate_batch_test
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = train_utils.create_model(hypes).to(device)
    load_state_dict_compatible(model, args.checkpoint)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pc_range = hypes['model']['args'].get('lidar_range', hypes.get('preprocess', {}).get('cav_lidar_range'))
    rows = []

    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            if index >= args.max_batches:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego_data = batch_data['ego']
            ego_data['compute_loss'] = True
            output = model(ego_data)
            maps = {
                key: tensor_to_map(value)
                for key, value in model.radar_distill.debug_maps.items()
            }
            sample_dir = output_dir / f'sample_{index:05d}'
            gt_boxes = None if args.no_gt_overlay else extract_gt_boxes(ego_data)
            row = {'sample': index}
            row.update(output.get('tb_dict', {}))
            for key, arr in maps.items():
                save_map(arr, sample_dir / f'{key}.png', key, gt_boxes, pc_range, args.color_percentile)
                row.update(summarize(arr, key))
            rgb_inputs = {}
            if 'radar_evidence_mask' in maps:
                rgb_inputs['radar'] = maps['radar_evidence_mask']
            if 'object_kd_mask' in maps:
                rgb_inputs['object'] = maps['object_kd_mask']
            if 'lidar_intensity_mask' in maps:
                rgb_inputs['lidar'] = maps['lidar_intensity_mask']
            if len(rgb_inputs) > 1:
                save_rgb_map(
                    rgb_inputs,
                    sample_dir / 'mask_rgb_overlay.png',
                    'red radar, green object KD, blue LiDAR intensity',
                    gt_boxes,
                    pc_range,
                )
            if all(key in maps for key in ('pfd_fp_mask', 'pfd_tp_mask', 'pfd_fn_mask')):
                save_custom_rgb_map(
                    maps['pfd_fp_mask'],
                    maps['pfd_tp_mask'],
                    maps['pfd_fn_mask'],
                    sample_dir / 'pfd_tp_fp_fn_overlay.png',
                    'PFD: red FP, green TP, blue FN',
                    gt_boxes,
                    pc_range,
                )
            if all(key in maps for key in ('low_main_afd_radar_only_mask', 'low_main_afd_overlap_mask', 'low_main_afd_lidar_active_mask')):
                save_custom_rgb_map(
                    maps['low_main_afd_radar_only_mask'],
                    maps['low_main_afd_overlap_mask'],
                    maps['low_main_afd_lidar_active_mask'],
                    sample_dir / 'afd_low_main_overlay.png',
                    'AFD low main: red radar-only, green overlap, blue LiDAR-active',
                    gt_boxes,
                    pc_range,
                )
            if all(key in maps for key in ('cma_radar_before_xconv4_mag', 'cma_radar_after_xconv4_mag', 'cma_radar_delta_xconv4_mag')):
                save_custom_rgb_map(
                    maps['cma_radar_before_xconv4_mag'],
                    maps['cma_radar_after_xconv4_mag'],
                    maps['cma_radar_delta_xconv4_mag'],
                    sample_dir / 'cma_before_after_delta_overlay.png',
                    'CMA: red before, green after, blue delta',
                    gt_boxes,
                    pc_range,
                )
            if all(key in maps for key in ('radar_evidence_mask', 'object_proto_radar_support_mask', 'object_proto_context_mask')):
                save_custom_rgb_map(
                    maps['radar_evidence_mask'],
                    maps['object_proto_radar_support_mask'],
                    maps['object_proto_context_mask'],
                    sample_dir / 'proto_support_context_overlay.png',
                    'Proto KD: red radar evidence, green object support, blue context',
                    gt_boxes,
                    pc_range,
                )
            if all(key in maps for key in ('semantic_radar_heatmap_mask', 'semantic_teacher_heatmap_mask', 'semantic_heatmap_weight_mask')):
                save_custom_rgb_map(
                    maps['semantic_radar_heatmap_mask'],
                    maps['semantic_teacher_heatmap_mask'],
                    maps['semantic_heatmap_weight_mask'],
                    sample_dir / 'semantic_teacher_radar_weight_overlay.png',
                    'Semantic KD: red radar response, green teacher response, blue KD weight',
                    gt_boxes,
                    pc_range,
                )
            rows.append(row)
            print(f'[sample {index}] saved object-KD masks to {sample_dir}')

    write_summary(output_dir, rows)
    print(f'[summary] wrote {output_dir / "summary.csv"}')


if __name__ == '__main__':
    main()
