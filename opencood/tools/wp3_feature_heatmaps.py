import argparse
import csv
import importlib
import os
from pathlib import Path

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
    parser = argparse.ArgumentParser(
        description='Dump WP3 PillarNet student/teacher BEV feature heatmaps.'
    )
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--student_checkpoint', default='')
    parser.add_argument('--teacher_checkpoint', default='')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_gt_overlay', action='store_true')
    parser.add_argument(
        '--color_percentile',
        type=float,
        default=99.0,
        help='Upper percentile for heatmap color scaling. Use 100 for raw min/max scaling.',
    )
    return parser.parse_args()


def load_state_dict_compatible(model, checkpoint_path, label):
    if not checkpoint_path:
        print(f'[{label}] no checkpoint provided; using current initialization')
        return
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
    print(f'[{label}] loaded {len(compatible)} tensors from {checkpoint_path}')


def create_teacher(hypes):
    kd_flag = hypes.get('kd_flag', {})
    teacher_name = kd_flag.get('teacher_model')
    teacher_args = kd_flag.get('teacher_model_config')
    if not teacher_name or not teacher_args:
        return None

    model_lib = importlib.import_module('opencood.models.' + teacher_name)
    target_name = teacher_name.replace('_', '')
    teacher_cls = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_name.lower():
            teacher_cls = cls
            break
    if teacher_cls is None:
        raise RuntimeError(f'Could not find teacher class for {teacher_name}')
    return teacher_cls(teacher_args)


def override_paths(hypes, args):
    if args.root_dir:
        hypes['root_dir'] = args.root_dir
    if args.validate_dir:
        hypes['validate_dir'] = args.validate_dir
    if args.test_dir:
        hypes['test_dir'] = args.test_dir
    if args.split == 'train':
        return
    if args.split == 'val':
        hypes['validate_dir'] = hypes.get('validate_dir', hypes.get('root_dir'))
    if args.split == 'test':
        hypes['validate_dir'] = hypes.get('test_dir', hypes.get('validate_dir'))


def feature_heatmap(feature):
    feature = feature.detach().float()
    if feature.dim() == 4:
        feature = feature[0]
    return feature.abs().mean(dim=0).cpu().numpy()


def extract_gt_boxes(student_output):
    gt_boxes = student_output.get('gt_boxes_for_kd')
    if gt_boxes is None:
        return None
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    if gt_boxes.ndim == 3:
        gt_boxes = gt_boxes[0]
    if gt_boxes.shape[-1] >= 8:
        gt_boxes = gt_boxes[gt_boxes[:, 7] > 0]
    if gt_boxes.size == 0:
        return None
    return gt_boxes[:, :7]


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
        closed_x = np.r_[box_px, box_px[0]]
        closed_y = np.r_[box_py, box_py[0]]
        ax.plot(closed_x, closed_y, color='cyan', linewidth=1.2)
        ax.plot([cx, hx], [cy, hy], color='cyan', linewidth=1.0)


def save_map(array, path, title, cmap='magma', gt_boxes=None, pc_range=None, color_percentile=100.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), array)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    vmax = None
    if color_percentile < 100.0:
        positive = array[np.isfinite(array)]
        if positive.size > 0:
            vmax = float(np.percentile(positive, color_percentile))
            if vmax <= float(np.min(positive)):
                vmax = None
    image = ax.imshow(array, origin='lower', cmap=cmap, vmax=vmax)
    draw_gt_overlay(ax, gt_boxes, array.shape, pc_range)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_rgb_map(array, path, title, gt_boxes=None, pc_range=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), array)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.imshow(array, origin='lower')
    draw_gt_overlay(ax, gt_boxes, array.shape[:2], pc_range)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def resize_mask_like(mask, target_shape):
    if mask is None:
        return None
    if mask.shape == target_shape:
        return mask
    tensor = torch.from_numpy(mask).float()[None, None]
    tensor = torch.nn.functional.interpolate(tensor, size=target_shape, mode='nearest')
    return tensor[0, 0].numpy()


def summarize_array(array, prefix):
    return {
        f'{prefix}_min': float(np.min(array)),
        f'{prefix}_mean': float(np.mean(array)),
        f'{prefix}_std': float(np.std(array)),
        f'{prefix}_max': float(np.max(array)),
        f'{prefix}_nonzero': float(np.mean(array > 0.0)),
    }


def masked_mean(array, mask, threshold=0.0):
    valid = mask > threshold
    if not np.any(valid):
        return 0.0
    return float(np.mean(array[valid]))


def save_batch_maps(
    output_dir,
    index,
    student_output,
    teacher_output,
    pc_range=None,
    overlay_gt=True,
    color_percentile=100.0,
):
    batch_dir = output_dir / f'sample_{index:05d}'
    gt_boxes = extract_gt_boxes(student_output) if overlay_gt else None
    student_feature = student_output.get('feature')
    adapted_feature = student_output.get('adapted_feature')
    teacher_feature = None if teacher_output is None else teacher_output.get('teacher_feature')
    row = {'sample': index}
    student_map = None
    teacher_map = None
    diff = None
    motion_mask = None
    rcs_mask = None
    evidence_mask = None

    if student_feature is not None:
        student_map = feature_heatmap(student_feature)
        row.update(summarize_array(student_map, 'student_feature'))
        save_map(student_map, batch_dir / 'student_feature.png', 'radar student feature', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if adapted_feature is not None:
        adapted_map = feature_heatmap(adapted_feature)
        row.update(summarize_array(adapted_map, 'adapted_feature'))
        save_map(adapted_map, batch_dir / 'adapted_feature.png', 'adapted radar feature', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if teacher_feature is not None:
        teacher_map = feature_heatmap(teacher_feature)
        row.update(summarize_array(teacher_map, 'teacher_feature'))
        save_map(teacher_map, batch_dir / 'teacher_feature.png', 'lidar teacher feature', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if adapted_feature is not None and teacher_feature is not None and adapted_feature.shape == teacher_feature.shape:
        adapted_diff = feature_heatmap(adapted_feature - teacher_feature)
        row.update(summarize_array(adapted_diff, 'adapted_feature_abs_diff'))
        save_map(adapted_diff, batch_dir / 'adapted_feature_abs_diff.png', 'adapted teacher abs diff', cmap='viridis', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if student_feature is not None and teacher_feature is not None:
        if student_feature.shape == teacher_feature.shape:
            diff = feature_heatmap(student_feature - teacher_feature)
            row.update(summarize_array(diff, 'feature_abs_diff'))
            save_map(diff, batch_dir / 'feature_abs_diff.png', 'student teacher abs diff', cmap='viridis', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
        else:
            print(
                f'[sample {index}] feature shape mismatch: '
                f'student={tuple(student_feature.shape)} teacher={tuple(teacher_feature.shape)}'
            )
    if 'kd_motion_mask' in student_output:
        motion_mask = student_output['kd_motion_mask'][0, 0].detach().cpu().numpy()
        row.update(summarize_array(motion_mask, 'kd_motion_mask'))
        save_map(motion_mask, batch_dir / 'kd_motion_mask.png', 'motion KD mask', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if 'rcs_confidence_mask' in student_output:
        rcs_mask = student_output['rcs_confidence_mask'][0, 0].detach().cpu().numpy()
        row.update(summarize_array(rcs_mask, 'rcs_confidence_mask'))
        save_map(rcs_mask, batch_dir / 'rcs_confidence_mask.png', 'RCS confidence mask', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
    if 'radar_evidence_mask' in student_output:
        evidence_mask = student_output['radar_evidence_mask'][0, 0].detach().cpu().numpy()
        row.update(summarize_array(evidence_mask, 'radar_evidence_mask'))
        save_map(evidence_mask, batch_dir / 'radar_evidence_mask.png', 'histogram radar evidence mask', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)

    if diff is not None:
        if motion_mask is not None:
            motion_mask = resize_mask_like(motion_mask, diff.shape)
            motion_binary = motion_mask > 0.0
            background_binary = ~motion_binary
            masked_diff = diff * motion_binary.astype(diff.dtype)
            background_diff = diff * background_binary.astype(diff.dtype)
            save_map(masked_diff, batch_dir / 'masked_feature_abs_diff.png', 'feature abs diff inside motion mask', cmap='viridis', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
            save_map(background_diff, batch_dir / 'background_feature_abs_diff.png', 'feature abs diff outside motion mask', cmap='viridis', gt_boxes=gt_boxes, pc_range=pc_range, color_percentile=color_percentile)
            row['diff_mean_motion'] = masked_mean(diff, motion_mask)
            row['diff_mean_background'] = float(np.mean(diff[background_binary])) if np.any(background_binary) else 0.0
            row['diff_motion_to_background_ratio'] = row['diff_mean_motion'] / max(row['diff_mean_background'], 1e-8)
            row['motion_coverage'] = float(np.mean(motion_binary))

        if rcs_mask is not None:
            rcs_mask = resize_mask_like(rcs_mask, diff.shape)
            rcs_binary = rcs_mask > 0.0
            row['diff_mean_rcs'] = masked_mean(diff, rcs_mask)
            row['rcs_coverage'] = float(np.mean(rcs_binary))

        if motion_mask is not None and rcs_mask is not None:
            motion_norm = motion_mask / max(float(np.max(motion_mask)), 1e-8)
            rcs_norm = rcs_mask / max(float(np.max(rcs_mask)), 1e-8)
            overlap = np.zeros((*diff.shape, 3), dtype=np.float32)
            overlap[..., 0] = rcs_norm
            overlap[..., 1] = motion_norm
            overlap[..., 2] = np.minimum(motion_norm, rcs_norm)
            save_rgb_map(overlap, batch_dir / 'motion_rcs_overlap.png', 'RCS red, motion green, overlap blue', gt_boxes=gt_boxes, pc_range=pc_range)
            motion_binary = motion_mask > 0.0
            rcs_binary = rcs_mask > 0.0
            union = motion_binary | rcs_binary
            inter = motion_binary & rcs_binary
            row['motion_rcs_iou'] = float(np.sum(inter) / max(np.sum(union), 1))
            row['motion_rcs_overlap_fraction'] = float(np.sum(inter) / max(np.sum(motion_binary), 1))

        if evidence_mask is not None:
            evidence_mask = resize_mask_like(evidence_mask, diff.shape)
            evidence_binary = evidence_mask > 0.0
            row['diff_mean_radar_evidence'] = masked_mean(diff, evidence_mask)
            row['radar_evidence_coverage'] = float(np.mean(evidence_binary))

    return row


def write_summary(output_dir, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    summary_path = output_dir / 'summary.csv'
    with summary_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'[summary] wrote {summary_path}')


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
    student = train_utils.create_model(hypes).to(device)
    load_state_dict_compatible(student, args.student_checkpoint, 'student')
    student.eval()
    if hasattr(student, 'use_radar_evidence_mask'):
        print(
            '[student] radar_evidence_mask '
            f'enabled={student.use_radar_evidence_mask} '
            f'key={student.radar_evidence_mask_key} '
            f'rcs_center={student.evidence_rcs_center} '
            f'rcs_scale={student.evidence_rcs_scale} '
            f'doppler_center={student.evidence_doppler_center} '
            f'doppler_scale={student.evidence_doppler_scale} '
            f'rcs_alpha={student.evidence_rcs_alpha} '
            f'combine={student.evidence_combine}'
        )

    teacher = create_teacher(hypes)
    if teacher is not None:
        teacher = teacher.to(device)
        teacher_path = args.teacher_checkpoint or hypes.get('kd_flag', {}).get('teacher_path', '')
        load_state_dict_compatible(teacher, teacher_path, 'teacher')
        teacher.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    pc_range = hypes['model']['args'].get('lidar_range', hypes.get('preprocess', {}).get('cav_lidar_range'))

    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            if index >= args.max_batches:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego_data = batch_data['ego']
            student_output = student(ego_data)
            teacher_output = teacher(ego_data) if teacher is not None else None
            summary_rows.append(
                save_batch_maps(
                    output_dir,
                    index,
                    student_output,
                    teacher_output,
                    pc_range=pc_range,
                    overlay_gt=not args.no_gt_overlay,
                    color_percentile=args.color_percentile,
                )
            )
            print(f'[sample {index}] saved heatmaps to {output_dir / f"sample_{index:05d}"}')

    write_summary(output_dir, summary_rows)


if __name__ == '__main__':
    main()
