import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect WP3 PillarNet CenterHead predictions, postprocess noise, and KD gradient flow.'
    )
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--student_checkpoint', default='')
    parser.add_argument('--teacher_checkpoint', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--validate_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--max_batches', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--score_thresholds', default='0.03,0.05,0.10,0.20,0.30,0.50')
    parser.add_argument('--post_score_thresh', type=float, default=None)
    parser.add_argument('--post_max_obj', type=int, default=None)
    parser.add_argument('--post_nms_thresh', type=float, default=None)
    parser.add_argument('--run_grad_check', action='store_true')
    parser.add_argument(
        '--ignore_trainable_filter',
        action='store_true',
        help='Do not apply train_params.trainable_parameter_keywords before gradient check.',
    )
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


def load_state_dict_compatible(model, checkpoint_path, label):
    if not checkpoint_path:
        print(f'[{label}] no checkpoint provided')
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


def create_teacher(hypes, checkpoint_path):
    kd_flag = hypes.get('kd_flag', {})
    teacher_name = kd_flag.get('teacher_model')
    teacher_args = kd_flag.get('teacher_model_config')
    if not teacher_name or not teacher_args:
        raise RuntimeError('No kd_flag.teacher_model / teacher_model_config in YAML.')

    model_lib = importlib.import_module('opencood.models.' + teacher_name)
    target_name = teacher_name.replace('_', '')
    teacher_cls = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_name.lower():
            teacher_cls = cls
            break
    if teacher_cls is None:
        raise RuntimeError(f'Could not find teacher class for {teacher_name}')
    teacher = teacher_cls(teacher_args)
    load_state_dict_compatible(teacher, checkpoint_path or kd_flag.get('teacher_path', ''), 'teacher')
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def merge_teacher_outputs(student_output, teacher_output):
    for key, value in teacher_output.items():
        if key == 'teacher_feature' or key.startswith('teacher_'):
            student_output[key] = value


def class_name(class_names, label):
    index = int(label) - 1
    if 0 <= index < len(class_names):
        return class_names[index]
    return f'label_{int(label)}'


def score_summary(scores, thresholds):
    row = {
        'pred_count': int(scores.size),
        'score_mean': 0.0,
        'score_median': 0.0,
        'score_p90': 0.0,
        'score_p99': 0.0,
        'score_max': 0.0,
    }
    if scores.size:
        row.update({
            'score_mean': float(np.mean(scores)),
            'score_median': float(np.median(scores)),
            'score_p90': float(np.percentile(scores, 90)),
            'score_p99': float(np.percentile(scores, 99)),
            'score_max': float(np.max(scores)),
        })
    for threshold in thresholds:
        row[f'pred_ge_{threshold:.2f}'] = int(np.sum(scores >= threshold))
    return row


def head_heatmap_summary(pred_dicts, thresholds):
    row = {}
    all_scores = []
    for head_idx, pred_dict in enumerate(pred_dicts):
        hm = pred_dict['hm'].detach().sigmoid().float()
        scores = hm.reshape(-1).cpu().numpy()
        all_scores.append(scores)
        row[f'head{head_idx}_hm_mean'] = float(np.mean(scores))
        row[f'head{head_idx}_hm_p99'] = float(np.percentile(scores, 99))
        row[f'head{head_idx}_hm_max'] = float(np.max(scores))
        for threshold in thresholds:
            row[f'head{head_idx}_cells_ge_{threshold:.2f}'] = int(np.sum(scores >= threshold))
    if all_scores:
        scores = np.concatenate(all_scores)
        row['all_hm_mean'] = float(np.mean(scores))
        row['all_hm_p99'] = float(np.percentile(scores, 99))
        row['all_hm_max'] = float(np.max(scores))
        for threshold in thresholds:
            row[f'all_hm_cells_ge_{threshold:.2f}'] = int(np.sum(scores >= threshold))
    return row


def summarize_gradients(model):
    groups = {
        'adapter': ['kd_feature_adapter'],
        'radar_vfe': ['radar_vfe', 'radar_pillar_vfe', 'vfe'],
        'backbone_neck': ['backbone', 'neck', 'dense_backbone'],
        'center_head': ['head'],
        'other': [],
    }
    stats = {
        key: {'params': 0, 'trainable': 0, 'with_grad': 0, 'grad_norm_sum': 0.0, 'max_grad_norm': 0.0}
        for key in groups
    }

    for name, param in model.named_parameters():
        group = 'other'
        for candidate, needles in groups.items():
            if candidate == 'other':
                continue
            if any(needle in name for needle in needles):
                group = candidate
                break
        stat = stats[group]
        stat['params'] += int(param.numel())
        if param.requires_grad:
            stat['trainable'] += int(param.numel())
        if param.grad is not None:
            norm = float(param.grad.detach().float().norm().item())
            stat['with_grad'] += int(param.numel())
            stat['grad_norm_sum'] += norm
            stat['max_grad_norm'] = max(stat['max_grad_norm'], norm)
    return stats


def apply_trainable_filter(model, hypes, ignore_filter=False):
    keywords = hypes.get('train_params', {}).get('trainable_parameter_keywords', [])
    if ignore_filter or not keywords:
        return []
    for name, param in model.named_parameters():
        param.requires_grad_(any(keyword in name for keyword in keywords))
    return keywords


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attach_head_capture(model):
    captured = {}
    head_module = getattr(model, 'radar_head', None)
    if head_module is None:
        return captured, None

    def hook(_module, _inputs, output):
        if isinstance(output, dict) and 'radar_pred_dicts' in output:
            captured['radar_pred_dicts'] = output['radar_pred_dicts']

    return captured, head_module.register_forward_hook(hook)


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.score_thresholds.split(',') if x.strip()]
    output_dir = Path(args.output_dir)

    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_paths(hypes, args)
    override_postprocess(hypes, args)
    print(f"Dataset dir: {hypes.get('validate_dir') if args.split != 'train' else hypes.get('root_dir')}")

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
    load_state_dict_compatible(model, args.student_checkpoint, 'student')
    trainable_keywords = apply_trainable_filter(model, hypes, args.ignore_trainable_filter)
    if trainable_keywords:
        print(f'[grad] applied trainable filter: {trainable_keywords}')
    model.eval()
    captured_head, head_hook = attach_head_capture(model)

    teacher = None
    criterion = None
    if args.run_grad_check:
        teacher = create_teacher(hypes, args.teacher_checkpoint).to(device)
        teacher.eval()
        criterion = train_utils.create_loss(hypes)

    class_names = hypes['model']['args'].get('class_names', [])
    rows = []
    grad_rows = []
    valid_count = 0
    skipped_empty = 0

    for batch_idx, batch_data in enumerate(loader):
        if valid_count >= args.max_batches:
            break
        if batch_data is None:
            skipped_empty += 1
            print(f'[sample {batch_idx}] skipped empty/no-target batch')
            continue
        batch_data = train_utils.to_device(batch_data, device)

        with torch.no_grad():
            captured_head.clear()
            output = model(batch_data['ego'])

        final_dict = output.get('final_box_dict', [{}])[0]
        scores_tensor = final_dict.get('pred_scores', torch.zeros((0,), device=device))
        labels_tensor = final_dict.get('pred_labels', torch.zeros((0,), dtype=torch.long, device=device))
        scores = scores_tensor.detach().float().cpu().numpy()
        labels = labels_tensor.detach().long().cpu().numpy()

        row = {'sample': batch_idx, 'valid_index': valid_count}
        row.update(score_summary(scores, thresholds))
        pred_dicts = output.get('radar_pred_dicts', captured_head.get('radar_pred_dicts', []))
        row.update(head_heatmap_summary(pred_dicts, thresholds))
        for label in sorted(set(labels.tolist())):
            name = class_name(class_names, label)
            row[f'class_count_{name}'] = int(np.sum(labels == label))
        rows.append(row)

        print(
            f"[sample {batch_idx}] preds={row['pred_count']} "
            f"ge0.10={row.get('pred_ge_0.10', 0)} ge0.20={row.get('pred_ge_0.20', 0)} "
            f"max={row['score_max']:.3f} hm_max={row.get('all_hm_max', 0.0):.3f}"
        )

        if args.run_grad_check and not grad_rows:
            model.zero_grad(set_to_none=True)
            batch_data['ego']['compute_loss'] = True
            grad_output = model(batch_data['ego'])
            if grad_output is None:
                print(f'[sample {batch_idx}] skipped gradient check because model returned None')
                valid_count += 1
                continue
            with torch.no_grad():
                teacher_output = teacher(batch_data['ego'])
            merge_teacher_outputs(grad_output, teacher_output)
            loss = criterion(grad_output, batch_data['ego']['label_dict'])
            loss.backward()
            stats = summarize_gradients(model)
            for group, stat in stats.items():
                grad_row = {'group': group, 'loss': float(loss.detach().item())}
                grad_row.update(stat)
                grad_rows.append(grad_row)
        valid_count += 1

    write_csv(output_dir / 'centerhead_postprocess_summary.csv', rows)
    write_csv(output_dir / 'gradient_summary.csv', grad_rows)
    print(f'wrote {output_dir / "centerhead_postprocess_summary.csv"}')
    if grad_rows:
        print(f'wrote {output_dir / "gradient_summary.csv"}')
    if skipped_empty:
        print(f'skipped {skipped_empty} empty/no-target batches')
    if head_hook is not None:
        head_hook.remove()


if __name__ == '__main__':
    main()
