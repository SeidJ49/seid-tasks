import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description='WP3 loss-term gradient attribution for RadarDistill variants.')
    parser.add_argument('-y', '--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max_batches', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def override_split_paths(hypes, split):
    if split == 'val':
        hypes['validate_dir'] = hypes.get('validate_dir', hypes.get('root_dir'))
    elif split == 'test':
        hypes['validate_dir'] = hypes.get('test_dir', hypes.get('validate_dir'))


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
    print(f'loaded {len(compatible)} tensors from {checkpoint_path}')


def group_name(name):
    if name.startswith('radar_vfe'):
        return 'radar_vfe'
    if name.startswith('radar_backbone'):
        return 'radar_backbone'
    if name.startswith('radar_distill'):
        return 'radar_distill_bev'
    if name.startswith('radar_head.shared_conv'):
        return 'radar_head_shared'
    if name.startswith('radar_head.heads_list.0'):
        return 'radar_head_0_car'
    if name.startswith('radar_head.heads_list.1'):
        return 'radar_head_1_truck_other'
    if name.startswith('radar_head.heads_list.2'):
        return 'radar_head_2_bus_trailer'
    if name.startswith('radar_head.heads_list.3'):
        return 'radar_head_3_pedestrian'
    if name.startswith('radar_head.heads_list.4'):
        return 'radar_head_4_motor_bicycle'
    if name.startswith('radar_head'):
        return 'radar_head_other'
    if name.startswith(('lidar_vfe', 'teacher_backbone', 'teacher_bev_backbone', 'teacher_head')):
        return 'teacher_frozen'
    return 'other'


def summarize_gradients(model):
    rows = {}
    for name, param in model.named_parameters():
        group = group_name(name)
        row = rows.setdefault(group, {
            'params': 0,
            'trainable': 0,
            'with_grad': 0,
            'grad_norm_sum': 0.0,
            'max_grad_norm': 0.0,
        })
        row['params'] += int(param.numel())
        if param.requires_grad:
            row['trainable'] += int(param.numel())
        if param.grad is not None:
            norm = float(param.grad.detach().float().norm().item())
            row['with_grad'] += int(param.numel())
            row['grad_norm_sum'] += norm
            row['max_grad_norm'] = max(row['max_grad_norm'], norm)
    return rows


def collect_gradients(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().float().clone()
    return grads


def summarize_gradient_cosine(reference_grads, current_grads, model):
    rows = {}
    param_names = {name for name, _param in model.named_parameters()}
    for name in sorted(param_names):
        ref = reference_grads.get(name)
        cur = current_grads.get(name)
        if ref is None or cur is None:
            continue
        group = group_name(name)
        row = rows.setdefault(group, {
            'params_with_both_grad': 0,
            'dot': 0.0,
            'ref_norm_sq': 0.0,
            'cur_norm_sq': 0.0,
        })
        row['params_with_both_grad'] += int(ref.numel())
        row['dot'] += float((ref * cur).sum().item())
        row['ref_norm_sq'] += float((ref * ref).sum().item())
        row['cur_norm_sq'] += float((cur * cur).sum().item())

    for row in rows.values():
        denom = (row['ref_norm_sq'] ** 0.5) * (row['cur_norm_sq'] ** 0.5)
        row['cosine_vs_radar_head'] = row['dot'] / denom if denom > 0 else 0.0
        row['ref_grad_norm'] = row['ref_norm_sq'] ** 0.5
        row['cur_grad_norm'] = row['cur_norm_sq'] ** 0.5
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_loss_terms(model, batch):
    batch['ego']['compute_loss'] = True
    output = model(batch['ego'])
    radar_head_loss = output['radar_head_loss']
    distill_loss = output['distill_loss']
    tb = output.get('tb_dict', {})
    radar_distill = getattr(model, 'radar_distill', None)
    loss_terms = getattr(radar_distill, 'last_loss_terms', {}) if radar_distill is not None else {}
    object_kd_loss = loss_terms.get('object_kd', distill_loss.new_tensor(0.0))
    object_proto_kd_loss = loss_terms.get('object_proto_kd', distill_loss.new_tensor(0.0))
    semantic_heatmap_kd_loss = loss_terms.get('semantic_heatmap_kd', distill_loss.new_tensor(0.0))
    response_kd_loss = loss_terms.get('response_kd', distill_loss.new_tensor(0.0))
    proposal_error_kd_loss = loss_terms.get('proposal_error_kd', distill_loss.new_tensor(0.0))
    task_dense_feature_kd_loss = loss_terms.get('task_dense_feature_kd', distill_loss.new_tensor(0.0))
    learned_mask_loss = loss_terms.get('learned_mask', distill_loss.new_tensor(0.0))
    afd_loss = loss_terms.get('afd', distill_loss.new_tensor(0.0))
    pfd_loss = loss_terms.get('pfd', distill_loss.new_tensor(0.0))
    return {
        'radar_head': radar_head_loss,
        'distill_total': distill_loss,
        'object_kd': object_kd_loss,
        'object_proto_kd': object_proto_kd_loss,
        'semantic_heatmap_kd': semantic_heatmap_kd_loss,
        'response_kd': response_kd_loss,
        'proposal_error_kd': proposal_error_kd_loss,
        'task_dense_feature_kd': task_dense_feature_kd_loss,
        'learned_mask': learned_mask_loss,
        'afd': afd_loss,
        'pfd': pfd_loss,
        'total': output['loss'],
    }, tb


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    hypes = yaml_utils.load_yaml(args.hypes_yaml, args)
    override_split_paths(hypes, args.split)
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
    load_state_dict_compatible(model, args.checkpoint)
    model.train()

    rows = []
    cosine_rows = []
    term_rows = []
    valid = 0
    for batch_idx, batch in enumerate(loader):
        if valid >= args.max_batches:
            break
        if batch is None:
            continue
        batch = train_utils.to_device(batch, device)

        terms, tb = get_loss_terms(model, batch)
        term_rows.append({
            'sample': batch_idx,
            'radar_head_loss': float(terms['radar_head'].detach().item()),
            'distill_total_loss': float(terms['distill_total'].detach().item()),
            'object_kd_loss': float(terms['object_kd'].detach().item()),
            'object_proto_kd_loss': float(terms['object_proto_kd'].detach().item()),
            'semantic_heatmap_kd_loss': float(terms['semantic_heatmap_kd'].detach().item()),
            'response_kd_loss': float(terms['response_kd'].detach().item()),
            'proposal_error_kd_loss': float(terms['proposal_error_kd'].detach().item()),
            'task_dense_feature_kd_loss': float(terms['task_dense_feature_kd'].detach().item()),
            'learned_mask_loss': float(terms['learned_mask'].detach().item()),
            'afd_loss': float(terms['afd'].detach().item()),
            'pfd_loss': float(terms['pfd'].detach().item()),
            'total_loss': float(terms['total'].detach().item()),
            'object_proto_mask_mean': float(tb.get('object_proto_mask_mean', 0.0)),
            'object_proto_observability_mean': float(tb.get('object_proto_observability_mean', 0.0)),
            'semantic_heatmap_weight_mean': float(tb.get('semantic_heatmap_weight_mean', 0.0)),
            'response_pos_weight_mean': float(tb.get('response_pos_weight_mean', 0.0)),
            'response_neg_cell_mean': float(tb.get('response_neg_cell_mean', 0.0)),
            'task_dense_mask_mean': float(tb.get('task_dense_mask_mean', 0.0)),
            'task_dense_teacher_conf_mean': float(tb.get('task_dense_teacher_conf_mean', 0.0)),
            'learned_mask_prior_mean': float(tb.get('learned_mask_prior_mean', 0.0)),
            'learned_mask_pred_mean': float(tb.get('learned_mask_pred_mean', 0.0)),
            'learned_mask_kd_mean': float(tb.get('learned_mask_kd_mean', 0.0)),
            'proposal_fn_cell_mean': float(tb.get('proposal_fn_cell_mean', 0.0)),
            'proposal_fp_cell_mean': float(tb.get('proposal_fp_cell_mean', 0.0)),
            'proposal_fp_weight_mean': float(tb.get('proposal_fp_weight_mean', 0.0)),
            'object_kd_mask_mean': float(tb.get('object_kd_mask_mean', 0.0)),
            'radar_evidence_mask_mean': float(tb.get('radar_evidence_mask_mean', 0.0)),
            'lidar_intensity_mask_mean': float(tb.get('lidar_intensity_mask_mean', 0.0)),
        })

        loss_names = [
            'radar_head',
            'object_kd',
            'object_proto_kd',
            'semantic_heatmap_kd',
            'response_kd',
            'proposal_error_kd',
            'task_dense_feature_kd',
            'learned_mask',
            'afd',
            'pfd',
            'distill_total',
            'total',
        ]

        reference_grads = None
        for loss_name in loss_names:
            model.zero_grad(set_to_none=True)
            terms, _tb = get_loss_terms(model, batch)
            loss = terms[loss_name]
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            loss.backward()
            cur_grads = collect_gradients(model)
            if loss_name == 'radar_head':
                reference_grads = cur_grads
            grad_stats = summarize_gradients(model)
            for group, stat in grad_stats.items():
                row = {'sample': batch_idx, 'loss_name': loss_name, 'loss_value': float(loss.detach().item()), 'group': group}
                row.update(stat)
                rows.append(row)
            if reference_grads is not None and loss_name != 'radar_head':
                cosine_stats = summarize_gradient_cosine(reference_grads, cur_grads, model)
                for group, stat in cosine_stats.items():
                    row = {
                        'sample': batch_idx,
                        'loss_name': loss_name,
                        'loss_value': float(loss.detach().item()),
                        'group': group,
                    }
                    row.update(stat)
                    cosine_rows.append(row)
        valid += 1

    write_csv(output_dir / 'loss_terms.csv', term_rows)
    write_csv(output_dir / 'gradient_attribution.csv', rows)
    write_csv(output_dir / 'gradient_cosine_vs_radar_head.csv', cosine_rows)
    print(f'wrote {output_dir / "loss_terms.csv"}')
    print(f'wrote {output_dir / "gradient_attribution.csv"}')
    print(f'wrote {output_dir / "gradient_cosine_vs_radar_head.csv"}')


if __name__ == '__main__':
    main()
