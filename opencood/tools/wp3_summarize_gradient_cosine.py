import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize WP3 gradient cosine debug outputs.')
    parser.add_argument('output_dirs', nargs='+')
    return parser.parse_args()


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_dir(output_dir):
    output_dir = Path(output_dir)
    cosine_path = output_dir / 'gradient_cosine_vs_radar_head.csv'
    attr_path = output_dir / 'gradient_attribution.csv'
    terms_path = output_dir / 'loss_terms.csv'
    if not cosine_path.exists():
        print(f'\n{output_dir}: missing {cosine_path.name}')
        return

    print(f'\n== {output_dir} ==')

    if terms_path.exists():
        term_rows = read_rows(terms_path)
        for key in [
            'radar_head_loss',
            'distill_total_loss',
            'object_proto_kd_loss',
            'semantic_heatmap_kd_loss',
            'proposal_error_kd_loss',
            'afd_loss',
            'pfd_loss',
            'proposal_fp_cell_mean',
            'proposal_fn_cell_mean',
        ]:
            vals = [float(row.get(key, 0.0) or 0.0) for row in term_rows]
            if vals and any(v != 0.0 for v in vals):
                print(f'{key}: {mean(vals):.6g}')

    cosine_rows = read_rows(cosine_path)
    grouped = defaultdict(list)
    for row in cosine_rows:
        loss = row['loss_name']
        group = row['group']
        cosine = float(row.get('cosine_vs_radar_head', 0.0) or 0.0)
        cur_norm = float(row.get('cur_grad_norm', 0.0) or 0.0)
        ref_norm = float(row.get('ref_grad_norm', 0.0) or 0.0)
        grouped[(loss, group)].append((cosine, cur_norm, ref_norm))

    print('loss,group,cosine_vs_radar_head,kd_grad_norm,radar_head_grad_norm')
    for (loss, group), vals in sorted(grouped.items()):
        cos = mean(v[0] for v in vals)
        cur = mean(v[1] for v in vals)
        ref = mean(v[2] for v in vals)
        if abs(cos) < 1e-6 and cur < 1e-6:
            continue
        print(f'{loss},{group},{cos:.6g},{cur:.6g},{ref:.6g}')

    if attr_path.exists():
        attr_rows = read_rows(attr_path)
        by_loss = defaultdict(list)
        for row in attr_rows:
            by_loss[row['loss_name']].append(float(row.get('grad_norm_sum', 0.0) or 0.0))
        print('grad_norm_sum_by_loss')
        for loss, vals in sorted(by_loss.items()):
            print(f'{loss}: {mean(vals):.6g}')


def main():
    args = parse_args()
    for output_dir in args.output_dirs:
        summarize_dir(output_dir)


if __name__ == '__main__':
    main()
