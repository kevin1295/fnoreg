import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('160_to_160', '160_to_80', '80_to_160', '80_to_80')
METRICS = ('dice', 'folding_percent', 'sdlogJ')


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize baseline, static-gate, and dynamic-gate results')
    parser.add_argument('--grid_dir', default='evaluations/afg_resolution_grid')
    parser.add_argument('--static_dir', default='evaluations/static_gate_resolution_grid')
    parser.add_argument('--output_dir', default='evaluations/static_gate_resolution_grid/comparison')
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2002)
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap(delta, samples, seed):
    clusters = np.asarray([idx if idx < 200 else 399 - idx for idx in range(len(delta))])
    unique_clusters = np.unique(clusters)
    cluster_values = {cluster: delta[clusters == cluster] for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        means[sample_idx] = np.concatenate([cluster_values[cluster] for cluster in selected]).mean()
    return np.percentile(means, [2.5, 97.5])


def load_frames(grid_dir, static_dir, setting):
    return {
        'baseline': pd.read_csv(grid_dir / setting / 'baseline' / 'per_pair_metrics.csv'),
        'static': pd.read_csv(static_dir / setting / 'per_pair_metrics.csv'),
        'dynamic': pd.read_csv(grid_dir / setting / 'afg' / 'per_pair_metrics.csv'),
    }


def build_combined_summary(grid_dir, static_dir):
    baseline_dynamic = pd.read_csv(grid_dir / 'summary.csv')
    static = pd.read_csv(static_dir / 'summary.csv')
    rows = []
    for setting in SETTINGS:
        for method in ('baseline', 'static', 'dynamic'):
            if method == 'static':
                row = static[static['setting'] == setting].iloc[0].to_dict()
            else:
                grid_method = 'afg' if method == 'dynamic' else 'baseline'
                row = baseline_dynamic[
                    (baseline_dynamic['setting'] == setting)
                    & (baseline_dynamic['method'] == grid_method)
                ].iloc[0].to_dict()
            row['method'] = method
            rows.append(row)
    return rows


def build_paired_summary(grid_dir, static_dir, samples, seed):
    rows = []
    comparisons = (
        ('static_minus_baseline', 'static', 'baseline'),
        ('dynamic_minus_static', 'dynamic', 'static'),
        ('dynamic_minus_baseline', 'dynamic', 'baseline'),
    )
    for setting_idx, setting in enumerate(SETTINGS):
        frames = load_frames(grid_dir, static_dir, setting)
        for comparison_idx, (name, left, right) in enumerate(comparisons):
            for metric_idx, metric in enumerate(METRICS):
                delta = (frames[left][metric] - frames[right][metric]).to_numpy()
                low, high = cluster_bootstrap(
                    delta,
                    samples,
                    seed + 10000 * setting_idx + 1000 * comparison_idx + 100 * metric_idx,
                )
                improved = delta > 0 if metric == 'dice' else delta < 0
                rows.append(
                    {
                        'setting': setting,
                        'comparison': name,
                        'metric': metric,
                        'mean_delta': float(delta.mean()),
                        'ci95_low': float(low),
                        'ci95_high': float(high),
                        'improved_fraction': float(improved.mean()),
                    }
                )
    return rows


def gate_statistics(static_dir):
    rows = []
    for exp_num, resolution in ((145, '160x192'), (146, '80x96')):
        state = torch.load(
            ROOT / f'experiments_fourier/oasis_exp{exp_num}/weights.pth',
            map_location='cpu',
        )
        gates = torch.stack(
            [
                torch.sigmoid(state[f'fno_blocks.{layer_idx}.spectral_conv.freq_gate'])
                for layer_idx in range(12)
            ]
        )
        rows.append(
            {
                'gate_source': 'independently_trained_static_gate',
                'train_resolution': resolution,
                'experiment': exp_num,
                'mean': float(gates.mean()),
                'std': float(gates.std(unbiased=False)),
                'min': float(gates.min()),
                'max': float(gates.max()),
            }
        )
    dynamic_mean = torch.load(
        ROOT / 'evaluations/afg_gate_ablation/80_to_160/mean_gates.pt',
        map_location='cpu',
    )
    rows.append(
        {
            'gate_source': 'dynamic_model_training_set_mean',
            'train_resolution': '80x96',
            'experiment': 148,
            'mean': float(dynamic_mean.mean()),
            'std': float(dynamic_mean.std(unbiased=False)),
            'min': float(dynamic_mean.min()),
            'max': float(dynamic_mean.max()),
        }
    )
    return rows


def main():
    args = parse_args()
    grid_dir = resolve(args.grid_dir)
    static_dir = resolve(args.static_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined_summary(grid_dir, static_dir)
    paired = build_paired_summary(
        grid_dir,
        static_dir,
        args.bootstrap_samples,
        args.seed,
    )
    gates = gate_statistics(static_dir)
    write_csv(output_dir / 'three_model_summary.csv', combined)
    write_csv(output_dir / 'paired_bootstrap.csv', paired)
    write_csv(output_dir / 'gate_parameter_statistics.csv', gates)
    with (output_dir / 'manifest.json').open('w') as file:
        json.dump(
            {
                'settings': list(SETTINGS),
                'bootstrap_samples': args.bootstrap_samples,
                'seed': args.seed,
                'static_training_resolution_assignment': {
                    '160x192': 145,
                    '80x96': 146,
                },
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f'Gate model comparison saved to {output_dir}')


if __name__ == '__main__':
    main()
