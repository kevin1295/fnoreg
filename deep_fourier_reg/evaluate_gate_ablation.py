import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import dataloaders
import utils
from evaluate_oasis import load_model, load_weights, resolve_path
from models import SpatialTransform


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODES = ('normal', 'identity', 'mean', 'shuffled')
METRICS = ('dice', 'folding_percent', 'sdlogJ')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate inference-time AFG gate interventions')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--config_file', default='params.json')
    parser.add_argument('--exp_num', type=int, default=148)
    parser.add_argument('--ckpt_epoch', type=int, default=-1)
    parser.add_argument('--size', type=int, default=160)
    parser.add_argument('--calibration_size', type=int, default=80)
    parser.add_argument('--calibration_pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2002)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--output_dir', default='evaluations/afg_gate_ablation/80_to_160')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--allow_cpu', action='store_true')
    return parser.parse_args()


def write_json(path, value):
    with path.open('w') as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f'Cannot write empty CSV: {path}')
    if fieldnames is None:
        fieldnames = rows[0].keys()
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validation_pair_indices(pair_idx, subject_count):
    if pair_idx < subject_count - 1:
        return pair_idx, pair_idx + 1
    return (subject_count - 1) * 2 - pair_idx, (subject_count - 1) * 2 - pair_idx - 1


def training_pair_indices(pair_idx, subject_count):
    moving_idx = pair_idx // subject_count
    remainder = pair_idx % subject_count
    fixed_idx = remainder if remainder != moving_idx else remainder + 1
    return moving_idx, fixed_idx


def collect_mean_gates(model, dataset, sample_indices, device, height, width, quiet):
    model.set_gate_overrides(None)
    gate_sums = None
    resize = transforms.Resize(size=(height, width))
    iterator = tqdm(sample_indices, desc='Calibrating mean gates', ncols=100, disable=quiet)
    with torch.no_grad():
        for dataset_idx in iterator:
            moving, fixed = dataset[int(dataset_idx)]
            moving = resize(torch.as_tensor(moving).unsqueeze(0)).to(device).float()
            fixed = resize(torch.as_tensor(fixed).unsqueeze(0)).to(device).float()
            model(moving, fixed)
            gates = model.get_last_gates()
            if gate_sums is None:
                gate_sums = [gate.double() for gate in gates]
            else:
                for layer_idx, gate in enumerate(gates):
                    gate_sums[layer_idx].add_(gate.double())
    model.set_gate_overrides(None)
    return torch.stack([gate_sum.div(len(sample_indices)).float() for gate_sum in gate_sums])


def build_shuffle_mapping(pair_subject_indices, seed):
    pair_count = len(pair_subject_indices)
    rng = np.random.default_rng(seed)
    cost = np.empty((pair_count, pair_count), dtype=np.float64)
    for target_idx, target_subjects in enumerate(pair_subject_indices):
        target_set = set(target_subjects)
        for donor_idx, donor_subjects in enumerate(pair_subject_indices):
            self_penalty = 100.0 if target_idx == donor_idx else 0.0
            overlap_penalty = 1.0 if target_set.intersection(donor_subjects) else 0.0
            cost[target_idx, donor_idx] = self_penalty + overlap_penalty + rng.random() * 1e-6
    target_indices, donor_indices = linear_sum_assignment(cost)
    mapping = np.empty(pair_count, dtype=np.int64)
    mapping[target_indices] = donor_indices
    return mapping


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate_mode(
    model,
    mode,
    test_loader,
    test_dataset,
    transform,
    device,
    test_height,
    test_width,
    mean_gates,
    normal_gates,
    shuffle_mapping,
    max_pairs,
    quiet,
):
    records = []
    collected_gates = []
    subject_count = test_dataset.n
    resize = transforms.Resize(size=(test_height, test_width))
    resize_labels = transforms.Resize(
        size=(test_height, test_width),
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    iterator = tqdm(test_loader, desc=f'Evaluating {mode}', ncols=100, disable=quiet)
    with torch.no_grad():
        for pair_idx, (moving, fixed, moving_labels, fixed_labels) in enumerate(iterator):
            if max_pairs is not None and pair_idx >= max_pairs:
                break
            moving = resize(moving.to(device).float())
            fixed = resize(fixed.to(device).float())
            moving_labels = resize_labels(moving_labels.to(device).float())
            fixed_labels = resize_labels(fixed_labels.to(device).float())

            if mode == 'normal':
                model.set_gate_overrides(None)
            elif mode == 'identity':
                model.set_gate_overrides([torch.ones_like(gate) for gate in mean_gates])
            elif mode == 'mean':
                model.set_gate_overrides(list(mean_gates))
            elif mode == 'shuffled':
                donor_idx = int(shuffle_mapping[pair_idx])
                model.set_gate_overrides(list(normal_gates[donor_idx]))
            else:
                raise ValueError(f'Unsupported gate mode: {mode}')

            synchronize(device)
            start_time = time.perf_counter()
            flow = model(moving, fixed)
            synchronize(device)
            inference_time = time.perf_counter() - start_time

            if mode == 'normal':
                collected_gates.append(torch.stack(model.get_last_gates()))

            flow_pixels = flow.clone()
            flow_pixels[:, 0] *= (test_height - 1) / 2
            flow_pixels[:, 1] *= (test_width - 1) / 2
            jacobian = utils.jacobian_determinant(flow_pixels.cpu().numpy().squeeze(0))
            folding_percent = 100 * np.mean(jacobian < 0)
            positive_jacobian = jacobian[jacobian > 0]
            sdlogj = float(np.std(np.log(positive_jacobian))) if positive_jacobian.size else float('nan')

            _, warped_labels = transform(
                moving_labels,
                flow.permute(0, 2, 3, 1),
                mod='nearest',
            )
            warped_label_np = warped_labels[0].long().cpu().numpy()
            fixed_label_np = fixed_labels[0].long().cpu().numpy()
            moving_label_np = moving_labels[0].long().cpu().numpy()
            moving_idx, fixed_idx = validation_pair_indices(pair_idx, subject_count)
            moving_subject = str(test_dataset.folders[moving_idx])
            fixed_subject = str(test_dataset.folders[fixed_idx])
            cluster_id = '|'.join(sorted((moving_subject, fixed_subject)))
            records.append(
                {
                    'pair_idx': pair_idx,
                    'moving_subject': moving_subject,
                    'fixed_subject': fixed_subject,
                    'cluster_id': cluster_id,
                    'initial_dice': float(utils.dice(moving_label_np, fixed_label_np)),
                    'dice': float(utils.dice(warped_label_np, fixed_label_np)),
                    'folding_percent': float(folding_percent),
                    'sdlogJ': sdlogj,
                    'inference_time_seconds': float(inference_time),
                }
            )
    model.set_gate_overrides(None)
    gates = torch.stack(collected_gates) if collected_gates else None
    return records, gates


def collect_test_gates(model, test_loader, device, height, width, max_pairs, quiet):
    model.set_gate_overrides(None)
    resize = transforms.Resize(size=(height, width))
    collected_gates = []
    iterator = tqdm(test_loader, desc=f'Collecting gates at {height}x{width}', ncols=100, disable=quiet)
    with torch.no_grad():
        for pair_idx, (moving, fixed, _, _) in enumerate(iterator):
            if max_pairs is not None and pair_idx >= max_pairs:
                break
            moving = resize(moving.to(device).float())
            fixed = resize(fixed.to(device).float())
            model(moving, fixed)
            collected_gates.append(torch.stack(model.get_last_gates()))
    model.set_gate_overrides(None)
    return torch.stack(collected_gates)


def summarize_records(mode, records, weights_path, test_height, test_width):
    summary = {
        'mode': mode,
        'weights': str(weights_path),
        'test_resolution': [test_height, test_width],
        'num_pairs': len(records),
        'mean_initial_dice': float(np.mean([row['initial_dice'] for row in records])),
        'mean_dice': float(np.mean([row['dice'] for row in records])),
        'std_dice': float(np.std([row['dice'] for row in records])),
        'mean_folding_percent': float(np.mean([row['folding_percent'] for row in records])),
        'std_folding_percent': float(np.std([row['folding_percent'] for row in records])),
        'mean_sdlogJ': float(np.nanmean([row['sdlogJ'] for row in records])),
        'std_sdlogJ': float(np.nanstd([row['sdlogJ'] for row in records])),
        'mean_inference_time_seconds': float(np.mean([row['inference_time_seconds'] for row in records])),
        'jacobian_scaling': {
            'flow_channel_0': f'(H - 1) / 2 = {(test_height - 1) / 2}',
            'flow_channel_1': f'(W - 1) / 2 = {(test_width - 1) / 2}',
        },
    }
    return summary


def cluster_bootstrap_interval(differences, cluster_ids, samples, seed):
    unique_clusters = np.unique(cluster_ids)
    cluster_values = {
        cluster: differences[cluster_ids == cluster]
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        bootstrap_means[sample_idx] = np.mean(
            np.concatenate([cluster_values[cluster] for cluster in selected])
        )
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def compare_modes(all_records, bootstrap_samples, seed):
    normal_by_pair = {row['pair_idx']: row for row in all_records['normal']}
    paired_rows = []
    aggregate_rows = []
    for mode, records in all_records.items():
        summary_values = {
            'mode': mode,
            'num_pairs': len(records),
            'mean_dice': float(np.mean([row['dice'] for row in records])),
            'mean_folding_percent': float(np.mean([row['folding_percent'] for row in records])),
            'mean_sdlogJ': float(np.nanmean([row['sdlogJ'] for row in records])),
        }
        if mode == 'normal':
            for metric in METRICS:
                summary_values[f'mean_delta_{metric}'] = 0.0
                summary_values[f'ci95_low_delta_{metric}'] = 0.0
                summary_values[f'ci95_high_delta_{metric}'] = 0.0
                summary_values[f'improved_fraction_{metric}'] = 0.0
            aggregate_rows.append(summary_values)
            continue

        mode_differences = {metric: [] for metric in METRICS}
        cluster_ids = []
        for row in records:
            normal_row = normal_by_pair[row['pair_idx']]
            cluster_ids.append(row['cluster_id'])
            paired_row = {
                'mode': mode,
                'pair_idx': row['pair_idx'],
                'moving_subject': row['moving_subject'],
                'fixed_subject': row['fixed_subject'],
                'cluster_id': row['cluster_id'],
            }
            for metric in METRICS:
                difference = row[metric] - normal_row[metric]
                improved = difference > 0 if metric == 'dice' else difference < 0
                mode_differences[metric].append(difference)
                paired_row[f'normal_{metric}'] = normal_row[metric]
                paired_row[f'intervention_{metric}'] = row[metric]
                paired_row[f'delta_{metric}'] = difference
                paired_row[f'improved_{metric}'] = int(improved)
            paired_rows.append(paired_row)

        cluster_ids = np.asarray(cluster_ids)
        for metric_idx, metric in enumerate(METRICS):
            differences = np.asarray(mode_differences[metric], dtype=np.float64)
            low, high = cluster_bootstrap_interval(
                differences,
                cluster_ids,
                bootstrap_samples,
                seed + 100 * metric_idx + 1000 * MODES.index(mode),
            )
            if metric == 'dice':
                improved = differences > 0
            else:
                improved = differences < 0
            summary_values[f'mean_delta_{metric}'] = float(np.mean(differences))
            summary_values[f'ci95_low_delta_{metric}'] = low
            summary_values[f'ci95_high_delta_{metric}'] = high
            summary_values[f'improved_fraction_{metric}'] = float(np.mean(improved))
        aggregate_rows.append(summary_values)
    return paired_rows, aggregate_rows


def build_shuffle_rows(mapping, pair_subject_indices, test_dataset):
    rows = []
    for target_idx, donor_idx in enumerate(mapping):
        target_subject_indices = pair_subject_indices[target_idx]
        donor_subject_indices = pair_subject_indices[int(donor_idx)]
        target_subjects = [str(test_dataset.folders[idx]) for idx in target_subject_indices]
        donor_subjects = [str(test_dataset.folders[idx]) for idx in donor_subject_indices]
        rows.append(
            {
                'target_pair_idx': target_idx,
                'donor_pair_idx': int(donor_idx),
                'target_moving_subject': target_subjects[0],
                'target_fixed_subject': target_subjects[1],
                'donor_moving_subject': donor_subjects[0],
                'donor_fixed_subject': donor_subjects[1],
                'shares_subject': int(bool(set(target_subjects).intersection(donor_subjects))),
            }
        )
    return rows


def summarize_gate_variation(mean_gates, normal_gates, shuffle_mapping):
    test_mean_gates = normal_gates.mean(dim=0)
    test_std_gates = normal_gates.std(dim=0, unbiased=False)
    shuffled_gates = normal_gates[torch.as_tensor(shuffle_mapping, dtype=torch.long)]
    rows = []
    for layer_idx in range(normal_gates.shape[1]):
        calibration_gate = mean_gates[layer_idx]
        test_mean_gate = test_mean_gates[layer_idx]
        test_std_gate = test_std_gates[layer_idx]
        matched_gates = normal_gates[:, layer_idx]
        donor_gates = shuffled_gates[:, layer_idx]
        rows.append(
            {
                'layer': layer_idx + 1,
                'calibration_gate_mean': float(calibration_gate.mean()),
                'calibration_gate_min': float(calibration_gate.min()),
                'calibration_gate_max': float(calibration_gate.max()),
                'test_gate_mean': float(test_mean_gate.mean()),
                'test_gate_min': float(matched_gates.min()),
                'test_gate_max': float(matched_gates.max()),
                'test_between_sample_std_mean': float(test_std_gate.mean()),
                'test_between_sample_std_max': float(test_std_gate.max()),
                'calibration_test_mean_abs_difference': float(
                    (calibration_gate - test_mean_gate).abs().mean()
                ),
                'test_mean_abs_distance_from_identity': float(
                    (1 - matched_gates).abs().mean()
                ),
                'matched_shuffled_mean_abs_difference': float(
                    (matched_gates - donor_gates).abs().mean()
                ),
            }
        )
    return rows


def summarize_resolution_gate_shift(low_resolution_gates, full_resolution_gates):
    rows = []
    for layer_idx in range(full_resolution_gates.shape[1]):
        low_gates = low_resolution_gates[:, layer_idx]
        full_gates = full_resolution_gates[:, layer_idx]
        rows.append(
            {
                'layer': layer_idx + 1,
                'low_resolution_gate_mean': float(low_gates.mean()),
                'full_resolution_gate_mean': float(full_gates.mean()),
                'same_sample_mean_abs_difference': float((low_gates - full_gates).abs().mean()),
                'low_resolution_between_sample_std_mean': float(
                    low_gates.std(dim=0, unbiased=False).mean()
                ),
                'full_resolution_between_sample_std_mean': float(
                    full_gates.std(dim=0, unbiased=False).mean()
                ),
            }
        )
    return rows


def git_value(*arguments):
    try:
        return subprocess.run(
            ['git', *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_analysis_summary(path, aggregate_rows):
    lines = [
        '# AFG gate intervention summary',
        '',
        'All deltas are intervention minus normal. Dice is better when positive; folding rate and sdlogJ are better when negative.',
        '',
        '| Mode | Dice | Folding rate (%) | sdlogJ |',
        '|---|---:|---:|---:|',
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['mode']} | {row['mean_dice']:.6f} | "
            f"{row['mean_folding_percent']:.6f} | {row['mean_sdlogJ']:.6f} |"
        )
    lines.extend(['', '## Paired changes', ''])
    for row in aggregate_rows:
        if row['mode'] == 'normal':
            continue
        lines.append(f"### {row['mode']}")
        for metric in METRICS:
            lines.append(
                f"- {metric}: mean delta {row[f'mean_delta_{metric}']:.6f}, "
                f"cluster-bootstrap 95% CI "
                f"[{row[f'ci95_low_delta_{metric}']:.6f}, "
                f"{row[f'ci95_high_delta_{metric}']:.6f}], "
                f"improved fraction {row[f'improved_fraction_{metric}']:.3f}."
            )
        lines.append('')
    path.write_text('\n'.join(lines))


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA is unavailable; pass --allow_cpu only for a small smoke test')

    config_path = resolve_path(args.config_file, SCRIPT_DIR)
    params = pd.read_json(config_path)
    config_dir = config_path.parent
    oasis_folders_path = resolve_path(params['oasis_folders_path'][0], config_dir)
    oasis_path = resolve_path(params['oasis_path'][0], config_dir)
    weights_root = resolve_path(params['weights_path'][0], config_dir)
    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    with oasis_folders_path.open() as file:
        oasis_folders = [line.strip() for line in file]
    train_dataset = dataloaders.TrainOasis2d(range(0, 201), str(oasis_path), oasis_folders)
    test_dataset = dataloaders.ValidationOasis2d(range(213, 414), str(oasis_path), oasis_folders)

    exp_folder = weights_root / f'oasis_exp{args.exp_num}'
    with (exp_folder / 'metadata.json').open() as file:
        exp_metadata = json.load(file)
    if exp_metadata['model_name'] != 'adaptive_freq_gated_convfno':
        raise ValueError('Gate ablation requires an adaptive_freq_gated_convfno checkpoint')
    model = load_model(exp_metadata['model_name'], exp_metadata['model_config'], device)
    weights_path = load_weights(model, exp_folder, args.ckpt_epoch, device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    calibration_count = min(args.calibration_pairs, len(train_dataset))
    calibration_indices = rng.choice(len(train_dataset), size=calibration_count, replace=False)
    calibration_height = args.calibration_size
    calibration_width = int(round(192 / 160 * calibration_height))
    mean_gates = collect_mean_gates(
        model,
        train_dataset,
        calibration_indices,
        device,
        calibration_height,
        calibration_width,
        args.quiet,
    )
    torch.save(mean_gates, output_dir / 'mean_gates.pt')

    calibration_rows = []
    for dataset_idx in calibration_indices:
        moving_idx, fixed_idx = training_pair_indices(int(dataset_idx), train_dataset.n)
        calibration_rows.append(
            {
                'dataset_idx': int(dataset_idx),
                'moving_index': moving_idx,
                'fixed_index': fixed_idx,
                'moving_subject': str(train_dataset.folders[moving_idx]),
                'fixed_subject': str(train_dataset.folders[fixed_idx]),
            }
        )
    write_json(
        output_dir / 'calibration_indices.json',
        {
            'seed': args.seed,
            'requested_pairs': args.calibration_pairs,
            'actual_pairs': calibration_count,
            'resolution': [calibration_height, calibration_width],
            'samples': calibration_rows,
        },
    )

    test_pair_count = len(test_dataset) if args.max_pairs is None else min(args.max_pairs, len(test_dataset))
    pair_subject_indices = [
        validation_pair_indices(pair_idx, test_dataset.n)
        for pair_idx in range(test_pair_count)
    ]
    shuffle_mapping = build_shuffle_mapping(pair_subject_indices, args.seed)
    shuffle_rows = build_shuffle_rows(shuffle_mapping, pair_subject_indices, test_dataset)
    write_csv(output_dir / 'shuffle_mapping.csv', shuffle_rows)

    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    transform = SpatialTransform().to(device)
    test_height = args.size
    test_width = int(round(192 / 160 * test_height))
    all_records = {}
    summaries = []
    normal_gates = None
    for mode in MODES:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        records, collected_gates = evaluate_mode(
            model,
            mode,
            test_loader,
            test_dataset,
            transform,
            device,
            test_height,
            test_width,
            mean_gates,
            normal_gates,
            shuffle_mapping,
            args.max_pairs,
            args.quiet,
        )
        if mode == 'normal':
            normal_gates = collected_gates
            torch.save(normal_gates, output_dir / 'normal_gates.pt')
        all_records[mode] = records
        summary = summarize_records(mode, records, weights_path, test_height, test_width)
        summaries.append(summary)
        write_csv(mode_dir / 'per_pair_metrics.csv', records)
        write_json(mode_dir / 'summary.json', summary)

    paired_rows, aggregate_rows = compare_modes(all_records, args.bootstrap_samples, args.seed)
    write_csv(output_dir / 'paired_differences.csv', paired_rows)
    write_csv(output_dir / 'aggregate_summary.csv', aggregate_rows)
    write_json(output_dir / 'aggregate_summary.json', aggregate_rows)
    gate_statistics = summarize_gate_variation(mean_gates, normal_gates, shuffle_mapping)
    write_csv(output_dir / 'gate_statistics.csv', gate_statistics)
    write_json(output_dir / 'gate_statistics.json', gate_statistics)
    low_resolution_test_gates = collect_test_gates(
        model,
        test_loader,
        device,
        calibration_height,
        calibration_width,
        args.max_pairs,
        args.quiet,
    )
    torch.save(low_resolution_test_gates, output_dir / 'test_gates_80x96.pt')
    resolution_gate_statistics = summarize_resolution_gate_shift(
        low_resolution_test_gates,
        normal_gates,
    )
    write_csv(output_dir / 'resolution_gate_statistics.csv', resolution_gate_statistics)
    write_json(output_dir / 'resolution_gate_statistics.json', resolution_gate_statistics)
    write_analysis_summary(output_dir / 'analysis_summary.md', aggregate_rows)

    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'experiment': args.exp_num,
        'model_name': exp_metadata['model_name'],
        'weights': str(weights_path),
        'modes': list(MODES),
        'test_resolution': [test_height, test_width],
        'test_pairs': test_pair_count,
        'calibration_resolution': [calibration_height, calibration_width],
        'calibration_pairs': calibration_count,
        'seed': args.seed,
        'bootstrap_samples': args.bootstrap_samples,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'git_commit': git_value('rev-parse', 'HEAD'),
        'git_status': git_value('status', '--short'),
        'code_sha256': {
            'main.py': sha256_file(REPO_ROOT / 'main.py'),
            'deep_fourier_reg/fno.py': sha256_file(SCRIPT_DIR / 'fno.py'),
            'deep_fourier_reg/evaluate_gate_ablation.py': sha256_file(Path(__file__)),
        },
    }
    write_json(output_dir / 'run_manifest.json', manifest)
    print(json.dumps(aggregate_rows, indent=2, ensure_ascii=False))
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
