import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run_script(script, arguments):
    command = [sys.executable, str(REPO_ROOT / script), *arguments]
    print('$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def add_common_gpu_argument(parser):
    parser.add_argument('--gpu_num', type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(description='Unified FNOReg experiment entry point')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train = subparsers.add_parser('train', help='Train or resume a 2D OASIS experiment')
    add_common_gpu_argument(train)
    train.add_argument('--size', type=int, default=160)
    train.add_argument('--exp_num', type=int, default=-1)
    train.add_argument('--ckpt_epoch', type=int, default=-1)
    train.add_argument('--config_file', default='params.json')

    evaluate = subparsers.add_parser('evaluate', help='Evaluate one experiment')
    add_common_gpu_argument(evaluate)
    evaluate.add_argument('--exp_num', type=int, required=True)
    evaluate.add_argument('--size', type=int, default=None)
    evaluate.add_argument('--ckpt_epoch', type=int, default=-1)
    evaluate.add_argument('--config_file', default='params.json')
    evaluate.add_argument('--output_dir', default=None)
    evaluate.add_argument('--num_workers', type=int, default=0)
    evaluate.add_argument('--no_visuals', action='store_true')
    evaluate.add_argument('--quiet', action='store_true')

    visualize = subparsers.add_parser('visualize', help='Visualize validation pairs and gates')
    add_common_gpu_argument(visualize)
    visualize.add_argument('--exp_num', type=int, required=True)
    visualize.add_argument('--num_pairs', type=int, default=5)
    visualize.add_argument('--size', type=int, default=None)
    visualize.add_argument('--no_gate_plot', action='store_true')

    compare = subparsers.add_parser('compare', help='Compare baseline and AFG-FNOReg')
    add_common_gpu_argument(compare)
    compare.add_argument('--exp_fnoreg', type=int, default=143)
    compare.add_argument('--exp_ours', type=int, default=148)
    compare.add_argument('--size', type=int, default=None)
    compare.add_argument('--max_pairs', type=int, default=None)

    grid = subparsers.add_parser('evaluate-grid', help='Evaluate the complete AFG resolution grid')
    add_common_gpu_argument(grid)
    grid.add_argument('--baseline_full', type=int, default=141)
    grid.add_argument('--baseline_half', type=int, default=143)
    grid.add_argument('--afg_full', type=int, default=147)
    grid.add_argument('--afg_half', type=int, default=148)
    grid.add_argument('--output_dir', default='evaluations/afg_resolution_grid')
    grid.add_argument('--num_workers', type=int, default=0)
    grid.add_argument('--reuse', action='store_true', help='Reuse existing per-run summary files')

    gate_ablation = subparsers.add_parser(
        'evaluate-gate-ablation',
        help='Evaluate inference-time AFG gate interventions',
    )
    add_common_gpu_argument(gate_ablation)
    gate_ablation.add_argument('--exp_num', type=int, default=148)
    gate_ablation.add_argument('--ckpt_epoch', type=int, default=-1)
    gate_ablation.add_argument('--size', type=int, default=160)
    gate_ablation.add_argument('--calibration_size', type=int, default=80)
    gate_ablation.add_argument('--calibration_pairs', type=int, default=1000)
    gate_ablation.add_argument('--seed', type=int, default=2002)
    gate_ablation.add_argument('--bootstrap_samples', type=int, default=10000)
    gate_ablation.add_argument('--output_dir', default='evaluations/afg_gate_ablation/80_to_160')
    gate_ablation.add_argument('--num_workers', type=int, default=0)
    gate_ablation.add_argument('--max_pairs', type=int, default=None)
    gate_ablation.add_argument('--quiet', action='store_true')
    gate_ablation.add_argument('--allow_cpu', action='store_true')

    static_grid = subparsers.add_parser(
        'evaluate-static-grid',
        help='Evaluate the static frequency-gated resolution grid',
    )
    add_common_gpu_argument(static_grid)
    static_grid.add_argument('--static_full', type=int, default=145)
    static_grid.add_argument('--static_half', type=int, default=146)
    static_grid.add_argument('--output_dir', default='evaluations/static_gate_resolution_grid')
    static_grid.add_argument('--num_workers', type=int, default=0)
    static_grid.add_argument('--reuse', action='store_true')

    subparsers.add_parser('paper-figures', help='Regenerate paper figure assets')
    return parser


def optional_argument(arguments, name, value):
    if value is not None:
        arguments.extend([name, str(value)])


def run_train(args):
    arguments = [
        '--gpu_num', str(args.gpu_num),
        '--size', str(args.size),
        '--config_file', args.config_file,
    ]
    if args.exp_num >= 0:
        arguments.extend(['--exp_num', str(args.exp_num)])
    if args.ckpt_epoch >= 0:
        arguments.extend(['--ckpt_epoch', str(args.ckpt_epoch)])
    run_script('deep_fourier_reg/train_oasis.py', arguments)


def run_evaluate(args, output_dir=None):
    arguments = [
        '--gpu_num', str(args.gpu_num),
        '--exp_num', str(args.exp_num),
        '--config_file', args.config_file,
        '--num_workers', str(args.num_workers),
    ]
    optional_argument(arguments, '--size', args.size)
    optional_argument(arguments, '--output_dir', output_dir or args.output_dir)
    if args.ckpt_epoch >= 0:
        arguments.extend(['--ckpt_epoch', str(args.ckpt_epoch)])
    if args.no_visuals:
        arguments.append('--no_visuals')
    if args.quiet:
        arguments.append('--quiet')
    run_script('deep_fourier_reg/evaluate_oasis.py', arguments)


def run_visualize(args):
    arguments = [
        '--gpu_num', str(args.gpu_num),
        '--exp_num', str(args.exp_num),
        '--num_pairs', str(args.num_pairs),
    ]
    optional_argument(arguments, '--size', args.size)
    if args.no_gate_plot:
        arguments.append('--no_gate_plot')
    run_script('deep_fourier_reg/visualize_validation.py', arguments)


def run_compare(args):
    arguments = [
        '--gpu_num', str(args.gpu_num),
        '--exp_fnoreg', str(args.exp_fnoreg),
        '--exp_ours', str(args.exp_ours),
    ]
    optional_argument(arguments, '--size', args.size)
    optional_argument(arguments, '--max_pairs', args.max_pairs)
    run_script('deep_fourier_reg/compare_models.py', arguments)


def run_grid(args):
    output_root = (REPO_ROOT / args.output_dir).resolve()
    runs = [
        ('160_to_160', 'baseline', args.baseline_full, 160, 160),
        ('160_to_160', 'afg', args.afg_full, 160, 160),
        ('160_to_80', 'baseline', args.baseline_full, 160, 80),
        ('160_to_80', 'afg', args.afg_full, 160, 80),
        ('80_to_160', 'baseline', args.baseline_half, 80, 160),
        ('80_to_160', 'afg', args.afg_half, 80, 160),
        ('80_to_80', 'baseline', args.baseline_half, 80, 80),
        ('80_to_80', 'afg', args.afg_half, 80, 80),
    ]
    summaries = []
    for setting, method, exp_num, train_size, test_size in runs:
        run_output = output_root / setting / method
        evaluate_args = argparse.Namespace(
            gpu_num=args.gpu_num,
            exp_num=exp_num,
            size=test_size,
            ckpt_epoch=-1,
            config_file='params.json',
            output_dir=str(run_output),
            num_workers=args.num_workers,
            no_visuals=True,
            quiet=True,
        )
        if not args.reuse or not (run_output / 'summary.json').exists():
            run_evaluate(evaluate_args)
        with (run_output / 'summary.json').open() as file:
            summary = json.load(file)
        summary.update(
            {
                'setting': setting,
                'method': method,
                'train_resolution': f'{train_size}x{int(train_size * 192 / 160)}',
                'test_resolution': f'{test_size}x{int(test_size * 192 / 160)}',
            }
        )
        summaries.append(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / 'summary.json').open('w') as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)
    columns = [
        'setting',
        'method',
        'experiment',
        'model_name',
        'train_resolution',
        'test_resolution',
        'mean_dice',
        'std_dice',
        'mean_folding_percent',
        'std_folding_percent',
        'mean_sdlogJ',
        'std_sdlogJ',
        'mean_inference_time_seconds',
    ]
    with (output_root / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    print(f'Grid summary saved to {output_root}')


def run_gate_ablation(args):
    arguments = [
        '--gpu_num', str(args.gpu_num),
        '--exp_num', str(args.exp_num),
        '--size', str(args.size),
        '--calibration_size', str(args.calibration_size),
        '--calibration_pairs', str(args.calibration_pairs),
        '--seed', str(args.seed),
        '--bootstrap_samples', str(args.bootstrap_samples),
        '--output_dir', args.output_dir,
        '--num_workers', str(args.num_workers),
    ]
    if args.ckpt_epoch >= 0:
        arguments.extend(['--ckpt_epoch', str(args.ckpt_epoch)])
    optional_argument(arguments, '--max_pairs', args.max_pairs)
    if args.quiet:
        arguments.append('--quiet')
    if args.allow_cpu:
        arguments.append('--allow_cpu')
    run_script('deep_fourier_reg/evaluate_gate_ablation.py', arguments)


def run_static_grid(args):
    output_root = (REPO_ROOT / args.output_dir).resolve()
    runs = [
        ('160_to_160', args.static_full, 160, 160),
        ('160_to_80', args.static_full, 160, 80),
        ('80_to_160', args.static_half, 80, 160),
        ('80_to_80', args.static_half, 80, 80),
    ]
    summaries = []
    for setting, exp_num, train_size, test_size in runs:
        run_output = output_root / setting
        evaluate_args = argparse.Namespace(
            gpu_num=args.gpu_num,
            exp_num=exp_num,
            size=test_size,
            ckpt_epoch=-1,
            config_file='params.json',
            output_dir=str(run_output),
            num_workers=args.num_workers,
            no_visuals=True,
            quiet=True,
        )
        if not args.reuse or not (run_output / 'summary.json').exists():
            run_evaluate(evaluate_args)
        with (run_output / 'summary.json').open() as file:
            summary = json.load(file)
        summary.update(
            {
                'setting': setting,
                'method': 'static_frequency_gate',
                'train_resolution': f'{train_size}x{int(train_size * 192 / 160)}',
                'test_resolution': f'{test_size}x{int(test_size * 192 / 160)}',
            }
        )
        summaries.append(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / 'summary.json').open('w') as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)
    columns = [
        'setting',
        'method',
        'experiment',
        'model_name',
        'train_resolution',
        'test_resolution',
        'mean_dice',
        'std_dice',
        'mean_folding_percent',
        'std_folding_percent',
        'mean_sdlogJ',
        'std_sdlogJ',
        'mean_inference_time_seconds',
    ]
    with (output_root / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    assignment = {
        'static_full_experiment': args.static_full,
        'static_half_experiment': args.static_half,
        'assignment_basis': (
            'Training timestamps show about 70 minutes per 10 epochs for the full-resolution run '
            'and about 31 minutes per 10 epochs for the half-resolution run; the latter metadata '
            'description appears to be stale.'
        ),
    }
    with (output_root / 'training_resolution_assignment.json').open('w') as file:
        json.dump(assignment, file, indent=2, ensure_ascii=False)
    run_script(
        'scripts/summarize_static_gate_results.py',
        [
            '--grid_dir', 'evaluations/afg_resolution_grid',
            '--static_dir', args.output_dir,
            '--output_dir', f'{args.output_dir}/comparison',
        ],
    )
    print(f'Static gate grid summary saved to {output_root}')


def main():
    args = build_parser().parse_args()
    if args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'visualize':
        run_visualize(args)
    elif args.command == 'compare':
        run_compare(args)
    elif args.command == 'evaluate-grid':
        run_grid(args)
    elif args.command == 'evaluate-gate-ablation':
        run_gate_ablation(args)
    elif args.command == 'evaluate-static-grid':
        run_static_grid(args)
    elif args.command == 'paper-figures':
        run_script('scripts/build_paper_figures.py', [])


if __name__ == '__main__':
    main()
