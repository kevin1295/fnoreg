import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from tqdm import tqdm

import dataloaders
import utils
from fno import AdaptiveFreqGatedFNOReg, FNOReg, FreqGatedFNOReg, GatedFNOReg, MyFNO
from models import DeepUNet2d, FourierNet, SpatialTransform
from plot_utils import dft_amplitude, plotter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_model(model_name, model_cfg, device):
    if model_name == 'fno':
        return MyFNO(model_cfg).to(device)
    if model_name == 'convfno':
        return FNOReg(model_cfg).to(device)
    if model_name == 'gated_convfno':
        return GatedFNOReg(model_cfg).to(device)
    if model_name == 'freq_gated_convfno':
        return FreqGatedFNOReg(model_cfg).to(device)
    if model_name == 'adaptive_freq_gated_convfno':
        return AdaptiveFreqGatedFNOReg(model_cfg).to(device)
    if model_name == 'fouriernet':
        model = FourierNet(**model_cfg).to(device)
        model.patch_size = (160, 192)
        return model
    if model_name == 'deepunet':
        return DeepUNet2d(model_cfg).to(device)
    raise ValueError(f'Incorrect model name: {model_name}')


def load_weights(model, exp_folder, ckpt_epoch, device):
    if ckpt_epoch >= 0:
        weights_path = exp_folder / f'ckpt_epoch{ckpt_epoch}.pt'
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        return weights_path

    weights_path = exp_folder / 'weights.pth'
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return weights_path

    weights_path = exp_folder / 'weights.pt'
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return weights_path


def save_last_visuals(output_dir, model_name, fixed, moving, warped, flow):
    fixed_np = fixed.squeeze((0, 1)).detach().cpu().numpy()
    moving_np = moving.squeeze((0, 1)).detach().cpu().numpy()
    warped_np = warped.squeeze((0, 1)).detach().cpu().numpy()
    flow_np = flow.squeeze(0).detach().cpu().numpy()

    _, _, vis_before = utils.vis(fixed_np, moving_np)
    _, _, vis_after = utils.vis(fixed_np, warped_np)
    cv2.imwrite(str(output_dir / f'plot_{model_name}.png'), plotter(warped_np, flow_np))
    cv2.imwrite(str(output_dir / f'dft_{model_name}.png'), dft_amplitude(flow_np))
    cv2.imwrite(str(output_dir / 'vis_before.png'), vis_before)
    cv2.imwrite(str(output_dir / 'vis_after.png'), vis_after)
    cv2.imwrite(str(output_dir / 'fixed.png'), fixed_np * 255)
    cv2.imwrite(str(output_dir / 'moving.png'), moving_np * 255)
    cv2.imwrite(str(output_dir / f'moved_{model_name}.png'), warped_np * 255)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a 2D OASIS registration experiment')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number')
    parser.add_argument('--config_file', type=str, default='params.json', help='Config filename or path')
    parser.add_argument('--exp_num', type=int, required=True, help='Experiment number')
    parser.add_argument('--ckpt_epoch', type=int, default=-1, help='Checkpoint epoch; final weights if omitted')
    parser.add_argument('--size', type=int, default=None, help='Test image smaller dimension; default 160')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for metrics and visualizations')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--no_visuals', action='store_true', help='Do not save last-pair visualizations')
    parser.add_argument('--quiet', action='store_true', help='Disable progress bar')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_path(args.config_file, SCRIPT_DIR)
    params = pd.read_json(config_path)
    config_dir = config_path.parent
    oasis_folders_path = resolve_path(params['oasis_folders_path'][0], config_dir)
    oasis_path = resolve_path(params['oasis_path'][0], config_dir)
    weights_root = resolve_path(params['weights_path'][0], config_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with oasis_folders_path.open() as file:
        oasis_folders = [line.strip() for line in file]
    oasis_test = dataloaders.ValidationOasis2d(range(213, 414), str(oasis_path), oasis_folders)

    exp_folder = weights_root / f'oasis_exp{args.exp_num}'
    with (exp_folder / 'metadata.json').open() as file:
        exp_metadata = json.load(file)
    model_name = exp_metadata['model_name']
    model = load_model(model_name, exp_metadata['model_config'], device)
    utils.count_parameters(model)
    weights_path = load_weights(model, exp_folder, args.ckpt_epoch, device)
    model.eval()

    test_height = args.size or 160
    test_width = int(192 / 160 * test_height)
    if args.output_dir:
        output_dir = resolve_path(args.output_dir, REPO_ROOT)
    else:
        output_dir = REPO_ROOT / 'evaluations' / f'exp_{args.exp_num}' / f'test_{test_height}x{test_width}'
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = SpatialTransform().to(device)
    test_gen = Data.DataLoader(
        dataset=oasis_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    resize = None
    resize_labels = None
    if args.size is not None:
        resize_size = (test_height, test_width)
        resize = transforms.Resize(size=resize_size)
        resize_labels = transforms.Resize(
            size=resize_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    records = []
    last_batch = None
    iterator = tqdm(test_gen, ncols=100, disable=args.quiet)
    with torch.no_grad():
        for pair_idx, (moving, fixed, moving_labels, fixed_labels) in enumerate(iterator):
            moving = moving.to(device).float()
            fixed = fixed.to(device).float()
            moving_labels = moving_labels.to(device).float()
            fixed_labels = fixed_labels.to(device).float()

            if resize is not None:
                moving = resize(moving)
                fixed = resize(fixed)
                moving_labels = resize_labels(moving_labels)
                fixed_labels = resize_labels(fixed_labels)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            flow = model(moving, fixed)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - start_time

            height, width = flow.shape[-2:]
            flow_pixels = flow.clone()
            flow_pixels[:, 0] *= (height - 1) / 2
            flow_pixels[:, 1] *= (width - 1) / 2
            jacobian = utils.jacobian_determinant(flow_pixels.cpu().numpy().squeeze(0))
            folding_percent = 100 * np.mean(jacobian < 0)
            positive_jacobian = jacobian[jacobian > 0]
            sample_sdlogj = float(np.std(np.log(positive_jacobian))) if positive_jacobian.size else float('nan')

            _, warped = transform(moving, flow.permute(0, 2, 3, 1))
            _, warped_labels = transform(
                moving_labels,
                flow.permute(0, 2, 3, 1),
                mod='nearest',
            )
            warped_label_np = warped_labels[0].long().cpu().numpy()
            fixed_label_np = fixed_labels[0].long().cpu().numpy()
            moving_label_np = moving_labels[0].long().cpu().numpy()
            dice = utils.dice(warped_label_np, fixed_label_np)
            initial_dice = utils.dice(moving_label_np, fixed_label_np)

            records.append(
                {
                    'pair_idx': pair_idx,
                    'initial_dice': float(initial_dice),
                    'dice': float(dice),
                    'folding_percent': float(folding_percent),
                    'sdlogJ': sample_sdlogj,
                    'inference_time_seconds': float(inference_time),
                }
            )
            last_batch = (fixed, moving, warped, flow)

    dice_values = np.array([record['dice'] for record in records])
    initial_dice_values = np.array([record['initial_dice'] for record in records])
    folding_values = np.array([record['folding_percent'] for record in records])
    sdlogj_values = np.array([record['sdlogJ'] for record in records])
    inference_times = np.array([record['inference_time_seconds'] for record in records])
    summary = {
        'experiment': args.exp_num,
        'model_name': model_name,
        'weights': str(weights_path),
        'test_resolution': [test_height, test_width],
        'num_pairs': len(records),
        'mean_initial_dice': float(np.mean(initial_dice_values)),
        'mean_dice': float(np.mean(dice_values)),
        'std_dice': float(np.std(dice_values)),
        'mean_folding_percent': float(np.mean(folding_values)),
        'std_folding_percent': float(np.std(folding_values)),
        'mean_sdlogJ': float(np.nanmean(sdlogj_values)),
        'std_sdlogJ': float(np.nanstd(sdlogj_values)),
        'mean_inference_time_seconds': float(np.mean(inference_times)),
        'jacobian_scaling': {
            'flow_channel_0': f'(H - 1) / 2 = {(test_height - 1) / 2}',
            'flow_channel_1': f'(W - 1) / 2 = {(test_width - 1) / 2}',
        },
    }

    with (output_dir / 'summary.json').open('w') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with (output_dir / 'per_pair_metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    if not args.no_visuals and last_batch is not None:
        save_last_visuals(output_dir, model_name, *last_batch)

    print(f'--- Evaluation results for {model_name} (exp {args.exp_num}) ---')
    print(f'Test resolution: {test_height}x{test_width}')
    print(f'Mean initial Dice: {summary["mean_initial_dice"]:.6f}')
    print(f'Mean Dice: {summary["mean_dice"]:.6f} ± {summary["std_dice"]:.6f}')
    print(
        f'Mean folded pixels: {summary["mean_folding_percent"]:.6f}% '
        f'± {summary["std_folding_percent"]:.6f}%'
    )
    print(f'Mean sdlogJ: {summary["mean_sdlogJ"]:.6f} ± {summary["std_sdlogJ"]:.6f}')
    print(f'Mean inference time: {summary["mean_inference_time_seconds"]:.6f} s')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
