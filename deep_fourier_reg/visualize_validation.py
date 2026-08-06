"""
Validation set registration visualization + adaptive gate analysis.

Loads a trained model, runs inference on a subset of validation pairs,
and saves per-pair diagnostic images. For AdaptiveFreqGatedFNOReg models,
also collects and visualizes the frequency gate weights (per-sample + aggregate).

Usage:
    python3 deep_fourier_reg/visualize_validation.py --exp_num 150 --num_pairs 5 --gpu_num 0
    python3 deep_fourier_reg/visualize_validation.py --exp_num 150 --num_pairs 5 --gpu_num 0 --size 80
    python3 deep_fourier_reg/visualize_validation.py --exp_num 150 --num_pairs 5 --no_gate_plot

Output: visualizations/exp_{exp_num}/
    pairs/
        pair_{i:02d}_fixed.png
        pair_{i:02d}_moving.png
        pair_{i:02d}_warped.png
        pair_{i:02d}_overlay_before.png
        pair_{i:02d}_overlay_after.png
        pair_{i:02d}_flow.png
        pair_{i:02d}_dft.png
        pair_{i:02d}_gate.png           (adaptive gate models only)
        pair_{i:02d}_gate_radial.png    (adaptive gate models only)
        ...
    freq_gate/                          (adaptive gate models only)
        gate_heatmap_mean.png
        gate_heatmap_std.png
        gate_radial_profile.png
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.utils.data as Data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

sys.path.insert(0, os.path.dirname(__file__))
import utils
import dataloaders
from models import SpatialTransform
from fno import MyFNO, FNOReg, GatedFNOReg, FreqGatedFNOReg, AdaptiveFreqGatedFNOReg
from plot_utils import plotter, dft_amplitude, flow_colorized


def load_model(exp_num, device, weights_path, exp_metadata):
    """Instantiate and load the trained model."""
    model_cfg = exp_metadata['model_config']
    model_name = exp_metadata['model_name']

    if model_name == 'fno':
        model = MyFNO(model_cfg).to(device)
    elif model_name == 'convfno':
        model = FNOReg(model_cfg).to(device)
    elif model_name == 'gated_convfno':
        model = GatedFNOReg(model_cfg).to(device)
    elif model_name == 'freq_gated_convfno':
        model = FreqGatedFNOReg(model_cfg).to(device)
    elif model_name == 'adaptive_freq_gated_convfno':
        model = AdaptiveFreqGatedFNOReg(model_cfg).to(device)
    elif model_name == 'fouriernet':
        from models import FourierNet
        model = FourierNet(**model_cfg).to(device)
        model.patch_size = (160, 192)
    elif model_name == 'deepunet':
        from models import DeepUNet2d
        model = DeepUNet2d(model_cfg).to(device)
    else:
        raise Exception(f'Unknown model name: {model_name}')

    exp_folder = os.path.join(weights_path, f'oasis_exp{exp_num}')
    try:
        weights_file = os.path.join(exp_folder, 'weights.pth')
        model.load_state_dict(torch.load(weights_file, map_location=device))
    except Exception:
        weights_file = os.path.join(exp_folder, 'weights.pt')
        model.load_state_dict(torch.load(weights_file, map_location=device)['model_state_dict'])

    model.eval()
    return model


def save_image(img, path):
    """Save a (H, W) or (H, W, 3) float32 numpy array as PNG."""
    if img.ndim == 2:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    elif img.ndim == 3:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Gate visualization helpers
# ---------------------------------------------------------------------------

def plot_single_gate_heatmap(gates, out_path):
    """Plot per-layer gate heatmaps for a single sample."""
    n_layers = len(gates)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    vmin = min(g.min().item() for g in gates)
    vmax = max(g.max().item() for g in gates)

    for i, gate in enumerate(gates):
        gate_np = gate.numpy()
        gate_shifted = np.fft.fftshift(gate_np)
        im = axes[i].imshow(gate_shifted, cmap='viridis', origin='lower',
                            vmin=vmin, vmax=vmax)
        axes[i].set_title(f'第 {i + 1} 层')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(n_layers, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_single_radial_profile(gates, out_path):
    """Plot gate vs frequency radius for a single sample."""
    h0, h1 = gates[0].shape
    n_layers = len(gates)
    radii = np.sqrt(np.arange(h0)[:, None] ** 2 + np.arange(h1)[None, :] ** 2)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for layer_idx, gate in enumerate(gates):
        gate_np = gate.numpy()
        r_flat = radii.flatten()
        v_flat = gate_np.flatten()
        order = np.argsort(r_flat)
        ax.plot(r_flat[order], v_flat[order], alpha=0.3,
                color=colors[layer_idx], linewidth=0.5)

    ax.set_xlabel('频率半径')
    ax.set_ylabel('门控权重')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_aggregate_heatmap(means, out_path):
    """Plot per-layer mean (or std) gate heatmaps across samples."""
    n_layers = len(means)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    vmin = min(m.min() for m in means)
    vmax = max(m.max() for m in means)

    for i, arr in enumerate(means):
        shifted = np.fft.fftshift(arr)
        im = axes[i].imshow(shifted, cmap='viridis', origin='lower',
                            vmin=vmin, vmax=vmax)
        axes[i].set_title(f'第 {i + 1} 层')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(n_layers, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_aggregate_radial_profile(means, stds, out_path):
    """Plot mean gate vs frequency radius with std band across samples."""
    h0, h1 = means[0].shape
    n_layers = len(means)
    radii = np.sqrt(np.arange(h0)[:, None] ** 2 + np.arange(h1)[None, :] ** 2)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    all_radii, all_values = [], []
    for layer_idx, mean_arr in enumerate(means):
        r_flat = radii.flatten()
        v_flat = 1.0 - mean_arr.flatten()
        order = np.argsort(r_flat)
        ax.plot(r_flat[order], v_flat[order], alpha=0.25,
                color=colors[layer_idx], linewidth=0.5)
        all_radii.append(r_flat)
        all_values.append(v_flat)

    all_radii = np.concatenate(all_radii)
    all_values = np.concatenate(all_values)

    max_r = np.sqrt((h0 - 1) ** 2 + (h1 - 1) ** 2)
    bins = np.linspace(0, max_r, 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(all_radii, bins)

    bin_means, bin_stds = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_means.append(all_values[mask].mean())
            bin_stds.append(all_values[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    valid = ~np.isnan(bin_means)

    ax.plot(bin_centers[valid], bin_means[valid], 'r-', linewidth=2,
            label='跨样本与层的平均门控衰减量')
    ax.fill_between(bin_centers[valid],
                    bin_means[valid] - bin_stds[valid],
                    bin_means[valid] + bin_stds[valid],
                    alpha=0.2, color='red')

    ax.set_xlabel('频率半径（距直流分量的距离）')
    ax.set_ylabel('门控衰减量 1−g')
    ax.legend()
    y_max = max(all_values.max(), np.nanmax(bin_means + bin_stds))
    ax.set_ylim(0, y_max * 1.08)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4), useMathText=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# FFT amplitude diff & LF energy ratio helpers
# ---------------------------------------------------------------------------

def plot_fft_diff_heatmap(fft_before_list, fft_after_list, out_path):
    """Plot per-layer FFT amplitude difference (after - before) as a heatmap grid.

    Uses diverging colormap (RdYlBu) centered at 0:
      - Red/positive: gate amplified this frequency mode
      - Blue/negative: gate suppressed this frequency mode
    """
    n_layers = len(fft_before_list)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    # Compute symmetric range for diverging colormap
    diffs = []
    for i in range(n_layers):
        diff = fft_after_list[i].numpy() - fft_before_list[i].numpy()
        diffs.append(diff)
    max_abs = max(max(abs(d.min()), abs(d.max())) for d in diffs)
    vmin, vmax = -max_abs, max_abs

    for i, diff in enumerate(diffs):
        shifted = np.fft.fftshift(diff)
        im = axes[i].imshow(shifted, cmap='RdBu_r', origin='lower',
                            vmin=vmin, vmax=vmax)
        axes[i].set_title(f'第 {i + 1} 层')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(n_layers, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def compute_lf_ratio(fft_amp, radii, max_r):
    """Compute low-frequency energy ratio for a single FFT amplitude map.

    LF region = frequency radius < max_r / 3.
    """
    h0, h1 = fft_amp.shape
    lf_mask = radii < (max_r / 3)
    total = fft_amp.sum()
    lf_energy = fft_amp[lf_mask].sum()
    return (lf_energy / total).item() if total > 0 else 0.0


def plot_lf_energy_curve(all_lf_before, all_lf_after, out_path):
    """Plot cross-layer LF energy ratio curve (before vs after gating).

    Args:
        all_lf_before: list[list[float]], all_lf_before[layer][sample]
        all_lf_after:  list[list[float]], all_lf_after[layer][sample]
    """
    n_layers = len(all_lf_before)
    n_samples = len(all_lf_before[0])

    before_mean = np.array([np.mean(all_lf_before[l]) for l in range(n_layers)])
    before_std = np.array([np.std(all_lf_before[l]) for l in range(n_layers)])
    after_mean = np.array([np.mean(all_lf_after[l]) for l in range(n_layers)])
    after_std = np.array([np.std(all_lf_after[l]) for l in range(n_layers)])

    layers = np.arange(1, n_layers + 1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ax.plot(layers, before_mean, 'b-o', linewidth=2, markersize=6, label='门控前')
    ax.fill_between(layers, before_mean - before_std, before_mean + before_std,
                    alpha=0.15, color='blue')

    ax.plot(layers, after_mean, 'r-s', linewidth=2, markersize=6, label='门控后')
    ax.fill_between(layers, after_mean - after_std, after_mean + after_std,
                    alpha=0.15, color='red')

    ax.set_xlabel('Fourier 层')
    ax.set_ylabel('低频能量占比')
    ax.legend()
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize validation results + adaptive gate analysis')
    parser.add_argument('--exp_num', type=int, required=True, help='Experiment number')
    parser.add_argument('--num_pairs', type=int, default=5, help='Number of pairs to visualize')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number')
    parser.add_argument('--size', type=int, default=None,
                        help='Resize images to given smaller dim (None = full resolution)')
    parser.add_argument('--no_gate_plot', action='store_true',
                        help='Skip gate visualization even for adaptive gate models')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load config ---
    config_dir = os.path.dirname(__file__)
    import pandas as pd
    params = pd.read_json(os.path.join(config_dir, 'params.json'))
    def resolve_config_path(path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))

    WEIGHTS_PATH = resolve_config_path(params['weights_path'][0])
    OASIS_PATH = resolve_config_path(params['oasis_path'][0])
    OASIS_FOLDERS_PATH = resolve_config_path(params['oasis_folders_path'][0])

    # --- Load experiment metadata ---
    exp_folder = os.path.join(WEIGHTS_PATH, f'oasis_exp{args.exp_num}')
    with open(os.path.join(exp_folder, 'metadata.json'), 'r') as f:
        exp_metadata = json.loads(f.read())

    model = load_model(args.exp_num, device, WEIGHTS_PATH, exp_metadata)
    utils.count_parameters(model)
    model_name = exp_metadata['model_name']
    print(f'Loaded model from exp {args.exp_num}: {model_name}')

    is_freq_gated = isinstance(model, (FreqGatedFNOReg, AdaptiveFreqGatedFNOReg))
    do_gate_plot = is_freq_gated and not args.no_gate_plot

    # --- Load validation subjects ---
    oasis_folders = []
    with open(OASIS_FOLDERS_PATH, 'r') as f:
        for line in f:
            oasis_folders.append(line.strip('\n'))

    val_indices = range(213, 414)
    val_dataset = dataloaders.ValidationOasis2d(val_indices, OASIS_PATH, oasis_folders)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    transform = SpatialTransform().to(device)

    # --- Output directories ---
    pairs_out = os.path.join('visualizations', f'exp_{args.exp_num}', 'pairs')
    os.makedirs(pairs_out, exist_ok=True)

    gate_agg_out = os.path.join('visualizations', f'exp_{args.exp_num}', 'freq_gate')
    if do_gate_plot:
        os.makedirs(gate_agg_out, exist_ok=True)
        n_layers = len(model.fno_blocks)
        all_gates = [[] for _ in range(n_layers)]
        # For LF energy ratio: track per-layer, per-sample ratios
        h0, h1 = model.fno_blocks[0].spectral_conv.half_total_n_modes
        radii = np.sqrt(np.arange(h0)[:, None] ** 2 + np.arange(h1)[None, :] ** 2)
        max_r = np.sqrt((h0 - 1) ** 2 + (h1 - 1) ** 2)
        all_lf_before = [[] for _ in range(n_layers)]
        all_lf_after = [[] for _ in range(n_layers)]

    # --- Process pairs ---
    print(f'Processing {args.num_pairs} validation pairs...')

    with torch.no_grad():
        for i, (moving, fixed, moving_labels, fixed_labels) in enumerate(val_loader):
            if i >= args.num_pairs:
                break

            moving = moving.to(device)
            fixed = fixed.to(device)

            if args.size is not None:
                h, w = moving.shape[-2], moving.shape[-1]
                aspect = w / h
                new_h, new_w = args.size, int(args.size * aspect)
                moving = torch.nn.functional.interpolate(moving, size=(new_h, new_w), mode='bilinear')
                fixed = torch.nn.functional.interpolate(fixed, size=(new_h, new_w), mode='bilinear')

            # Forward pass
            f_xy = model(moving, fixed)

            # Warp
            _, warped = transform(moving, f_xy.permute(0, 2, 3, 1))

            # Convert to numpy
            fixed_np = fixed.squeeze().cpu().numpy()
            moving_np = moving.squeeze().cpu().numpy()
            warped_np = warped.squeeze().cpu().numpy()
            flow_np = f_xy.squeeze().cpu().numpy()

            # --- Save registration images ---
            save_image(fixed_np, os.path.join(pairs_out, f'pair_{i:02d}_fixed.png'))
            save_image(moving_np, os.path.join(pairs_out, f'pair_{i:02d}_moving.png'))
            save_image(warped_np, os.path.join(pairs_out, f'pair_{i:02d}_warped.png'))

            _, _, overlay_before = utils.vis(moving_np, fixed_np)
            _, _, overlay_after = utils.vis(warped_np, fixed_np)
            save_image(overlay_before / 255.0, os.path.join(pairs_out, f'pair_{i:02d}_overlay_before.png'))
            save_image(overlay_after / 255.0, os.path.join(pairs_out, f'pair_{i:02d}_overlay_after.png'))

            flow_rgb = flow_colorized(flow_np).astype(np.float32) / 255.0
            save_image(flow_rgb, os.path.join(pairs_out, f'pair_{i:02d}_flow.png'))

            dft_img = dft_amplitude(flow_np) / 255.0
            save_image(dft_img, os.path.join(pairs_out, f'pair_{i:02d}_dft.png'))

            # --- Per-sample gate + FFT diff visualization ---
            if do_gate_plot:
                sample_gates = []
                sample_fft_before = []
                sample_fft_after = []
                for layer_idx, layer in enumerate(model.fno_blocks):
                    sc = layer.spectral_conv
                    gate = sc.last_gate.clone()
                    sample_gates.append(gate)
                    all_gates[layer_idx].append(gate)

                    fft_b = sc.last_fft_before.clone()  # [h0, h1]
                    fft_a = sc.last_fft_after.clone()
                    sample_fft_before.append(fft_b)
                    sample_fft_after.append(fft_a)

                    # LF energy ratios
                    all_lf_before[layer_idx].append(compute_lf_ratio(fft_b, radii, max_r))
                    all_lf_after[layer_idx].append(compute_lf_ratio(fft_a, radii, max_r))

                plot_single_gate_heatmap(
                    sample_gates,
                    os.path.join(pairs_out, f'pair_{i:02d}_gate.png'))
                plot_single_radial_profile(
                    sample_gates,
                    os.path.join(pairs_out, f'pair_{i:02d}_gate_radial.png'))
                plot_fft_diff_heatmap(
                    sample_fft_before, sample_fft_after,
                    os.path.join(pairs_out, f'pair_{i:02d}_fft_diff.png'))

            print(f'  Pair {i:02d} done')

    # --- Aggregate gate visualization ---
    if do_gate_plot and all_gates[0]:
        print('Computing aggregate gate statistics...')
        n_samples = len(all_gates[0])
        means = []
        stds = []
        for layer_idx in range(n_layers):
            stacked = torch.stack(all_gates[layer_idx])
            means.append(stacked.mean(dim=0).numpy())
            stds.append(stacked.std(dim=0).numpy())

        np.savez(
            os.path.join(gate_agg_out, 'gate_statistics.npz'),
            means=np.stack(means),
            stds=np.stack(stds),
        )

        plot_aggregate_heatmap(
            means,
            os.path.join(gate_agg_out, 'gate_heatmap_mean.png'))

        plot_aggregate_heatmap(
            stds,
            os.path.join(gate_agg_out, 'gate_heatmap_std.png'))

        plot_aggregate_radial_profile(
            means, stds,
            os.path.join(gate_agg_out, 'gate_radial_profile.png'))

        plot_lf_energy_curve(
            all_lf_before, all_lf_after,
            os.path.join(gate_agg_out, 'lf_energy_ratio.png'))

    print(f'All images saved to visualizations/exp_{args.exp_num}/')


if __name__ == '__main__':
    main()
