"""
Gate weight visualization for GatedFNOReg.

Captures SE block sigmoid gate weights and spectral conv outputs via
PyTorch forward hooks, then generates interpretability visualizations:
  - gate_heatmap.png         : mean gate activation per layer x channel
  - gate_distribution.png    : violin plots of gate values per layer
  - gate_profile.png         : per-layer mean gate activation (bar chart)
  - freq_gate_correlation.png: gate weight vs spectral energy scatter
  - channel_spectral_energy.png: log-DFT amplitude per channel heatmap
  - resolution_comparison.png: full-res vs half-res gate comparison (--compare mode)

Usage:
  python visualize_gate_weights.py --exp_num 142 --ckpt_epoch 60
  python visualize_gate_weights.py --exp_num 142 --ckpt_epoch 60 --size 80
  python visualize_gate_weights.py --compare A.npz B.npz
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import dataloaders
import utils
from fno import MyFNO, FNOReg, GatedFNOReg
from models import FourierNet, DeepUNet2d

# ─── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Visualize GatedFNOReg gate weights')
parser.add_argument('--gpu_num', type=int, default=0, help='GPU device number')
parser.add_argument('--config_file', type=str, default='params.json', help='JSON config file')
parser.add_argument('--exp_num', type=int, default=0, help='Experiment number')
parser.add_argument('--ckpt_epoch', type=int, default=-1, help='Checkpoint epoch (-1 for final)')
parser.add_argument('--size', type=int, default=None,
                    help='Resize test images to given smaller dim (None = full res)')
parser.add_argument('--num_samples', type=int, default=-1,
                    help='Number of test pairs for gate aggregation (-1 = all)')
parser.add_argument('--spectral_samples', type=int, default=20,
                    help='Number of samples for spectral analysis (memory-saving)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: experiments_fourier/oasis_exp{num}/gate_vis/)')
parser.add_argument('--no_save_npz', action='store_true', help='Do not save raw gate data as NPZ')
parser.add_argument('--compare', nargs=2, metavar=('NPZ1', 'NPZ2'),
                    help='Compare two saved gate NPZ files (labels inferred from filenames)')
args = parser.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model_and_config(exp_num, config_file, device):
    """Load model, its config, and experiment metadata.
    Mirrors the loading logic in evaluate_oasis.py lines 26-61.
    """
    params = pd.read_json(config_file)
    WEIGHTS_PATH = params['weights_path'][0]

    exp_folder = os.path.join(WEIGHTS_PATH, f'oasis_exp{exp_num}')
    with open(os.path.join(exp_folder, 'metadata.json'), 'r') as f:
        exp_metadata = json.loads(f.read())

    model_cfg = exp_metadata['model_config']
    model_name = exp_metadata['model_name']

    if model_name == 'fno':
        model = MyFNO(model_cfg).to(device)
    elif model_name == 'convfno':
        model = FNOReg(model_cfg).to(device)
    elif model_name == 'gated_convfno':
        model = GatedFNOReg(model_cfg).to(device)
    elif model_name == 'fouriernet':
        model = FourierNet(**model_cfg).to(device)
        model.patch_size = (160, 192)
    elif model_name == 'deepunet':
        model = DeepUNet2d(model_cfg).to(device)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # Load weights
    if args.ckpt_epoch < 0:
        weights_path = os.path.join(exp_folder, 'weights.pth')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(exp_folder, 'weights.pt')
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        weights_path = os.path.join(exp_folder, f'ckpt_epoch{args.ckpt_epoch}.pt')
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    return model, model_cfg, model_name, params


def build_test_loader(params, max_samples=-1):
    """Build DataLoader for OASIS 2D test set."""
    OASIS_FOLDERS_PATH = params['oasis_folders_path'][0]
    OASIS_PATH = params['oasis_path'][0]

    oasis_folders = []
    with open(OASIS_FOLDERS_PATH, 'r') as f:
        for line in f:
            oasis_folders.append(line.strip('\n'))

    test_dataset = dataloaders.ValidationOasis2d(
        range(213, 414), OASIS_PATH, oasis_folders
    )
    loader = Data.DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=2)
    return loader, oasis_folders


# ─── Hook Infrastructure ─────────────────────────────────────────────────────

class GateWeightCollector:
    """Collects SE gate weights and spectral conv outputs via forward hooks."""

    def __init__(self, model, n_layers=12, spectral_samples=20):
        self.model = model
        self.n_layers = n_layers
        self.spectral_samples = spectral_samples
        self._hooks = []
        self._gate_data = defaultdict(list)       # layer_idx -> list of [B, 32] np arrays
        self._spectral_data = defaultdict(list)   # layer_idx -> list of [B, 32, H, W] np arrays
        self._spectral_count = defaultdict(int)

    def _make_gate_hook(self, idx):
        def hook(module, input, output):
            # output is [B, C] — sigmoid gate values from SEblock.fc Sequential
            self._gate_data[idx].append(output.detach().cpu().float().numpy())
        return hook

    def _make_spectral_hook(self, idx):
        collector = self
        def hook(module, input, output):
            # output is [B, C, H, W] from FactorizedSpectralConv2d
            if collector._spectral_count[idx] >= collector.spectral_samples:
                return
            collector._spectral_data[idx].append(
                output.detach().cpu().float().numpy()
            )
            collector._spectral_count[idx] += 1
        return hook

    def register(self):
        """Register hooks on all GatedFourierLayer2d layers in model.fno_blocks."""
        for i, layer in enumerate(self.model.fno_blocks):
            if hasattr(layer, 'gate'):
                h1 = layer.gate.fc.register_forward_hook(self._make_gate_hook(i))
                self._hooks.append(h1)
            if hasattr(layer, 'spectral_conv'):
                h2 = layer.spectral_conv.register_forward_hook(self._make_spectral_hook(i))
                self._hooks.append(h2)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self._gate_data.clear()
        self._spectral_data.clear()
        self._spectral_count.clear()

    def gate_array(self):
        """Return stacked gate weights: shape [n_layers, N, n_channels]."""
        arrs = []
        for i in range(self.n_layers):
            if i in self._gate_data and len(self._gate_data[i]) > 0:
                arrs.append(np.stack(self._gate_data[i], axis=0))  # [N, C]
            else:
                arrs.append(np.empty((0,)))
        return arrs

    def spectral_array(self):
        """Return stacked spectral outputs: list of [K, C, H, W] per layer."""
        arrs = []
        for i in range(self.n_layers):
            if i in self._spectral_data and len(self._spectral_data[i]) > 0:
                arrs.append(np.stack(self._spectral_data[i], axis=0))
            else:
                arrs.append(np.empty((0,)))
        return arrs


# ─── Spectral Analysis ───────────────────────────────────────────────────────

def channel_spectral_energy(spectral_out):
    """Compute per-channel DFT energy and centroid from spatial feature map.

    Args:
        spectral_out: [K, C, H, W] numpy array (spectral conv outputs)

    Returns:
        energy: [C] — mean total log-DFT-energy per channel (across K samples)
        centroid: [C] — mean spectral centroid per channel
    """
    K, C, H, W = spectral_out.shape
    energies = np.zeros((K, C))
    centroids = np.zeros((K, C))

    y, x = np.meshgrid(
        np.fft.fftfreq(H, d=1.0 / H),
        np.fft.fftfreq(W, d=1.0 / W),
        indexing='ij'
    )
    dist = np.sqrt(x ** 2 + y ** 2)

    for k in range(K):
        for c in range(C):
            fft = np.fft.fft2(spectral_out[k, c])
            amp = np.abs(np.fft.fftshift(fft))
            total = np.sum(amp)
            if total > 0:
                energies[k, c] = np.log1p(total)
                centroids[k, c] = np.sum(dist * amp) / total
            else:
                energies[k, c] = 0.0
                centroids[k, c] = 0.0

    return energies.mean(axis=0), centroids.mean(axis=0)   # each [C]


# ─── Visualization Functions ─────────────────────────────────────────────────

def _set_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass  # fall back to default


def plot_gate_heatmap(gate_mean, gate_std, save_path, title_suffix=''):
    """Heatmap: layers x channels, color = mean gate activation."""
    n_layers, n_ch = gate_mean.shape
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(gate_mean.T, aspect='auto', cmap='RdYlGn',
                   vmin=0.0, vmax=1.0, origin='lower')
    cbar = plt.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label('Mean Gate Activation', fontsize=12)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'Layer {i+1}' for i in range(n_layers)], rotation=45, ha='right')
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f'Ch {i}' for i in range(n_ch)])
    ax.set_xlabel('Fourier Layer', fontsize=13)
    ax.set_ylabel('Channel', fontsize=13)
    ax.set_title(f'Gate Weight Heatmap{title_suffix}', fontsize=15, fontweight='bold')

    # Annotate cells with value (small font)
    for i in range(n_layers):
        for j in range(n_ch):
            val = gate_mean[i, j]
            color = 'white' if val < 0.4 or val > 0.7 else 'black'
            ax.text(i, j, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_gate_distribution(gate_arrs, save_path, title_suffix=''):
    """Violin plots per layer showing distribution of channel gate weights."""
    n_layers = len(gate_arrs)
    fig, ax = plt.subplots(figsize=(16, 6))

    data_for_violin = []
    positions = []
    for i in range(n_layers):
        if gate_arrs[i].size > 0:
            # gate_arrs[i] is [N, C]; across all samples and channels
            vals = gate_arrs[i].ravel()  # [N * C]
            data_for_violin.append(vals)
            positions.append(i + 1)

    if not data_for_violin:
        print('  No gate data for distribution plot')
        plt.close(fig)
        return

    vp = ax.violinplot(data_for_violin, positions=positions,
                       showmeans=True, showmedians=True, widths=0.7)
    # Style violins
    for body in vp['bodies']:
        body.set_facecolor('#5DADE2')
        body.set_alpha(0.7)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Neutral (0.5)')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Layer {p}' for p in positions])
    ax.set_xlabel('Fourier Layer', fontsize=13)
    ax.set_ylabel('Gate Weight', fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title(f'Gate Weight Distribution per Layer{title_suffix}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_gate_profile(gate_mean, gate_std, save_path, title_suffix=''):
    """Bar chart: per-layer mean gate activation across all channels."""
    n_layers = gate_mean.shape[0]
    layer_means = gate_mean.mean(axis=1)  # [n_layers]
    layer_stds = gate_std.mean(axis=1)    # [n_layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, n_layers + 1)
    bars = ax.bar(x, layer_means, yerr=layer_stds, capsize=4,
                  color=plt.cm.RdYlGn(layer_means), edgecolor='gray', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Neutral (0.5)')

    # Value labels on bars
    for i, (xi, val) in enumerate(zip(x, layer_means)):
        ax.text(xi, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)

    ax.set_xlabel('Fourier Layer', fontsize=13)
    ax.set_ylabel('Mean Gate Activation', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_title(f'Mean Gate Activation per Layer{title_suffix}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_freq_gate_correlation(gate_mean, spectral_energy, save_path, title_suffix=''):
    """Scatter: gate weight vs log-DFT energy, colored by layer."""
    n_layers, n_ch = gate_mean.shape
    if spectral_energy is None or spectral_energy.size == 0:
        print('  Skipping frequency-gate correlation (no spectral data)')
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab20
    all_gates = []
    all_energies = []
    for i in range(n_layers):
        g = gate_mean[i]      # [C]
        e = spectral_energy[i]  # [C]
        ax.scatter(e, g, c=[cmap(i / n_layers)], label=f'Layer {i+1}',
                   alpha=0.7, edgecolors='none', s=40)
        all_gates.append(g)
        all_energies.append(e)

    all_gates = np.concatenate(all_gates)
    all_energies = np.concatenate(all_energies)
    # Pearson r
    if len(all_gates) > 2:
        corr = np.corrcoef(all_energies, all_gates)[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
                fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Log-DFT Energy per Channel', fontsize=13)
    ax.set_ylabel('Mean Gate Weight', fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title(f'Gate Weight vs Spectral Energy{title_suffix}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='lower right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_channel_spectral_heatmap(spectral_energy, save_path, title_suffix=''):
    """Heatmap: layers x channels showing log-DFT energy per channel."""
    n_layers, n_ch = spectral_energy.shape
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(spectral_energy.T, aspect='auto', cmap='viridis',
                   origin='lower')
    cbar = plt.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label('Log-DFT Energy', fontsize=12)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'Layer {i+1}' for i in range(n_layers)], rotation=45, ha='right')
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f'Ch {i}' for i in range(n_ch)])
    ax.set_xlabel('Fourier Layer', fontsize=13)
    ax.set_ylabel('Channel', fontsize=13)
    ax.set_title(f'Channel Spectral Energy{title_suffix}', fontsize=15, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


def plot_resolution_comparison(npz1_path, npz2_path, save_dir):
    """Compare two saved gate NPZ files (full-res vs half-res)."""
    d1 = np.load(npz1_path)
    d2 = np.load(npz2_path)
    g1 = d1['gate_mean']  # [L, C]
    g2 = d2['gate_mean']

    label1 = d1.get('resolution_label', os.path.basename(npz1_path).replace('.npz', ''))
    label2 = d2.get('resolution_label', os.path.basename(npz2_path).replace('.npz', ''))

    n_layers, n_ch = g1.shape

    fig, axes = plt.subplots(1, 3, figsize=(24, 7),
                              gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # Left: heatmap for full-res
    im1 = axes[0].imshow(g1.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    axes[0].set_title(f'Gate Weights — {label1}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[0], shrink=0.85)

    # Middle: heatmap for half-res
    im2 = axes[1].imshow(g2.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    axes[1].set_title(f'Gate Weights — {label2}', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[1], shrink=0.85)

    # Right: scatter comparison
    diff = g1 - g2
    sc = axes[2].scatter(g1.ravel(), g2.ravel(), c=np.tile(np.arange(n_layers), n_ch),
                         cmap='tab20', alpha=0.6, s=30, edgecolors='none')
    axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    axes[2].set_xlabel(f'Gate Weight — {label1}', fontsize=12)
    axes[2].set_ylabel(f'Gate Weight — {label2}', fontsize=12)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    mae = np.mean(np.abs(diff))
    axes[2].set_title(f'Gate Weight Comparison (MAE={mae:.4f})', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(sc, ax=axes[2], shrink=0.85)
    cbar.set_label('Layer', fontsize=10)

    fig.suptitle('Resolution Comparison: Gate Weight Patterns', fontsize=16, fontweight='bold')
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'resolution_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')

    # Also print summary stats
    print(f'\n  Resolution Comparison Summary:')
    print(f'    {label1}: mean gate = {g1.mean():.4f} ± {g1.std():.4f}')
    print(f'    {label2}: mean gate = {g2.mean():.4f} ± {g2.std():.4f}')
    print(f'    Mean Absolute Difference (MAE) = {mae:.4f}')
    print(f'    Max Absolute Difference = {np.max(np.abs(diff)):.4f}')


def save_gate_npz(gate_arrs, spectral_arrs, gate_mean, gate_std, spectral_energy,
                  save_path, resolution_label='1x'):
    """Save aggregated gate data to NPZ."""
    save_dict = {
        'resolution_label': resolution_label,
        'gate_mean': gate_mean,
        'gate_std': gate_std,
    }
    for i, arr in enumerate(gate_arrs):
        if arr.size > 0:
            save_dict[f'gate_layer_{i}'] = arr
    if spectral_energy is not None and spectral_energy.size > 0:
        save_dict['spectral_energy'] = spectral_energy
    for i, arr in enumerate(spectral_arrs):
        if arr.size > 0:
            # only save first 5 for file size
            save_dict[f'spectral_layer_{i}'] = arr[:5]

    np.savez_compressed(save_path, **save_dict)
    print(f'  Saved NPZ data to {save_path}')


# ─── Inference Loop ──────────────────────────────────────────────────────────

def run_collection(model, test_loader, collector, device, resize_size, max_samples):
    """Run inference on test set, collecting gate weights and spectral outputs."""
    resize_transform = None
    if resize_size is not None:
        resize_transform = transforms.Resize(size=resize_size)
    else:
        resize_transform = None

    count = 0
    for moving, fixed, moving_labels, fixed_labels in tqdm(test_loader, ncols=100,
                                                             desc='Collecting gate weights'):
        moving = moving.to(device).float()
        fixed = fixed.to(device).float()

        if resize_transform is not None:
            moving = resize_transform(moving)
            fixed = resize_transform(fixed)

        with torch.no_grad():
            _ = model(moving, fixed)

        count += 1
        if max_samples > 0 and count >= max_samples:
            break

    print(f'  Collected gate weights from {count} test pairs')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    _set_style()

    # ── Compare mode ──
    if args.compare is not None:
        npz1, npz2 = args.compare
        out_dir = args.output_dir or os.path.dirname(npz1)
        _ensure_dir(out_dir)
        plot_resolution_comparison(npz1, npz2, out_dir)
        print('Done.')
        return

    # ── Normal mode ──
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model, model_cfg, model_name, params = load_model_and_config(
        args.exp_num, args.config_file, device
    )
    print(f'Loaded model: {model_name} (exp {args.exp_num})')

    if model_name != 'gated_convfno':
        print(f'WARNING: model is "{model_name}", not "gated_convfno". '
              'Gate hooks may not find SEblock targets. Gate visualization will be skipped.')
        # Still proceed — hooks simply won't fire if no SEblock exists

    # Build data loader
    test_loader, _ = build_test_loader(params, max_samples=args.num_samples)

    # Determine resize size
    if args.size is not None:
        resize_size = (args.size, int(192 / 160 * args.size))
        resolution_label = f'{args.size / 160:.1f}x'
    else:
        resize_size = None
        resolution_label = '1x'

    # Setup output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        WEIGHTS_PATH = params['weights_path'][0]
        out_dir = os.path.join(WEIGHTS_PATH, f'oasis_exp{args.exp_num}', 'gate_vis')
    _ensure_dir(out_dir)

    # Determine number of Fourier layers (in case non-standard config)
    if hasattr(model, 'fno_blocks'):
        n_layers = len(model.fno_blocks)
    else:
        n_layers = model_cfg.get('n_layers', 12)
    n_channels = model_cfg.get('hidden_channels', 32)
    print(f'  FNO layers: {n_layers}, hidden channels: {n_channels}')
    print(f'  Resolution: {resolution_label}')
    print(f'  Output dir: {out_dir}')

    # Register hooks
    collector = GateWeightCollector(model, n_layers=n_layers,
                                    spectral_samples=args.spectral_samples)
    collector.register()

    try:
        # Run inference
        run_collection(model, test_loader, collector, device,
                       resize_size, args.num_samples)
    finally:
        collector.remove()

    # ── Aggregate ──
    gate_arrs = collector.gate_array()        # list of [N, C] per layer
    spectral_arrs = collector.spectral_array() # list of [K, C, H, W] per layer

    # Build gate_mean [L, C] and gate_std [L, C]
    has_gate_data = any(arr.size > 0 for arr in gate_arrs)

    if has_gate_data:
        gate_mean = np.zeros((n_layers, n_channels))
        gate_std = np.zeros((n_layers, n_channels))
        for i in range(n_layers):
            if gate_arrs[i].size > 0:
                # gate_arrs[i] is [N, C]
                gate_mean[i] = gate_arrs[i].mean(axis=0)
                gate_std[i] = gate_arrs[i].std(axis=0)
            else:
                gate_mean[i] = np.nan
                gate_std[i] = np.nan
        print(f'  Gate weights aggregated: mean={gate_mean[~np.isnan(gate_mean)].mean():.4f}, '
              f'min={gate_mean[~np.isnan(gate_mean)].min():.4f}, '
              f'max={gate_mean[~np.isnan(gate_mean)].max():.4f}')
    else:
        print('  No gate data collected (model may not have SE blocks)')
        gate_mean = np.full((n_layers, n_channels), np.nan)
        gate_std = np.full((n_layers, n_channels), np.nan)

    # Compute spectral energy
    has_spectral = any(arr.size > 0 for arr in spectral_arrs)
    spectral_energy = None
    spectral_centroid = None
    if has_spectral:
        se_list = []
        sc_list = []
        for i in range(n_layers):
            if spectral_arrs[i].size > 0:
                e, c = channel_spectral_energy(spectral_arrs[i])
                se_list.append(e)
                sc_list.append(c)
            else:
                se_list.append(np.full(n_channels, np.nan))
                sc_list.append(np.full(n_channels, np.nan))
        spectral_energy = np.array(se_list)    # [L, C]
        spectral_centroid = np.array(sc_list)  # [L, C]
        print(f'  Spectral energy computed from up to {args.spectral_samples} samples')

    # ── Generate Plots ──
    title_suffix = f' ({resolution_label} res)'
    if has_gate_data:
        plot_gate_heatmap(gate_mean, gate_std,
                          os.path.join(out_dir, 'gate_heatmap.png'), title_suffix)
        plot_gate_distribution(gate_arrs,
                               os.path.join(out_dir, 'gate_distribution.png'), title_suffix)
        plot_gate_profile(gate_mean, gate_std,
                          os.path.join(out_dir, 'gate_profile.png'), title_suffix)

    if has_spectral:
        plot_channel_spectral_heatmap(spectral_energy,
                                      os.path.join(out_dir, 'channel_spectral_energy.png'),
                                      title_suffix)

    if has_gate_data and has_spectral:
        plot_freq_gate_correlation(gate_mean, spectral_energy,
                                   os.path.join(out_dir, 'freq_gate_correlation.png'),
                                   title_suffix)

    # ── Save NPZ ──
    if not args.no_save_npz and has_gate_data:
        npz_name = f'gate_data_{resolution_label.replace(".", "p")}.npz'
        npz_path = os.path.join(out_dir, npz_name)
        save_gate_npz(gate_arrs, spectral_arrs, gate_mean, gate_std, spectral_energy,
                      npz_path, resolution_label)

    print(f'\nAll visualizations saved to: {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
