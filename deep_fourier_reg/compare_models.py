"""
Side-by-side comparison of convfno (FNOReg) vs adaptive_freq_gated_convfno.

Loads two trained models, runs inference on the full validation set, and saves
per-sample diagnostic images plus aggregate improvement distribution plots.

Usage:
    # 0.5x resolution (matching training)
    python3 deep_fourier_reg/compare_models.py --gpu_num 0 --size 80
    # full resolution
    python3 deep_fourier_reg/compare_models.py --gpu_num 0

Output: visualizations/comparison/
    samples/pair_XXX/
        wrapped_fnoreg.png          # warped moving image (baseline)
        wrapped_ours.png            # warped moving image (ours)
        error_fnoreg.png            # overlay: warped(fnoreg) vs fixed
        error_ours.png              # overlay: warped(ours) vs fixed
        fixed.png                   # reference fixed image
        moving.png                  # original moving image
    per_sample_metrics.csv          # per-pair dice + folding
    dice_improvement_distribution.png
    folding_improvement_distribution.png
    dice_scatter.png
    folding_scatter.png
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import utils
import dataloaders
from models import SpatialTransform
from fno import FNOReg, AdaptiveFreqGatedFNOReg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(exp_num, device, weights_path, model_cls):
    """Instantiate and load a trained model from an experiment folder.

    Args:
        exp_num: experiment number
        device: torch device
        weights_path: root experiments directory
        model_cls: model class (FNOReg or AdaptiveFreqGatedFNOReg)

    Returns:
        model (eval mode), model_name (str)
    """
    exp_folder = os.path.join(weights_path, f'oasis_exp{exp_num}')
    with open(os.path.join(exp_folder, 'metadata.json'), 'r') as f:
        metadata = json.loads(f.read())

    model_cfg = metadata['model_config']
    model = model_cls(model_cfg).to(device)

    try:
        weights_file = os.path.join(exp_folder, 'weights.pth')
        state = torch.load(weights_file, map_location=device)
        model.load_state_dict(state)
    except Exception:
        weights_file = os.path.join(exp_folder, 'weights.pt')
        state = torch.load(weights_file, map_location=device)
        model.load_state_dict(state['model_state_dict'])

    model.eval()
    return model, metadata['model_name']


def save_grayscale(img, path):
    """Save (H, W) float32 numpy as 8-bit PNG."""
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def save_overlay(img_rgb, path):
    """Save an (H, W, 3) float32 RGB overlay as PNG."""
    img = np.clip(img_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_improvement_distribution(deltas, xlabel, out_path, ref_line=0):
    """Histogram + KDE of improvement values (ours - baseline).

    Positive = ours better.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mean_val = np.mean(deltas)
    median_val = np.median(deltas)

    ax.hist(deltas, bins=40, density=True, color='steelblue',
            edgecolor='white', linewidth=0.5, alpha=0.7, label='Histogram')
    # Simple numpy-based KDE
    if len(deltas) > 1:
        try:
            # Silverman's rule for bandwidth
            bw = 1.06 * np.std(deltas) * len(deltas) ** (-0.2) + 1e-9
            xs = np.linspace(deltas.min(), deltas.max(), 200)
            kde_y = np.zeros_like(xs)
            for d in deltas:
                kde_y += np.exp(-0.5 * ((xs - d) / bw) ** 2)
            kde_y /= (len(deltas) * bw * np.sqrt(2 * np.pi))
            ax.plot(xs, kde_y, '-', color='darkblue', linewidth=2, label='KDE')
        except Exception:
            pass
    ax.axvline(ref_line, color='gray', linestyle='--', linewidth=1.5, label=f'No difference (0)')
    ax.axvline(mean_val, color='red', linestyle='-', linewidth=2,
               label=f'Mean: {mean_val:+.4f}')
    ax.axvline(median_val, color='darkorange', linestyle='-', linewidth=2,
               label=f'Median: {median_val:+.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'{xlabel} Distribution (N={len(deltas)} pairs)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_scatter(x, y, xlabel, ylabel, out_path):
    """Scatter plot: baseline (x) vs ours (y), one point per sample.

    Points above the diagonal = ours better.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lims = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6, label='Equal performance')
    ax.scatter(x, y, c='steelblue', alpha=0.5, s=12, edgecolors='none')

    above = (y > x).sum()
    below = (y < x).sum()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs {xlabel}\n'
                 f'Ours better: {above} / Worse: {below} / Total: {len(x)}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare convfno vs adaptive_freq_gated_convfno on validation set')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number')
    parser.add_argument('--config_file', type=str, default='params.json',
                        help='JSON config file name')
    parser.add_argument('--size', type=int, default=None,
                        help='Resize images to given smaller dim (None = full resolution, '
                             '80 = 0.5x matching exp143/148 training)')
    parser.add_argument('--exp_fnoreg', type=int, default=143,
                        help='convfno experiment number')
    parser.add_argument('--exp_ours', type=int, default=148,
                        help='adaptive_freq_gated_convfno experiment number')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Limit number of pairs (None = all)')
    args = parser.parse_args()

    # --- Setup ---
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(__file__)
    config_path = args.config_file if os.path.isabs(args.config_file) else os.path.join(script_dir, args.config_file)
    config_path = os.path.normpath(config_path)
    params = pd.read_json(config_path)
    config_dir = os.path.dirname(config_path)

    def resolve_config_path(path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))

    WEIGHTS_PATH = resolve_config_path(params['weights_path'][0])
    OASIS_PATH = resolve_config_path(params['oasis_path'][0])
    OASIS_FOLDERS_PATH = resolve_config_path(params['oasis_folders_path'][0])

    # --- Output dirs ---
    out_root = os.path.join('visualizations', 'comparison')
    samples_dir = os.path.join(out_root, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # --- Load models ---
    print(f'Loading convfno from exp {args.exp_fnoreg} ...')
    model_fnoreg, name_fnoreg = load_model(
        args.exp_fnoreg, device, WEIGHTS_PATH, FNOReg)
    utils.count_parameters(model_fnoreg)
    print(f'  Loaded: {name_fnoreg}')

    print(f'Loading adaptive_freq_gated_convfno from exp {args.exp_ours} ...')
    model_ours, name_ours = load_model(
        args.exp_ours, device, WEIGHTS_PATH, AdaptiveFreqGatedFNOReg)
    utils.count_parameters(model_ours)
    print(f'  Loaded: {name_ours}')

    # --- Data ---
    oasis_folders = []
    with open(OASIS_FOLDERS_PATH, 'r') as f:
        for line in f:
            oasis_folders.append(line.strip('\n'))

    val_dataset = dataloaders.ValidationOasis2d(
        range(213, 414), OASIS_PATH, oasis_folders)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                  num_workers=0)

    transform = SpatialTransform().to(device)

    # --- Inference loop ---
    print(f'Processing {len(val_dataset)} validation pairs...')
    if args.max_pairs:
        print(f'  (limited to {args.max_pairs} pairs)')

    records = []  # per-pair metrics
    import torchvision.transforms as transforms

    with torch.no_grad():
        for pair_idx, (moving, fixed, moving_labels, fixed_labels) in enumerate(val_loader):
            if args.max_pairs and pair_idx >= args.max_pairs:
                break

            moving = moving.to(device).float()
            fixed = fixed.to(device).float()
            moving_labels = moving_labels.to(device).float()
            fixed_labels = fixed_labels.to(device).float()

            # Resize if requested
            if args.size is not None:
                h_orig, w_orig = moving.shape[-2], moving.shape[-1]
                aspect = w_orig / h_orig
                new_h, new_w = args.size, int(args.size * aspect)
                resize_img = transforms.Resize(size=(new_h, new_w))
                resize_lbl = transforms.Resize(
                    size=(new_h, new_w),
                    interpolation=transforms.InterpolationMode.NEAREST)
                moving_rs = resize_img(moving)
                fixed_rs = resize_img(fixed)
                moving_labels_rs = resize_lbl(moving_labels)
                fixed_labels_rs = resize_lbl(fixed_labels)
            else:
                moving_rs, fixed_rs = moving, fixed
                moving_labels_rs, fixed_labels_rs = moving_labels, fixed_labels

            # --- Forward: convfno (baseline) ---
            f_xy_fnoreg = model_fnoreg(moving_rs, fixed_rs)
            _, warped_fnoreg = transform(
                moving_rs, f_xy_fnoreg.permute(0, 2, 3, 1))
            _, warped_labels_fnoreg = transform(
                moving_labels_rs, f_xy_fnoreg.permute(0, 2, 3, 1), mod='nearest')

            # Metrics: convfno
            dice_fno = utils.dice(
                warped_labels_fnoreg.squeeze().long().cpu().numpy(),
                fixed_labels_rs.squeeze().long().cpu().numpy())
            # Compute Jacobian scaling factors from actual resized spatial dims
            h, w = moving_rs.shape[-2], moving_rs.shape[-1]
            scale_h = (h - 1) / 2
            scale_w = (w - 1) / 2

            f_xy_J_fno = torch.clone(f_xy_fnoreg)
            f_xy_J_fno[:, 0] *= scale_h
            f_xy_J_fno[:, 1] *= scale_w
            J_fno = utils.jacobian_determinant(
                f_xy_J_fno.detach().cpu().numpy().squeeze(0))
            fold_fno = 100 * (J_fno < 0).sum() / J_fno.size

            # --- Forward: adaptive_freq_gated_convfno (ours) ---
            f_xy_ours = model_ours(moving_rs, fixed_rs)
            _, warped_ours = transform(
                moving_rs, f_xy_ours.permute(0, 2, 3, 1))
            _, warped_labels_ours = transform(
                moving_labels_rs, f_xy_ours.permute(0, 2, 3, 1), mod='nearest')

            # Metrics: ours
            dice_ours = utils.dice(
                warped_labels_ours.squeeze().long().cpu().numpy(),
                fixed_labels_rs.squeeze().long().cpu().numpy())
            f_xy_J_ours = torch.clone(f_xy_ours)
            f_xy_J_ours[:, 0] *= scale_h
            f_xy_J_ours[:, 1] *= scale_w
            J_ours = utils.jacobian_determinant(
                f_xy_J_ours.detach().cpu().numpy().squeeze(0))
            fold_ours = 100 * (J_ours < 0).sum() / J_ours.size

            # --- Record ---
            records.append({
                'pair_idx': pair_idx,
                'dice_fnoreg': dice_fno,
                'dice_ours': dice_ours,
                'dice_delta': dice_ours - dice_fno,
                'folding_fnoreg': fold_fno,
                'folding_ours': fold_ours,
                'folding_delta': fold_ours - fold_fno,
            })

            # --- Save per-sample images ---
            pair_dir = os.path.join(samples_dir, f'pair_{pair_idx:03d}')
            os.makedirs(pair_dir, exist_ok=True)

            fixed_np = fixed_rs.squeeze().cpu().numpy()
            moving_np = moving_rs.squeeze().cpu().numpy()
            warped_fno_np = warped_fnoreg.squeeze().cpu().numpy()
            warped_ours_np = warped_ours.squeeze().cpu().numpy()

            # Warped images
            save_grayscale(fixed_np, os.path.join(pair_dir, 'fixed.png'))
            save_grayscale(moving_np, os.path.join(pair_dir, 'moving.png'))
            save_grayscale(warped_fno_np,
                           os.path.join(pair_dir, 'wrapped_fnoreg.png'))
            save_grayscale(warped_ours_np,
                           os.path.join(pair_dir, 'wrapped_ours.png'))

            # Error overlays: vis(fixed, warped) → combined RGB
            _, _, err_fno = utils.vis(fixed_np, warped_fno_np)
            _, _, err_ours = utils.vis(fixed_np, warped_ours_np)
            save_overlay(err_fno, os.path.join(pair_dir, 'error_fnoreg.png'))
            save_overlay(err_ours, os.path.join(pair_dir, 'error_ours.png'))

            if (pair_idx + 1) % 50 == 0:
                print(f'  {pair_idx + 1} pairs done')

    # --- Save CSV ---
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_root, 'per_sample_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path} ({len(df)} pairs)')

    # --- Print summary ---
    print('\n' + '=' * 60)
    print('Summary Statistics')
    print('=' * 60)
    print(f'{"Metric":<30} {"convfno":>12} {"ours":>12} {"delta":>12}')
    print('-' * 66)
    print(f'{"Mean Dice":<30} {df["dice_fnoreg"].mean():>12.4f} '
          f'{df["dice_ours"].mean():>12.4f} '
          f'{df["dice_delta"].mean():>+12.4f}')
    print(f'{"Median Dice":<30} {df["dice_fnoreg"].median():>12.4f} '
          f'{df["dice_ours"].median():>12.4f} '
          f'{(df["dice_ours"].median() - df["dice_fnoreg"].median()):>+12.4f}')
    print(f'{"Std Dice":<30} {df["dice_fnoreg"].std():>12.4f} '
          f'{df["dice_ours"].std():>12.4f}')
    print(f'{"Mean Folding %":<30} {df["folding_fnoreg"].mean():>12.4f} '
          f'{df["folding_ours"].mean():>12.4f} '
          f'{df["folding_delta"].mean():>+12.4f}')
    print(f'{"Median Folding %":<30} {df["folding_fnoreg"].median():>12.4f} '
          f'{df["folding_ours"].median():>12.4f} '
          f'{(df["folding_ours"].median() - df["folding_fnoreg"].median()):>+12.4f}')
    print(f'{"Std Folding %":<30} {df["folding_fnoreg"].std():>12.4f} '
          f'{df["folding_ours"].std():>12.4f}')
    print(f'{"Dice improved pairs":<30} '
          f'{(df["dice_delta"] > 0).sum():>12} / {len(df)}')
    print(f'{"Folding improved pairs":<30} '
          f'{(df["folding_delta"] < 0).sum():>12} / {len(df)}')
    print(f'{"Both improved pairs":<30} '
          f'{((df["dice_delta"] > 0) & (df["folding_delta"] < 0)).sum():>12} / {len(df)}')
    print('=' * 60)

    # --- Distribution plots ---
    # Dice: positive delta = ours better
    plot_improvement_distribution(
        df['dice_delta'].values,
        xlabel='Δ Dice (Ours − FNOReg)',
        out_path=os.path.join(out_root, 'dice_improvement_distribution.png'))

    # Folding: negative delta = ours better (less folding)
    fold_delta_improvement = -df['folding_delta'].values  # positive = ours better
    plot_improvement_distribution(
        fold_delta_improvement,
        xlabel='Δ Folding Reduction (FNOReg − Ours) [pp]',
        out_path=os.path.join(out_root, 'folding_improvement_distribution.png'))

    # Scatter plots
    plot_scatter(
        df['dice_fnoreg'].values, df['dice_ours'].values,
        xlabel='Dice (convfno)', ylabel='Dice (ours)',
        out_path=os.path.join(out_root, 'dice_scatter.png'))
    plot_scatter(
        df['folding_fnoreg'].values, df['folding_ours'].values,
        xlabel='Folding % (convfno)', ylabel='Folding % (ours)',
        out_path=os.path.join(out_root, 'folding_scatter.png'))

    print(f'\nAll outputs saved to {os.path.abspath(out_root)}/')


if __name__ == '__main__':
    main()
