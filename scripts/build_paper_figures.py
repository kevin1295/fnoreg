from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper_figures"
COMPARISON_DIR = ROOT / "visualizations" / "comparison"
GRID_DIR = ROOT / "evaluations" / "afg_resolution_grid"
GATE_ABLATION_DIR = ROOT / "evaluations" / "afg_gate_ablation" / "80_to_160"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def export_qualitative_panels(pair_idx=117):
    pair_dir = COMPARISON_DIR / "samples" / f"pair_{pair_idx:03d}"
    panels = [
        ("moving.png", "figure2a_moving.png"),
        ("fixed.png", "figure2b_fixed.png"),
        ("wrapped_fnoreg.png", "figure2c_baseline_warped.png"),
        ("wrapped_ours.png", "figure2d_afg_warped.png"),
        ("error_fnoreg.png", "figure2e_baseline_overlay.png"),
        ("error_ours.png", "figure2f_afg_overlay.png"),
    ]
    for source_name, output_name in panels:
        image = Image.open(pair_dir / source_name)
        image.save(OUTPUT_DIR / output_name)


def build_dice_scatter():
    baseline_metrics = pd.read_csv(GRID_DIR / "80_to_160" / "baseline" / "per_pair_metrics.csv")
    afg_metrics = pd.read_csv(GATE_ABLATION_DIR / "normal" / "per_pair_metrics.csv")
    if not baseline_metrics["pair_idx"].equals(afg_metrics["pair_idx"]):
        raise ValueError("Baseline and AFG evaluation pairs are not aligned")
    baseline = baseline_metrics["dice"].to_numpy()
    afg = afg_metrics["dice"].to_numpy()
    improved = afg > baseline
    lower = min(baseline.min(), afg.min()) - 0.01
    upper = max(baseline.max(), afg.max()) + 0.01

    fig, axis = plt.subplots(figsize=(6.4, 6.1))
    axis.scatter(
        baseline[~improved],
        afg[~improved],
        s=20,
        alpha=0.65,
        color="#B8B8B8",
        edgecolors="none",
        label=f"Lower with AFG-FNOReg ({(~improved).sum()} pairs)",
    )
    axis.scatter(
        baseline[improved],
        afg[improved],
        s=20,
        alpha=0.72,
        color="#2878B5",
        edgecolors="none",
        label=f"Higher with AFG-FNOReg ({improved.sum()} pairs)",
    )
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#555555", linewidth=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Baseline FNOReg Dice")
    axis.set_ylabel("AFG-FNOReg Dice")
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(loc="upper left", frameon=True)
    axis.text(
        0.98,
        0.04,
        f"Mean: {baseline.mean():.4f} → {afg.mean():.4f}\nMean difference: {np.mean(afg - baseline):+.4f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure3_dice_scatter.png", bbox_inches="tight")
    plt.close(fig)


def build_folding_scatter():
    baseline_metrics = pd.read_csv(GRID_DIR / "80_to_160" / "baseline" / "per_pair_metrics.csv")
    afg_metrics = pd.read_csv(GATE_ABLATION_DIR / "normal" / "per_pair_metrics.csv")
    if not baseline_metrics["pair_idx"].equals(afg_metrics["pair_idx"]):
        raise ValueError("Baseline and AFG evaluation pairs are not aligned")
    baseline = baseline_metrics["folding_percent"].to_numpy()
    afg = afg_metrics["folding_percent"].to_numpy()
    lower_mask = afg < baseline
    higher_mask = afg > baseline
    equal_mask = ~(lower_mask | higher_mask)
    upper = max(baseline.max(), afg.max()) * 1.03

    fig, axis = plt.subplots(figsize=(6.4, 6.1))
    axis.scatter(
        baseline[higher_mask],
        afg[higher_mask],
        s=20,
        alpha=0.62,
        color="#B8B8B8",
        edgecolors="none",
        label=f"Higher with AFG-FNOReg ({higher_mask.sum()} pairs)",
    )
    axis.scatter(
        baseline[lower_mask],
        afg[lower_mask],
        s=20,
        alpha=0.72,
        color="#2878B5",
        edgecolors="none",
        label=f"Lower with AFG-FNOReg ({lower_mask.sum()} pairs)",
    )
    if equal_mask.any():
        axis.scatter(
            baseline[equal_mask],
            afg[equal_mask],
            s=24,
            alpha=0.85,
            color="#D9A441",
            edgecolors="none",
            label=f"Equal ({equal_mask.sum()} pairs)",
        )
    axis.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1.2)
    axis.set_xlim(0, upper)
    axis.set_ylim(0, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Baseline FNOReg folding rate (%)")
    axis.set_ylabel("AFG-FNOReg folding rate (%)")
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(loc="upper left", frameon=True)
    axis.text(
        0.98,
        0.04,
        f"Mean: {baseline.mean():.4f}% → {afg.mean():.4f}%\nMean difference: {np.mean(afg - baseline):+.4f} pp",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure4_folding_scatter.png", bbox_inches="tight")
    plt.close(fig)


def build_gate_resolution_diagnostic():
    statistics = pd.read_csv(GATE_ABLATION_DIR / "resolution_gate_statistics.csv")
    layers = statistics["layer"].to_numpy()
    positions = np.arange(len(layers))
    width = 0.38
    low_attenuation = 1.0 - statistics["low_resolution_gate_mean"].to_numpy()
    full_attenuation = 1.0 - statistics["full_resolution_gate_mean"].to_numpy()
    positive_attenuation = np.concatenate(
        [low_attenuation[low_attenuation > 0], full_attenuation[full_attenuation > 0]]
    )
    zero_floor = positive_attenuation.min() * 0.55
    full_attenuation_for_plot = np.where(full_attenuation > 0, full_attenuation, zero_floor)

    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.bar(
        positions - width / 2,
        low_attenuation,
        width=width,
        color="#2878B5",
        label="80×96 input",
    )
    axis.bar(
        positions + width / 2,
        full_attenuation_for_plot,
        width=width,
        color="#D97706",
        label="160×192 input",
    )
    zero_mask = full_attenuation == 0
    axis.scatter(
        positions[zero_mask] + width / 2,
        np.full(zero_mask.sum(), zero_floor),
        marker="v",
        s=24,
        color="#9A4F00",
        zorder=3,
    )
    axis.set_xlabel("Fourier layer")
    axis.set_ylabel("Mean gate attenuation ($1-\\bar{g}$)")
    axis.set_xticks(positions, layers)
    axis.set_yscale("log")
    axis.set_ylim(bottom=zero_floor * 0.65)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7, which="both")
    axis.legend(loc="upper center", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure5a_gate_resolution_attenuation.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure5a_gate_resolution_attenuation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.bar(
        positions - width / 2,
        statistics["low_resolution_between_sample_std_mean"],
        width=width,
        color="#2878B5",
        label="80×96 input",
    )
    axis.bar(
        positions + width / 2,
        statistics["full_resolution_between_sample_std_mean"],
        width=width,
        color="#D97706",
        label="160×192 input",
    )
    axis.set_xlabel("Fourier layer")
    axis.set_ylabel("Mean between-sample gate standard deviation")
    axis.set_xticks(positions, layers)
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7, which="both")
    axis.legend(loc="upper center", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure5b_gate_resolution_variability.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure5b_gate_resolution_variability.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for obsolete in [
        "figure1_qualitative_registration.png",
        "figure1a_moving.png",
        "figure1b_fixed.png",
        "figure1c_baseline_warped.png",
        "figure1d_afg_warped.png",
        "figure1e_baseline_overlay.png",
        "figure1f_afg_overlay.png",
        "figure2_dice_scatter.png",
        "figure3_folding_scatter.png",
        "figure4a_gate_heatmap_mean.png",
        "figure4b_gate_radial_profile.png",
        "figure3_gate_diagnostic.png",
        "figure3a_gate_heatmap_mean.png",
        "figure3b_gate_radial_profile.png",
        "figure5a_gate_heatmap_mean.png",
        "figure5b_gate_radial_profile.png",
        "figure6a_gate_resolution_mean.png",
        "figure6a_gate_resolution_mean.pdf",
        "figure6b_gate_resolution_variability.png",
        "figure6b_gate_resolution_variability.pdf",
    ]:
        (OUTPUT_DIR / obsolete).unlink(missing_ok=True)
    configure_style()
    export_qualitative_panels()
    build_dice_scatter()
    build_folding_scatter()
    build_gate_resolution_diagnostic()
    print(f"Paper figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
