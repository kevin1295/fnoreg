# AFG-FNOReg: Adaptive Frequency-Gated Fourier Neural Operator for Cross-Resolution Medical Image Registration

**AFG-FNOReg** extends the [FNOReg](https://link.springer.com/chapter/10.1007/978-3-031-78201-5_11) architecture with an *adaptive frequency-gating* mechanism: each Fourier layer computes an input-dependent gate from the magnitude of its spectral output and rescales the retained Fourier modes per-sample and per-frequency, improving registration accuracy and deformation regularity when training and test resolutions differ.

The base registration framework (FNO layers between FFC encoders/decoders, unsupervised loss, OASIS-1 setup) comes from the official FNOReg implementation by its original authors; this repository builds the AFG study on top of it. The AFG manuscript is in preparation.

## Frequency gating in a nutshell

For each Fourier layer, the spectral convolution produces complex features `z` for the upper and lower frequency blocks. The gate is derived from the mean magnitude over output channels,

```
e(x) = Mean_c | z+(x) |
g(x) = sigmoid( a ⊙ e(x) + b )
```

and the same gate is broadcast along the channel dimension and multiplied back into both frequency blocks. Because `e(x)` depends on the current input pair, the gate is *input-adaptive* even though `a` and `b` are shared across samples. Model variants included for controlled comparison:

| Config key | Class | Description |
|---|---|---|
| `convfno` | `FNOReg` | Baseline: standard spectral convolution, no gate |
| `gated_convfno` | `GatedFNOReg` | SE-block gating on spectral output channels |
| `freq_gated_convfno` | `FreqGatedFNOReg` | Static (input-independent) per-frequency gate |
| `adaptive_freq_gated_convfno` | `AdaptiveFreqGatedFNOReg` | **AFG (this work)**: input-dependent per-frequency gate |

## Experiments and key results

Results below are from a **single training run** per model (see `paper_draft`/manuscript for details) on 2D OASIS-1 across four train→test resolution combinations (full = 160×192, half = 80×96). Mean Dice over the validation set:

| Train → Test | Baseline FNOReg | AFG-FNOReg |
|:---:|:---:|:---:|
| 160 → 160 | 0.7739 | **0.7741** |
| 160 → 80 | 0.7455 | 0.7455 |
| 80 → 160 | 0.7654 | **0.7694** |
| 80 → 80 | 0.7389 | **0.7438** |

AFG-FNOReg shows a higher mean Dice in three settings, is on par in the fourth, and has a lower mean folding-pixel percentage in all four. Inference-time interventions on the trained AFG model indicate that frequency rescaling is functional (disabling it with an identity gate clearly degrades both Dice and regularity), but that *sample-specific* gate matching is not the main driver of the gain — freezing the mean gate performs at least as well. Diagnostics also show the learned gates saturate near 1, more so at higher input resolution.

## Installation

Python 3.10.12, PyTorch 2.1.0 + CUDA 11.8.

```bash
# uv-based environment (recommended)
git clone https://github.com/kevin1295/fnoreg.git fnoreg
cd fnoreg
uv sync
```

GPU commands must go through the helper below so the CUDA 11.8 NVRTC library is visible to cuDNN:

```bash
bash scripts/uv_gpu.sh python main.py train --gpu_num 0 --size 160
```

The legacy `pip install -r requirements.txt` / venv workflow remains available, and the original training/evaluation scripts under `deep_fourier_reg/` are kept for compatibility.

## Usage

All maintained 2D commands are launched from the repository root through `main.py`. `--size 160` is full resolution, `--size 80` half resolution.

```bash
# Train a new experiment
bash scripts/uv_gpu.sh python main.py train --gpu_num 0 --size 160

# Resume experiment 148 from epoch 70
bash scripts/uv_gpu.sh python main.py train --gpu_num 0 --size 80 --exp_num 148 --ckpt_epoch 70

# Evaluate one experiment (Dice / folding / sdlogJ + inference time)
bash scripts/uv_gpu.sh python main.py evaluate --gpu_num 0 --exp_num 148 --size 160

# Full train-resolution × test-resolution grid (baseline + AFG)
bash scripts/uv_gpu.sh python main.py evaluate-grid --gpu_num 0

# Inference-time gate interventions (normal / identity / mean / shuffled)
bash scripts/uv_gpu.sh python main.py evaluate-gate-ablation --gpu_num 0 --exp_num 148

# Visualize registration pairs and adaptive gates
bash scripts/uv_gpu.sh python main.py visualize --gpu_num 0 --exp_num 147 --num_pairs 20

# Pair-by-pair comparison of two experiments
bash scripts/uv_gpu.sh python main.py compare --gpu_num 0
```

Model selection and training hyperparameters live in `deep_fourier_reg/params.json` (set `model_name` to one of the config keys above). Relative paths in the config are resolved from the config file location.

## Dataset

We use the preprocessed [OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/) dataset from [Adrian Dalca's repository](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md). After downloading the 2D (and optionally 3D) data, update the `oasis_path` / `oasis_folders_path` fields in `deep_fourier_reg/params.json`.

## Reproducing the original FNOReg results

This repository is a superset of the official FNOReg implementation. To reproduce the published FNOReg paper results, download the pretrained checkpoints with `./download_ckpt.sh` (2D) and `./download_ckpt3d.sh` (3D), then follow `instructions_to_reproduce.md`.

## Acknowledgements

- Hoopes et al. [Learning the Effect of Registration Hyperparameters with HyperMorph](https://arxiv.org/abs/2203.16680) — preprocessed data collection.
- Chen, Junyu, et al. [TransMorph: Transformer for Unsupervised Medical Image Registration](https://www.sciencedirect.com/science/article/pii/S1361841522002432) — source code used to train the VoxelMorph and TransMorph baselines.
- The FNOReg baseline framework is from [FNOReg: Resolution-Robust Medical Image Registration Method Based on Fourier Neural Operator](https://link.springer.com/chapter/10.1007/978-3-031-78201-5_11) (MICCAI 2024 workshop).

## Citation

Citation details for AFG-FNOReg will be added here once the manuscript is accepted.
