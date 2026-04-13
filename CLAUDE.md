# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Supervised DWI denoising and DTI prediction for a variable-`N` diffusion dataset. A deep learning model (QSpaceUNet) is trained to predict clean 6D DTI tensors directly from degraded DWI volumes, compared against Patch2Self and MP-PCA baselines.

## Common commands

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the Zarr dataset from NIfTI
python3 build_pretext_dataset.py --data_dir dataset/dataset_v1 --output dataset/pretext_dataset_new.zarr

# Inspect dataset (GUI)
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr
# Dataset summary (no GUI)
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr --summary-only

# Train
python3 -m research.train --zarr_path dataset/pretext_dataset_new.zarr --out_dir research/runs/run_01

# Monitor training
tensorboard --logdir research/runs/run_01/tb

# Evaluate (test subjects only, or all)
python3 -m research.evaluate --checkpoint research/runs/run_01/best_model.pt --zarr_path dataset/pretext_dataset_new.zarr
python3 -m research.evaluate --checkpoint research/runs/run_01/best_model.pt --zarr_path dataset/pretext_dataset_new.zarr --eval_all
```

## Architecture

The pipeline is `slice-wise DWI -> q-space conditioning -> 2D U-Net -> 6D DTI tensor`.

**Data flow across key files:**

1. **`functions.py`** - Core DWI loading (NIfTI + bval/bvec), Rician noise degradation, and DTI fitting via DIPY's `TensorModel`. Used by `build_pretext_dataset.py` to create the Zarr store.
2. **`research/dataset.py`** (`DWISliceDataset`) - Loads one axial slice per sample from Zarr. Normalizes DWI by mean b0, scales DTI target by a computed `dti_scale`, pads all subjects to a shared `max_n`, produces a `vol_mask` and pre-computes a `brain_mask` per subject using `functions.brain_mask()`.
3. **`research/model.py`** (`QSpaceUNet`) - Q-space encoder (1x1 conv + FiLM conditioning from gradient MLP) compresses variable-N DWI into fixed features, then a 4-level 2D U-Net (64->128->256->512) predicts 6 DTI channels. Uses Cholesky parameterization to guarantee PSD tensors. `DTILoss` combines tensor MSE + FA/MD MAE (computed via Frobenius norms, no eigendecomposition -- MPS-safe), masked to brain voxels only.
4. **`research/train.py`** - AdamW + cosine annealing, gradient clipping, early stopping (patience=25). Subject-level split: 7 train / 2 val / 2 test. Logs scalars and image panels to TensorBoard. Loss and metrics are computed only within brain mask.
5. **`research/utils.py`** - Shared DWI-space metrics (PSNR, SSIM, RMSE, MAE, NRMSE), DTI-space metrics (FA/MD/ADC RMSE/MAE/R2), `sanitize_dti6d()` for eigenvalue clamping, and visualization helpers. Used by training and evaluation for consistent metric computation.
6. **`research/evaluate.py`** - Evaluates the trained model and baselines (Patch2Self, MP-PCA) on test subjects. All methods go through a shared `_compute_dti_metrics()` pipeline that sanitizes DTI tensors (eigenvalue clamping to `[0, MAX_DIFFUSIVITY]`) before computing metrics. Computes a shared brain mask per subject and applies it to all methods. Generates per-subject prediction plots and comparison plots (FA/ADC maps across all methods). Also produces metric CSVs and a bar-chart summary.

**Device handling:** `research/train.py` auto-selects CUDA > MPS > CPU. The FA/MD computation avoids eigendecomposition specifically for MPS compatibility.

## Dataset contract

Zarr store at `dataset/pretext_dataset_new.zarr` with per-subject groups (`sub-XX_ses-Y`):

| Array | Shape | Description |
|---|---|---|
| `input_dwi` | `(X, Y, Z, N)` | noisy DWI (input) |
| `target_dwi` | `(X, Y, Z, N)` | clean DWI (target) |
| `target_dti_6d` | `(X, Y, Z, 6)` | DTI tensor `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]` |
| `bvals` | `(N,)` | b-values |
| `bvecs` | `(3, N)` | gradient directions |

11 biological subjects, 18 sessions total. Spatial dims `(130, 132, 25)`, `N` varies (130 or 258). Train/val/test split operates on biological subject IDs to prevent data leakage.

## Shared configuration

All shared hyperparameters live in **`config.py`** at the project root. Every script imports from here instead of hardcoding values. This ensures consistency across data prep, training, baselines, and evaluation. Key parameters include `B0_THRESHOLD`, `KEEP_FRACTION`, noise levels, subject splits, training hyperparameters, `MAX_DIFFUSIVITY` (eigenvalue cap for DTI sanitization), and baseline-specific settings. CLI argument defaults also reference `config.py`.

## Development rules

- Keep `README.md` and `CLAUDE.md` aligned with actual repo structure.
- Do not commit datasets, Zarr stores, QC plots, model checkpoints, images, or CSVs (all in `.gitignore`).
- Reuse helpers in `functions.py` and `research/utils.py` instead of duplicating DWI/DTI logic.
- Add new hyperparameters to `config.py` rather than hardcoding them in individual scripts.
- Treat `dataset/` as large, local-only data.
- The `research/` module is imported as a package (`python -m research.train`), not run directly.
