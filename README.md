# DW_THI_Project

Supervised DWI denoising and DTI prediction for a variable-`N` diffusion dataset. A deep learning model (QSpaceUNet) is trained to predict clean 6D DTI tensors directly from degraded DWI volumes, compared against Patch2Self and MP-PCA baselines.

## Current focus

- Supervised prediction: `input_dwi (X,Y,Z,N)` → `target_dti_6d (X,Y,Z,6)`
- 2D slice-based training with q-space encoding for variable `N`
- Tensor-aware training via combined MSE + differentiable FA/MD loss
- Subject-level train/val/test split with held-out evaluation
- Integrated evaluation of Patch2Self and MP-PCA baselines alongside the DL model
- Lightweight PyQt dataset inspection for raw DWI and derived DTI maps

## Dataset contract

The supervised Zarr dataset lives at `dataset/pretext_dataset_new.zarr/` and contains per-subject groups:

| Array | Shape | Description |
|---|---|---|
| `input_dwi` | `(X, Y, Z, N)` | noisy / degraded DWI |
| `target_dwi` | `(X, Y, Z, N)` | clean DWI target |
| `target_dti_6d` | `(X, Y, Z, 6)` | tensor target `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]` |
| `bvals` | `(N,)` | diffusion b-values |
| `bvecs` | `(3, N)` | diffusion gradient directions |

The dataset contains 11 biological subjects (18 sessions total, 7 subjects have 2 sessions).
Zarr groups are named by subject and session (e.g., `sub-01_ses-1`).
Spatial dims are `(130, 132, 25)`. `N` varies across subjects (130 or 258 volumes).
The train/val/test split operates on biological subject IDs to prevent data leakage.

## Repository structure

```text
DW_THI_Project/
├── functions.py                 # Core DWI loading, degradation, and DTI helpers
├── build_pretext_dataset.py     # NIfTI -> Zarr dataset builder
├── config.py                    # Shared hyperparameters (imported by all scripts)
├── visualizer.py                # PyQt6 viewer for the Zarr dataset
├── research/
│   ├── dataset.py               # Zarr slice-based PyTorch dataset
│   ├── model.py                 # QSpaceUNet (q-space encoder + 2D U-Net)
│   ├── loss.py                  # Masked DTI tensor + FA/MD loss
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation with integrated baselines
│   ├── utils.py                 # Shared DWI/DTI metrics, DTI sanitization, plotting
│   └── __init__.py              # Python package marker
├── requirements.txt             # Python dependencies
└── dti_prep.ipynb               # Exploratory notebook
```

## QSpaceUNet

`research/model.py` implements the model:

- **Q-space encoder**: 1x1 conv compresses `N` DWI volumes into `C` feature channels, with FiLM conditioning from a gradient table MLP (bvals/bvecs -> scale + shift)
- **2D U-Net backbone**: configurable encoder-decoder channel stack (default `64->128->256`) with skip connections, GroupNorm, and spatial dropout
- **Output head**: predicts 6-channel DTI tensor `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]`; `--cholesky` switches to a positive-semidefinite Cholesky parameterization

Variable `N` is handled by zero-padding to `max_n` with a volume mask. The gradient FiLM conditioning adapts features to the acquisition protocol.

## Training

`research/train.py` trains on 2D axial slices with:

- **Loss**: MSE on 6D tensor + lambda x (FA MAE + MD MAE), computed within a brain mask
- **FA/MD computation**: Frobenius norm formulation (no eigendecomposition -- MPS-safe)
- **Optimizer**: AdamW with cosine annealing, gradient clipping
- **Split**: train / val / test by biological subject ID, with all sessions grouped to prevent leakage. Defaults are `VAL_SUBJECTS = ["sub-05", "sub-11"]` and `TEST_SUBJECTS = ["sub-03", "sub-04"]` in `config.py`.
- **Augmentation**: random horizontal/vertical flips
- **Early stopping**: patience=25 on validation loss

### Weights & Biases monitoring

`research/train.py` starts one W&B run per training job and stores local run files in `{out_dir}/wandb/`.

**Scalars logged every epoch:**

| Tag | Description |
|---|---|
| `train_loss`, `val_loss` | Total combined loss |
| `train_tensor_mse`, `val_tensor_mse` | 6D tensor MSE component |
| `train_fa_mae`, `val_fa_mae` | FA MAE auxiliary loss component |
| `train_md_mae`, `val_md_mae` | MD MAE auxiliary loss component |
| `learning_rate` | Current learning rate |
| `epoch_time_s` | Wall-clock time per epoch |
| `baseline_patch2self_fa_rmse` | Patch2Self mean FA RMSE (flat reference line) |
| `baseline_mppca_fa_rmse` | MP-PCA mean FA RMSE (flat reference line) |

**Images logged every `--vis_every` epochs (default: 1):**

`val_prediction` -- a 2x4 panel showing for a fixed validation slice: target FA, predicted FA, FA error map, FA scatter plot (with RMSE and R2), and the same four panels for ADC.

**Automatically tracked by W&B:**

- Hyperparameters and derived run config (dataset split, model size, AMP/compile settings, loader settings)
- System metrics including GPU utilization, GPU memory, CPU usage, RAM, disk I/O, and network traffic

## Evaluation

`research/evaluate.py` runs the trained QSpaceUNet alongside Patch2Self and MP-PCA baselines on the same test subjects. All three methods are evaluated through a single shared metric pipeline (`_compute_dti_metrics`) which:

1. Sanitizes DTI tensors via eigenvalue clamping to `[0, MAX_DIFFUSIVITY]` so all methods are compared under identical physical constraints
2. Computes tensor RMSE, FA metrics (RMSE, MAE, NRMSE, R2), and ADC metrics (RMSE, MAE, NRMSE, R2) within a shared brain mask

Baseline denoised DWI is clipped to non-negative before DTI fitting to prevent unstable estimates from negative signal values.

Outputs:
- Per-method and per-subject metric CSVs
- Per-subject prediction plots (FA/ADC maps)
- Side-by-side comparison plots across all methods
- Summary bar chart of mean metrics

## Research modules

Only `research.train` and `research.evaluate` are command-line entry points. The other files are imported by those scripts.

| File | Usage |
|---|---|
| `research/dataset.py` | Defines `DWISliceDataset`, which preloads selected Zarr subjects into RAM, builds axial 2D samples, computes target-side brain masks from clean `target_dwi`, normalizes DWI by mean b0 signal, scales/clips tensor targets, pads variable `N`, and returns `vol_mask` plus `brain_mask`. |
| `research/model.py` | Defines `QSpaceUNet`, `QSpaceEncoder`, `UNet2D`, and optional Cholesky-to-tensor conversion. Imported by training, evaluation, and the viewer. |
| `research/loss.py` | Defines `DTILoss`, a masked tensor MSE plus optional FA/MD MAE auxiliary loss controlled by `--lambda_scalar`. |
| `research/utils.py` | Shared metric and plotting helpers: DWI metrics, DTI fitting, tensor sanitization, FA/ADC conversion, scalar-map metrics, automatic plot slice/volume selection, denoising plots, and prediction plots. |
| `research/train.py` | CLI for supervised QSpaceUNet training. Writes checkpoints, `history.json`, and W&B logs. |
| `research/evaluate.py` | CLI for checkpoint evaluation. Runs QSpaceUNet and, unless skipped, Patch2Self and MP-PCA baselines through the same brain-mask metric pipeline. |
| `research/__init__.py` | Marks `research` as an importable package for `python3 -m research.train` and `python3 -m research.evaluate`. |

### `research.train` options

```bash
python3 -m research.train \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/runs/run_01 \
  --epochs 150 \
  --batch_size 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --lambda_scalar 0.3 \
  --patience 25 \
  --vis_every 10 \
  --num_workers 0
```

Useful optional overrides:
- `--test_subjects sub-03 sub-04` and `--val_subjects sub-05 sub-11` override the biological-subject split.
- `--feat_dim 128` and `--channels 64 128 256` override the q-space feature width and U-Net channel stack.
- `--cholesky` enables positive-semidefinite tensor output parameterization.

Training outputs in `--out_dir`:
- `best_model.pt` and `last_model.pt`
- `history.json`
- `wandb/` local Weights & Biases run files

### `research.evaluate` options

```bash
python3 -m research.evaluate \
  --zarr_path dataset/default_dataset.zarr \
  --out_dir research/results
```

The default checkpoint is `research/runs/run_new_data_2_aug/best_model.pt`.

Subject selection:
- Default: checkpoint `test_subjects`; falls back to all Zarr groups if the checkpoint has no test split metadata.
- `--subjects sub-03 sub-04` accepts biological subject IDs and expands to matching session keys.
- `--subjects sub-03_ses-1` also accepts exact Zarr group keys.
- `--eval_all` evaluates every group in the Zarr store.

Evaluation controls:
- `--eval_repeats 3` controls how many independent corruptions are sampled per subject.
- `--eval_keep_fraction_min 0.5 --eval_keep_fraction_max 0.7` controls the sampled central k-space mask size.
- `--eval_noise_min 0.01 --eval_noise_max 0.10` controls the sampled relative Gaussian noise range.
- `--eval_seed 1234` controls reproducible repeat sampling.
- `--skip_baselines` runs only QSpaceUNet.
- `--skip_patch2self` / `--patch2self` disable or enable only Patch2Self.
- `--skip_mppca` / `--mppca` disable or enable only MP-PCA.
- `--p2s_model ridge --p2s_alpha 0.1` overrides Patch2Self model parameters.
- `--p2s_b0_denoising` / `--p2s_no_b0_denoising` toggles b0 denoising for Patch2Self.
- `--p2s_clip_negative` / `--p2s_no_clip_negative` toggles Patch2Self negative clipping.
- `--p2s_shift_intensity` / `--p2s_no_shift_intensity` toggles Patch2Self intensity shifting.
- `--sweep_patch2self` runs a validation-subject Patch2Self sweep and exits.
- `--skip_plot` disables PNG plot generation.
- `--b0_threshold 50` overrides the b0/DWI split threshold.
- `--plot_repeat 0` selects which repeat is visualized when multiple repeats are used.
- `--plot_slice_idx` and `--plot_volume_idx` force the axial slice and DWI volume used for saved plots; otherwise they are selected automatically.

Patch2Self validation sweep:

```bash
python3 -m research.evaluate \
  --sweep_patch2self \
  --eval_repeats 3 \
  --out_dir research/results/p2s_sweep
```

The sweep defaults to validation subjects from `config.py` unless `--subjects` is supplied. It writes `patch2self_sweep.csv` and `patch2self_sweep_summary.csv`; use the best row from the summary for a final test evaluation, for example:

```bash
python3 -m research.evaluate \
  --p2s_model ridge \
  --p2s_alpha 0.1 \
  --p2s_no_b0_denoising \
  --p2s_clip_negative
```

Evaluation outputs in `--out_dir`:
- `metrics_research.csv` with one row per subject/repeat plus mean/std rows
- `metrics_per_subject.csv` for backward-compatible research-model results with the same repeat rows
- `metrics_patch2self.csv` and `metrics_mppca.csv` when baselines are enabled
- `comparison_metrics.csv`, `comparison_per_subject.csv`, and `comparison_metrics.png` when any baseline is enabled
- `patch2self_sweep.csv` and `patch2self_sweep_summary.csv` when `--sweep_patch2self` is used
- `prediction_example_<subject>_repeat-00.png` for the selected plot repeat unless `--skip_plot` is used
- `comparison_<subject>_repeat-00.png` when any baseline and plots are enabled

## Shared configuration

All shared hyperparameters live in `config.py` at the project root. Every script imports from here instead of hardcoding values. This ensures consistency across data prep, training, baselines, and evaluation. Key parameters include `B0_THRESHOLD`, `KEEP_FRACTION`, noise levels, subject splits, training hyperparameters, `MAX_DIFFUSIVITY`, and baseline-specific settings. CLI argument defaults also reference `config.py`.

## Visualization tool

`visualizer.py` is a lightweight desktop viewer for the Zarr dataset.

- Subject selector for browsing the 18 stored subject groups
- Plane switcher for axial, coronal, and sagittal views
- Slice and diffusion-volume sliders that work with variable-`N` acquisitions
- Side-by-side panels for `input_dwi`, `target_dwi`, absolute difference, FA, MD, and color-FA
- Metadata sidebar with source file, shell counts, current `bval` / `bvec`, and a b-value scatter plot

For quick checks without opening the GUI, `--summary-only` prints the dataset shape and subject count in the terminal.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

Build the supervised Zarr dataset:

```bash
python3 build_pretext_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/pretext_dataset_new.zarr
```

Inspect the dataset:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr
```

Print a dataset summary without launching Qt:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr --summary-only
```

Train the DL model:

```bash
python3 -m research.train \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/runs/run_01 \
  --epochs 150 \
  --batch_size 8
```

Track training in Weights & Biases:

```bash
wandb login
```

Then launch training normally, or add `--wandb_mode offline` if you want local-only tracking without cloud sync.

Evaluate the trained model and baselines on test subjects:

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_01/best_model.pt \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/results
```

Skip baselines for a quick model-only evaluation:

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_01/best_model.pt \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --skip_baselines
```

Evaluate on all subjects (not just test split):

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_01/best_model.pt \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --eval_all
```
