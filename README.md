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

### TensorBoard monitoring

`research/train.py` writes TensorBoard event files to `{out_dir}/tb/`.

**Scalars logged every epoch:**

| Tag | Description |
|---|---|
| `loss/train`, `loss/val` | Total combined loss |
| `tensor_mse/train`, `tensor_mse/val` | 6D tensor MSE component |
| `fa_mae/train`, `fa_mae/val` | FA MAE auxiliary loss component |
| `md_mae/train`, `md_mae/val` | MD MAE auxiliary loss component |
| `lr` | Current learning rate |
| `baselines/fa_rmse/patch2self` | Patch2Self mean FA RMSE (flat reference line) |
| `baselines/fa_rmse/mppca` | MP-PCA mean FA RMSE (flat reference line) |

**Images logged every `--vis_every` epochs (default: 10):**

`val_prediction` -- a 2x4 panel showing for a fixed validation slice: target FA, predicted FA, FA error map, FA scatter plot (with RMSE and R2), and the same four panels for ADC.

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
| `research/train.py` | CLI for supervised QSpaceUNet training. Writes checkpoints, `history.json`, and TensorBoard logs. |
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
- `tb/` TensorBoard event files

### `research.evaluate` options

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_01/best_model.pt \
  --zarr_path dataset/default_dataset.zarr \
  --out_dir research/results
```

Subject selection:
- Default: checkpoint `test_subjects`; falls back to all Zarr groups if the checkpoint has no test split metadata.
- `--subjects sub-03 sub-04` accepts biological subject IDs and expands to matching session keys.
- `--subjects sub-03_ses-1` also accepts exact Zarr group keys.
- `--eval_all` evaluates every group in the Zarr store.

Evaluation controls:
- `--skip_baselines` runs only QSpaceUNet.
- `--skip_plot` disables PNG plot generation.
- `--b0_threshold 50` overrides the b0/DWI split threshold.
- `--plot_slice_idx` and `--plot_volume_idx` force the axial slice and DWI volume used for saved plots; otherwise they are selected automatically.

Evaluation outputs in `--out_dir`:
- `metrics_research.csv`
- `metrics_per_subject.csv` for backward-compatible research-model results
- `metrics_patch2self.csv` and `metrics_mppca.csv` when baselines are enabled
- `comparison_metrics.csv`, `comparison_per_subject.csv`, and `comparison_metrics.png` when all methods are available
- `prediction_example_<subject>.png` for each evaluated subject unless `--skip_plot` is used
- `comparison_<subject>.png` for each evaluated subject when baselines and plots are enabled

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

Monitor training with TensorBoard:

```bash
tensorboard --logdir research/runs/run_01/tb
```

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
