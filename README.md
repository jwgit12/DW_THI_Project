# DW_THI_Project

Deep-learning and baseline-denoising experiments for diffusion-weighted MRI (DW-MRI), with a focus on:

- denoising degraded DWI signal,
- reconstructing diffusion directions that were masked during training,
- predicting 6-component DTI tensors from noisy, partial diffusion inputs.

The repository contains:

- a full data-preparation pipeline from DWI NIfTI to Zarr,
- a direction-invariant residual U-Net denoising model with EMA training,
- classical baselines (MPPCA, Patch2Self) with per-subject evaluation,
- a PyQt6 dataset inspector,
- shared evaluation utilities (PSNR, SSIM, DTI metrics).

## Why this project exists

DW-MRI is noisy and relatively low-resolution. That makes downstream analysis (tractography, FA/MD quantification, microstructure studies) sensitive to image quality.

This project frames the problem as a pretext learning task:

- Input: noisy DWI plus missing gradient directions.
- Targets: clean DWI and clean DTI (6 independent tensor coefficients).

The hypothesis is that forcing the model to infer hidden gradient directions helps it learn diffusion structure priors that transfer to denoising/reconstruction.

## Current project status

- Pretext Zarr dataset at `dataset/pretext_dataset.zarr` with **18 subjects**
- Dataset generation params: `keep_fraction=0.5`, `noise_min=0.01`, `noise_max=0.05`
- Subject tensor shapes: `input_dwi (130, 132, 25, 258)`, `target_dti_6d (130, 132, 25, 6)`
- Training is early-stage; metrics not yet near clinical-quality targets

## Repository map

```text
DW_THI_Project/
├── functions.py                 # Core DWI utilities (loading, degradation, DTI)
├── build_pretext_dataset.py     # NIfTI -> Zarr preprocessing pipeline
├── visualizer.py                # PyQt6 dataset inspector GUI
├── requirements.txt             # Python dependencies
├── dti_prep.ipynb               # Exploratory notebook
├── baselines/
│   ├── utils.py                 # Shared metrics (PSNR, SSIM, DTI helpers)
│   ├── mppca/
│   │   ├── mppca.py             # MP-PCA denoising evaluation
│   │   └── results/             # Per-subject CSV metrics
│   └── patch2self/
│       ├── patch2self.py        # Patch2Self denoising evaluation
│       └── results/             # Per-subject CSV metrics
├── ml/
│   ├── denoise_dataset.py       # PyTorch Dataset (2D axial slices from Zarr)
│   ├── denoise_model.py         # DenoiseUNet architecture
│   ├── train_denoise.py         # Training loop (EMA, mixed precision, multi-loss)
│   ├── checkpoints/             # Saved model weights (best_denoise.pt)
│   └── runs/                    # TensorBoard logs
└── dataset/
    ├── dataset_v1/              # Raw NIfTI DWI files
    ├── dataset_v2/              # Additional raw data
    ├── dataset_v3/              # Additional raw data
    └── pretext_dataset.zarr/    # Processed training data (18 subjects)
```

## End-to-end pipeline

### 1) Source data discovery and loading

`functions.py` handles DWI dataset discovery and loading:

- finds `*_dwi.nii.gz` files,
- expects sidecar gradients at matching paths (`*.bval`, `*.bvec`),
- builds dipy `gradient_table` objects.

### 2) Synthetic degradation and targets

`build_pretext_dataset.py` performs per-subject preprocessing:

- noisy input creation:
  - k-space center-crop style masking via `keep_fraction`,
  - additive Gaussian noise with random level in `[noise_min, noise_max]`.
- clean target creation:
  - dipy tensor fit,
  - conversion from full 3x3 to 6-component representation (`Dxx, Dxy, Dyy, Dxz, Dyz, Dzz`).

Output is written to a Zarr store with compressed arrays.

### 3) Training sample construction

`ml/denoise_dataset.py` (`DWIDenoiseDataset`):

- samples 2D axial slices from each subject,
- optional augmentation (flips, 180-degree rotations),
- optional patch-wise training (random crops via `--patch_size`),
- per-slice percentile-based normalization,
- position encoding: normalized (y, x) coordinates appended per direction,
- custom collate function to pad variable direction counts per batch.

### 4) Model architecture

`ml/denoise_model.py` defines `DenoiseUNet`:

- shared direction-wise encoder operating on 7-channel per-direction input:
  - signal (1ch),
  - normalized b-value (1ch),
  - 3 b-vector components (3ch),
  - normalized position encoding (2ch).
- masked mean aggregation across valid (non-padded) directions,
- 4-level decoder with CBAM (Channel + Spatial) attention,
- residual learning: output = input + predicted residual,
- chunked direction processing (`dir_chunk_size`) to control memory.

Model size presets: `tiny` (16 features), `small` (24), `base` (48).

### 5) Training and validation

`ml/train_denoise.py` includes:

- loss: 70% Charbonnier + 20% MS-SSIM + 10% Edge (spatial gradients),
- EMA weight smoothing (decay=0.999),
- mixed precision (bfloat16 on MPS, float16 on CUDA),
- linear warmup + cosine annealing LR schedule,
- gradient accumulation and clipping,
- metrics: PSNR, SSIM in DWI domain,
- TensorBoard scalar/image logging,
- best-checkpoint saving to `ml/checkpoints/best_denoise.pt`.

## Baseline: MPPCA

`baselines/mppca/mppca.py` evaluates Marcenko-Pastur PCA denoising:

- uses dipy's `mppca` implementation,
- computes DWI-space metrics (PSNR, SSIM, RMSE, MAE, NRMSE),
- computes DTI-space metrics (FA-MAE, MD-MAE),
- outputs per-subject CSV to `baselines/mppca/results/`.

## Baseline: Patch2Self

`baselines/patch2self/patch2self.py` evaluates Patch2Self self-supervised denoising:

- uses dipy's Patch2Self implementation,
- same metric suite as MPPCA baseline,
- outputs per-subject CSV to `baselines/patch2self/results/`.

Shared evaluation utilities live in `baselines/utils.py`.

## Data format contract (Zarr)

Per subject group (`subject_XXX`):

| Array           | Shape          | Type    | Description                          |
|-----------------|----------------|---------|--------------------------------------|
| `input_dwi`     | (X, Y, Z, N)  | float32 | Noisy + low-res DWI                 |
| `target_dwi`    | (X, Y, Z, N)  | float32 | Clean ground truth DWI              |
| `target_dti_6d` | (X, Y, Z, 6)  | float32 | DTI tensor (6 independent coeffs)   |
| `bvals`         | (N,)           | float32 | b-values                            |
| `bvecs`         | (3, N)         | float32 | Gradient vectors                    |

Root attrs include generation provenance (`num_subjects`, `keep_fraction`, `noise_min`, `noise_max`).

## Environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

### Build pretext dataset

```bash
python3 build_pretext_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/pretext_dataset.zarr \
  --keep_fraction 0.5 \
  --noise_min 0.01 \
  --noise_max 0.05
```

### Inspect dataset interactively

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset.zarr
```

### Train denoising model

```bash
python3 ml/train_denoise.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --model_size base \
  --epochs 100 \
  --batch_size 2 \
  --accum_steps 4 \
  --augment
```

### Open TensorBoard

```bash
tensorboard --logdir ml/runs/
```

## Training CLI reference (`ml/train_denoise.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--zarr_path` | `dataset/pretext_dataset.zarr` | Path to Zarr dataset |
| `--model_size` | `base` | Model preset: `tiny`, `small`, `base` |
| `--base_features` | (from preset) | Override base feature width |
| `--drop_path` | `0.1` | Stochastic depth rate |
| `--patch_size` | `None` | Random crop size (full slice if unset) |
| `--augment` | `False` | Enable data augmentation |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `2` | Batch size |
| `--accum_steps` | `4` | Gradient accumulation steps |
| `--lr` | `1e-3` | Learning rate |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm |
| `--ema_decay` | `0.999` | EMA weight decay |
| `--dir_chunk_size` | `16` | Directions processed per chunk |
| `--alpha_ssim` | `0.2` | MS-SSIM loss weight |
| `--alpha_edge` | `0.1` | Edge loss weight |
| `--seed` | `42` | Random seed |
| `--num_workers` | `0` | DataLoader workers |
| `--log_dir` | `ml/runs/denoise` | TensorBoard log directory |
| `--ckpt_dir` | `ml/checkpoints` | Checkpoint directory |

## Apple Silicon notes

The training code includes MPS-aware paths:

- mixed precision with bfloat16 autocast,
- channels-last memory layout,
- direction chunking to reduce peak memory,
- periodic cache cleanup.

If you hit OOM: lower `--dir_chunk_size`, `--batch_size`, or `--model_size`.

## Known limitations / next priorities

- Validation metrics are still far from clinical-quality targets.
- No fixed train/val split artifact is persisted (split generated in code with seed 42).
- No standalone test harness for the deep learning pipeline.
- No experiment config/version registry.

Recommended next improvements:

1. Add experiment config files and reproducible run manifests.
2. Add model unit/integration tests for data contracts and loss/mask behavior.
3. Track train/val subject IDs in saved checkpoint metadata.
4. Introduce baseline-vs-model evaluation scripts on held-out subjects.

## Dataset references

- Fiber architecture in the ventromedial striatum and its relation with the bed nucleus of the stria terminalis:
  [OpenNeuro ds003047 v1.0.0](https://openneuro.org/datasets/ds003047/versions/1.0.0)

Potential additional datasets:

- [DWI Traveling Human Phantom Study (ds000206)](https://openneuro.org/datasets/ds000206/versions/00002)
- [SUDMEX_CONN (ds003346)](https://openneuro.org/datasets/ds003346/versions/1.1.2)
- [SCA2 Diffusion Tensor Imaging (ds001378)](https://openneuro.org/datasets/ds001378/versions/00003)
