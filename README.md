# DW_THI_Project

Deep-learning and baseline-denoising experiments for diffusion-weighted MRI (DW-MRI), with a focus on:

- denoising degraded DWI signal,
- reconstructing diffusion directions that were masked during training,
- predicting 6-component DTI tensors from noisy, partial diffusion inputs.

The repository currently contains:

- a full data-preparation pipeline from DWI NIfTI to Zarr,
- a channel-invariant U-Net pretext model and training loop,
- a PyQt dataset inspector,
- a research-grade Torch MPPCA baseline implementation with dipy comparisons.

## Why this project exists

DW-MRI is noisy and relatively low-resolution. That makes downstream analysis (tractography, FA/MD quantification, microstructure studies) sensitive to image quality.

This project frames the problem as a pretext learning task:

- Input: noisy DWI plus missing gradient directions.
- Targets: clean DWI and clean DTI (6 independent tensor coefficients).

The hypothesis is that forcing the model to infer hidden gradient directions helps it learn diffusion structure priors that transfer to denoising/reconstruction.

## Current project status

Current local artifacts in this workspace indicate:

- pretext Zarr dataset exists at `dataset/pretext_dataset.zarr`
- dataset subjects: `18`
- dataset generation params (stored in Zarr attrs):
  - `keep_fraction=0.5`
  - `noise_min=0.01`
  - `noise_max=0.05`
- first subject tensor shapes:
  - `input_dwi`: `(130, 132, 25, 258)`
  - `target_dwi`: `(130, 132, 25, 258)`
  - `target_dti_6d`: `(130, 132, 25, 6)`
  - `bvals`: `(258,)`
  - `bvecs`: `(3, 258)`

A checkpoint exists at `ml/checkpoints/best_model.pt` with recorded metrics:

- epoch: `2`
- val_loss: `0.1947`
- val_psnr: `18.71 dB`
- val_ssim: `0.5813`
- val_fa_mae: `0.2938`
- val_md_mae: `0.0751`

These metrics suggest the training setup is functional but early-stage and not yet near target clinical-quality reconstruction.

## Repository map

```text
DW_THI_Project/
├── functions.py
├── build_pretext_dataset.py
├── visualizer.py
├── benchmark_mppca.py
├── plot_mppca_diff.py
├── baselines/
│   └── mppca/
│       └── mppca_torch.py
├── ml/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── checkpoints/
│   └── runs/
├── dataset/
│   ├── dataset_v1/
│   └── pretext_dataset.zarr/
├── dti_prep.ipynb
└── requirements.txt
```

## End-to-end pipeline

### 1) Source data discovery and loading

`functions.py` handles DWI dataset discovery and loading:

- finds `*_dwi.nii.gz` files,
- expects sidecar gradients at matching paths:
  - `*.bval`
  - `*.bvec`
- builds dipy `gradient_table` objects.

### 2) Synthetic degradation and targets

`build_pretext_dataset.py` performs per-subject preprocessing:

- noisy input creation (`lowres_noise`):
  - k-space center-crop style masking via `keep_fraction`,
  - additive Gaussian noise with random level in `[noise_min, noise_max]`.
- clean target creation:
  - dipy tensor fit,
  - conversion from full 3x3 to 6-component representation
    (`Dxx, Dxy, Dyy, Dxz, Dyz, Dzz`).

Output is written to a Zarr store with compressed arrays.

### 3) Training sample construction

`ml/dataset.py`:

- samples 2D axial slices from each subject,
- performs per-slice normalization,
- randomly masks a fraction of diffusion directions (`mask_fraction`, default `0.4`),
- uses a custom collate function to pad variable direction counts (`N`) per batch.

### 4) Model architecture

`ml/model.py` defines `PretextUNet`:

- shared direction-wise encoder operating on 6-channel per-direction input:
  - signal,
  - mask-flag,
  - normalized b-value,
  - 3 b-vector components.
- masked mean aggregation across valid (non-padded, unmasked) directions,
- two decoding heads:
  - DWI direction reconstruction head,
  - DTI 6-channel tensor prediction head.
- chunked direction processing (`dir_chunk_size`) to control memory.

### 5) Training and validation

`ml/train.py` includes:

- loss:
  - DWI: Charbonnier + SSIM blend,
  - DTI: Charbonnier,
  - weighted combination via `lambda_dwi` and `lambda_dti`.
- metrics:
  - PSNR/SSIM in DWI domain,
  - FA-MAE and MD-MAE derived from predicted tensors.
- logging/checkpointing:
  - TensorBoard scalar/image logging,
  - best-checkpoint saving to `ml/checkpoints/best_model.pt`.

## Baseline: Torch MPPCA

`baselines/mppca/mppca_torch.py` provides a PyTorch MPPCA implementation with:

- automatic device selection (`CUDA > MPS > CPU`),
- MPS-aware eigendecomposition fallback to CPU,
- chunked large-volume inference,
- built-in test suite for sanity and dipy agreement checks.

Related scripts:

- `benchmark_mppca.py`: benchmark Torch (MPS/CPU) vs dipy on a dataset block,
- `plot_mppca_diff.py`: generate qualitative comparison image (`mppca_comparison.png`).

## Baseline: Patch2Self (trainable)

`baselines/patch2self/patch2self_trainable.py` provides a trainable Patch2Self
baseline with:

- leave-one-volume-out self-supervised linear regression (J-invariant),
- separate handling of b0 and DWI groups via `b0_threshold`,
- optional CountSketch-style row sketching for faster fitting,
- model choices: `ols`, `ridge`, `lasso`,
- saved fitted models (`coefficients` + `intercepts`) per subject.

Main runner:

- `baselines/patch2self/train_patch2self_baseline.py`

## Data format contract (Zarr)

Per subject group (`subject_XXX`):

- `input_dwi`: `(X, Y, Z, N)` float32
- `target_dwi`: `(X, Y, Z, N)` float32
- `target_dti_6d`: `(X, Y, Z, 6)` float32
- `bvals`: `(N,)` float32
- `bvecs`: `(3, N)` float32
- `attrs.source_path`: original DWI path

Root attrs include generation provenance (`num_subjects`, `keep_fraction`, `noise_min`, `noise_max`).

## Environment and install

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your shell has no `python` alias, use `python3` (or `./.venv/bin/python`) for commands below.

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

### Train pretext model

```bash
python3 ml/train.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --epochs 100 \
  --batch_size 2 \
  --accum_steps 4 \
  --dir_chunk_size 16 \
  --base_features 32 \
  --num_workers 0
```

### Open TensorBoard

```bash
tensorboard --logdir ml/runs/
```

### Benchmark MPPCA baseline

```bash
python3 benchmark_mppca.py --device auto --repeats 3
```

### Train/evaluate Patch2Self baseline

```bash
python3 baselines/patch2self/train_patch2self_baseline.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --subject subject_000 \
  --model ols \
  --sketch_fraction 0.30 \
  --save_models
```

### Run MPPCA comparison plot

```bash
python3 plot_mppca_diff.py
```

## Training CLI reference (`ml/train.py`)

- `--zarr_path` (default: `dataset/pretext_dataset.zarr`)
- `--epochs` (default: `100`)
- `--batch_size` (default: `2`)
- `--accum_steps` (default: `4`)
- `--dir_chunk_size` (default: `16`)
- `--lr` (default: `1e-3`)
- `--mask_fraction` (default: `0.4`)
- `--alpha_ssim` (default: `0.2`)
- `--lambda_dwi` (default: `1.0`)
- `--lambda_dti` (default: `1.0`)
- `--base_features` (default: `32`)
- `--num_workers` (default: `0`)
- `--log_dir` (default: `ml/runs/pretext`)
- `--ckpt_dir` (default: `ml/checkpoints`)

## Apple Silicon notes

The training code already includes MPS-aware paths:

- optional mixed precision with bfloat16 autocast,
- channels-last memory layout,
- direction chunking to reduce peak memory,
- periodic cache cleanup.

If you hit OOM:

- lower `--dir_chunk_size`,
- lower `--batch_size`,
- lower `--base_features`.

## Known limitations / next priorities

- Validation metrics in current checkpoint are still far from ideal quality targets.
- No fixed random split artifact is persisted yet (split is generated in code with seed 42).
- No explicit experiment config/version registry yet.
- No standalone test harness for the deep model path (only MPPCA module ships with integrated tests).

Recommended next improvements:

1. Add experiment config files and reproducible run manifests.
2. Add model unit/integration tests for data contracts and loss/mask behavior.
3. Track train/val subject IDs in saved metadata.
4. Introduce baseline-vs-model evaluation scripts on held-out subjects.

## Dataset references

- Fiber architecture in the ventromedial striatum and its relation with the bed nucleus of the stria terminalis:
  [OpenNeuro ds003047 v1.0.0](https://openneuro.org/datasets/ds003047/versions/1.0.0)

Potential additional datasets:

- [DWI Traveling Human Phantom Study (ds000206)](https://openneuro.org/datasets/ds000206/versions/00002)
- [SUDMEX_CONN (ds003346)](https://openneuro.org/datasets/ds003346/versions/1.1.2)
- [SCA2 Diffusion Tensor Imaging (ds001378)](https://openneuro.org/datasets/ds001378/versions/00003)
