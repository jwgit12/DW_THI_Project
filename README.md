# DW_THI_Project

Production-ready DWI preprocessing, QSpaceUNet training, and evaluation for
clean DWI -> 6D DTI tensor prediction.

The stable pipeline lives in `src/dw_thi/`, with thin root entry points for
preprocessing, training, evaluation, and visualization.

## Pipeline

```text
Raw DWI NIfTI + bval/bvec
  -> build_pretext_dataset.py
  -> dataset/default_clean.zarr
  -> train.py
  -> evaluate.py
```

## Dataset Contract

The production dataset is `dataset/default_clean.zarr`.

Each subject group contains:

| Array | Shape | Description |
|---|---:|---|
| `target_dwi` | `(X, Y, Z, N)` | Clean DWI |
| `target_dti_6d` | `(X, Y, Z, 6)` | `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]` |
| `brain_mask` | `(X, Y, Z)` | DIPY `median_otsu` mask from mean b0 |
| `bvals` | `(N,)` | Diffusion b-values |
| `bvecs` | `(3, N)` | Diffusion directions |

Noise and k-space degradation are applied on the fly during training and
evaluation. No degraded DWI is stored.

## Build The Dataset

```bash
python build_pretext_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/default_clean.zarr
```

`build_pretext_dataset.py` computes `brain_mask` with
`dipy.segment.mask.median_otsu` on the mean b0 volume and stores it in Zarr.
Older Zarr stores without `brain_mask` still run, but training will recompute
the mask at startup and warn you to rebuild for production.

## Train

```bash
python train.py
```

Defaults come from `config.py`, including:

- Dataset: `dataset/default_clean.zarr`
- Output: `runs/production`
- TensorBoard logs: `runs/production/tb`
- Subject split: `VAL_SUBJECTS` and `TEST_SUBJECTS`
- Model, optimizer, augmentation, AMP, compile, and DataLoader settings

Useful overrides:

```bash
python train.py --epochs 200 --batch_size 8 --out_dir runs/my_run
tensorboard --logdir runs/production/tb
```

Training uses CUDA automatically when available, Apple MPS on M-series Macs,
and CPU as a fallback. CUDA keeps the fast path: channels-last convolutions,
AMP, fused AdamW when available, optional `torch.compile` on Linux CUDA, and
GPU-side k-space degradation.

## Evaluate

```bash
python evaluate.py \
  --checkpoint runs/production/best_model.pt \
  --zarr_path dataset/default_clean.zarr
```

Evaluation writes CSV metrics and plots to `runs/evaluation` by default. It
can run QSpaceUNet alone or include Patch2Self and MP-PCA baselines:

```bash
python evaluate.py --skip_baselines
python evaluate.py --no-mppca
python evaluate.py --no-patch2self
```

## Configuration

All production defaults live in `config.py`. CLI arguments reference those
defaults and are meant for temporary overrides, not as a second config system.

Key sections:

- Paths: dataset, QC, training output, evaluation output
- Degradation: k-space keep fraction and Gaussian noise ranges
- Brain mask: DIPY median-otsu parameters
- Training: epochs, batch size, optimizer, warmup, patience
- Runtime: AMP dtype, channels-last, compile mode, worker count
- Evaluation and baseline settings

## Source Layout

```text
src/dw_thi/
  preprocessing.py  # DWI discovery/loading, DTI targets, median_otsu masks, Zarr build
  dataset.py        # Zarr-backed PyTorch slice dataset
  augment.py        # CPU/GPU k-space degradation
  model.py          # QSpaceUNet
  loss.py           # Masked tensor + FA/MD/edge loss
  train.py          # TensorBoard training loop
  evaluate.py       # Evaluation and baselines
  runtime.py        # CUDA/MPS/runtime helpers
  utils.py          # Metrics and plotting helpers
```

Root scripts (`build_pretext_dataset.py`, `train.py`, `evaluate.py`) are thin
entry points into this package.
