# DW_THI_Project

DWI preprocessing, QSpaceUNet training, and evaluation. Two pipelines share
the same Zarr dataset format and on-the-fly degradation:

- **Standard FA/MD** — predict the 6D DTI tensor (`src/dw_thi/`)
- **fODF** — predict single-shell CSD SH coefficients (`src/fodf/`)

Shared infrastructure (preprocessing, augmentation, runtime helpers, metric
utilities) lives in `src/dw_thi/` and is imported by both packages. Root-level
entry points dispatch to the requested pipeline.

## Pipeline

For the fODF-specific reference flow, see
[`src/fodf/Pipeline.md`](Pipeline.md).

```text
Raw DWI NIfTI + bval/bvec
  -> build_dataset.py            # standard target_dti_6d
  -> build_fodf_dataset.py       # standard target + target_fodf_sh
  -> dataset/default_clean.zarr
  -> train.py --training {standard,f-odf}
  -> evaluate.py --training {standard,f-odf}
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
# Standard FA/MD targets only
python build_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/default_clean.zarr

# fODF: same DTI targets plus single-shell CSD SH coefficients
python build_fodf_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/default_odf.zarr
```

`build_dataset.py` computes `brain_mask` with
`dipy.segment.mask.median_otsu` on the mean b0 volume and stores it in Zarr.
Older Zarr stores without `brain_mask` still run, but training will recompute
the mask at startup and warn you to rebuild for production.

## Train

```bash
python train.py
python train.py --training standard
python train.py --training f-odf
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
python train.py --training f-odf --epochs 220 --batch_size 8
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

python evaluate.py --training f-odf \
  --checkpoint runs/production_fodf_context_l4/best_model.pt \
  --zarr_path dataset/default_odf.zarr
```

Evaluation writes CSV metrics and plots to `runs/evaluation` (standard) or
`runs/evaluation_fodf` (fODF) by default. Both pipelines support comparison
against Patch2Self and MP-PCA baselines:

```bash
python evaluate.py --skip_baselines
python evaluate.py --no-mppca
python evaluate.py --no-patch2self
```

## Configuration

All production defaults live in `config.py`. CLI arguments reference those
defaults and are meant for temporary overrides, not as a second config system.
The standard FA/MD path uses the unprefixed defaults; fODF uses the `FODF_*`
section through `src/fodf/defaults.py`.

Key sections:

- Paths: dataset, QC, training output, evaluation output
- Degradation: k-space keep fraction and Gaussian noise ranges
- Brain mask: DIPY median-otsu parameters
- Training: epochs, batch size, optimizer, warmup, patience
- Runtime: AMP dtype, channels-last, compile mode, worker count
- Evaluation and baseline settings

## Source Layout

```text
src/dw_thi/                     # standard FA/MD pipeline + shared infra
  preprocessing.py              # DWI discovery/loading, DTI/SH targets, masks, Zarr build
  dataset.py                    # Zarr-backed PyTorch slice dataset
  augment.py                    # CPU/GPU k-space degradation (shared)
  runtime.py                    # CUDA/MPS/runtime helpers (shared)
  utils.py                      # DTI metrics + plotting helpers (shared)
  model.py                      # QSpaceUNet (DTI head)
  loss.py                       # Charbonnier tensor + FA/MD + edge loss
  train.py                      # TensorBoard training loop
  evaluate.py                   # Evaluation and baselines

src/fodf/                       # fODF pipeline (sibling of dw_thi)
  defaults.py                   # Reads FODF_* knobs from config.py
  dataset.py                    # 2.5D context slices + fODF SH targets
  model.py                      # QSpaceUNet variant emitting SH coefficients
  loss.py                       # Band/peak/anisotropic correlation losses
  train.py                      # fODF training loop
  evaluate.py                   # fODF eval + MRtrix CSD baseline
  Pipeline.md                   # Detailed transformation reference
```

Root scripts dispatch to the right package:

| Script | Notes |
|---|---|
| `build_dataset.py` | Standard build (clean DWI + DTI targets + brain mask) |
| `build_fodf_dataset.py` | Same plus single-shell CSD SH targets |
| `train.py` | Dispatcher: `--training {standard,f-odf}` |
| `evaluate.py` | Dispatcher: `--training {standard,f-odf}` |
| `visualizer.py` | Unified Qt viewer (handles standard, fODF, or both — pass `--checkpoint` for auto-routing or `--dti_checkpoint`/`--fodf_checkpoint` explicitly) |
