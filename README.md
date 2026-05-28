# DW_THI_Project

Supervised DWI denoising and DTI prediction. A `QSpaceUNet` is trained to
predict clean 6D diffusion tensors directly from degraded diffusion-weighted
images. The Zarr store holds only clean targets; k-space cutout and magnitude
noise are sampled on the fly so every epoch sees a fresh corruption of the same
clean slice. The model is compared against Patch2Self and MP-PCA baselines.

## Pipeline

```text
Raw DWI NIfTI + bval/bvec
  -> build_dataset.py            # brain mask + DTI fit, clean targets only
  -> dataset/default_clean.zarr
  -> train.py                    # QSpaceUNet, on-the-fly degradation
  -> evaluate.py                 # metrics + Patch2Self / MP-PCA baselines
  -> report.py                   # self-contained HTML run report
```

## Dataset Contract

The production dataset is `dataset/default_clean.zarr`. Each subject group
contains:

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
python build_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/default_clean.zarr
```

`build_dataset.py` computes `brain_mask` with `dipy.segment.mask.median_otsu`
on the mean b0 volume and stores it in Zarr. Older Zarr stores without
`brain_mask` still run, but training will recompute the mask at startup and
warn you to rebuild for production.

## Train

```bash
python train.py
python train.py --epochs 200 --batch_size 8 --out_dir runs/my_run
tensorboard --logdir runs/production/tb
```

Defaults come from `config.py`, including:

- Dataset: `dataset/default_clean.zarr`
- Output: `runs/production`
- TensorBoard logs: `runs/production/tb`
- Subject split: `VAL_SUBJECTS` and `TEST_SUBJECTS`
- Model, optimizer, augmentation, AMP, compile, and DataLoader settings

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

Evaluation writes CSV metrics and plots to `runs/evaluation` by default, and
compares against Patch2Self and MP-PCA baselines:

```bash
python evaluate.py --skip_baselines
python evaluate.py --no-mppca
python evaluate.py --no-patch2self
```

## Report

`report.py` reads a run's `config.json` and `history.json` and renders a
single self-contained HTML file that explains the data flow and embeds the
training curves:

```bash
python report.py --run_dir runs/production
python report.py --run_dir runs/production --out report.html
```

## Visualize

```bash
python visualizer.py --zarr_path dataset/default_clean.zarr
python visualizer.py --zarr_path dataset/default_clean.zarr \
  --checkpoint runs/production/best_model.pt
```

The Qt viewer shows clean vs degraded DWI, the predicted vs ground-truth DTI
maps, and deterministic tractography from the principal eigenvector field.

## Configuration

All production defaults live in `config.py`. CLI arguments reference those
defaults and are meant for temporary overrides, not as a second config system.
`mps_config.py` is an optional smaller-model variant for local MPS runs.

Key sections:

- Paths: dataset, QC, training output, evaluation output
- Degradation: k-space keep fraction, noise level range, and noise model
  (Rician / chi / Gaussian)
- Brain mask: DIPY median-otsu parameters
- Training: epochs, batch size, optimizer, warmup, patience
- Runtime: AMP dtype, channels-last, compile mode, worker count
- Evaluation and baseline settings

## Source Layout

```text
src/dw_thi/                     # pipeline + shared infra
  preprocessing.py              # DWI discovery/loading, DTI fit, masks, Zarr build
  dataset.py                    # Zarr-backed PyTorch slice dataset
  augment.py                    # CPU/GPU k-space degradation + noise
  runtime.py                    # CUDA/MPS/runtime helpers
  utils.py                      # DTI metrics + plotting helpers
  model.py                      # QSpaceUNet (q-space encoder + 2D U-Net)
  loss.py                       # Charbonnier tensor + FA/MD + edge loss
  train.py                      # TensorBoard training loop
  evaluate.py                   # Evaluation and baselines

build_dataset.py                # Build the clean Zarr dataset
train.py                        # Train QSpaceUNet
evaluate.py                     # Evaluate + Patch2Self / MP-PCA baselines
report.py                       # Self-contained HTML run report
visualizer.py                   # Qt viewer (DWI / DTI / tractography)
```
