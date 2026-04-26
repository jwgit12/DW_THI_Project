# Data Pipeline — Precise Reference Map

Traces every transformation applied to data from raw NIfTI files through Zarr storage, on-the-fly degradation, augmentation, model forward pass, and loss computation.

---

## Phase 1 — Dataset Build (`build_pretext_dataset.py` → `preprocessing.py`)

Run once. Stores only clean data.

### 1.1 Discover files
[`preprocessing.py:65` `find_dwi_datasets`](src/dw_thi/preprocessing.py)
- Globs `*_dwi.nii.gz` under `--data_dir`
- Checks for matching `.bval` / `.bvec` sidecars
- Parses BIDS entities (sub, ses, run) → unique Zarr group key via [`parse_dwi_entities:34`](src/dw_thi/preprocessing.py)

### 1.2 Load raw DWI
[`preprocessing.py:90` `load_dwi_dataset`](src/dw_thi/preprocessing.py)
- `nibabel.load` → `get_fdata(dtype=float32)` → `(X, Y, Z, N)` float32
- `dipy.io.read_bvals_bvecs` → bvals `(N,)`, bvecs `(3, N)`
- `dipy.core.gradients.gradient_table` with `B0_THRESHOLD=50`

### 1.3 Brain mask
[`preprocessing.py:153` `compute_brain_mask_from_dwi`](src/dw_thi/preprocessing.py)
- [`mean_b0_volume:141`](src/dw_thi/preprocessing.py): averages all volumes with bval < 50 → `(X, Y, Z)` reference
- `dipy.segment.mask.median_otsu` with `median_radius=4`, `numpass=4`, `dilate=1`, `finalize_mask=True`
- Output: `(X, Y, Z)` bool mask

### 1.4 DTI fit
[`preprocessing.py:112` `compute_dti`](src/dw_thi/preprocessing.py)
- `dipy.reconst.dti.TensorModel(gtab, fit_method="WLS")`
- `.fit(data, mask=brain_mask)` — only brain voxels are fit
- Returns `quadratic_form` → `(X, Y, Z, 3, 3)` float32

### 1.5 Flatten to 6D
[`preprocessing.py:119` `tensor_to_6d`](src/dw_thi/preprocessing.py)
- Extracts 6 unique components: **[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]**
- Output: `(X, Y, Z, 6)` float32

### 1.6 Write Zarr
[`preprocessing.py:362`](src/dw_thi/preprocessing.py) inside `build_pretext_dataset`

Per subject group:
| Array | Shape | dtype |
|---|---|---|
| `target_dwi` | `(X, Y, Z, N)` | float32 |
| `target_dti_6d` | `(X, Y, Z, 6)` | float32 |
| `brain_mask` | `(X, Y, Z)` | uint8 |
| `bvals` | `(N,)` | float32 |
| `bvecs` | `(3, N)` | float32 |

No degraded data is stored. Degradation ranges are written only as metadata (`store.attrs["degradation_ranges"]`).

---

## Phase 2 — Dataset Initialization (`dataset.py`)

[`DWISliceDataset.__init__:82`](src/dw_thi/dataset.py)

Runs once at startup, builds index and global statistics.

- Loads `bvals`, `bvecs`, `brain_mask` for each subject into RAM (~few MB total). Large DWI/DTI arrays stay on disk.
- **`dti_scale`** [`dataset.py:205`](src/dw_thi/dataset.py): computed as `1 / percentile(abs(target_dti_6d), 99)` across all training subjects. Scales targets into O(1) range.
- **`max_bval`** [`dataset.py:202`](src/dw_thi/dataset.py): global maximum b-value across all subjects. Used to normalize bvals to [0, 1].
- **`canonical_hw`** [`dataset.py:198`](src/dw_thi/dataset.py): `(max_H, max_W)` across all slices and axes. All batches are padded to this.
- **`max_n`** [`dataset.py:176`](src/dw_thi/dataset.py): maximum number of DWI volumes across subjects. All batches are padded to this.
- **Sample index** [`dataset.py:184`](src/dw_thi/dataset.py): enumerates `(subject_key, axis, slice_index)` for every slice along every configured axis.

---

## Phase 3 — Per-Sample `__getitem__` (`dataset.py:215`)

Called by DataLoader for every sample. The full transform chain per slice:

### 3.1 Load 2D slice
[`dataset.py:228`](src/dw_thi/dataset.py)

Two paths:
- **Worker RAM path** (fast): arrays preloaded by [`preload_dataset_in_worker:418`](src/dw_thi/dataset.py) via `worker_init_fn`. Array indexed directly in RAM.
- **Lazy zarr path**: [`_get_zarr_group:37`](src/dw_thi/dataset.py) opens a cached store handle per process; reads one 2D slice from disk.

Slice extraction by `axis` (0=sagittal, 1=coronal, 2=axial):
- `clean`: `(H, W, N)` from `target_dwi`
- `tgt`: `(H, W, 6)` from `target_dti_6d`
- `bmask`: `(H, W)` from stored brain_mask

Transpose to channel-first [`dataset.py:263`](src/dw_thi/dataset.py):
- `clean_nhw`: `(N, H, W)`
- `tgt_chw`: `(6, H, W)`

### 3.2 On-the-fly degradation
[`dataset.py:275`](src/dw_thi/dataset.py)

**CPU path** (default, or `eval_mode`):
- Sample `kf ~ Uniform(KEEP_FRACTION_MIN=0.5, KEEP_FRACTION_MAX=0.7)` and `nl ~ Uniform(NOISE_MIN=0.01, NOISE_MAX=0.10)` per sample
- Call [`augment.py:73` `degrade_dwi_slice(clean_nhw, kf, nl, rng)`](src/dw_thi/augment.py):
  1. [`lowres_kspace_cutout:34`](src/dw_thi/augment.py): `scipy.fft.rfft2` on `(N, H, W)` → zero out rows `[ry : H-ry]` and columns `[rx :]` where `ry = int(H * kf / 2)`, `rx = int(W * kf / 2)` → `scipy.fft.irfft2` back to image space
  2. [`add_scaled_gaussian_noise:57`](src/dw_thi/augment.py): `sigma = nl * max(slice)` per volume → add `N(0, sigma)` noise

**GPU-deferred path** (CUDA only, `gpu_degrade=True`) [`dataset.py:285`](src/dw_thi/dataset.py):
- `clean_nhw` is returned unchanged; `degrade_kf`, `degrade_nl`, `b0_mask` added to the batch dict
- Actual degradation runs in `run_epoch` after the batch is on GPU via [`augment.py:120` `gpu_degrade_dwi_batch`](src/dw_thi/augment.py)

**Eval mode** [`dataset.py:279`](src/dw_thi/dataset.py):
- Fixed `kf=EVAL_KEEP_FRACTION=0.6`, `nl=EVAL_NOISE_LEVEL=0.055`
- Deterministic RNG seeded with `EVAL_DEGRADE_SEED + idx`

### 3.3 DTI target scaling
[`dataset.py:298`](src/dw_thi/dataset.py)
```
tgt_chw = clip(tgt_chw * dti_scale, -3.0, 3.0)
```
Scales the 6D tensor target into O(1) range using the global `dti_scale` from init.

### 3.4 b0 normalization of input
[`dataset.py:303`](src/dw_thi/dataset.py) (CPU path only; GPU path does this in `run_epoch`)
- Identifies b0 volumes: `bvals < B0_THRESHOLD=50`
- `mean_b0 = mean over b0 volumes` → `(H, W)`
- [`preprocessing.py:134` `compute_b0_norm`](src/dw_thi/preprocessing.py): `brain_voxels = mean_b0[mean_b0 > 0.1 * max]` → `norm = mean(brain_voxels)`
- `input_nhw = input_nhw / norm`

### 3.5 bval normalization
[`dataset.py:310`](src/dw_thi/dataset.py)
```
bvals_norm = bvals / max_bval
```
Normalizes b-values to [0, 1] using global `max_bval`.

### 3.6 Spatial augmentations (training only)
[`dataset.py:317`](src/dw_thi/dataset.py) — only when `augment=True`

**Flip** [`dataset.py:320`](src/dw_thi/dataset.py) — `AUG_FLIP=True`:
- Independent 50% chance to flip H-axis and W-axis
- Maps `(slice_axis, hw_axis)` → world axis (0=x, 1=y, 2=z)
- On flip: mirrors `input_nhw`, `tgt_chw`, `bmask_hw`
- Also applies:
  - [`_flip_dti6d_sign:55`](src/dw_thi/dataset.py): negates off-diagonal tensor components that touch the flipped axis (e.g., x-flip negates Dxy and Dxz at channels 1 and 3)
  - [`_flip_bvecs:63`](src/dw_thi/dataset.py): negates the corresponding bvec component

**Intensity jitter** [`dataset.py:341`](src/dw_thi/dataset.py) — `AUG_INTENSITY=0.1`:
```
scale ~ Uniform(0.9, 1.1)
input_nhw = input_nhw * scale
```

**Volume dropout** [`dataset.py:346`](src/dw_thi/dataset.py) — `AUG_VOLUME_DROPOUT=0.1`:
- Each volume independently zeroed with probability 0.1
- Zeroed volumes also set to 0 in `vol_mask`

### 3.7 Padding to batch shape
[`dataset.py:360`](src/dw_thi/dataset.py)

**Volume dimension**: zero-pad `input_nhw`, `bvals_norm`, `bvecs_n` from `N` to `max_n`.
`vol_mask` has 1 for real volumes, 0 for padding (and for dropout-zeroed volumes).

**Spatial dimensions** [`dataset.py:371`](src/dw_thi/dataset.py): zero-pad `input_nhw`, `tgt_chw`, `bmask_hw` from `(H, W)` to `canonical_hw`.

### 3.8 Output tensors
[`dataset.py:381`](src/dw_thi/dataset.py)

| Key | Shape | Description |
|---|---|---|
| `input` | `(max_n, H, W)` | degraded + normalized DWI |
| `target` | `(6, H, W)` | scaled clean DTI tensor |
| `bvals` | `(max_n,)` | normalized b-values in [0, 1] |
| `bvecs` | `(3, max_n)` | unit b-vectors |
| `vol_mask` | `(max_n,)` | 1=real, 0=pad/dropped |
| `brain_mask` | `(H, W)` | binary brain mask |
| `degrade_kf/nl/b0_mask` | scalars / `(max_n,)` | GPU-degrade path only |

---

## Phase 4 — GPU Degradation in `run_epoch` (CUDA training only)

[`train.py:532`](src/dw_thi/train.py)

When `degrade_kf` is present in the batch (GPU-degrade mode):

1. [`augment.py:120` `gpu_degrade_dwi_batch(signal, kf, nl)`](src/dw_thi/augment.py):
   - `torch.fft.rfft2` on `(B, N, H, W)`
   - Vectorized per-sample frequency masks (no Python loops): `row_keep`, `col_keep` → zero out high-frequency coefficients
   - `torch.fft.irfft2` → add `N(0, nl * max(slice))` noise
2. [`augment.py:166` `gpu_b0_normalize_batch(signal, b0_mask)`](src/dw_thi/augment.py):
   - GPU equivalent of the CPU b0 normalization in step 3.4

---

## Phase 5 — Model Forward Pass (`model.py`)

[`train.py:542`](src/dw_thi/train.py): `pred = model(signal, bvals, bvecs, vol_mask)`

### 5.1 Q-space encoder
[`model.py:75` `QSpaceEncoder.forward`](src/dw_thi/model.py)

Input: `signal (B, N, H, W)`, `bvals (B, N)`, `bvecs (B, 3, N)`, `vol_mask (B, N)`

1. Concatenate `[bval, bvec_x, bvec_y, bvec_z]` → `grad_info (B, N, 4)`
2. `grad_mlp`: Linear(4→128) → SiLU → Linear(128→128) → SiLU → Linear(128→feat_dim=64) → `e (B, N, 64)`
3. Zero out padded/dropped volumes: `e = e * vol_mask.unsqueeze(-1)`
4. Permutation-invariant aggregation: `einsum("bnhw,bnf->bfhw", signal, e)` → `(B, 64, H, W)`
5. Divide by effective volume count `n_eff = sum(vol_mask)` to decouple magnitude from N
6. `post`: two Conv2d(64→64, 3×3) + GroupNorm(8) + SiLU → `(B, 64, H, W)`

### 5.2 U-Net
[`model.py:159` `UNet2D.forward`](src/dw_thi/model.py)

Input: `(B, 64, H, W)`

- Reflect-pad to nearest multiple of `2^depth` (depth=3 → multiple of 8)
- **Encoder**: 3 levels of `ConvBlock` (Conv+GN+SiLU, dropout=0.1) + MaxPool2d(2)
  - 64→64, 64→128, 128→256
- **Bottleneck**: ConvBlock 256→512
- **Decoder**: 3 ConvTranspose2d(2×2) + skip concat + ConvBlock
  - 512→256, 256→128, 128→64
- **Head**: Conv2d(64→6, 1×1) → `(B, 6, H, W)`
- Crop back to original `(H, W)`

Each `ConvBlock` [`model.py:106`](src/dw_thi/model.py): `Conv2d → GroupNorm(8) → SiLU → Dropout2d → Conv2d → GroupNorm(8) → SiLU`

### 5.3 Cholesky output (when `cholesky=True`)
[`model.py:14` `cholesky_to_tensor6`](src/dw_thi/model.py)

Raw 6 channels interpreted as lower-triangular Cholesky factor `L`:
- Diagonal elements passed through `softplus` to guarantee positivity
- Computes `D = L @ L^T` → [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
- Guarantees the output tensor is positive semi-definite

---

## Phase 6 — Loss Computation (`loss.py`)

[`train.py:544`](src/dw_thi/train.py): `loss, metrics = criterion(pred.float(), target, mask=brain_mask)`

[`loss.py:81` `DTILoss.forward`](src/dw_thi/loss.py)

All terms are masked: only brain voxels contribute.

**Tensor term** (always active):
- `residual = pred - target` (both in dti_scale space)
- [`_charbonnier:39`](src/dw_thi/loss.py): `sqrt(residual² + eps²)` with `eps=1e-3` — behaves like MSE near zero, L1 for large residuals
- Normalized by `sum(brain_mask) * 6`

**Scalar terms** (`LAMBDA_SCALAR=0.3`):
- [`tensor6_to_fa_md:12`](src/dw_thi/loss.py): computes FA and MD analytically from 6D tensor using Frobenius norms (no eigendecomposition)
- `FA MAE` + `MD MAE` masked to brain voxels

**Edge term** (`LAMBDA_EDGE=0.1`):
- [`_spatial_grad_mag:48`](src/dw_thi/loss.py): `|Δx FA| + |Δy FA|` for pred and target
- MAE between gradient magnitudes, masked to brain

**Total loss**:
```
L = L_tensor + 0.3 * (L_FA + L_MD) + 0.1 * L_edge
```

---

## Phase 7 — Optimizer Step (`train.py`)

[`train.py:551`](src/dw_thi/train.py)

1. `optimizer.zero_grad(set_to_none=True)`
2. `scaler.scale(loss).backward()` (AMP) or `loss.backward()`
3. `scaler.unscale_(optimizer)` then `clip_grad_norm_(params, max_norm=1.0)`
4. `scaler.step(optimizer)` / `scaler.update()`

**Optimizer**: AdamW, `lr=1e-3`, `weight_decay=1e-4`, `betas=(0.9, 0.99)`, fused on CUDA

**LR schedule** [`train.py:874`](src/dw_thi/train.py):
- Epochs 1–5: `LinearLR` warmup from `lr×1e-3` to `lr`
- Epochs 6–150: `CosineAnnealingLR` from `lr` to `lr×0.01`

---

## Summary: Shape of Data Through the Pipeline

```
Raw NIfTI:         (X, Y, Z, N)  float32
Zarr target_dwi:   (X, Y, Z, N)  float32   [clean only]
Zarr target_dti:   (X, Y, Z, 6)  float32   [Dxx,Dxy,Dyy,Dxz,Dyz,Dzz]

DataLoader sample:
  input:           (max_n, H, W)  float32   [degraded, b0-normalized]
  target:          (6, H, W)      float32   [scaled by dti_scale]
  bvals:           (max_n,)       float32   [normalized to [0,1]]
  bvecs:           (3, max_n)     float32
  vol_mask:        (max_n,)       float32
  brain_mask:      (H, W)         float32

Q-space encoder:   (B, 64, H, W)  [permutation-invariant aggregation]
U-Net output:      (B, 6, H, W)   [raw]
After Cholesky:    (B, 6, H, W)   [PSD-guaranteed]
Loss:              scalar         [brain-masked Charbonnier + FA/MD + edge]
```
