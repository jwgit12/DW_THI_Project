# mppca_eval.py — MP-PCA Denoising Quality Evaluation

Evaluates [Marcenko-Pastur PCA (MP-PCA)](https://docs.dipy.org/stable/examples_built/preprocessing/denoise_mppca.html)
denoising quality against clean ground-truth targets stored in a Zarr dataset.
Nothing is written back to the store — only a metrics CSV is produced.

---

## Installation

```bash
pip install dipy zarr numpy pandas scikit-image
```

---

## Usage

```bash
# with all options explicit
python mppca.py \
    --zarr_path      /Users/jannis/PycharmProjects/DW_THI_Project/dataset/pretext_dataset.zarr \
    --out_dir        ./results \
    --n_jobs         6 \
    --patch_radius   2 \
    --pca_method     eig \
    --dti_fit_method WLS \
    --fa_mask_thresh 0.1
```

---

## Zarr layout required

```
pretext_dataset.zarr/
├── subject_000/
│   ├── input_dwi        (X, Y, Z, N)  float32   ← noisy DWI (input)
│   ├── bvals             (N,)          float32
│   ├── bvecs             (3, N)        float32
│   ├── target_dti_6d     (X, Y, Z, 6) float32   ← clean DTI tensor (target)
│   └── target_dwi        (X, Y, Z, N) float32   ← clean DWI (target)
└── subject_001/ ...
```

---

## What the script does

For each subject:

```
input_dwi
    │
    ▼
MP-PCA (mppca)
    │
    ├────────────────────────────────────────────┐
    ▼                                            ▼
denoised_dwi                             DTI fit (WLS)
    │                                            │
    ▼                                            ▼
DWI metrics                            denoised FA / ADC
(vs target_dwi)                                  │
                                                 ▼
                                       DTI metrics
                                 (vs FA/ADC from target_dti_6d)
```

---

## Output

```
./results/
└── metrics_per_subject.csv
```

One row per subject, plus `MEAN` and `STD` summary rows at the bottom:

```
subject,      dwi_psnr, dwi_ssim, dwi_rmse, dwi_mae, dwi_nrmse, fa_rmse, fa_r2, adc_rmse, adc_r2, ...
subject_000,  32.41,    0.923,    0.000821, ...
subject_001,  31.88,    0.919,    0.000904, ...
MEAN,         ...
STD,          ...
```

---

## Metrics

### DWI-space — denoised_dwi vs target_dwi

| Column | Metric | Better when |
|---|---|---|
| `dwi_psnr` | Peak Signal-to-Noise Ratio (dB) | Higher |
| `dwi_ssim` | Structural Similarity [0–1] | Higher |
| `dwi_rmse` | Root Mean Square Error | Lower |
| `dwi_mae` | Mean Absolute Error | Lower |
| `dwi_nrmse` | RMSE normalised by signal range — comparable across subjects | Lower |

SSIM is computed per axial slice and averaged across all slices and volumes.

### DTI-space — fitted FA/ADC vs target FA/ADC

Reported separately for **FA** (`fa_*`) and **ADC** (`adc_*`):

| Suffix | Metric | Better when |
|---|---|---|
| `_rmse` | Voxelwise root mean square error | Lower |
| `_mae` | Voxelwise mean absolute error | Lower |
| `_nrmse` | RMSE normalised by value range — comparable across subjects | Lower |
| `_r2` | Pearson R² — linear correlation [0–1] | Higher |

---

## Tensor component order

The stored `target_dti_6d` follows the order used by `functions.py tensor_to_6d`:

```
index 0: Dxx  |  index 1: Dxy  |  index 2: Dyy
index 3: Dxz  |  index 4: Dyz  |  index 5: Dzz
```

> **Note:** index 2 is **Dyy** and index 3 is **Dxz** — not the other way
> around. This is the most common source of silent bugs when reading DTI
> tensors from different software.

FA and ADC are derived identically from both the fitted and target tensors:

- FA uses the same formula as `compute_fa_from_tensor6` in `functions.py`
- ADC = mean of the three eigenvalues

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--zarr_path` | *(required)* | Path to the `.zarr` store |
| `--out_dir` | `./mppca_eval` | Directory for the output CSV |
| `--n_jobs` | `4` | Parallel workers |

### MP-PCA

| Flag | Default | Description |
|---|---|---|
| `--patch_radius` | `2` | Local patch radius in voxels (2 → 5×5×5 patches) |
| `--pca_method` | `eig` | PCA decomposition: `eig` (faster) or `svd` (occasionally more accurate) |
| `--b0_threshold` | `50` | b-value cutoff for b0 volumes (raise to ~70 for HCP 7T) |

### DTI

| Flag | Default | Description |
|---|---|---|
| `--dti_fit_method` | `WLS` | Fitting algorithm: `WLS` (robust), `OLS` (fast), `NLLS` (slow/accurate) |
| `--fa_mask_thresh` | `0.0` | Restrict DTI metrics to voxels where target FA exceeds this value. Set `0.1` for white-matter-only |
| `--brain_mask_frac` | `0.1` | Brain mask: fraction of max b0 signal. Voxels below this are excluded from DTI metrics |

---

## Choosing `--n_jobs`

Each worker holds one full `(X, Y, Z, N)` float32 volume plus a DTI fit in
memory simultaneously. Budget roughly `3 × subject_size` per worker.

| Setup | Recommended |
|---|---|
| Laptop | 2 |
| Workstation (16–32 cores) | 4–8 |
| HPC node | `ncores // 2` |
| Large subjects (e.g. HCP, ~8 GB each) | 2–3 |

---

## About MP-PCA

MP-PCA (Marcenko-Pastur PCA) exploits the Marcenko-Pastur distribution of
eigenvalues in random matrices to separate signal from noise components. It
operates on local patches (default 5×5×5 voxels) and automatically determines
how many principal components to retain based on noise statistics — no
hyperparameter tuning for the noise threshold is needed.

Key differences from Patch2Self:
- **Unsupervised**: does not use b-value information for denoising
- **Local PCA**: works on spatial patches rather than self-supervised regression
- **Automatic rank selection**: noise floor determined by Marcenko-Pastur law
- **Fast**: typically faster than Patch2Self since no regression is involved

---

## References

- Veraart J. et al. *Denoising of diffusion MRI using random matrix theory.* NeuroImage 2016.
- Veraart J. et al. *Diffusion MRI noise mapping using random matrix theory.* Magn. Reson. Med. 2016.
- Garyfallidis E. et al. *DIPY, a library for the analysis of diffusion MRI data.* Front. Neuroinformatics 2014.
