# patch2self_eval.py — Patch2Self Denoising Quality Evaluation

Evaluates [Patch2Self](https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html)
denoising quality against clean ground-truth targets stored in a Zarr dataset.
Nothing is written back to the store — the script writes a metrics CSV and an
example denoising PNG to the output directory.

---

## Installation

```bash
pip install dipy zarr numpy pandas scikit-image matplotlib
```

---

## Usage

```bash
# with all options explicit
python patch2self.py \
    --zarr_path      /Users/jannis/PycharmProjects/DW_THI_Project/dataset/pretext_dataset.zarr \
    --out_dir        ./results \
    --n_jobs         6 \
    --model          ols \
    --b0_threshold   50 \
    --dti_fit_method WLS \
    --fa_mask_thresh 0.1 \
    --plot_subject   subject_000
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
Patch2Self
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
├── metrics_per_subject.csv
└── denoising_example_subject_000.png
```

The PNG reproduces the notebook-style 2D slice view with:
- noisy input slice
- denoised slice
- target slice
- normalized difference map between target and denoised output
- k-space views for all four panels in a second row

By default, the script saves one example plot for the first valid subject. Use
`--plot_subject`, `--plot_slice_idx`, and `--plot_volume_idx` to override the
selection, or `--skip_plot` to disable it.

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
| `--out_dir` | `./patch2self_eval` | Directory for the output CSV |
| `--n_jobs` | `4` | Parallel workers |
| `--skip_plot` | off | Disable saving the denoising example PNG |
| `--plot_subject` | first valid subject | Subject key to visualize |
| `--plot_slice_idx` | auto | Axial slice index for the example plot |
| `--plot_volume_idx` | auto | DWI volume index for the example plot |

### Patch2Self

| Flag | Default | Description |
|---|---|---|
| `--model` | `ols` | Regression model: `ols`, `ridge`, or `lasso` |
| `--alpha` | `1.0` | Regularisation strength for `ridge` / `lasso` |
| `--b0_threshold` | `50` | b-value cutoff for b0 volumes (raise to ~70 for HCP 7T) |
| `--no_shift` | off | Disable `shift_intensity` — not recommended |
| `--clip_negative` | off | Clip negative values to 0 after denoising |
| `--skip_b0` | off | Skip denoising of b0 volumes (if b0s have artefacts) |

### DTI

| Flag | Default | Description |
|---|---|---|
| `--dti_fit_method` | `WLS` | Fitting algorithm: `WLS` (robust), `OLS` (fast), `NLLS` (slow/accurate) |
| `--fa_mask_thresh` | `0.0` | Restrict DTI metrics to voxels where target FA exceeds this value. Set `0.1` for white-matter-only |

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

## Expected metric ranges

| Metric | Poor | Acceptable | Good |
|---|---|---|---|
| `dwi_psnr` | < 28 dB | 28–33 dB | > 33 dB |
| `dwi_ssim` | < 0.80 | 0.80–0.92 | > 0.92 |
| `dwi_nrmse` | > 0.10 | 0.03–0.10 | < 0.03 |
| `fa_r2` | < 0.70 | 0.70–0.88 | > 0.88 |
| `adc_r2` | < 0.70 | 0.70–0.90 | > 0.90 |
| `fa_nrmse` | > 0.15 | 0.05–0.15 | < 0.05 |

---

## References

- Fadnavis S. et al. *Patch2Self: Denoising Diffusion MRI with Self-supervised Learning.* NeurIPS 2020.
- Fadnavis S. et al. *Patch2Self2: Self-supervised Denoising on Coresets via Matrix Sketching.* CVPR 2024.
- Basser P.J. & Pierpaoli C. *Microstructural and Physiological Features of Tissues Elucidated by Quantitative-Diffusion-Tensor MRI.* J. Magn. Reson. 1996.
- Garyfallidis E. et al. *DIPY, a library for the analysis of diffusion MRI data.* Front. Neuroinformatics 2014.
