# Patch2Self Research Notes (for this baseline)

## Primary sources used
- NeurIPS 2020 paper (original method): [Patch2Self: Denoising Diffusion MRI with Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/bc047286b224b7bfa73d4cb02de1238d-Paper.pdf)
- CVPR 2024 paper (acceleration with sketching): [Patch2Self2: Self-supervised Denoising on Coresets via Matrix Sketching](https://openaccess.thecvf.com/content/CVPR2024/papers/Fadnavis_Patch2Self2_Self-supervised_Denoising_on_Coresets_via_Matrix_Sketching_CVPR_2024_paper.pdf)
- Official implementation reference (DIPY source): [dipy.denoise.patch2self source docs](https://docs.dipy.org/stable/_modules/dipy/denoise/patch2self.html)

## What Patch2Self does
Patch2Self denoises one held-out diffusion volume `v_j` by learning a predictor from all other volumes `v_-j` (J-invariant self-supervision). The core assumption is that noise is independent across diffusion measurements, so predicting one volume from the others suppresses independent noise while preserving shared structure.

## How DIPY currently implements it (v1.12, `version=3`)
- Uses a per-volume linear regressor (default OLS-like behavior).
- Splits volumes by `b0_threshold` into b0 and DWI groups and fits each group separately.
- Can skip b0 denoising (`b0_denoising=False`, or automatically when only one b0 exists).
- Uses matrix sketching (CountSketch-style) to reduce training rows and speed up fitting.
- `model` options are `ols`, `ridge`, `lasso`.

## What we implemented in this folder
File: `patch2self_trainable.py`

- Explicit `fit` and `predict` API so you can "train" Patch2Self as a baseline and save fitted weights.
- Reproduces DIPY v3-style behavior:
  - hold-one-volume-out regression,
  - b0/DWI split using `b0_threshold`,
  - optional b0 denoising,
  - `ols/ridge/lasso`,
  - CountSketch-style row sketching via `sketch_fraction`.
- Saves models (`coefficients`, `intercepts`) to `.npz`.

File: `train_patch2self_baseline.py`

- Runs this baseline on the project's Zarr dataset.
- Supports subject selection, crop selection, direction limits, and saving denoised outputs.
- Reports DWI-domain baseline metrics against `target_dwi`: MSE, MAE, PSNR.

## Notes on "training" for Patch2Self
Patch2Self is self-supervised and typically trained per subject (or per scan), not as a global deep model over many epochs. In this baseline, "training" means fitting the per-volume linear predictors for each selected subject.
