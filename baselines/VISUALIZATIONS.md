# Baseline Visualization Suite

This project now includes a high-coverage baseline visualization pipeline:

- script: `baselines/visualize_baselines.py`
- outputs:
  - per-subject figure sets in `baselines/visualizations/run_*/subjects/<subject>/`
  - cross-subject summaries in `baselines/visualizations/run_*/summary/`
  - run-level `metrics.csv` and `report.md`

## What gets visualized

For each subject:

- qualitative DWI panels (`target`, each method, and absolute error maps)
- voxel value / residual distributions
- absolute error CDF + boxplots
- per-direction curves (`PSNR`, `MAE`, `MSE`) with `bvals`
- per-slice curves (`PSNR`, `MAE`)
- `b-value` vs error scatter plots
- runtime + metric summary table

Across subjects:

- method boxplots for `MSE`, `MAE`, `PSNR`
- runtime summary bars (mean ± std)
- per-subject PSNR improvements vs noisy baseline
- method score heatmap (`MSE`, `MAE`, `PSNR`, runtime)

## Example commands

Quick smoke run on one small crop:

```bash
python3 baselines/visualize_baselines.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --subject subject_000 \
  --x0 20 --x1 44 \
  --y0 20 --y1 44 \
  --z0 4 --z1 12 \
  --max_dirs 24 \
  --mppca_patch_radius 1 \
  --mppca_chunk_size 6
```

Fuller run over multiple subjects:

```bash
python3 baselines/visualize_baselines.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --max_subjects 4 \
  --max_dirs 64 \
  --methods mppca patch2self \
  --mppca_chunk_size 10 \
  --p2s_sketch_fraction 0.30
```

Patch2Self-only analysis:

```bash
python3 baselines/visualize_baselines.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --max_subjects 3 \
  --methods patch2self
```

