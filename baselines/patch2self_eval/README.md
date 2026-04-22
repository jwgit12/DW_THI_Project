# Patch2Self Evaluation

Standalone Patch2Self tuning and evaluation for the clean Zarr dataset.

The runner loads each subject through `research.dataset.DWISliceDataset`, applies
the same k-space cutout and Gaussian noise degradation used by the research
pipeline, runs Patch2Self, then fits a 6D DTI tensor from the denoised DWI for
metric evaluation against `target_dti_6d`.

## Default Run

```bash
python -m baselines.patch2self_eval.run
```

Defaults:

- evaluates all Zarr subject/session groups in `dataset/default_clean.zarr`
- uses 5 reproducible degradation trials with different k-space/noise settings
- tunes a compact Patch2Self grid on validation subjects from `config.py`
- ranks tuned configs by FA RMSE from the fitted 6D tensor
- writes outputs under `baselines/patch2self_eval/results`

Useful options:

```bash
# Evaluate a subset.
python -m baselines.patch2self_eval.run --subjects sub-03 sub-04

# Skip tuning and use the supplied Patch2Self parameters.
python -m baselines.patch2self_eval.run \
  --skip_tuning \
  --p2s_model ridge \
  --p2s_alpha 0.1 \
  --p2s_no_b0_denoising \
  --p2s_clip_negative

# Also fit/evaluate DTI directly on the degraded input.
python -m baselines.patch2self_eval.run --include_noisy_baseline

# Save fitted Patch2Self 6D tensors as compressed files.
python -m baselines.patch2self_eval.run --save_fitted_tensors
```

## Outputs

- `degradation_plan.csv`: the five k-space/noise settings and seeds
- `tuning_trials.csv`: per-config tuning rows
- `tuning_summary.csv`: mean/std tuning metrics grouped by config
- `best_config.json`: selected Patch2Self parameters
- `final_trials.csv`: one row per subject/repeat, evaluated on fitted 6D tensors
- `final_summary.csv`: mean/std final metrics
- `metrics_patch2self.csv`: Patch2Self-only final rows
- `fitted_dti6d/*.npz`: optional fitted 6D tensors when `--save_fitted_tensors` is set
