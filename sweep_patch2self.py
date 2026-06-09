"""Grid search for Patch2Self hyperparameters — no model checkpoint required.

Swept parameters:
  model           ols | ridge | lasso
  alpha           regularisation strength (ridge/lasso only)
  b0_threshold    b-value threshold separating b0 from DWI volumes
  b0_denoising    whether to include b0 volumes in the denoising
  shift_intensity intensity-shift bias correction
  clip_negative   clip negative denoised values to zero

Runs on VAL_SUBJECTS by default to avoid test-set leakage.
Saves to runs/sweep_patch2self/ :
  patch2self_grid.csv          one row per config × subject × repeat
  patch2self_grid_summary.csv  mean ± std per config, sorted by metric

Usage:
    python sweep_patch2self.py
    python sweep_patch2self.py --subjects sub-05 --eval_repeats 3
    python sweep_patch2self.py --models ols ridge lasso
    python sweep_patch2self.py --alphas 0.001 0.01 0.1 1.0 10.0
    python sweep_patch2self.py --b0_thresholds 50 75 100
    python sweep_patch2self.py --metric fa_r2 --out_dir runs/sweep_p2s_v2
"""

import argparse
import itertools
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.dw_thi.evaluate import _baseline_dti_metrics, _next_degradation_trial, _expand_subjects
from src.dw_thi.augment import degrade_dwi_volume
from src.dw_thi.preprocessing import compute_brain_mask_from_dwi
from src.dw_thi.runtime import path_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

METRIC_COLS = [
    "tensor_rmse",
    "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
    "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
]
HIGHER_IS_BETTER = {"fa_r2", "adc_r2"}


def _run_patch2self(noisy, bvals, model, alpha, b0_threshold,
                    b0_denoising, shift_intensity, clip_negative):
    from dipy.denoise.patch2self import patch2self
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return patch2self(
            noisy, bvals,
            model=model,
            alpha=alpha,
            b0_threshold=b0_threshold,
            b0_denoising=b0_denoising,
            shift_intensity=shift_intensity,
            clip_negative_vals=clip_negative,
            verbose=False,
        ).astype("float32")


def _build_grid(args) -> list[dict]:
    configs = []
    b0_denoising_values = [v.lower() == "true" for v in args.b0_denoising]
    shift_values = [v.lower() == "true" for v in args.shift_intensity]
    clip_values = [v.lower() == "true" for v in args.clip_negative]

    for model in args.models:
        alpha_values = args.alphas if model != "ols" else [args.alphas[0]]
        for alpha, b0_thr, b0_den, shift, clip in itertools.product(
            alpha_values, args.b0_thresholds,
            b0_denoising_values, shift_values, clip_values,
        ):
            configs.append(dict(
                model=model,
                alpha=float(alpha),
                b0_threshold=float(b0_thr),
                b0_denoising=bool(b0_den),
                shift_intensity=bool(shift),
                clip_negative=bool(clip),
            ))

    # Deduplicate (OLS alpha collapse creates dupes)
    unique, seen = [], set()
    for c in configs:
        key = (c["model"], c["alpha"], c["b0_threshold"],
               c["b0_denoising"], c["shift_intensity"], c["clip_negative"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def run_sweep(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = path_str(args.zarr_path)
    store = zarr.open_group(zarr_path, mode="r")
    all_keys = sorted(store.keys())

    if args.subjects:
        subjects = _expand_subjects(args.subjects, all_keys)
    else:
        subjects = _expand_subjects(cfg.VAL_SUBJECTS, all_keys) or all_keys

    if not subjects:
        log.error("No subjects found.")
        return

    configs = _build_grid(args)
    sort_ascending = args.metric not in HIGHER_IS_BETTER

    log.info(
        "Patch2Self grid search: %d configs × %d subjects × %d repeats",
        len(configs), len(subjects), args.eval_repeats,
    )
    log.info("Subjects: %s", subjects)

    eval_rng = np.random.default_rng(cfg.EVAL_DEGRADE_SEED)
    rows = []

    for subject_key in subjects:
        grp = store[subject_key]
        target_dti6d = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
        target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
        bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
        bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)
        if "brain_mask" in set(grp.array_keys()):
            mask_3d = np.asarray(grp["brain_mask"][:], dtype=bool)
        else:
            mask_3d = compute_brain_mask_from_dwi(target_dwi, bvals, cfg.B0_THRESHOLD)

        for repeat_idx in range(args.eval_repeats):
            trial = _next_degradation_trial(
                eval_rng, repeat_idx=repeat_idx,
                keep_fraction_range=(cfg.EVAL_KEEP_FRACTION_MIN, cfg.EVAL_KEEP_FRACTION_MAX),
                noise_range=(cfg.EVAL_NOISE_MIN, cfg.EVAL_NOISE_MAX),
            )
            input_dwi = degrade_dwi_volume(
                target_dwi,
                keep_fraction=trial["keep_fraction"],
                rel_noise_level=trial["noise_level"],
                seed=trial["degrade_seed"],
            )

            for cfg_idx, p2s_cfg in enumerate(configs):
                t0 = time.time()
                try:
                    denoised = _run_patch2self(
                        input_dwi, bvals,
                        model=p2s_cfg["model"],
                        alpha=p2s_cfg["alpha"],
                        b0_threshold=p2s_cfg["b0_threshold"],
                        b0_denoising=p2s_cfg["b0_denoising"],
                        shift_intensity=p2s_cfg["shift_intensity"],
                        clip_negative=p2s_cfg["clip_negative"],
                    )
                    metrics, _ = _baseline_dti_metrics(
                        denoised, target_dti6d, bvals, bvecs,
                        b0_threshold=p2s_cfg["b0_threshold"],
                        dti_fit_method=cfg.DTI_FIT_METHOD,
                        mask=mask_3d,
                    )
                except Exception as exc:
                    log.warning("FAIL  %s r=%d cfg=%d  %s", subject_key, repeat_idx, cfg_idx, exc)
                    continue

                row = {
                    "subject": subject_key,
                    "repeat": repeat_idx,
                    "keep_fraction": round(float(trial["keep_fraction"]), 6),
                    "noise_level": round(float(trial["noise_level"]), 6),
                    "config_idx": cfg_idx,
                    **p2s_cfg,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                row.update(metrics)
                rows.append(row)

                log.info(
                    "%-14s r=%02d cfg=%03d  model=%-6s alpha=%.4g  "
                    "b0thr=%.0f  b0den=%s  %s=%.6f  (%.1fs)",
                    subject_key, repeat_idx, cfg_idx,
                    p2s_cfg["model"], p2s_cfg["alpha"],
                    p2s_cfg["b0_threshold"], p2s_cfg["b0_denoising"],
                    args.metric, metrics[args.metric], time.time() - t0,
                )

    if not rows:
        log.error("No rows collected.")
        return

    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / "patch2self_grid.csv"
    raw_df.to_csv(raw_path, index=False)
    log.info("Saved raw rows -> %s", raw_path)

    group_cols = ["config_idx", "model", "alpha", "b0_threshold",
                  "b0_denoising", "shift_intensity", "clip_negative"]
    avail = [c for c in METRIC_COLS if c in raw_df.columns]
    summary = raw_df.groupby(group_cols, dropna=False)[avail].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(
        f"{args.metric}_mean", ascending=sort_ascending
    ).reset_index(drop=True)

    summary_path = out_dir / "patch2self_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved summary -> %s", summary_path)

    best = summary.iloc[0]
    print(f"\n{'=' * 72}")
    print("  Patch2Self Best Config")
    print(f"{'=' * 72}")
    print(
        f"model={best['model']}  alpha={best['alpha']:.6g}  "
        f"patch_radius=b0_threshold={best['b0_threshold']:.0f}  "
        f"b0_denoising={best['b0_denoising']}  "
        f"{args.metric}_mean={best[f'{args.metric}_mean']:.6f}"
    )
    print("\nTop-10 configs:")
    top_cols = ["model", "alpha", "b0_threshold", "b0_denoising",
                f"{args.metric}_mean", f"{args.metric}_std"]
    print(summary[top_cols].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Patch2Self hyperparameter grid search")
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default="runs/sweep_patch2self")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs to evaluate (default: VAL_SUBJECTS)")
    parser.add_argument("--eval_repeats", type=int, default=cfg.EVAL_REPEATS)

    parser.add_argument("--models", nargs="+", choices=["ols", "ridge", "lasso"],
                        default=["ols", "ridge"],
                        help="Models to include in the grid")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.01, 0.1, 1.0, 10.0],
                        help="Alpha values to sweep (for ridge/lasso)")
    parser.add_argument("--b0_thresholds", nargs="+", type=float,
                        default=[50.0, 100.0],
                        help="b0 threshold values to sweep")
    parser.add_argument("--b0_denoising", nargs="+", type=str.lower,
                        choices=["true", "false"], default=["true", "false"],
                        help="b0 denoising on/off")
    parser.add_argument("--shift_intensity", nargs="+", type=str.lower,
                        choices=["true", "false"], default=["true"],
                        help="Intensity shift correction on/off")
    parser.add_argument("--clip_negative", nargs="+", type=str.lower,
                        choices=["true", "false"], default=["true"],
                        help="Clip negative values on/off")
    parser.add_argument("--metric",
                        choices=["tensor_rmse", "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                                 "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"],
                        default="fa_rmse",
                        help="Metric used to rank configs")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
