"""Quick BM4D baseline evaluation — no model checkpoint required.

Runs BM4D on degraded DWI for the configured test subjects, fits DTI,
and prints a metric table (tensor RMSE, FA RMSE/R², ADC RMSE/R²).

Usage:
    python run_bm4d_baseline.py
    python run_bm4d_baseline.py --subjects sub-03 sub-04
    python run_bm4d_baseline.py --eval_repeats 1 --profile lc
    python run_bm4d_baseline.py --eval_all
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.dw_thi.evaluate import (
    _run_bm4d,
    _baseline_dti_metrics,
    _next_degradation_trial,
    _expand_subjects,
    BM4D_CFG,
)
from src.dw_thi.augment import degrade_dwi_volume
from src.dw_thi.preprocessing import compute_brain_mask_from_dwi
from src.dw_thi.runtime import path_str


def main(args):
    BM4D_CFG.update(sigma=args.sigma, profile=args.profile)

    zarr_path = path_str(args.zarr_path)
    store = zarr.open_group(zarr_path, mode="r")
    all_keys = sorted(store.keys())

    if args.eval_all:
        subjects = all_keys
    elif args.subjects:
        subjects = _expand_subjects(args.subjects, all_keys)
    else:
        subjects = _expand_subjects(cfg.TEST_SUBJECTS, all_keys) or all_keys

    if not subjects:
        print("No subjects found.")
        return

    print(f"BM4D baseline  |  profile={args.profile}  sigma={'auto (MAD)' if args.sigma is None else args.sigma}")
    print(f"Subjects: {subjects}  |  repeats={args.eval_repeats}")
    print()

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]

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
                eval_rng,
                repeat_idx=repeat_idx,
                keep_fraction_range=(cfg.EVAL_KEEP_FRACTION_MIN, cfg.EVAL_KEEP_FRACTION_MAX),
                noise_range=(cfg.EVAL_NOISE_MIN, cfg.EVAL_NOISE_MAX),
            )
            input_dwi = degrade_dwi_volume(
                target_dwi,
                keep_fraction=trial["keep_fraction"],
                rel_noise_level=trial["noise_level"],
                seed=trial["degrade_seed"],
            )

            t0 = time.time()
            denoised = _run_bm4d(input_dwi)
            metrics, _ = _baseline_dti_metrics(
                denoised, target_dti6d, bvals, bvecs,
                b0_threshold=cfg.B0_THRESHOLD,
                dti_fit_method=cfg.DTI_FIT_METHOD,
                mask=mask_3d,
            )
            elapsed = time.time() - t0

            row = {
                "subject": subject_key,
                "repeat": repeat_idx,
                "keep_fraction": round(trial["keep_fraction"], 4),
                "noise_level": round(trial["noise_level"], 4),
                "elapsed_s": round(elapsed, 1),
            }
            row.update(metrics)
            rows.append(row)

            print(
                f"  {subject_key}  r={repeat_idx}  keep={trial['keep_fraction']:.3f}  "
                f"noise={trial['noise_level']:.3f}  "
                f"tensor_rmse={metrics['tensor_rmse']:.5f}  "
                f"FA[rmse={metrics['fa_rmse']:.4f} r2={metrics['fa_r2']:.3f}]  "
                f"ADC[rmse={metrics['adc_rmse']:.2e} r2={metrics['adc_r2']:.3f}]  "
                f"({elapsed:.1f}s)"
            )

    if not rows:
        return

    df = pd.DataFrame(rows)

    print(f"\n{'=' * 72}")
    print("  BM4D — mean over all subjects and repeats")
    print(f"{'=' * 72}")
    for col in metric_cols:
        print(f"  {col:<16} {df[col].mean():.6f}  ±  {df[col].std():.6f}")

    out_path = Path(args.out_dir) / "metrics_bm4d_standalone.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM4D-only baseline evaluation")
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default=cfg.EVAL_OUT_DIR)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--eval_repeats", type=int, default=cfg.EVAL_REPEATS)
    parser.add_argument("--profile", choices=["np", "lc", "high"], default=cfg.BM4D_PROFILE,
                        help="BM4D profile: np (normal), lc (low complexity), high")
    parser.add_argument("--sigma", type=float, default=cfg.BM4D_SIGMA,
                        help="Fixed noise sigma per volume (default: auto via MAD)")
    main(parser.parse_args())
