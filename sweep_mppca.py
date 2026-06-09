"""Grid search for MP-PCA hyperparameters — no model checkpoint required.

Swept parameters:
  patch_radius    local patch radius in voxels (r → (2r+1)³ patch cube)
  pca_method      'eig' (faster) | 'svd' (occasionally more accurate)
  use_mask        whether to restrict denoising to brain-mask voxels

Note: dipy 1.12 mppca does not expose a tau_factor / threshold parameter.
The effective sampling density is fully determined by patch_radius.

Runs on VAL_SUBJECTS by default to avoid test-set leakage.
Saves to runs/sweep_mppca/ :
  mppca_grid.csv          one row per config × subject × repeat
  mppca_grid_summary.csv  mean ± std per config, sorted by metric

Usage:
    python sweep_mppca.py
    python sweep_mppca.py --subjects sub-05 --eval_repeats 5
    python sweep_mppca.py --patch_radii 1 2 3 4 5
    python sweep_mppca.py --pca_methods eig svd
    python sweep_mppca.py --use_mask true false
    python sweep_mppca.py --metric fa_r2 --out_dir runs/sweep_mppca_v2
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


def _run_mppca(noisy, mask, patch_radius, pca_method, use_mask):
    from dipy.denoise.localpca import mppca
    effective_mask = mask if use_mask else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mppca(
            noisy,
            patch_radius=patch_radius,
            pca_method=pca_method,
            mask=effective_mask,
            suppress_warning=True,
        ).astype("float32")


def _build_grid(args) -> list[dict]:
    use_mask_values = [v.lower() == "true" for v in args.use_mask]
    configs = []
    for radius, method, use_mask in itertools.product(
        args.patch_radii, args.pca_methods, use_mask_values,
    ):
        configs.append(dict(
            patch_radius=int(radius),
            pca_method=method,
            use_mask=bool(use_mask),
        ))
    # Deduplicate
    unique, seen = [], set()
    for c in configs:
        key = (c["patch_radius"], c["pca_method"], c["use_mask"])
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
        "MP-PCA grid search: %d configs × %d subjects × %d repeats",
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

            for cfg_idx, mppca_cfg in enumerate(configs):
                t0 = time.time()
                try:
                    denoised = _run_mppca(
                        input_dwi, mask_3d,
                        patch_radius=mppca_cfg["patch_radius"],
                        pca_method=mppca_cfg["pca_method"],
                        use_mask=mppca_cfg["use_mask"],
                    )
                    metrics, _ = _baseline_dti_metrics(
                        denoised, target_dti6d, bvals, bvecs,
                        b0_threshold=cfg.B0_THRESHOLD,
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
                    **mppca_cfg,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                row.update(metrics)
                rows.append(row)

                log.info(
                    "%-14s r=%02d cfg=%02d  radius=%d  method=%s  mask=%s  %s=%.6f  (%.1fs)",
                    subject_key, repeat_idx, cfg_idx,
                    mppca_cfg["patch_radius"], mppca_cfg["pca_method"], mppca_cfg["use_mask"],
                    args.metric, metrics[args.metric], time.time() - t0,
                )

    if not rows:
        log.error("No rows collected.")
        return

    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / "mppca_grid.csv"
    raw_df.to_csv(raw_path, index=False)
    log.info("Saved raw rows -> %s", raw_path)

    group_cols = ["config_idx", "patch_radius", "pca_method", "use_mask"]
    avail = [c for c in METRIC_COLS if c in raw_df.columns]
    summary = raw_df.groupby(group_cols, dropna=False)[avail].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(
        f"{args.metric}_mean", ascending=sort_ascending
    ).reset_index(drop=True)

    summary_path = out_dir / "mppca_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved summary -> %s", summary_path)

    best = summary.iloc[0]
    print(f"\n{'=' * 72}")
    print("  MP-PCA Best Config")
    print(f"{'=' * 72}")
    print(
        f"patch_radius={int(best['patch_radius'])}  "
        f"pca_method={best['pca_method']}  "
        f"use_mask={best['use_mask']}  "
        f"{args.metric}_mean={best[f'{args.metric}_mean']:.6f}"
    )
    print("\nAll configs (sorted):")
    top_cols = ["patch_radius", "pca_method", "use_mask",
                f"{args.metric}_mean", f"{args.metric}_std"]
    print(summary[top_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="MP-PCA hyperparameter grid search")
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default="runs/sweep_mppca")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs to evaluate (default: VAL_SUBJECTS)")
    parser.add_argument("--eval_repeats", type=int, default=cfg.EVAL_REPEATS)

    parser.add_argument("--patch_radii", nargs="+", type=int,
                        default=[1, 2, 3, 4],
                        help="Patch radius values to sweep (r → (2r+1)³ voxel patch)")
    parser.add_argument("--pca_methods", nargs="+", choices=["eig", "svd"],
                        default=["eig", "svd"],
                        help="PCA solvers to sweep")
    parser.add_argument("--use_mask", nargs="+", type=str.lower,
                        choices=["true", "false"], default=["true", "false"],
                        help="Whether to restrict denoising to brain-mask voxels")
    parser.add_argument("--metric",
                        choices=["tensor_rmse", "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                                 "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"],
                        default="fa_rmse",
                        help="Metric used to rank configs")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
