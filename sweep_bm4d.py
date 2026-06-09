"""Grid search for BM4D hyperparameters — no model checkpoint required.

Swept parameters (bm4d 4.x / BM4DProfile):
  sigma           noise estimate: 'auto' (MAD) or fixed float
  profile         'np' (default BM4DProfile) | 'refilter' (two-pass Wiener)
  patch_size      block size in voxels for both HT and Wiener steps
                    BM4DProfile.bs_ht = BM4DProfile.bs_wiener = (s, s, s)
                    default: 4 (HT) / 5 (Wiener)
  search_window   search window half-size for both steps
                    BM4DProfile.search_window_ht = _wiener = (w, w, w)
                    default: 7  (actual window = 2w+1 = 15 voxels)
  n_max           maximum number of similar blocks stacked per group
                    BM4DProfile.max_stack_size_ht  default: 16
                    BM4DProfile.max_stack_size_wiener default: 32

⚠ BM4D is SLOW (~minutes per volume × volumes per subject).
  Default grid: 2 profiles × 1 sigma = 2 configs.
  Start with the default, then selectively add --patch_sizes / --search_windows / --n_max_values.
  Always use --eval_repeats 1 and a single subject for exploratory sweeps.

Saves to runs/sweep_bm4d/ :
  bm4d_grid.csv          one row per config × subject × repeat
  bm4d_grid_summary.csv  mean ± std per config, sorted by metric

Usage:
    python sweep_bm4d.py                                       # 2 profiles, val subjects
    python sweep_bm4d.py --subjects sub-05 --eval_repeats 1    # fast single-subject run
    python sweep_bm4d.py --profiles np refilter --sigmas auto
    python sweep_bm4d.py --patch_sizes 4 5 6                   # extend grid (slow!)
    python sweep_bm4d.py --search_windows 5 7 9                # extend grid (slow!)
    python sweep_bm4d.py --n_max_values 8 16 32                # extend grid (slower)
    python sweep_bm4d.py --metric fa_r2 --out_dir runs/sweep_bm4d_v2
"""

import argparse
import itertools
import logging
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


# ─── BM4D Profile builder ────────────────────────────────────────────────────

def _build_bm4d_profile(profile_name: str, patch_size=None, search_window=None, n_max=None):
    """Create a BM4DProfile with optional parameter overrides.

    profile_name : 'np' → BM4DProfile(), 'refilter' → BM4DProfileRefilter()
    patch_size   : block size (int); sets bs_ht = bs_wiener = (s,s,s)
    search_window: search window half-size (int); sets search_window_ht = _wiener = (w,w,w)
    n_max        : max similar blocks (int); sets max_stack_size_ht and _wiener
    """
    from bm4d import BM4DProfile, BM4DProfileRefilter

    profile = BM4DProfileRefilter() if profile_name == "refilter" else BM4DProfile()

    if patch_size is not None:
        profile.bs_ht = (patch_size,) * 3
        profile.bs_wiener = (patch_size,) * 3

    if search_window is not None:
        profile.search_window_ht = (search_window,) * 3
        profile.search_window_wiener = (search_window,) * 3

    if n_max is not None:
        profile.max_stack_size_ht = int(n_max)
        profile.max_stack_size_wiener = int(n_max)

    return profile


def _run_bm4d_volume(vol_3d: np.ndarray, sigma: float | None, profile) -> np.ndarray:
    """Denoise a single 3D volume with BM4D."""
    from bm4d import bm4d
    if sigma is None:
        sigma = float(np.median(np.abs(vol_3d - np.median(vol_3d))) / 0.6745)
        sigma = max(sigma, 1e-6)
    return bm4d(vol_3d.astype(np.float64), sigma, profile=profile).astype(np.float32)


def _run_bm4d_4d(noisy: np.ndarray, sigma: float | None, profile) -> np.ndarray:
    """Apply BM4D to each DWI volume (4th dimension) independently."""
    N = noisy.shape[3]
    denoised = np.empty_like(noisy, dtype=np.float32)
    t_start = time.time()
    for n in range(N):
        vol = noisy[:, :, :, n]
        vol_sigma = sigma
        if vol_sigma is None:
            vol_sigma = float(np.median(np.abs(vol - np.median(vol))) / 0.6745)
            vol_sigma = max(vol_sigma, 1e-6)
        print(f"  BM4D  [{n + 1:2d}/{N}]  sigma={vol_sigma:.4f} ...", end="", flush=True)
        t0 = time.time()
        denoised[:, :, :, n] = _run_bm4d_volume(vol, sigma, profile)
        elapsed = time.time() - t0
        remaining = (time.time() - t_start) / (n + 1) * (N - n - 1)
        print(f"  {elapsed:.1f}s  (remaining ~{remaining:.0f}s)", flush=True)
    return denoised


# ─── Grid builder ────────────────────────────────────────────────────────────

def _parse_sigma(s: str) -> float | None:
    return None if s.lower() == "auto" else float(s)


def _build_grid(args) -> list[dict]:
    sigmas = [_parse_sigma(s) for s in args.sigmas]

    patch_sizes = args.patch_sizes if args.patch_sizes else [None]
    search_windows = args.search_windows if args.search_windows else [None]
    n_max_values = args.n_max_values if args.n_max_values else [None]

    configs = []
    for profile, sigma, ps, sw, nm in itertools.product(
        args.profiles, sigmas, patch_sizes, search_windows, n_max_values,
    ):
        configs.append(dict(
            profile=profile,
            sigma=sigma,
            patch_size=None if ps is None else int(ps),
            search_window=None if sw is None else int(sw),
            n_max=None if nm is None else int(nm),
        ))

    # Deduplicate
    unique, seen = [], set()
    for c in configs:
        key = (c["profile"], c["sigma"], c["patch_size"], c["search_window"], c["n_max"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _sigma_label(sigma) -> str:
    return "auto" if sigma is None else f"{sigma:.4g}"


def _param_label(v) -> str:
    return "default" if v is None else str(v)


# ─── Sweep runner ────────────────────────────────────────────────────────────

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

    # Estimate runtime
    n_runs = len(configs) * len(subjects) * args.eval_repeats
    log.info(
        "BM4D grid search: %d configs × %d subjects × %d repeats = %d total BM4D runs",
        len(configs), len(subjects), args.eval_repeats, n_runs,
    )
    log.info("Subjects: %s", subjects)
    log.warning(
        "BM4D is slow. With ~%d DWI volumes/subject, expect ~%d–%d minutes total.",
        32, n_runs * 5, n_runs * 15,
    )

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

            for cfg_idx, bm_cfg in enumerate(configs):
                log.info(
                    "Config %d/%d: profile=%s sigma=%s patch=%s search_win=%s n_max=%s",
                    cfg_idx + 1, len(configs),
                    bm_cfg["profile"], _sigma_label(bm_cfg["sigma"]),
                    _param_label(bm_cfg["patch_size"]),
                    _param_label(bm_cfg["search_window"]),
                    _param_label(bm_cfg["n_max"]),
                )
                t0 = time.time()
                try:
                    bm4d_profile = _build_bm4d_profile(
                        bm_cfg["profile"],
                        patch_size=bm_cfg["patch_size"],
                        search_window=bm_cfg["search_window"],
                        n_max=bm_cfg["n_max"],
                    )
                    denoised = _run_bm4d_4d(input_dwi, bm_cfg["sigma"], bm4d_profile)
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
                    "bm4d_profile": bm_cfg["profile"],
                    "bm4d_sigma": bm_cfg["sigma"],
                    "bm4d_patch_size": bm_cfg["patch_size"],
                    "bm4d_search_window": bm_cfg["search_window"],
                    "bm4d_n_max": bm_cfg["n_max"],
                    "elapsed_s": round(time.time() - t0, 2),
                }
                row.update(metrics)
                rows.append(row)

                log.info(
                    "  %-14s r=%02d cfg=%02d  profile=%s  sigma=%s  patch=%s  "
                    "search=%s  n_max=%s  %s=%.6f  (%.0fs)",
                    subject_key, repeat_idx, cfg_idx,
                    bm_cfg["profile"], _sigma_label(bm_cfg["sigma"]),
                    _param_label(bm_cfg["patch_size"]),
                    _param_label(bm_cfg["search_window"]),
                    _param_label(bm_cfg["n_max"]),
                    args.metric, metrics[args.metric], time.time() - t0,
                )

    if not rows:
        log.error("No rows collected.")
        return

    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / "bm4d_grid.csv"
    raw_df.to_csv(raw_path, index=False)
    log.info("Saved raw rows -> %s", raw_path)

    group_cols = ["config_idx", "bm4d_profile", "bm4d_sigma",
                  "bm4d_patch_size", "bm4d_search_window", "bm4d_n_max"]
    avail = [c for c in METRIC_COLS if c in raw_df.columns]
    summary = raw_df.groupby(group_cols, dropna=False)[avail].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(
        f"{args.metric}_mean", ascending=sort_ascending
    ).reset_index(drop=True)

    summary_path = out_dir / "bm4d_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved summary -> %s", summary_path)

    best = summary.iloc[0]
    print(f"\n{'=' * 72}")
    print("  BM4D Best Config")
    print(f"{'=' * 72}")
    print(
        f"profile={best['bm4d_profile']}  "
        f"sigma={_sigma_label(best['bm4d_sigma'])}  "
        f"patch_size={_param_label(best['bm4d_patch_size'])}  "
        f"search_window={_param_label(best['bm4d_search_window'])}  "
        f"n_max={_param_label(best['bm4d_n_max'])}  "
        f"{args.metric}_mean={best[f'{args.metric}_mean']:.6f}"
    )
    print("\nAll configs (sorted):")
    top_cols = ["bm4d_profile", "bm4d_sigma", "bm4d_patch_size",
                "bm4d_search_window", "bm4d_n_max",
                f"{args.metric}_mean", f"{args.metric}_std"]
    print(summary[top_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="BM4D hyperparameter grid search")
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default="runs/sweep_bm4d")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs (default: VAL_SUBJECTS). Use 1 subject for speed.")
    parser.add_argument("--eval_repeats", type=int, default=1,
                        help="Number of degradation repeats per subject (default: 1 for speed)")

    parser.add_argument("--profiles", nargs="+", choices=["np", "refilter"],
                        default=["np", "refilter"],
                        help="BM4D profiles: np (default) | refilter (two-pass Wiener)")
    parser.add_argument("--sigmas", nargs="+", default=["auto"],
                        help="Sigma values: 'auto' (MAD per volume) or float, e.g. 5.0 10.0")

    parser.add_argument("--patch_sizes", nargs="+", type=int, default=[],
                        help="Block sizes to sweep (e.g. 4 5 6). "
                             "Omit to keep profile defaults. Sets bs_ht = bs_wiener.")
    parser.add_argument("--search_windows", nargs="+", type=int, default=[],
                        help="Search window half-sizes to sweep (e.g. 5 7 9). "
                             "Actual window = 2w+1. Omit to keep profile defaults.")
    parser.add_argument("--n_max_values", nargs="+", type=int, default=[],
                        help="Max similar-block counts to sweep (e.g. 8 16 32). "
                             "Omit to keep profile defaults.")

    parser.add_argument("--metric",
                        choices=["tensor_rmse", "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                                 "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"],
                        default="fa_rmse",
                        help="Metric used to rank configs")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
