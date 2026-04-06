"""
MP-PCA Denoising Quality Evaluation — Zarr Dataset
====================================================
For each subject this script:
  1. Loads input_dwi (noisy), bvals, bvecs, target_dwi, target_dti_6d
  2. Runs Marcenko-Pastur PCA denoising on input_dwi  →  denoised_dwi
  3. Fits a DTI TensorModel to denoised_dwi  →  denoised_dti_6d
  4. Derives FA and ADC maps from both denoised and target tensors
  5. Reports two sets of metrics:

     A) DWI-space  (denoised_dwi vs target_dwi)
        PSNR, SSIM, RMSE, MAE, NRMSE

     B) DTI-space  (denoised FA/ADC vs target FA/ADC derived from target_dti_6d)
        RMSE, MAE, NRMSE, R² per map

Output CSV written to --out_dir:
  metrics_per_subject.csv   — one wide row per subject + MEAN/STD footer

Zarr layout required:
    pretext_dataset.zarr/
    ├── subject_000/
    │   ├── input_dwi        (X, Y, Z, N)  float32
    │   ├── bvals             (N,)          float32
    │   ├── bvecs             (3, N)        float32
    │   ├── target_dti_6d     (X, Y, Z, 6) float32  [Dxx,Dxy,Dyy,Dxz,Dyz,Dzz]
    │   └── target_dwi        (X, Y, Z, N) float32
    └── ...

Usage:
    python mppca.py
    python mppca.py --zarr_path /data/pretext_dataset.zarr --n_jobs 6
"""

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Support running this file directly (python baselines/mppca/mppca.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.utils import (
    dwi_metrics,
    dti6d_to_evals,
    evals_to_adc,
    evals_to_fa,
    fit_dti_to_6d,
    scalar_map_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ZARR_PATH       = "/path/to/pretext_dataset.zarr"
OUT_DIR         = "./mppca_eval"
N_JOBS          = 4

# MP-PCA
PATCH_RADIUS    = 2         # local patch radius in voxels (2 → 5×5×5 patches)
PCA_METHOD      = "eig"     # 'eig' (faster) | 'svd' (occasionally more accurate)

# DTI fitting
B0_THRESHOLD    = 50        # raise to ~60-80 for HCP 7T
DTI_FIT_METHOD  = "WLS"     # 'WLS' (robust) | 'OLS' | 'NLLS'
FA_MASK_THRESH  = 0.0       # DTI metrics restricted to voxels where target FA >
                             # this value. 0 = all voxels. Try 0.1 for WM only.
BRAIN_MASK_FRAC = 0.1       # Fraction of max b0 signal used as brain mask threshold
                             # for DTI metrics. Background voxels produce garbage
                             # tensors that dominate RMSE/R². 0 = no mask.
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# ══════════════════════════════════════════════════════════════════════════════
# Per-subject worker  (runs in subprocess)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_subject(zarr_path: str, subject_key: str, cfg: dict) -> dict | str:
    """Full evaluation pipeline for one subject."""
    from dipy.denoise.localpca import mppca

    t0 = time.time()
    try:
        # ── 1. Load ───────────────────────────────────────────────────────────
        store        = zarr.open(zarr_path, mode="r")
        grp          = store[subject_key]
        noisy        = grp["input_dwi"][:]              # (X,Y,Z,N)
        target_dwi   = grp["target_dwi"][:]             # (X,Y,Z,N)
        target_dti6d = grp["target_dti_6d"][:]          # (X,Y,Z,6)
        bvals        = grp["bvals"][:].astype(float)    # (N,)
        bvecs        = grp["bvecs"][:].astype(float)    # (3,N) in zarr → transpose

        # ── 2. Denoise with MP-PCA ────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = mppca(
                noisy,
                patch_radius=cfg["patch_radius"],
                pca_method=cfg["pca_method"],
            ).astype(np.float32)

        # ── 3. DWI-space metrics ──────────────────────────────────────────────
        dwi_m = dwi_metrics(target_dwi, denoised)

        # ── 4. Fit DTI to denoised DWI ────────────────────────────────────────
        denoised_dti6d = fit_dti_to_6d(
            denoised, bvals,
            bvecs_n3     = bvecs.T,           # (N,3)
            fit_method   = cfg["dti_fit_method"],
            b0_threshold = cfg["b0_threshold"],
        )

        # ── 5. FA and ADC from denoised tensor and target tensor ──────────────
        evals_den = dti6d_to_evals(denoised_dti6d)
        evals_tgt = dti6d_to_evals(target_dti6d)

        fa_den  = evals_to_fa(evals_den)
        fa_tgt  = evals_to_fa(evals_tgt)
        adc_den = evals_to_adc(evals_den)
        adc_tgt = evals_to_adc(evals_tgt)

        # Brain mask from mean b=0 signal — excludes background voxels where
        # DTI fitting is undefined and produces extreme outlier eigenvalues
        b0_idx = bvals < cfg["b0_threshold"]
        if b0_idx.sum() > 0 and cfg["brain_mask_frac"] > 0:
            mean_b0 = target_dwi[..., b0_idx].mean(axis=-1)
            brain_mask = mean_b0 > cfg["brain_mask_frac"] * mean_b0.max()
        else:
            brain_mask = np.ones(target_dwi.shape[:3], dtype=bool)

        # Optional FA threshold on top of brain mask
        if cfg["fa_mask_thresh"] > 0:
            brain_mask = brain_mask & (fa_tgt > cfg["fa_mask_thresh"])

        # ── 6. DTI-space metrics ──────────────────────────────────────────────
        fa_m  = scalar_map_metrics(fa_tgt,  fa_den,  mask=brain_mask)
        adc_m = scalar_map_metrics(adc_tgt, adc_den, mask=brain_mask)

        elapsed = time.time() - t0
        return {
            "subject":    subject_key,
            "elapsed_s":  round(elapsed, 2),
            # ── DWI metrics ──────────────────────────────────────
            "dwi_psnr":   round(dwi_m["psnr"],   4),
            "dwi_ssim":   round(dwi_m["ssim"],   4),
            "dwi_rmse":   round(dwi_m["rmse"],   6),
            "dwi_mae":    round(dwi_m["mae"],    6),
            "dwi_nrmse":  round(dwi_m["nrmse"],  6),
            "dwi_n_vols": dwi_m["n_volumes"],
            # ── FA metrics ───────────────────────────────────────
            "fa_rmse":    round(fa_m["rmse"],   6),
            "fa_mae":     round(fa_m["mae"],    6),
            "fa_nrmse":   round(fa_m["nrmse"],  6),
            "fa_r2":      round(fa_m["r2"],     4),
            # ── ADC metrics (units = mm²/s) ──────────────────────
            "adc_rmse":   round(adc_m["rmse"],  8),
            "adc_mae":    round(adc_m["mae"],   8),
            "adc_nrmse":  round(adc_m["nrmse"], 6),
            "adc_r2":     round(adc_m["r2"],    4),
        }

    except Exception as exc:
        return f"FAIL  {subject_key}  —  {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    zarr_path = args.zarr_path
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(zarr_path).exists():
        log.error("Zarr store not found: %s", zarr_path)
        return

    store    = zarr.open(zarr_path, mode="r")
    required = {"input_dwi", "target_dwi", "target_dti_6d", "bvals", "bvecs"}
    subjects = sorted(
        k for k in store.keys()
        if isinstance(store[k], zarr.Group) and required.issubset(store[k].keys())
    )
    if not subjects:
        log.error("No valid subjects (each needs: %s).", ", ".join(required))
        return

    log.info("Found %d subjects  |  workers=%d  patch_radius=%d  pca_method=%s  "
             "dti_fit=%s  b0_thr=%d  brain_mask=%.2f  fa_mask=%.2f",
             len(subjects), args.n_jobs, args.patch_radius, args.pca_method,
             args.dti_fit_method, args.b0_threshold,
             args.brain_mask_frac, args.fa_mask_thresh)

    cfg = dict(
        patch_radius   = args.patch_radius,
        pca_method     = args.pca_method,
        b0_threshold   = args.b0_threshold,
        dti_fit_method = args.dti_fit_method,
        fa_mask_thresh = args.fa_mask_thresh,
        brain_mask_frac= args.brain_mask_frac,
    )

    # ── Parallel evaluation ───────────────────────────────────────────────────
    t_start = time.time()
    rows    = []
    n_fail  = 0

    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = {
            pool.submit(evaluate_subject, zarr_path, subj, cfg): subj
            for subj in subjects
        }
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, str):
                log.warning("✗  %s", result)
                n_fail += 1
            else:
                log.info(
                    "✓  %-14s  "
                    "DWI[PSNR=%5.2f SSIM=%.3f RMSE=%.5f]  "
                    "FA[RMSE=%.4f R²=%.3f]  "
                    "ADC[RMSE=%.2e R²=%.3f]  (%.1fs)",
                    result["subject"],
                    result["dwi_psnr"],  result["dwi_ssim"], result["dwi_rmse"],
                    result["fa_rmse"],   result["fa_r2"],
                    result["adc_rmse"],  result["adc_r2"],
                    result["elapsed_s"],
                )
                rows.append(result)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    metric_cols = [
        "dwi_psnr", "dwi_ssim", "dwi_rmse", "dwi_mae",  "dwi_nrmse",
        "fa_rmse",  "fa_mae",   "fa_nrmse",  "fa_r2",
        "adc_rmse", "adc_mae",  "adc_nrmse", "adc_r2",
    ]

    summary = df[metric_cols].agg(["mean", "std"]).round(6).reset_index()
    summary.columns = ["subject"] + metric_cols
    summary["subject"] = summary["subject"].str.upper()
    df_out = pd.concat([df, summary], ignore_index=True)

    out_path = out_dir / "metrics_per_subject.csv"
    df_out.to_csv(out_path, index=False)

    total = time.time() - t_start
    log.info("─" * 72)
    log.info("Done.  %d/%d succeeded  |  total: %.1fs",
             len(subjects) - n_fail, len(subjects), total)
    log.info("Saved → %s", out_path)

    # ── Console summary ───────────────────────────────────────────────────────
    if not df.empty:
        print("\n── DWI metrics (denoised_dwi vs target_dwi) " + "─" * 25)
        print(df[["subject", "dwi_psnr", "dwi_ssim",
                  "dwi_rmse", "dwi_mae", "dwi_nrmse"]].to_string(index=False))

        print("\n── DTI metrics (fitted FA/ADC vs target_dti_6d) " + "─" * 18)
        print(df[["subject",
                  "fa_rmse",  "fa_mae",  "fa_nrmse",  "fa_r2",
                  "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"]].to_string(index=False))

        print("\n  MEAN  " + "  ".join(
            f"{c}={df[c].mean():.4f}" for c in metric_cols
        ))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MP-PCA denoising + DTI fitting vs zarr ground-truth."
    )
    parser.add_argument("--zarr_path",      default=ZARR_PATH)
    parser.add_argument("--out_dir",        default=OUT_DIR,
                        help="Output directory for CSV (default: ./mppca_eval)")
    parser.add_argument("--n_jobs",         type=int,   default=N_JOBS)
    # MP-PCA
    parser.add_argument("--patch_radius",   type=int,   default=PATCH_RADIUS,
                        help="Local patch radius in voxels (2 → 5×5×5 patches)")
    parser.add_argument("--pca_method",     default=PCA_METHOD,
                        choices=["eig", "svd"],
                        help="PCA decomposition method: 'eig' (faster) or 'svd'")
    parser.add_argument("--b0_threshold",   type=int,   default=B0_THRESHOLD)
    # DTI
    parser.add_argument("--dti_fit_method", default=DTI_FIT_METHOD,
                        choices=["WLS", "OLS", "NLLS"],
                        help="DTI fitting algorithm (default: WLS)")
    parser.add_argument("--fa_mask_thresh", type=float, default=FA_MASK_THRESH,
                        help="Restrict DTI metrics to voxels with target FA above "
                             "this threshold (0=all voxels, 0.1=white matter only)")
    parser.add_argument("--brain_mask_frac", type=float, default=BRAIN_MASK_FRAC,
                        help="Brain mask: fraction of max b0 signal. Voxels below "
                             "this are excluded from DTI metrics (default: 0.1)")

    main(parser.parse_args())
