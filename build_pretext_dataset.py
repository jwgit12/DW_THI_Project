"""
patch2self_eval.py — Patch2Self Denoising Quality Evaluation
=============================================================
For each subject in a Zarr store:
  1. Loads input_dwi (noisy), bvals, bvecs, target_dwi, target_dti_6d
  2. Runs Patch2Self denoising
  3. Fits a DTI tensor model to the denoised DWI
  4. Derives FA and ADC from both fitted and target tensors
  5. Computes DWI-space and DTI-space metrics

Nothing is written back to the Zarr store.
Output: metrics_per_subject.csv in --out_dir.

Zarr layout required:
    pretext_dataset.zarr/
    ├── subject_000/
    │   ├── input_dwi        (X, Y, Z, N)  float32
    │   ├── bvals             (N,)          float32
    │   ├── bvecs             (3, N)        float32
    │   ├── target_dti_6d     (X, Y, Z, 6) float32  [Dxx,Dxy,Dyy,Dxz,Dyz,Dzz]
    │   └── target_dwi        (X, Y, Z, N) float32
    └── subject_001/ ...

Usage:
    python patch2self_eval.py --zarr_path /data/pretext_dataset.zarr
    python patch2self_eval.py --zarr_path /data/pretext_dataset.zarr --n_jobs 6 --fa_mask_thresh 0.1
"""

import argparse
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit here or use CLI flags
# ─────────────────────────────────────────────────────────────────────────────
ZARR_PATH       = "/path/to/pretext_dataset.zarr"
OUT_DIR         = "./patch2self_eval"
N_JOBS          = 4

# Patch2Self
MODEL           = "ols"    # 'ols' | 'ridge' | 'lasso'
ALPHA           = 1.0      # regularisation for ridge/lasso only
B0_THRESHOLD    = 50       # raise to ~70 for HCP 7T
SHIFT_INTENSITY = True     # recommended: prevents negative values
CLIP_NEGATIVE   = False    # keep False when SHIFT_INTENSITY=True
B0_DENOISING    = True     # set False if b0 volumes are artefacted

# DTI
DTI_FIT_METHOD  = "WLS"    # 'WLS' (robust) | 'OLS' (fast) | 'NLLS' (slow/accurate)
FA_MASK_THRESH  = 0.0      # 0 = all voxels; 0.1 = white-matter only
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DWI-space metrics
# ══════════════════════════════════════════════════════════════════════════════

def _rmse(r, e):
    return float(np.sqrt(np.mean((r.astype(np.float64) - e.astype(np.float64)) ** 2)))

def _mae(r, e):
    return float(np.mean(np.abs(r.astype(np.float64) - e.astype(np.float64))))

def _nrmse(r, e):
    d = float(r.max() - r.min())
    return _rmse(r, e) / d if d != 0 else float("nan")

def _psnr(r, e):
    d = float(r.max() - r.min())
    return float(sk_psnr(r, e, data_range=d)) if d != 0 else float("nan")

def _ssim_3d(r, e):
    """Mean SSIM over axial slices of one 3-D volume."""
    d = float(r.max() - r.min())
    if d == 0:
        return float("nan")
    return float(np.nanmean([
        sk_ssim(r[..., z], e[..., z], data_range=d)
        for z in range(r.shape[2])
    ]))

def dwi_metrics(ref: np.ndarray, est: np.ndarray) -> dict:
    """Mean metrics across all N volumes. ref/est: (X,Y,Z,N)."""
    N = ref.shape[-1]
    acc = {k: [] for k in ["psnr", "ssim", "rmse", "mae", "nrmse"]}
    for n in range(N):
        r, e = ref[..., n], est[..., n]
        acc["psnr"].append(_psnr(r, e))
        acc["ssim"].append(_ssim_3d(r, e))
        acc["rmse"].append(_rmse(r, e))
        acc["mae"].append(_mae(r, e))
        acc["nrmse"].append(_nrmse(r, e))
    return {k: float(np.nanmean(v)) for k, v in acc.items()} | {"n_volumes": N}


# ══════════════════════════════════════════════════════════════════════════════
# DTI helpers
# ══════════════════════════════════════════════════════════════════════════════

def dti6d_to_evals(dti_6d: np.ndarray) -> np.ndarray:
    """
    (X,Y,Z,6) → eigenvalues (X,Y,Z,3) sorted descending (λ1 ≥ λ2 ≥ λ3).

    Component order matches functions.py tensor_to_6d:
        idx 0: Dxx  |  idx 1: Dxy  |  idx 2: Dyy
        idx 3: Dxz  |  idx 4: Dyz  |  idx 5: Dzz
    """
    X, Y, Z    = dti_6d.shape[:3]
    dxx, dxy   = dti_6d[..., 0], dti_6d[..., 1]
    dyy, dxz   = dti_6d[..., 2], dti_6d[..., 3]   # note: Dyy=2, Dxz=3
    dyz, dzz   = dti_6d[..., 4], dti_6d[..., 5]

    flat = np.stack([
        dxx.ravel(), dxy.ravel(), dxz.ravel(),
        dxy.ravel(), dyy.ravel(), dyz.ravel(),
        dxz.ravel(), dyz.ravel(), dzz.ravel(),
    ], axis=-1).reshape(-1, 3, 3).astype(np.float64)

    evals = np.linalg.eigvalsh(flat)[..., ::-1]    # ascending → descending
    return evals.reshape(X, Y, Z, 3).astype(np.float32)


def evals_to_fa(evals: np.ndarray) -> np.ndarray:
    """FA matching functions.py compute_fa_from_tensor6."""
    md  = evals.mean(axis=-1, keepdims=True)
    num = np.sqrt(((evals - md) ** 2).sum(axis=-1))
    den = np.sqrt((evals ** 2).sum(axis=-1) + 1e-12)
    return np.clip(np.sqrt(1.5) * num / den, 0.0, 1.0).astype(np.float32)


def evals_to_adc(evals: np.ndarray) -> np.ndarray:
    """ADC = mean diffusivity = mean of eigenvalues."""
    return (evals.sum(axis=-1) / 3.0).astype(np.float32)


def fit_dti_to_6d(dwi: np.ndarray, bvals: np.ndarray,
                  bvecs_n3: np.ndarray, method: str, b0_thr: float) -> np.ndarray:
    """
    Fit TensorModel → 6-component tensor in functions.py order:
        [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]

    bvecs_n3: (N,3) — pass bvecs.T when zarr stores (3,N).
    """
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    gtab = gradient_table(bvals, bvecs=bvecs_n3, b0_threshold=b0_thr)
    q    = TensorModel(gtab, fit_method=method).fit(dwi).quadratic_form
    return np.stack([
        q[..., 0, 0],   # Dxx → idx 0
        q[..., 0, 1],   # Dxy → idx 1
        q[..., 1, 1],   # Dyy → idx 2
        q[..., 0, 2],   # Dxz → idx 3
        q[..., 1, 2],   # Dyz → idx 4
        q[..., 2, 2],   # Dzz → idx 5
    ], axis=-1).astype(np.float32)


def scalar_metrics(ref: np.ndarray, est: np.ndarray,
                   mask: np.ndarray | None = None) -> dict:
    """Voxelwise RMSE, MAE, NRMSE, R² between two 3-D scalar maps."""
    r, e = ref.ravel().astype(np.float64), est.ravel().astype(np.float64)
    if mask is not None:
        m = mask.ravel(); r, e = r[m], e[m]
    valid = np.isfinite(r) & np.isfinite(e)
    r, e  = r[valid], e[valid]
    if len(r) == 0:
        return {k: float("nan") for k in ["rmse", "mae", "nrmse", "r2"]}
    rmse  = float(np.sqrt(np.mean((r - e) ** 2)))
    mae   = float(np.mean(np.abs(r - e)))
    denom = float(r.max() - r.min())
    nrmse = rmse / denom if denom > 0 else float("nan")
    r2    = float(np.corrcoef(r, e)[0, 1] ** 2) \
            if r.std() > 0 and e.std() > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# Per-subject worker  (runs in subprocess)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_subject(zarr_path: str, subject_key: str, cfg: dict) -> dict | str:
    """Denoise one subject and compute all metrics. Returns dict or error string."""
    from dipy.denoise.patch2self import patch2self

    t0 = time.time()
    try:
        # ── Load ─────────────────────────────────────────────────────────────
        grp          = zarr.open(zarr_path, mode="r")[subject_key]
        noisy        = grp["input_dwi"][:]              # (X,Y,Z,N)
        target_dwi   = grp["target_dwi"][:]             # (X,Y,Z,N)
        target_dti6d = grp["target_dti_6d"][:]          # (X,Y,Z,6)
        bvals        = grp["bvals"][:].astype(float)    # (N,)
        bvecs        = grp["bvecs"][:].astype(float)    # (3,N) in zarr

        # ── Denoise ──────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = patch2self(
                noisy,
                bvals,
                model              = cfg["model"],
                alpha              = cfg["alpha"],
                b0_threshold       = cfg["b0_threshold"],
                shift_intensity    = cfg["shift_intensity"],
                clip_negative_vals = cfg["clip_negative"],
                b0_denoising       = cfg["b0_denoising"],
                verbose            = False,
            ).astype(np.float32)

        # ── DWI metrics ───────────────────────────────────────────────────────
        dwi_m = dwi_metrics(target_dwi, denoised)

        # ── Fit DTI on denoised DWI ───────────────────────────────────────────
        denoised_dti6d = fit_dti_to_6d(
            denoised, bvals,
            bvecs_n3 = bvecs.T,                         # (3,N) → (N,3) for dipy
            method   = cfg["dti_fit_method"],
            b0_thr   = cfg["b0_threshold"],
        )

        # ── FA and ADC from both tensors ──────────────────────────────────────
        evals_den = dti6d_to_evals(denoised_dti6d)
        evals_tgt = dti6d_to_evals(target_dti6d)

        fa_den,  fa_tgt  = evals_to_fa(evals_den),  evals_to_fa(evals_tgt)
        adc_den, adc_tgt = evals_to_adc(evals_den), evals_to_adc(evals_tgt)

        # Optional: restrict DTI metrics to voxels above FA threshold
        fa_mask = (fa_tgt > cfg["fa_mask_thresh"]) if cfg["fa_mask_thresh"] > 0 else None

        # ── DTI metrics ───────────────────────────────────────────────────────
        fa_m  = scalar_metrics(fa_tgt,  fa_den,  mask=fa_mask)
        adc_m = scalar_metrics(adc_tgt, adc_den, mask=fa_mask)

        return {
            "subject":    subject_key,
            "elapsed_s":  round(time.time() - t0, 2),
            # DWI
            "dwi_psnr":   round(dwi_m["psnr"],  4),
            "dwi_ssim":   round(dwi_m["ssim"],  4),
            "dwi_rmse":   round(dwi_m["rmse"],  6),
            "dwi_mae":    round(dwi_m["mae"],   6),
            "dwi_nrmse":  round(dwi_m["nrmse"], 6),
            "dwi_n_vols": dwi_m["n_volumes"],
            # FA
            "fa_rmse":    round(fa_m["rmse"],   6),
            "fa_mae":     round(fa_m["mae"],    6),
            "fa_nrmse":   round(fa_m["nrmse"],  6),
            "fa_r2":      round(fa_m["r2"],     4),
            # ADC
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

    required = {"input_dwi", "target_dwi", "target_dti_6d", "bvals", "bvecs"}
    store    = zarr.open(zarr_path, mode="r")
    subjects = sorted(
        k for k in store.keys()
        if isinstance(store[k], zarr.Group) and required.issubset(store[k].keys())
    )
    if not subjects:
        log.error("No valid subjects found. Each group needs: %s", ", ".join(required))
        return

    log.info("Found %d subjects  |  workers=%d  model=%s  dti=%s  b0_thr=%d  fa_mask=%.2f",
             len(subjects), args.n_jobs, args.model,
             args.dti_fit_method, args.b0_threshold, args.fa_mask_thresh)

    cfg = dict(
        model          = args.model,
        alpha          = args.alpha,
        b0_threshold   = args.b0_threshold,
        shift_intensity= not args.no_shift,
        clip_negative  = args.clip_negative,
        b0_denoising   = not args.skip_b0,
        dti_fit_method = args.dti_fit_method,
        fa_mask_thresh = args.fa_mask_thresh,
    )

    # ── Parallel evaluation ───────────────────────────────────────────────────
    t_start = time.time()
    rows    = []
    n_fail  = 0

    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = {pool.submit(evaluate_subject, zarr_path, s, cfg): s
                   for s in subjects}
        for f in as_completed(futures):
            result = f.result()
            if isinstance(result, str):
                log.warning("✗  %s", result)
                n_fail += 1
            else:
                log.info(
                    "✓  %-14s  DWI[PSNR=%5.2f SSIM=%.3f RMSE=%.5f]  "
                    "FA[RMSE=%.4f R²=%.3f]  ADC[RMSE=%.2e R²=%.3f]  (%.1fs)",
                    result["subject"],
                    result["dwi_psnr"], result["dwi_ssim"], result["dwi_rmse"],
                    result["fa_rmse"],  result["fa_r2"],
                    result["adc_rmse"], result["adc_r2"],
                    result["elapsed_s"],
                )
                rows.append(result)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    metric_cols = [
        "dwi_psnr", "dwi_ssim", "dwi_rmse", "dwi_mae",  "dwi_nrmse",
        "fa_rmse",  "fa_mae",   "fa_nrmse",  "fa_r2",
        "adc_rmse", "adc_mae",  "adc_nrmse", "adc_r2",
    ]
    df      = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    summary = df[metric_cols].agg(["mean", "std"]).round(6).reset_index()
    summary.columns = ["subject"] + metric_cols
    summary["subject"] = summary["subject"].str.upper()
    df_out  = pd.concat([df, summary], ignore_index=True)

    out_path = out_dir / "metrics_per_subject.csv"
    df_out.to_csv(out_path, index=False)

    log.info("─" * 72)
    log.info("Done.  %d/%d succeeded  |  total: %.1fs",
             len(subjects) - n_fail, len(subjects), time.time() - t_start)
    log.info("Saved → %s", out_path)

    # ── Console summary ───────────────────────────────────────────────────────
    if not df.empty:
        print("\n── DWI metrics (denoised vs target_dwi) " + "─" * 28)
        print(df[["subject", "dwi_psnr", "dwi_ssim",
                  "dwi_rmse", "dwi_mae", "dwi_nrmse"]].to_string(index=False))
        print("\n── DTI metrics (fitted FA/ADC vs target_dti_6d) " + "─" * 18)
        print(df[["subject",
                  "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                  "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"]].to_string(index=False))
        print("\n  MEAN  " + "  ".join(f"{c}={df[c].mean():.4f}" for c in metric_cols))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Patch2Self denoising quality vs zarr ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr_path",      default=ZARR_PATH,
                        help="Path to the .zarr store")
    parser.add_argument("--out_dir",        default=OUT_DIR,
                        help="Directory for output CSV")
    parser.add_argument("--n_jobs",         type=int,   default=N_JOBS,
                        help="Parallel workers")
    # Patch2Self
    g = parser.add_argument_group("Patch2Self")
    g.add_argument("--model",          default=MODEL, choices=["ols", "ridge", "lasso"])
    g.add_argument("--alpha",          type=float, default=ALPHA,
                   help="Regularisation for ridge/lasso")
    g.add_argument("--b0_threshold",   type=int,   default=B0_THRESHOLD,
                   help="b-value cutoff for b0 volumes (raise to ~70 for HCP 7T)")
    g.add_argument("--no_shift",       action="store_true",
                   help="Disable shift_intensity (not recommended)")
    g.add_argument("--clip_negative",  action="store_true",
                   help="Clip negative values to 0 after denoising")
    g.add_argument("--skip_b0",        action="store_true",
                   help="Skip denoising of b0 volumes")
    # DTI
    g = parser.add_argument_group("DTI")
    g.add_argument("--dti_fit_method", default=DTI_FIT_METHOD,
                   choices=["WLS", "OLS", "NLLS"],
                   help="DTI fitting algorithm")
    g.add_argument("--fa_mask_thresh", type=float, default=FA_MASK_THRESH,
                   help="Restrict DTI metrics to voxels with target FA above this "
                        "value (0=all voxels, 0.1=white matter only)")

    main(parser.parse_args())