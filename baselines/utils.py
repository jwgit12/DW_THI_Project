import numpy as np
import pandas as pd
import zarr
import time
import warnings
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

# ─────────────────────────────────────────────────────────────────────────────
# DWI-space metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def _rmse(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ref.astype(np.float64) - est.astype(np.float64)) ** 2)))

def _mae(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.mean(np.abs(ref.astype(np.float64) - est.astype(np.float64))))

def _nrmse(ref: np.ndarray, est: np.ndarray) -> float:
    denom = float(ref.max() - ref.min())
    return _rmse(ref, est) / denom if denom != 0 else float("nan")

def _psnr(ref: np.ndarray, est: np.ndarray) -> float:
    dr = float(ref.max() - ref.min())
    return float(sk_psnr(ref, est, data_range=dr)) if dr != 0 else float("nan")

def _ssim_3d(ref: np.ndarray, est: np.ndarray) -> float:
    dr = float(ref.max() - ref.min())
    if dr == 0:
        return float("nan")
    return float(np.nanmean([
        sk_ssim(ref[..., z], est[..., z], data_range=dr)
        for z in range(ref.shape[2])
    ]))

def dwi_metrics(ref_4d: np.ndarray, est_4d: np.ndarray) -> dict:
    N = ref_4d.shape[-1]
    accum = {k: [] for k in ["psnr", "ssim", "rmse", "mae", "nrmse"]}
    for n in range(N):
        r, e = ref_4d[..., n], est_4d[..., n]
        accum["psnr"].append(_psnr(r, e))
        accum["ssim"].append(_ssim_3d(r, e))
        accum["rmse"].append(_rmse(r, e))
        accum["mae"].append(_mae(r, e))
        accum["nrmse"].append(_nrmse(r, e))
    return {k: float(np.nanmean(v)) for k, v in accum.items()} | {"n_volumes": N}

# ═════════════════════════════════════════════════════════════════════════════════
# DTI helpers
# ═════════════════════
def dti6d_to_evals(dti_6d: np.ndarray) -> np.ndarray:
    X, Y, Z = dti_6d.shape[:3]
    dxx = dti_6d[..., 0]; dxy = dti_6d[..., 1]
    dyy = dti_6d[..., 2]; dxz = dti_6d[..., 3]
    dyz = dti_6d[..., 4]; dzz = dti_6d[..., 5]

    flat = np.stack([
        dxx.ravel(), dxy.ravel(), dxz.ravel(),
        dxy.ravel(), dyy.ravel(), dyz.ravel(),
        dxz.ravel(), dyz.ravel(), dzz.ravel(),
    ], axis=-1).reshape(-1, 3, 3).astype(np.float64)

    evals = np.linalg.eigvalsh(flat)[..., ::-1]
    return evals.reshape(X, Y, Z, 3).astype(np.float32)

def evals_to_fa(evals: np.ndarray) -> np.ndarray:
    md  = evals.mean(axis=-1, keepdims=True)
    num = np.sqrt(((evals - md) ** 2).sum(axis=-1))
    den = np.sqrt((evals ** 2).sum(axis=-1) + 1e-12)
    return np.clip(np.sqrt(1.5) * num / den, 0.0, 1.0).astype(np.float32)

def evals_to_adc(evals: np.ndarray) -> np.ndarray:
    return ((evals[..., 0] + evals[..., 1] + evals[..., 2]) / 3.0).astype(np.float32)

def fit_dti_to_6d(dwi_4d: np.ndarray, bvals: np.ndarray,
                  bvecs_n3: np.ndarray, fit_method: str,
                  b0_threshold: float) -> np.ndarray:
    gtab  = gradient_table(bvals, bvecs=bvecs_n3, b0_threshold=b0_threshold)
    q     = TensorModel(gtab, fit_method=fit_method).fit(dwi_4d).quadratic_form
    return np.stack([
        q[..., 0, 0],   # Dxx
        q[..., 0, 1],   # Dxy
        q[..., 1, 1],   # Dyy
        q[..., 0, 2],   # Dxz
        q[..., 1, 2],   # Dyz
        q[..., 2, 2],   # Dzz
    ], axis=-1).astype(np.float32)

def scalar_map_metrics(ref: np.ndarray, est: np.ndarray,
                       mask: np.ndarray | None = None) -> dict:
    r = ref.ravel().astype(np.float64)
    e = est.ravel().astype(np.float64)

    if mask is not None:
        m  = mask.ravel()
        r, e = r[m], e[m]

    valid = np.isfinite(r) & np.isfinite(e)
    r, e  = r[valid], e[valid]

    if len(r) == 0:
        return {"rmse": np.nan, "mae": np.nan, "nrmse": np.nan, "r2": np.nan}

    rmse  = float(np.sqrt(np.mean((r - e) ** 2)))
    mae   = float(np.mean(np.abs(r - e)))
    denom = float(r.max() - r.min())
    nrmse = rmse / denom if denom > 0 else float("nan")
    r2    = float(np.corrcoef(r, e)[0, 1] ** 2) if (r.std() > 0 and e.std() > 0) else float("nan")

    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}
