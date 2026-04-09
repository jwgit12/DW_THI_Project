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


# ─────────────────────────────────────────────────────────────────────────────
# Denoising visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo = float(image.min())
    hi = float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def _show_kspace(image: np.ndarray) -> np.ndarray:
    kspace = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
    return np.log1p(np.abs(kspace)).astype(np.float32)


def select_plot_indices(
    dwi_4d: np.ndarray,
    bvals: np.ndarray,
    b0_threshold: float,
    brain_mask_frac: float = 0.1,
    slice_idx: int | None = None,
    volume_idx: int | None = None,
) -> tuple[int, int]:
    n_slices = int(dwi_4d.shape[2])
    n_volumes = int(dwi_4d.shape[3])

    if volume_idx is None:
        dwi_indices = np.where(bvals >= b0_threshold)[0]
        if dwi_indices.size:
            volume_idx = int(dwi_indices[len(dwi_indices) // 2])
        else:
            volume_idx = n_volumes // 2
    elif not 0 <= volume_idx < n_volumes:
        raise ValueError(f"volume_idx={volume_idx} is out of bounds for {n_volumes} volumes")

    if slice_idx is None:
        b0_indices = np.where(bvals < b0_threshold)[0]
        if b0_indices.size:
            ref_volume = dwi_4d[..., b0_indices].mean(axis=-1)
        else:
            ref_volume = dwi_4d[..., volume_idx]

        ref_max = float(ref_volume.max())
        if ref_max > 0.0 and brain_mask_frac > 0.0:
            brain_mask = ref_volume > (brain_mask_frac * ref_max)
            slice_scores = brain_mask.sum(axis=(0, 1))
            if np.any(slice_scores):
                slice_idx = int(np.argmax(slice_scores))
            else:
                slice_idx = n_slices // 2
        else:
            slice_idx = n_slices // 2
    elif not 0 <= slice_idx < n_slices:
        raise ValueError(f"slice_idx={slice_idx} is out of bounds for {n_slices} slices")

    return int(slice_idx), int(volume_idx)


def save_denoising_slice_plot(
    noisy_dwi: np.ndarray,
    denoised_dwi: np.ndarray,
    bvals: np.ndarray,
    out_path: str | Path,
    subject_key: str,
    b0_threshold: float,
    target_dwi: np.ndarray | None = None,
    brain_mask_frac: float = 0.1,
    slice_idx: int | None = None,
    volume_idx: int | None = None,
    before_label: str = "Before denoising",
    after_label: str = "After denoising",
    target_label: str = "Target",
) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    slice_idx, volume_idx = select_plot_indices(
        dwi_4d=noisy_dwi,
        bvals=bvals,
        b0_threshold=b0_threshold,
        brain_mask_frac=brain_mask_frac,
        slice_idx=slice_idx,
        volume_idx=volume_idx,
    )

    before = noisy_dwi[:, :, slice_idx, volume_idx]
    after = denoised_dwi[:, :, slice_idx, volume_idx]
    before_norm = _normalize_slice(before)
    after_norm = _normalize_slice(after)
    before_kspace = _show_kspace(before_norm)
    after_kspace = _show_kspace(after_norm)
    target_norm = None
    target_kspace = None
    if target_dwi is not None:
        target = target_dwi[:, :, slice_idx, volume_idx]
        target_norm = _normalize_slice(target)
        target_kspace = _show_kspace(target_norm)
        diff_norm = target_norm - after_norm
        diff_kspace = target_kspace - after_kspace
        diff_title = "Difference map (target - denoised)"
    else:
        diff_norm = before_norm - after_norm
        diff_kspace = before_kspace - after_kspace
        diff_title = "Difference map (normalized)"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spatial_panels = [
        (before_norm, "gray", before_label),
        (after_norm, "gray", after_label),
    ]
    if target_norm is not None:
        spatial_panels.append((target_norm, "gray", target_label))
    spatial_panels.append((diff_norm, "bwr", diff_title))

    kspace_panels = [
        (before_kspace, "gray", f"{before_label} k-space"),
        (after_kspace, "gray", f"{after_label} k-space"),
    ]
    if target_kspace is not None:
        kspace_panels.append((target_kspace, "gray", f"{target_label} k-space"))
    kspace_panels.append((diff_kspace, "gray", "Difference k-space"))

    n_cols = len(spatial_panels)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    axes = np.atleast_2d(axes)

    diff_im = None
    for axis, (panel, cmap, title) in zip(axes[0], spatial_panels):
        diff_im = axis.imshow(np.rot90(panel, 1), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    diff_k_im = None
    for axis, (panel, cmap, title) in zip(axes[1], kspace_panels):
        diff_k_im = axis.imshow(np.rot90(panel, 1), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    fig.colorbar(diff_im, ax=axes[0, -1], fraction=0.046, pad=0.04)
    fig.colorbar(diff_k_im, ax=axes[1, -1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{subject_key} | z={slice_idx} | volume={volume_idx}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "subject": subject_key,
        "slice_idx": slice_idx,
        "volume_idx": volume_idx,
        "out_path": str(out_path),
    }
