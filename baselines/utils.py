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

def dti6d_to_scalar_maps(dti_6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals = dti6d_to_evals(dti_6d)
    return evals_to_fa(evals), evals_to_adc(evals)

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
def _normalize_slices_together(*images: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(image, dtype=np.float32) for image in images]
    stacked = np.stack(arrays, axis=0)
    lo = float(stacked.min())
    hi = float(stacked.max())
    if hi <= lo:
        return [np.zeros_like(image, dtype=np.float32) for image in arrays]
    scale = hi - lo
    return [((image - lo) / scale).astype(np.float32) for image in arrays]


def _show_kspace(image: np.ndarray) -> np.ndarray:
    kspace = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
    return np.log1p(np.abs(kspace)).astype(np.float32)


def _robust_limits(*images: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0),
                   default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    valid_values = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            valid_values.append(finite)

    if not valid_values:
        return default

    values = np.concatenate(valid_values)
    lo, hi = np.percentile(values, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return default
    return float(lo), float(hi)


def _symmetric_limits(image: np.ndarray, fallback: float = 1.0) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return -fallback, fallback
    vmax = float(np.max(np.abs(finite)))
    if vmax <= 0.0:
        vmax = fallback
    return -vmax, vmax


def _ensure_bvecs_n3(bvecs: np.ndarray) -> np.ndarray:
    bvecs = np.asarray(bvecs, dtype=np.float32)
    if bvecs.ndim != 2:
        raise ValueError(f"bvecs must be 2D, got shape {bvecs.shape}")
    if bvecs.shape[1] == 3:
        return bvecs
    if bvecs.shape[0] == 3:
        return bvecs.T
    raise ValueError(f"bvecs must have shape (N, 3) or (3, N), got {bvecs.shape}")


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
    bvecs: np.ndarray | None = None,
    target_dti6d: np.ndarray | None = None,
    dti_fit_method: str = "WLS",
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
    spatial_slices = [before, after]
    target = None
    if target_dwi is not None:
        target = target_dwi[:, :, slice_idx, volume_idx]
        spatial_slices.append(target)
    normalized_slices = _normalize_slices_together(*spatial_slices)
    before_norm = normalized_slices[0]
    after_norm = normalized_slices[1]
    target_norm = normalized_slices[2] if target is not None else None
    before_kspace = _show_kspace(before_norm)
    after_kspace = _show_kspace(after_norm)
    target_kspace = None
    if target_norm is not None:
        target_kspace = _show_kspace(target_norm)

    diff_norm = (target_norm - after_norm) if target_norm is not None else (before_norm - after_norm)
    diff_kspace = (target_kspace - after_kspace) if target_kspace is not None else (before_kspace - after_kspace)
    diff_title = "Difference map (target - denoised)" if target_norm is not None else "Difference map (normalized)"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_specs = [
        {
            "panels": [
                (before_norm, before_label),
                (after_norm, after_label),
            ] + ([(target_norm, target_label)] if target_norm is not None else []),
            "diff": (diff_norm, diff_title),
            "base_cmap": "gray",
            "diff_cmap": "bwr",
            "base_limits": (0.0, 1.0),
            "diff_limits": _symmetric_limits(diff_norm),
        },
        {
            "panels": [
                (before_kspace, f"{before_label} k-space"),
                (after_kspace, f"{after_label} k-space"),
            ] + ([(target_kspace, f"{target_label} k-space")] if target_kspace is not None else []),
            "diff": (diff_kspace, "Difference k-space"),
            "base_cmap": "gray",
            "diff_cmap": "bwr",
            "base_limits": _robust_limits(
                before_kspace,
                after_kspace,
                *( [target_kspace] if target_kspace is not None else [] ),
            ),
            "diff_limits": _symmetric_limits(diff_kspace),
        },
    ]

    if bvecs is not None:
        bvecs_n3 = _ensure_bvecs_n3(bvecs)
        before_dti6d = fit_dti_to_6d(
            noisy_dwi,
            bvals,
            bvecs_n3=bvecs_n3,
            fit_method=dti_fit_method,
            b0_threshold=b0_threshold,
        )
        after_dti6d = fit_dti_to_6d(
            denoised_dwi,
            bvals,
            bvecs_n3=bvecs_n3,
            fit_method=dti_fit_method,
            b0_threshold=b0_threshold,
        )
        if target_dti6d is None and target_dwi is not None:
            target_dti6d = fit_dti_to_6d(
                target_dwi,
                bvals,
                bvecs_n3=bvecs_n3,
                fit_method=dti_fit_method,
                b0_threshold=b0_threshold,
            )

        before_fa, before_adc = dti6d_to_scalar_maps(before_dti6d)
        after_fa, after_adc = dti6d_to_scalar_maps(after_dti6d)
        target_fa = target_adc = None
        if target_dti6d is not None:
            target_fa, target_adc = dti6d_to_scalar_maps(target_dti6d)

        fa_before_slice = before_fa[:, :, slice_idx]
        fa_after_slice = after_fa[:, :, slice_idx]
        fa_target_slice = target_fa[:, :, slice_idx] if target_fa is not None else None
        fa_diff_slice = (fa_target_slice - fa_after_slice) if fa_target_slice is not None else (fa_before_slice - fa_after_slice)

        adc_before_slice = before_adc[:, :, slice_idx]
        adc_after_slice = after_adc[:, :, slice_idx]
        adc_target_slice = target_adc[:, :, slice_idx] if target_adc is not None else None
        adc_diff_slice = (adc_target_slice - adc_after_slice) if adc_target_slice is not None else (adc_before_slice - adc_after_slice)

        row_specs.extend([
            {
                "panels": [
                    (fa_before_slice, f"{before_label} FA"),
                    (fa_after_slice, f"{after_label} FA"),
                ] + ([(fa_target_slice, f"{target_label} FA")] if fa_target_slice is not None else []),
                "diff": (fa_diff_slice, "FA difference"),
                "base_cmap": "viridis",
                "diff_cmap": "bwr",
                "base_limits": (0.0, 1.0),
                "diff_limits": _symmetric_limits(fa_diff_slice),
            },
            {
                "panels": [
                    (adc_before_slice, f"{before_label} ADC"),
                    (adc_after_slice, f"{after_label} ADC"),
                ] + ([(adc_target_slice, f"{target_label} ADC")] if adc_target_slice is not None else []),
                "diff": (adc_diff_slice, "ADC difference"),
                "base_cmap": "magma",
                "diff_cmap": "bwr",
                "base_limits": _robust_limits(
                    adc_before_slice,
                    adc_after_slice,
                    *( [adc_target_slice] if adc_target_slice is not None else [] ),
                ),
                "diff_limits": _symmetric_limits(adc_diff_slice),
            },
        ])

    n_cols = max(len(row["panels"]) + 1 for row in row_specs)
    fig, axes = plt.subplots(len(row_specs), n_cols, figsize=(6 * n_cols, 4.5 * len(row_specs)))
    axes = np.atleast_2d(axes)

    for row_idx, row in enumerate(row_specs):
        row_axes = np.atleast_1d(axes[row_idx])
        for axis in row_axes:
            axis.axis("off")

        base_vmin, base_vmax = row["base_limits"]
        for col_idx, (panel, title) in enumerate(row["panels"]):
            axis = row_axes[col_idx]
            axis.imshow(
                np.rot90(panel, 1),
                cmap=row["base_cmap"],
                vmin=base_vmin,
                vmax=base_vmax,
            )
            axis.set_title(title)
            axis.axis("off")

        diff_axis = row_axes[len(row["panels"])]
        diff_vmin, diff_vmax = row["diff_limits"]
        diff_im = diff_axis.imshow(
            np.rot90(row["diff"][0], 1),
            cmap=row["diff_cmap"],
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        diff_axis.set_title(row["diff"][1])
        diff_axis.axis("off")
        fig.colorbar(diff_im, ax=diff_axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{subject_key} | z={slice_idx} | volume={volume_idx} | DTI fit={dti_fit_method}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "subject": subject_key,
        "slice_idx": slice_idx,
        "volume_idx": volume_idx,
        "out_path": str(out_path),
    }


def save_prediction_slice_plot(
    input_dwi: np.ndarray,
    pred_dti6d: np.ndarray,
    target_dti6d: np.ndarray,
    bvals: np.ndarray,
    out_path: str | Path,
    subject_key: str,
    b0_threshold: float,
    target_dwi: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
    dti_fit_method: str = "WLS",
    brain_mask_frac: float = 0.1,
    slice_idx: int | None = None,
    volume_idx: int | None = None,
) -> dict:
    """Save a multi-row comparison plot for a DL model evaluation.

    Shows:
      Row 1 – DWI spatial: noisy input vs target (if available)
      Row 2 – DWI k-space: same
      Row 3 – FA maps: input (fitted) | predicted | target | diff (target - pred)
      Row 4 – ADC maps: same layout

    When *bvecs* is None the DWI rows are omitted and only DTI-space
    rows are produced using the pre-computed *pred_dti6d* and *target_dti6d*.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    slice_idx, volume_idx = select_plot_indices(
        dwi_4d=input_dwi,
        bvals=bvals,
        b0_threshold=b0_threshold,
        brain_mask_frac=brain_mask_frac,
        slice_idx=slice_idx,
        volume_idx=volume_idx,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_specs = []

    # ── DWI rows (only when we can show at least two images) ────────────────
    if target_dwi is not None:
        noisy_slice = input_dwi[:, :, slice_idx, volume_idx]
        target_slice = target_dwi[:, :, slice_idx, volume_idx]
        norm_noisy, norm_target = _normalize_slices_together(noisy_slice, target_slice)
        diff_dwi = norm_target - norm_noisy

        noisy_k = _show_kspace(norm_noisy)
        target_k = _show_kspace(norm_target)
        diff_k = target_k - noisy_k

        row_specs.append({
            "panels": [
                (norm_noisy, "Input (noisy)"),
                (norm_target, "Target"),
            ],
            "diff": (diff_dwi, "Difference (target − input)"),
            "base_cmap": "gray",
            "diff_cmap": "bwr",
            "base_limits": (0.0, 1.0),
            "diff_limits": _symmetric_limits(diff_dwi),
        })
        row_specs.append({
            "panels": [
                (noisy_k, "Input k-space"),
                (target_k, "Target k-space"),
            ],
            "diff": (diff_k, "Difference k-space"),
            "base_cmap": "gray",
            "diff_cmap": "bwr",
            "base_limits": _robust_limits(noisy_k, target_k),
            "diff_limits": _symmetric_limits(diff_k),
        })

    # ── DTI rows ─────────────────────────────────────────────────────────────
    # Fit DTI from noisy input only when bvecs are available
    input_fa = input_adc = None
    if bvecs is not None:
        bvecs_n3 = _ensure_bvecs_n3(bvecs)
        input_dti6d = fit_dti_to_6d(
            input_dwi, bvals, bvecs_n3=bvecs_n3,
            fit_method=dti_fit_method, b0_threshold=b0_threshold,
        )
        input_fa, input_adc = dti6d_to_scalar_maps(input_dti6d)

    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_dti6d)
    target_fa, target_adc = dti6d_to_scalar_maps(target_dti6d)

    fa_diff = target_fa[:, :, slice_idx] - pred_fa[:, :, slice_idx]
    adc_diff = target_adc[:, :, slice_idx] - pred_adc[:, :, slice_idx]

    fa_panels = []
    adc_panels = []
    if input_fa is not None:
        fa_panels.append((input_fa[:, :, slice_idx], "Input FA (fitted)"))
        adc_panels.append((input_adc[:, :, slice_idx], "Input ADC (fitted)"))
    fa_panels += [
        (pred_fa[:, :, slice_idx], "Predicted FA"),
        (target_fa[:, :, slice_idx], "Target FA"),
    ]
    adc_panels += [
        (pred_adc[:, :, slice_idx], "Predicted ADC"),
        (target_adc[:, :, slice_idx], "Target ADC"),
    ]

    row_specs.append({
        "panels": fa_panels,
        "diff": (fa_diff, "FA difference (target − pred)"),
        "base_cmap": "viridis",
        "diff_cmap": "bwr",
        "base_limits": (0.0, 1.0),
        "diff_limits": _symmetric_limits(fa_diff),
    })
    row_specs.append({
        "panels": adc_panels,
        "diff": (adc_diff, "ADC difference (target − pred)"),
        "base_cmap": "magma",
        "diff_cmap": "bwr",
        "base_limits": _robust_limits(
            pred_adc[:, :, slice_idx], target_adc[:, :, slice_idx],
            *([input_adc[:, :, slice_idx]] if input_adc is not None else []),
        ),
        "diff_limits": _symmetric_limits(adc_diff),
    })

    n_cols = max(len(row["panels"]) + 1 for row in row_specs)
    fig, axes = plt.subplots(len(row_specs), n_cols, figsize=(6 * n_cols, 4.5 * len(row_specs)))
    axes = np.atleast_2d(axes)

    for row_idx, row in enumerate(row_specs):
        row_axes = np.atleast_1d(axes[row_idx])
        for axis in row_axes:
            axis.axis("off")

        base_vmin, base_vmax = row["base_limits"]
        for col_idx, (panel, title) in enumerate(row["panels"]):
            axis = row_axes[col_idx]
            axis.imshow(np.rot90(panel, 1), cmap=row["base_cmap"], vmin=base_vmin, vmax=base_vmax)
            axis.set_title(title)
            axis.axis("off")

        diff_axis = row_axes[len(row["panels"])]
        diff_vmin, diff_vmax = row["diff_limits"]
        diff_im = diff_axis.imshow(
            np.rot90(row["diff"][0], 1),
            cmap=row["diff_cmap"],
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        diff_axis.set_title(row["diff"][1])
        diff_axis.axis("off")
        fig.colorbar(diff_im, ax=diff_axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{subject_key} | z={slice_idx} | volume={volume_idx} | DTI fit={dti_fit_method}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "subject": subject_key,
        "slice_idx": slice_idx,
        "volume_idx": volume_idx,
        "out_path": str(out_path),
    }
