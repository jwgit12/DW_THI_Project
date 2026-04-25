"""Evaluate a trained QSpaceUNet on test subjects.

Produces a CSV with DTI-level metrics (tensor RMSE, FA, ADC).

By default each subject is evaluated multiple times with independent
k-space cutouts and noise realisations. Patch2Self and MP-PCA are also
run unless disabled, and comparison plots + metric tables are saved.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint runs/production/best_model.pt --subjects sub-10 sub-11
    python evaluate.py --checkpoint runs/production/best_model.pt --skip_baselines
    python evaluate.py --checkpoint runs/production/best_model.pt --eval_repeats 5 --skip_mppca
    python evaluate.py --sweep_patch2self --eval_repeats 3
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
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .utils import (
    dti6d_to_scalar_maps,
    fit_dti_to_6d,
    sanitize_dti6d,
    scalar_map_metrics,
    save_prediction_slice_plot,
    select_plot_indices,
    _robust_limits,
    _symmetric_limits,
)
from .preprocessing import compute_b0_norm, compute_brain_mask_from_dwi
from .augment import degrade_dwi_volume
from .model import QSpaceUNet
from .runtime import (
    amp_dtype_from_name,
    autocast_context,
    configure_torch_runtime,
    get_device,
    maybe_channels_last,
    maybe_compile_model,
    path_str,
    require_cuda_if_requested,
    resolve_project_path,
)
import config as cfg


def _load_input_dwi(
    grp,
    target_dwi: np.ndarray | None = None,
    keep_fraction: float = cfg.EVAL_KEEP_FRACTION,
    noise_level: float = cfg.EVAL_NOISE_LEVEL,
    seed: int = cfg.EVAL_DEGRADE_SEED,
) -> np.ndarray:
    """Return the degraded DWI for eval.

    Uses the stored ``input_dwi`` array when present (back-compat with v1
    Zarr stores) and otherwise synthesises a reproducible degraded volume
    from ``target_dwi`` using the on-the-fly helpers.
    """
    if "input_dwi" in grp.array_keys():
        return np.asarray(grp["input_dwi"][:], dtype=np.float32)
    clean = target_dwi if target_dwi is not None else np.asarray(grp["target_dwi"][:], dtype=np.float32)
    return degrade_dwi_volume(
        clean, keep_fraction=keep_fraction, rel_noise_level=noise_level, seed=seed,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline defaults (match the standalone scripts)
# ─────────────────────────────────────────────────────────────────────────────
P2S_CFG = dict(
    model=cfg.P2S_MODEL, alpha=cfg.P2S_ALPHA, b0_threshold=cfg.B0_THRESHOLD,
    shift_intensity=cfg.P2S_SHIFT_INTENSITY, clip_negative=cfg.P2S_CLIP_NEGATIVE,
    b0_denoising=cfg.P2S_B0_DENOISING, dti_fit_method=cfg.DTI_FIT_METHOD,
)
MPPCA_CFG = dict(
    patch_radius=cfg.MPPCA_PATCH_RADIUS, pca_method=cfg.MPPCA_PCA_METHOD,
    b0_threshold=cfg.B0_THRESHOLD, dti_fit_method=cfg.DTI_FIT_METHOD,
)


def predict_subject(
    model: torch.nn.Module,
    zarr_path: str | Path,
    subject_key: str,
    device: torch.device,
    b0_threshold: float = cfg.B0_THRESHOLD,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
    input_dwi: np.ndarray | None = None,
    batch_size: int = 16,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
) -> np.ndarray:
    """Run inference on a full 3D subject, slice by slice.

    Returns predicted DTI tensor (X, Y, Z, 6) as float32 numpy array.
    """
    store = zarr.open_group(path_str(zarr_path), mode="r")
    grp = store[subject_key]

    if input_dwi is None:
        input_dwi = _load_input_dwi(grp)  # (X, Y, Z, N)
    else:
        input_dwi = np.asarray(input_dwi, dtype=np.float32)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)  # (3, N)

    X, Y, Z, N = input_dwi.shape
    max_n = getattr(model, "max_n", getattr(model, "_orig_mod", model).max_n)

    # Normalise bvals using the same max_bval as training
    bvals_norm = bvals / max_bval

    # Pad to max_n
    if N < max_n:
        pad = max_n - N
        bvals_norm = np.pad(bvals_norm, (0, pad))
        bvecs = np.pad(bvecs, ((0, 0), (0, pad)))
        input_dwi = np.pad(input_dwi, ((0, 0), (0, 0), (0, 0), (0, pad)))

    vol_mask = np.zeros(max_n, dtype=np.float32)
    vol_mask[:N] = 1.0

    # Prepare gradient tensors (shared across slices)
    non_blocking = device.type == "cuda"
    bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs_t = torch.from_numpy(bvecs).unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(device, non_blocking=non_blocking)

    # Compute normalisation factor from mean b0
    b0_idx = bvals[:N] < b0_threshold
    if b0_idx.any():
        mean_b0_vol = input_dwi[:, :, :, :N][..., b0_idx].mean(axis=-1)  # (X, Y, Z)
    else:
        mean_b0_vol = input_dwi[:, :, :, :N].mean(axis=-1)

    pred_dti = np.zeros((X, Y, Z, 6), dtype=np.float32)

    signals = np.ascontiguousarray(input_dwi.transpose(2, 3, 0, 1), dtype=np.float32)
    for z in range(Z):
        b0_norm = compute_b0_norm(mean_b0_vol[:, :, z])
        if b0_norm > 0:
            signals[z] /= np.float32(b0_norm)

    model.eval()
    batch_size = max(1, int(batch_size))
    with torch.inference_mode(), autocast_context(
        device, enabled=amp_enabled, dtype=amp_dtype,
    ):
        for start in range(0, Z, batch_size):
            end = min(start + batch_size, Z)
            signal_t = torch.from_numpy(signals[start:end]).to(
                device, non_blocking=non_blocking,
            )
            signal_t = maybe_channels_last(signal_t, channels_last)
            n_batch = end - start
            pred = model(
                signal_t,
                bvals_t.expand(n_batch, -1),
                bvecs_t.expand(n_batch, -1, -1),
                vol_mask_t.expand(n_batch, -1),
            )  # (B, 6, H, W)
            pred_dti[:, :, start:end, :] = (
                pred.permute(2, 3, 0, 1).float().cpu().numpy()
            )

    # Unscale from training range back to physical units
    pred_dti = pred_dti / dti_scale

    return pred_dti


# ─────────────────────────────────────────────────────────────────────────────
# Baseline runners
# ─────────────────────────────────────────────────────────────────────────────
def _run_patch2self(
    noisy: np.ndarray,
    bvals: np.ndarray,
    p2s_cfg: dict | None = None,
) -> np.ndarray:
    from dipy.denoise.patch2self import patch2self
    p2s_cfg = P2S_CFG if p2s_cfg is None else p2s_cfg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return patch2self(
            noisy, bvals, model=p2s_cfg["model"], alpha=p2s_cfg["alpha"],
            b0_threshold=p2s_cfg["b0_threshold"],
            shift_intensity=p2s_cfg["shift_intensity"],
            clip_negative_vals=p2s_cfg["clip_negative"],
            b0_denoising=p2s_cfg["b0_denoising"], verbose=False,
        ).astype(np.float32)


def _run_mppca(noisy: np.ndarray) -> np.ndarray:
    from dipy.denoise.localpca import mppca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mppca(
            noisy, patch_radius=MPPCA_CFG["patch_radius"],
            pca_method=MPPCA_CFG["pca_method"],
            suppress_warning=True,
        ).astype(np.float32)


def _compute_dti_metrics(
    pred_dti6d: np.ndarray,
    target_dti6d: np.ndarray,
    mask: np.ndarray | None = None,
    max_diffusivity: float = cfg.MAX_DIFFUSIVITY,
) -> dict:
    """Compute DTI metrics from predicted and target 6D tensors.

    Both tensors are sanitized (eigenvalue clamped to [0, max_diffusivity])
    before metric computation so that all methods are evaluated under
    identical physical constraints.
    """
    pred_clean = sanitize_dti6d(pred_dti6d, max_eigenvalue=max_diffusivity)
    tgt_clean = sanitize_dti6d(target_dti6d, max_eigenvalue=max_diffusivity)

    diff = pred_clean - tgt_clean
    if mask is not None:
        diff_brain = diff[mask]
        tensor_rmse = float(np.sqrt(np.mean(diff_brain ** 2)))
    else:
        tensor_rmse = float(np.sqrt(np.mean(diff ** 2)))

    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_clean)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(tgt_clean)
    fa_m = scalar_map_metrics(tgt_fa, pred_fa, mask=mask)
    adc_m = scalar_map_metrics(tgt_adc, pred_adc, mask=mask)

    return {
        "tensor_rmse": round(tensor_rmse, 6),
        "fa_rmse": round(fa_m["rmse"], 6),
        "fa_mae": round(fa_m["mae"], 6),
        "fa_nrmse": round(fa_m["nrmse"], 6),
        "fa_r2": round(fa_m["r2"], 4),
        "adc_rmse": round(adc_m["rmse"], 8),
        "adc_mae": round(adc_m["mae"], 8),
        "adc_nrmse": round(adc_m["nrmse"], 6),
        "adc_r2": round(adc_m["r2"], 4),
    }


def _baseline_dti_metrics(
    denoised_dwi: np.ndarray,
    target_dti6d: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    b0_threshold: float,
    dti_fit_method: str = "WLS",
    mask: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """Fit DTI from denoised DWI and compute metrics vs target.

    Clips DWI to non-negative before fitting to prevent unstable DTI
    estimates from negative signal values introduced during degradation.
    """
    bvecs_n3 = bvecs.T if bvecs.shape[0] == 3 else bvecs
    denoised_dwi = np.maximum(denoised_dwi, 0.0)
    fitted_dti6d = fit_dti_to_6d(
        denoised_dwi, bvals, bvecs_n3=bvecs_n3,
        fit_method=dti_fit_method, b0_threshold=b0_threshold,
    )
    metrics = _compute_dti_metrics(fitted_dti6d, target_dti6d, mask=mask)
    return metrics, fitted_dti6d


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a single subject/repeat (all enabled methods)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subject(
    model: torch.nn.Module,
    zarr_path: str | Path,
    subject_key: str,
    device: torch.device,
    b0_threshold: float = cfg.B0_THRESHOLD,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
    repeat_idx: int = 0,
    keep_fraction: float = cfg.EVAL_KEEP_FRACTION,
    noise_level: float = cfg.EVAL_NOISE_LEVEL,
    degrade_seed: int = cfg.EVAL_DEGRADE_SEED,
    run_patch2self: bool = True,
    run_mppca: bool = True,
    p2s_cfg: dict | None = None,
    infer_batch_size: int = 16,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
) -> tuple[dict, dict]:
    """Full evaluation pipeline for one subject and one degradation repeat.

    Returns
    -------
    metrics : dict
        Keys: 'qspaceunet', and optionally 'patch2self', 'mppca'.
        Each value is a dict of scalar metrics plus degradation metadata.
    arrays : dict
        Raw arrays needed for visualization.
    """
    t0 = time.time()

    store = zarr.open_group(path_str(zarr_path), mode="r")
    grp = store[subject_key]
    target_dti6d = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
    target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
    input_dwi = degrade_dwi_volume(
        target_dwi,
        keep_fraction=keep_fraction,
        rel_noise_level=noise_level,
        seed=degrade_seed,
    )
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)

    # Brain mask is part of the production Zarr contract. Older stores without
    # it still run via the same DIPY median_otsu fallback used by the builder.
    if "brain_mask" in set(grp.array_keys()):
        mask_3d = np.asarray(grp["brain_mask"][:], dtype=bool)
    else:
        mask_3d = compute_brain_mask_from_dwi(target_dwi, bvals, b0_threshold)

    # ── QSpaceUNet model ──────────────────────────────────────────────────
    pred_dti6d = predict_subject(
        model, zarr_path, subject_key, device,
        b0_threshold=b0_threshold,
        dti_scale=dti_scale,
        max_bval=max_bval,
        input_dwi=input_dwi,
        batch_size=infer_batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )

    qspaceunet_elapsed = time.time() - t0
    row_meta = {
        "subject": subject_key,
        "repeat": int(repeat_idx),
        "keep_fraction": round(float(keep_fraction), 6),
        "noise_level": round(float(noise_level), 6),
        "degrade_seed": int(degrade_seed),
    }
    qspaceunet_metrics = _compute_dti_metrics(pred_dti6d, target_dti6d, mask=mask_3d)
    qspaceunet_metrics.update(row_meta)
    qspaceunet_metrics["elapsed_s"] = round(qspaceunet_elapsed, 2)

    all_metrics = {"qspaceunet": qspaceunet_metrics}
    arrays = {
        "input_dwi": input_dwi,
        "target_dwi": target_dwi,
        "target_dti6d": target_dti6d,
        "bvals": bvals,
        "bvecs": bvecs,
        "qspaceunet_dti6d": pred_dti6d,
        "brain_mask_3d": mask_3d,
        "repeat": int(repeat_idx),
        "keep_fraction": float(keep_fraction),
        "noise_level": float(noise_level),
        "degrade_seed": int(degrade_seed),
    }

    # ── Baselines ─────────────────────────────────────────────────────────
    if run_patch2self:
        p2s_cfg = P2S_CFG if p2s_cfg is None else p2s_cfg
        t1 = time.time()
        p2s_denoised = _run_patch2self(input_dwi, bvals, p2s_cfg=p2s_cfg)
        p2s_metrics, p2s_dti6d = _baseline_dti_metrics(
            p2s_denoised, target_dti6d, bvals, bvecs, b0_threshold,
            dti_fit_method=p2s_cfg["dti_fit_method"], mask=mask_3d,
        )
        p2s_metrics.update(row_meta)
        p2s_metrics.update(
            {
                "p2s_model": p2s_cfg["model"],
                "p2s_alpha": p2s_cfg["alpha"],
                "p2s_b0_denoising": p2s_cfg["b0_denoising"],
                "p2s_clip_negative": p2s_cfg["clip_negative"],
                "p2s_shift_intensity": p2s_cfg["shift_intensity"],
                "p2s_dti_fit_method": p2s_cfg["dti_fit_method"],
            }
        )
        p2s_metrics["elapsed_s"] = round(time.time() - t1, 2)
        all_metrics["patch2self"] = p2s_metrics
        arrays["patch2self_dti6d"] = p2s_dti6d

    if run_mppca:
        t2 = time.time()
        mppca_denoised = _run_mppca(input_dwi)
        mppca_metrics, mppca_dti6d = _baseline_dti_metrics(
            mppca_denoised, target_dti6d, bvals, bvecs, b0_threshold,
            mask=mask_3d,
        )
        mppca_metrics.update(row_meta)
        mppca_metrics["elapsed_s"] = round(time.time() - t2, 2)
        all_metrics["mppca"] = mppca_metrics
        arrays["mppca_dti6d"] = mppca_dti6d

    return all_metrics, arrays


# ─────────────────────────────────────────────────────────────────────────────
# Comparison visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plot(
    arrays: dict,
    subject_key: str,
    out_path: Path,
    b0_threshold: float,
    slice_idx: int | None = None,
    volume_idx: int | None = None,
) -> dict:
    """Save a multi-row comparison plot showing all methods side by side.

    Rows: FA maps, ADC maps.
    Columns: Input (fitted) | Patch2Self | MP-PCA | QSpaceUNet | Target | Best-diff
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    input_dwi = arrays["input_dwi"]
    bvals = arrays["bvals"]
    bvecs = arrays["bvecs"]
    target_dti6d = arrays["target_dti6d"]

    slice_idx, volume_idx = select_plot_indices(
        dwi_4d=input_dwi, bvals=bvals, b0_threshold=b0_threshold,
        slice_idx=slice_idx, volume_idx=volume_idx,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fit DTI from noisy input for reference
    bvecs_n3 = bvecs.T if bvecs.shape[0] == 3 else bvecs
    input_dti6d = fit_dti_to_6d(
        input_dwi, bvals, bvecs_n3=bvecs_n3,
        fit_method="WLS", b0_threshold=b0_threshold,
    )

    # Collect all DTI tensors: (label, dti6d)
    methods = [("Input (noisy)", input_dti6d)]
    if "patch2self_dti6d" in arrays:
        methods.append(("Patch2Self", arrays["patch2self_dti6d"]))
    if "mppca_dti6d" in arrays:
        methods.append(("MP-PCA", arrays["mppca_dti6d"]))
    methods.append(("QSpaceUNet", arrays["qspaceunet_dti6d"]))
    methods.append(("Target", target_dti6d))

    # Brain mask for this slice
    mask_2d = arrays.get("brain_mask_3d")
    if mask_2d is not None:
        mask_2d = mask_2d[:, :, slice_idx].astype(np.float32)

    # Derive FA and ADC for all methods
    fa_maps = []
    adc_maps = []
    for label, dti6d in methods:
        fa, adc = dti6d_to_scalar_maps(dti6d)
        fa_slice = fa[:, :, slice_idx]
        adc_slice = adc[:, :, slice_idx]
        if mask_2d is not None:
            fa_slice = fa_slice * mask_2d
            adc_slice = adc_slice * mask_2d
        fa_maps.append((label, fa_slice))
        adc_maps.append((label, adc_slice))

    tgt_fa_slice = fa_maps[-1][1]
    tgt_adc_slice = adc_maps[-1][1]

    # Build figure: 2 rows (FA, ADC) x (N methods + 1 diff column)
    n_methods = len(methods)
    n_cols = n_methods + 1  # extra column for best method's diff map
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 4.5 * 2))

    # Find best non-target, non-input method for diff map
    method_diffs_fa = {}
    method_diffs_adc = {}
    for label, fa_slice in fa_maps[1:-1]:  # skip Input and Target
        method_diffs_fa[label] = tgt_fa_slice - fa_slice
    for label, adc_slice in adc_maps[1:-1]:
        method_diffs_adc[label] = tgt_adc_slice - adc_slice

    # ── Row 0: FA ─────────────────────────────────────────────────────────
    for col, (label, fa_slice) in enumerate(fa_maps):
        ax = axes[0, col]
        ax.imshow(np.rot90(fa_slice, 1), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    # Diff column: show QSpaceUNet diff.
    if "QSpaceUNet" in method_diffs_fa:
        diff_fa = method_diffs_fa["QSpaceUNet"]
    elif method_diffs_fa:
        diff_fa = next(iter(method_diffs_fa.values()))
    else:
        diff_fa = np.zeros_like(tgt_fa_slice)
    vmin_fa, vmax_fa = _symmetric_limits(diff_fa)
    ax = axes[0, -1]
    im_fa = ax.imshow(np.rot90(diff_fa, 1), cmap="bwr", vmin=vmin_fa, vmax=vmax_fa)
    ax.set_title("FA diff (target - ours)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im_fa, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 1: ADC ────────────────────────────────────────────────────────
    # Noisy input DTI fitting can produce extreme eigenvalues (ADC >> 1),
    # so derive the colour range from the target only.
    tgt_adc_for_limits = adc_maps[-1][1]  # last entry is "Target"
    adc_vmin, adc_vmax = _robust_limits(tgt_adc_for_limits)

    for col, (label, adc_slice) in enumerate(adc_maps):
        ax = axes[1, col]
        ax.imshow(np.rot90(adc_slice, 1), cmap="magma", vmin=adc_vmin, vmax=adc_vmax)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    if "QSpaceUNet" in method_diffs_adc:
        diff_adc = method_diffs_adc["QSpaceUNet"]
    elif method_diffs_adc:
        diff_adc = next(iter(method_diffs_adc.values()))
    else:
        diff_adc = np.zeros_like(tgt_adc_slice)
    vmin_adc, vmax_adc = _symmetric_limits(diff_adc)
    ax = axes[1, -1]
    im_adc = ax.imshow(np.rot90(diff_adc, 1), cmap="bwr", vmin=vmin_adc, vmax=vmax_adc)
    ax.set_title("ADC diff (target - ours)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im_adc, ax=ax, fraction=0.046, pad=0.04)

    degrade_info = ""
    if "repeat" in arrays:
        degrade_info = (
            f"  |  repeat={arrays['repeat']}  "
            f"keep={arrays.get('keep_fraction', float('nan')):.3f}  "
            f"noise={arrays.get('noise_level', float('nan')):.3f}"
        )
    fig.suptitle(
        f"Method Comparison  |  {subject_key}  |  z={slice_idx}{degrade_info}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"subject": subject_key, "slice_idx": slice_idx, "volume_idx": volume_idx, "out_path": str(out_path)}


def save_metric_comparison(
    all_rows: dict[str, list[dict]],
    out_dir: Path,
) -> Path:
    """Save a bar-chart + table comparing metrics across methods.

    Parameters
    ----------
    all_rows : dict mapping method name -> list of per-subject metric dicts
    out_dir : output directory

    Returns the path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]
    higher_is_better = {"fa_r2", "adc_r2"}

    # Build mean dataframe
    means = {}
    for method, rows in all_rows.items():
        df = pd.DataFrame(rows)
        means[method] = {c: df[c].mean() for c in metric_cols if c in df.columns}
    means_df = pd.DataFrame(means).T  # methods x metrics

    # Save CSV
    csv_path = out_dir / "comparison_metrics.csv"
    means_df.round(6).to_csv(csv_path)
    log.info("Saved metric comparison CSV -> %s", csv_path)

    # Per-subject comparison CSV
    subject_rows = []
    for method, rows in all_rows.items():
        for row in rows:
            r = {"method": method}
            r.update(row)
            subject_rows.append(r)
    subject_df = pd.DataFrame(subject_rows)
    per_subj_csv = out_dir / "comparison_per_subject.csv"
    subject_df.to_csv(per_subj_csv, index=False)
    log.info("Saved per-subject comparison CSV -> %s", per_subj_csv)

    # Bar chart
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    method_names = list(means_df.index)
    display_names = {
        "qspaceunet": "QSpaceUNet",
        "patch2self": "Patch2Self",
        "mppca": "MP-PCA",
    }
    colors = {"qspaceunet": "#2196F3", "patch2self": "#FF9800", "mppca": "#4CAF50"}
    bar_colors = [colors.get(m, "#9E9E9E") for m in method_names]

    for i, metric in enumerate(metric_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = [means_df.loc[m, metric] if metric in means_df.columns else 0 for m in method_names]
        labels = [display_names.get(m, m) for m in method_names]
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)

        # Highlight best
        if metric in higher_is_better:
            best_idx = int(np.argmax(vals))
        else:
            best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Remove unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metric Comparison (mean over evaluation repeats)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plot_path = out_dir / "comparison_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved metric comparison plot -> %s", plot_path)

    return plot_path


def _validate_eval_args(args) -> None:
    if args.eval_repeats < 1:
        raise ValueError("--eval_repeats must be >= 1")
    if args.plot_repeat < 0:
        raise ValueError("--plot_repeat must be >= 0")
    if args.plot_repeat >= args.eval_repeats:
        raise ValueError("--plot_repeat must be less than --eval_repeats")
    if not (0.0 < args.eval_keep_fraction_min <= args.eval_keep_fraction_max <= 1.0):
        raise ValueError(
            "--eval_keep_fraction_min/max must satisfy 0 < min <= max <= 1"
        )
    if not (0.0 <= args.eval_noise_min <= args.eval_noise_max):
        raise ValueError("--eval_noise_min/max must satisfy 0 <= min <= max")
    if args.p2s_alpha < 0:
        raise ValueError("--p2s_alpha must be >= 0")
    if any(alpha < 0 for alpha in args.p2s_sweep_alphas):
        raise ValueError("--p2s_sweep_alphas values must be >= 0")


def _next_degradation_trial(
    rng: np.random.Generator,
    repeat_idx: int,
    keep_fraction_range: tuple[float, float],
    noise_range: tuple[float, float],
) -> dict:
    keep_min, keep_max = keep_fraction_range
    noise_min, noise_max = noise_range
    if keep_min == keep_max:
        keep_fraction = float(keep_min)
    else:
        keep_fraction = float(rng.uniform(keep_min, keep_max))
    if noise_min == noise_max:
        noise_level = float(noise_min)
    else:
        noise_level = float(rng.uniform(noise_min, noise_max))
    return {
        "repeat_idx": repeat_idx,
        "keep_fraction": keep_fraction,
        "noise_level": noise_level,
        "degrade_seed": int(rng.integers(0, np.iinfo(np.int32).max)),
    }


def _bool_choices(values: list[str]) -> list[bool]:
    return [value.lower() == "true" for value in values]


def _p2s_cfg_from_args(args, overrides: dict | None = None) -> dict:
    p2s_cfg = dict(P2S_CFG)
    p2s_cfg.update(
        model=args.p2s_model,
        alpha=float(args.p2s_alpha),
        b0_threshold=float(args.b0_threshold),
        shift_intensity=bool(args.p2s_shift_intensity),
        clip_negative=bool(args.p2s_clip_negative),
        b0_denoising=bool(args.p2s_b0_denoising),
        dti_fit_method=args.p2s_dti_fit_method,
    )
    if overrides:
        p2s_cfg.update(overrides)
    return p2s_cfg


def _iter_p2s_sweep_configs(args) -> list[dict]:
    configs = []
    b0_values = _bool_choices(args.p2s_sweep_b0_denoising)
    clip_values = _bool_choices(args.p2s_sweep_clip_negative)
    shift_values = _bool_choices(args.p2s_sweep_shift_intensity)
    for model in args.p2s_sweep_models:
        alpha_values = args.p2s_sweep_alphas if model != "ols" else [args.p2s_alpha]
        for alpha, b0_denoising, clip_negative, shift_intensity in itertools.product(
            alpha_values, b0_values, clip_values, shift_values,
        ):
            configs.append(
                _p2s_cfg_from_args(
                    args,
                    overrides={
                        "model": model,
                        "alpha": float(alpha),
                        "b0_denoising": b0_denoising,
                        "clip_negative": clip_negative,
                        "shift_intensity": shift_intensity,
                    },
                )
            )
    # Deduplicate configs after OLS alpha collapsing while preserving order.
    unique = []
    seen = set()
    for p2s_cfg in configs:
        key = (
            p2s_cfg["model"],
            p2s_cfg["alpha"],
            p2s_cfg["b0_denoising"],
            p2s_cfg["clip_negative"],
            p2s_cfg["shift_intensity"],
            p2s_cfg["dti_fit_method"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(p2s_cfg)
    return unique


def _expand_subjects(subject_ids: list[str] | None, all_keys: list[str]) -> list[str]:
    if not subject_ids:
        return []
    subjects = []
    seen = set()
    for subject_id in subject_ids:
        if subject_id in all_keys:
            matches = [subject_id]
        else:
            matches = [k for k in all_keys if k.rsplit("_ses-", 1)[0] == subject_id]
        for match in matches:
            if match not in seen:
                seen.add(match)
                subjects.append(match)
    return subjects


def _select_eval_subjects(
    all_keys: list[str],
    args,
    checkpoint_subjects: list[str] | None = None,
    default_subjects: list[str] | None = None,
) -> list[str]:
    if args.eval_all:
        return all_keys
    if args.subjects:
        return _expand_subjects(args.subjects, all_keys)
    if default_subjects:
        subjects = _expand_subjects(default_subjects, all_keys)
        if subjects:
            return subjects
    if checkpoint_subjects:
        subjects = _expand_subjects(checkpoint_subjects, all_keys)
        if subjects:
            return subjects
    return all_keys


def _matches_plot_subject(subject_key: str, requested: str | None) -> bool:
    if requested is None:
        return True
    return subject_key == requested or subject_key.rsplit("_ses-", 1)[0] == requested


def _plot_key(subject_key: str, repeat_idx: int, eval_repeats: int) -> str:
    if eval_repeats <= 1:
        return subject_key
    return f"{subject_key}_repeat-{repeat_idx:02d}"


def run_patch2self_sweep(
    args,
    subjects: list[str],
    out_dir: Path,
) -> pd.DataFrame | None:
    """Evaluate a Patch2Self hyperparameter grid on shared degraded inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p2s_configs = _iter_p2s_sweep_configs(args)
    if not p2s_configs:
        log.error("No Patch2Self sweep configurations were generated.")
        return None

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]
    higher_is_better = {"fa_r2", "adc_r2"}
    sort_ascending = args.p2s_sweep_metric not in higher_is_better

    log.info(
        "Running Patch2Self sweep: %d configs x %d subjects x %d repeats",
        len(p2s_configs), len(subjects), args.eval_repeats,
    )

    store = zarr.open_group(path_str(args.zarr_path), mode="r")
    eval_rng = np.random.default_rng(args.eval_seed)
    rows = []

    for subject_key in subjects:
        grp = store[subject_key]
        target_dti6d = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
        target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
        bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
        bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)
        mask_3d = compute_brain_mask_from_dwi(target_dwi, bvals, args.b0_threshold)

        for repeat_idx in range(args.eval_repeats):
            trial = _next_degradation_trial(
                eval_rng,
                repeat_idx=repeat_idx,
                keep_fraction_range=(
                    args.eval_keep_fraction_min,
                    args.eval_keep_fraction_max,
                ),
                noise_range=(args.eval_noise_min, args.eval_noise_max),
            )
            input_dwi = degrade_dwi_volume(
                target_dwi,
                keep_fraction=trial["keep_fraction"],
                rel_noise_level=trial["noise_level"],
                seed=trial["degrade_seed"],
            )

            for config_idx, p2s_cfg in enumerate(p2s_configs):
                t0 = time.time()
                try:
                    p2s_denoised = _run_patch2self(input_dwi, bvals, p2s_cfg=p2s_cfg)
                    metrics, _ = _baseline_dti_metrics(
                        p2s_denoised,
                        target_dti6d,
                        bvals,
                        bvecs,
                        args.b0_threshold,
                        dti_fit_method=p2s_cfg["dti_fit_method"],
                        mask=mask_3d,
                    )
                except Exception as exc:
                    log.warning(
                        "Patch2Self sweep failed subject=%s repeat=%d config=%d: %s",
                        subject_key, repeat_idx, config_idx, exc,
                    )
                    continue

                row = {
                    "subject": subject_key,
                    "repeat": repeat_idx,
                    "keep_fraction": round(float(trial["keep_fraction"]), 6),
                    "noise_level": round(float(trial["noise_level"]), 6),
                    "degrade_seed": int(trial["degrade_seed"]),
                    "config_idx": config_idx,
                    "p2s_model": p2s_cfg["model"],
                    "p2s_alpha": p2s_cfg["alpha"],
                    "p2s_b0_denoising": p2s_cfg["b0_denoising"],
                    "p2s_clip_negative": p2s_cfg["clip_negative"],
                    "p2s_shift_intensity": p2s_cfg["shift_intensity"],
                    "p2s_dti_fit_method": p2s_cfg["dti_fit_method"],
                    "elapsed_s": round(time.time() - t0, 2),
                }
                row.update(metrics)
                rows.append(row)

                log.info(
                    "%-14s r=%02d cfg=%02d model=%s alpha=%.3g b0=%s clip=%s "
                    "%s=%.6f",
                    subject_key, repeat_idx, config_idx,
                    p2s_cfg["model"], p2s_cfg["alpha"],
                    p2s_cfg["b0_denoising"], p2s_cfg["clip_negative"],
                    args.p2s_sweep_metric, metrics[args.p2s_sweep_metric],
                )

    if not rows:
        log.error("Patch2Self sweep produced no successful rows.")
        return None

    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / "patch2self_sweep.csv"
    raw_df.to_csv(raw_path, index=False)
    log.info("Saved Patch2Self sweep rows -> %s", raw_path)

    group_cols = [
        "config_idx",
        "p2s_model",
        "p2s_alpha",
        "p2s_b0_denoising",
        "p2s_clip_negative",
        "p2s_shift_intensity",
        "p2s_dti_fit_method",
    ]
    available_metrics = [c for c in metric_cols if c in raw_df.columns]
    summary = raw_df.groupby(group_cols, dropna=False)[available_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary = summary.sort_values(
        f"{args.p2s_sweep_metric}_mean",
        ascending=sort_ascending,
    ).reset_index(drop=True)

    summary_path = out_dir / "patch2self_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved Patch2Self sweep summary -> %s", summary_path)

    best = summary.iloc[0]
    print(f"\n{'=' * 72}")
    print("  Patch2Self Sweep Best Config")
    print(f"{'=' * 72}")
    print(
        "model={model}  alpha={alpha:.6g}  b0_denoising={b0}  "
        "clip_negative={clip}  shift_intensity={shift}  "
        "{metric}_mean={score:.6f}".format(
            model=best["p2s_model"],
            alpha=best["p2s_alpha"],
            b0=best["p2s_b0_denoising"],
            clip=best["p2s_clip_negative"],
            shift=best["p2s_shift_intensity"],
            metric=args.p2s_sweep_metric,
            score=best[f"{args.p2s_sweep_metric}_mean"],
        )
    )
    print("\nTop configs:")
    top_cols = [
        "p2s_model",
        "p2s_alpha",
        "p2s_b0_denoising",
        "p2s_clip_negative",
        "p2s_shift_intensity",
        f"{args.p2s_sweep_metric}_mean",
        f"{args.p2s_sweep_metric}_std",
    ]
    print(summary[top_cols].head(8).to_string(index=False))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    _validate_eval_args(args)

    zarr_path = path_str(args.zarr_path)
    out_dir = resolve_project_path(args.out_dir)
    checkpoint_path = resolve_project_path(args.checkpoint)
    args.zarr_path = zarr_path
    args.out_dir = str(out_dir)

    device = get_device()
    log.info("Device: %s", device)
    require_cuda_if_requested(device, args.require_cuda)
    configure_torch_runtime(device, deterministic=args.deterministic)
    amp_dtype = amp_dtype_from_name(device, args.amp_dtype)
    amp_enabled = bool(args.amp and amp_dtype is not None)
    channels_last = bool(args.channels_last and device.type == "cuda")
    if amp_enabled:
        log.info("AMP: enabled (%s)", str(amp_dtype).replace("torch.", ""))
    else:
        log.info("AMP: disabled")

    store = zarr.open_group(zarr_path, mode="r")
    all_keys = sorted(store.keys())

    if args.sweep_patch2self:
        subjects = _select_eval_subjects(
            all_keys,
            args,
            default_subjects=cfg.VAL_SUBJECTS,
        )
        if not subjects:
            log.error("No subjects selected for Patch2Self sweep.")
            return
        log.info("Patch2Self sweep subjects: %s", subjects)
        run_patch2self_sweep(args, subjects, out_dir)
        return

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    max_n = ckpt["max_n"]
    feat_dim = ckpt.get("feat_dim", 64)
    channels = tuple(ckpt.get("channels", [64, 128, 256, 512]))
    cholesky = ckpt.get("cholesky", False)
    dti_scale = ckpt.get("dti_scale", 1.0)
    max_bval = ckpt.get("max_bval", 1000.0)
    log.info("DTI scale factor: %.4f, max_bval: %.1f, cholesky: %s", dti_scale, max_bval, cholesky)

    raw_model = QSpaceUNet(max_n=max_n, feat_dim=feat_dim, channels=channels, cholesky=cholesky).to(device)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)
    model, is_compiled = maybe_compile_model(
        raw_model,
        setting=args.compile,
        device=device,
        mode=args.compile_mode,
    )
    model.eval()
    log.info("torch.compile: %s", f"enabled (mode={args.compile_mode})" if is_compiled else "disabled")
    log.info("Loaded checkpoint from epoch %d (val_loss=%.6f)", ckpt["epoch"], ckpt["val_loss"])

    # Determine subjects to evaluate
    subjects = _select_eval_subjects(
        all_keys,
        args,
        checkpoint_subjects=ckpt.get("test_subjects", []),
    )
    if not subjects:
        log.error("No subjects selected for evaluation.")
        return

    run_patch2self = (not args.skip_baselines) and args.patch2self
    run_mppca = (not args.skip_baselines) and args.mppca
    p2s_cfg = _p2s_cfg_from_args(args)
    log.info(
        "Evaluating %d subjects x %d repeats  (Patch2Self=%s, MP-PCA=%s)",
        len(subjects), args.eval_repeats, run_patch2self, run_mppca,
    )
    log.info(
        "Eval degradation: keep_fraction=[%.3f, %.3f], noise=[%.3f, %.3f], seed=%d",
        args.eval_keep_fraction_min, args.eval_keep_fraction_max,
        args.eval_noise_min, args.eval_noise_max,
        args.eval_seed,
    )

    # Run evaluation
    qspaceunet_rows = []
    p2s_rows = []
    mppca_rows = []
    plot_arrays = {}
    eval_rng = np.random.default_rng(args.eval_seed)

    for subj in subjects:
        for repeat_idx in range(args.eval_repeats):
            trial = _next_degradation_trial(
                eval_rng,
                repeat_idx=repeat_idx,
                keep_fraction_range=(
                    args.eval_keep_fraction_min,
                    args.eval_keep_fraction_max,
                ),
                noise_range=(args.eval_noise_min, args.eval_noise_max),
            )
            try:
                all_metrics, arrays = evaluate_subject(
                    model, zarr_path, subj, device,
                    b0_threshold=args.b0_threshold,
                    dti_scale=dti_scale,
                    max_bval=max_bval,
                    repeat_idx=trial["repeat_idx"],
                    keep_fraction=trial["keep_fraction"],
                    noise_level=trial["noise_level"],
                    degrade_seed=trial["degrade_seed"],
                    run_patch2self=run_patch2self,
                    run_mppca=run_mppca,
                    p2s_cfg=p2s_cfg,
                    infer_batch_size=args.infer_batch_size,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                )
                rm = all_metrics["qspaceunet"]
                log.info(
                    "%-14s  r=%02d  keep=%.3f noise=%.3f  tensor_rmse=%.5f  "
                    "FA[rmse=%.4f r2=%.3f]  ADC[rmse=%.2e r2=%.3f]  (%.1fs)",
                    rm["subject"], rm["repeat"], rm["keep_fraction"], rm["noise_level"],
                    rm["tensor_rmse"], rm["fa_rmse"], rm["fa_r2"],
                    rm["adc_rmse"], rm["adc_r2"], rm["elapsed_s"],
                )
                qspaceunet_rows.append(rm)
                if "patch2self" in all_metrics:
                    p2s_rows.append(all_metrics["patch2self"])
                    log.info(
                        "  Patch2Self   tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  (%.1fs)",
                        all_metrics["patch2self"]["tensor_rmse"],
                        all_metrics["patch2self"]["fa_rmse"],
                        all_metrics["patch2self"]["fa_r2"],
                        all_metrics["patch2self"]["elapsed_s"],
                    )
                if "mppca" in all_metrics:
                    mppca_rows.append(all_metrics["mppca"])
                    log.info(
                        "  MP-PCA       tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  (%.1fs)",
                        all_metrics["mppca"]["tensor_rmse"],
                        all_metrics["mppca"]["fa_rmse"],
                        all_metrics["mppca"]["fa_r2"],
                        all_metrics["mppca"]["elapsed_s"],
                    )

                if (
                    not args.skip_plot
                    and repeat_idx == args.plot_repeat
                    and _matches_plot_subject(subj, args.plot_subject)
                ):
                    plot_arrays[_plot_key(subj, repeat_idx, args.eval_repeats)] = arrays
            except Exception as exc:
                log.warning("FAIL  %s repeat=%d  —  %s", subj, repeat_idx, exc)

    if not qspaceunet_rows:
        log.error("No subjects evaluated successfully.")
        return

    # ── Save CSVs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]

    def _save_method_csv(rows: list[dict], name: str) -> pd.DataFrame:
        sort_cols = [c for c in ("subject", "repeat") if c in rows[0]]
        df = pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)
        cols = [c for c in metric_cols if c in df.columns]
        summary = df[cols].agg(["mean", "std"]).round(6).reset_index()
        summary.columns = ["subject"] + cols
        summary["subject"] = summary["subject"].str.upper()
        df_out = pd.concat([df, summary], ignore_index=True)
        path = out_dir / f"metrics_{name}.csv"
        df_out.to_csv(path, index=False)
        log.info("Saved -> %s", path)
        return df

    df_qspaceunet = _save_method_csv(qspaceunet_rows, "qspaceunet")
    if p2s_rows:
        _save_method_csv(p2s_rows, "patch2self")
    if mppca_rows:
        _save_method_csv(mppca_rows, "mppca")

    # Also save the primary model CSV under a generic name for convenience.
    compat_sort_cols = [c for c in ("subject", "repeat") if c in qspaceunet_rows[0]]
    df_compat = pd.DataFrame(qspaceunet_rows).sort_values(compat_sort_cols).reset_index(drop=True)
    cols = [c for c in metric_cols if c in df_compat.columns]
    summary = df_compat[cols].agg(["mean", "std"]).round(6).reset_index()
    summary.columns = ["subject"] + cols
    summary["subject"] = summary["subject"].str.upper()
    pd.concat([df_compat, summary], ignore_index=True).to_csv(
        out_dir / "metrics_per_subject.csv", index=False
    )

    # ── Metric comparison ─────────────────────────────────────────────────
    comparison_rows = {"qspaceunet": qspaceunet_rows}
    if p2s_rows:
        comparison_rows["patch2self"] = p2s_rows
    if mppca_rows:
        comparison_rows["mppca"] = mppca_rows
    if len(comparison_rows) > 1:
        save_metric_comparison(comparison_rows, out_dir)

    # ── Visualization ─────────────────────────────────────────────────────
    if qspaceunet_rows and not args.skip_plot and plot_arrays:
        for plot_subject, arrs in plot_arrays.items():
            # Original prediction plot
            plot_path = out_dir / f"prediction_example_{plot_subject}.png"
            try:
                plot_meta = save_prediction_slice_plot(
                    input_dwi=arrs["input_dwi"],
                    pred_dti6d=arrs["qspaceunet_dti6d"],
                    target_dti6d=arrs["target_dti6d"],
                    bvals=arrs["bvals"],
                    out_path=plot_path,
                    subject_key=plot_subject,
                    b0_threshold=args.b0_threshold,
                    target_dwi=arrs["target_dwi"],
                    bvecs=arrs["bvecs"],
                    slice_idx=args.plot_slice_idx,
                    volume_idx=args.plot_volume_idx,
                )
                log.info("Saved prediction plot -> %s  (z=%d, volume=%d)",
                         plot_meta["out_path"], plot_meta["slice_idx"], plot_meta["volume_idx"])
            except Exception as exc:
                log.warning("Could not save prediction plot for %s: %s", plot_subject, exc)

            # All-methods comparison plot
            if "patch2self_dti6d" in arrs or "mppca_dti6d" in arrs:
                comp_path = out_dir / f"comparison_{plot_subject}.png"
                try:
                    comp_meta = save_comparison_plot(
                        arrays=arrs,
                        subject_key=plot_subject,
                        out_path=comp_path,
                        b0_threshold=args.b0_threshold,
                        slice_idx=args.plot_slice_idx,
                        volume_idx=args.plot_volume_idx,
                    )
                    log.info("Saved comparison plot -> %s  (z=%d)",
                             comp_meta["out_path"], comp_meta["slice_idx"])
                except Exception as exc:
                    log.warning("Could not save comparison plot for %s: %s", plot_subject, exc)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  QSpaceUNet  (evaluation repeats)")
    print(f"{'=' * 72}")
    display_cols = [
        "subject", "repeat", "keep_fraction", "noise_level",
        "tensor_rmse", "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]
    display_cols = [c for c in display_cols if c in df_qspaceunet.columns]
    print(df_qspaceunet[display_cols].to_string(index=False))
    print(f"\n  MEAN  " + "  ".join(f"{c}={df_qspaceunet[c].mean():.4f}" for c in metric_cols))

    if len(comparison_rows) > 1:
        print(f"\n{'─' * 72}")
        print("  Method Comparison (mean over evaluation repeats)")
        print(f"{'─' * 72}")
        comp = {"QSpaceUNet": df_qspaceunet}
        if p2s_rows:
            comp["Patch2Self"] = pd.DataFrame(p2s_rows)
        if mppca_rows:
            comp["MP-PCA"] = pd.DataFrame(mppca_rows)
        header = f"{'metric':<16}"
        for name in comp:
            header += f"  {name:>12}"
        header += f"  {'best':>12}"
        print(header)

        higher_better = {"fa_r2", "adc_r2"}
        for metric in metric_cols:
            line = f"  {metric:<14}"
            vals = {}
            for name, df in comp.items():
                v = df[metric].mean() if metric in df.columns else float("nan")
                vals[name] = v
                line += f"  {v:>12.6f}"

            # Determine best
            valid = {k: v for k, v in vals.items() if np.isfinite(v)}
            if valid:
                if metric in higher_better:
                    best = max(valid, key=valid.get)
                else:
                    best = min(valid, key=valid.get)
                line += f"  {best:>12}"
            print(line)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate QSpaceUNet on test subjects")
    parser.add_argument("--checkpoint", default=cfg.EVAL_DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default=cfg.EVAL_OUT_DIR)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Biological subject IDs or Zarr keys to evaluate (default: test subjects from checkpoint)")
    parser.add_argument("--eval_all", action="store_true",
                        help="Evaluate all subjects in the zarr store")
    parser.add_argument("--b0_threshold", type=float, default=cfg.B0_THRESHOLD)
    parser.add_argument("--eval_repeats", "--eval-repeats", type=int, default=cfg.EVAL_REPEATS,
                        help="Number of independent degraded inputs to evaluate per subject")
    parser.add_argument("--eval_seed", "--eval-seed", type=int, default=cfg.EVAL_DEGRADE_SEED,
                        help="Base seed for repeat degradation sampling")
    parser.add_argument("--eval_keep_fraction_min", "--eval-keep-fraction-min", type=float,
                        default=cfg.EVAL_KEEP_FRACTION_MIN,
                        help="Minimum central k-space keep fraction sampled during evaluation")
    parser.add_argument("--eval_keep_fraction_max", "--eval-keep-fraction-max", type=float,
                        default=cfg.EVAL_KEEP_FRACTION_MAX,
                        help="Maximum central k-space keep fraction sampled during evaluation")
    parser.add_argument("--eval_noise_min", "--eval-noise-min", type=float,
                        default=cfg.EVAL_NOISE_MIN,
                        help="Minimum relative Gaussian noise level sampled during evaluation")
    parser.add_argument("--eval_noise_max", "--eval-noise-max", type=float,
                        default=cfg.EVAL_NOISE_MAX,
                        help="Maximum relative Gaussian noise level sampled during evaluation")
    parser.add_argument("--infer_batch_size", "--infer-batch-size", type=int, default=cfg.EVAL_INFER_BATCH_SIZE,
                        help="Number of axial slices evaluated per GPU forward pass")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=cfg.AMP,
                           help="Enable CUDA automatic mixed precision (default)")
    amp_group.add_argument("--no_amp", "--no-amp", dest="amp", action="store_false",
                           help="Disable CUDA automatic mixed precision")
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default=cfg.AMP_DTYPE,
                        help="AMP dtype; auto prefers bf16 on RTX 40-series")
    parser.add_argument("--bf16", dest="amp_dtype", action="store_const", const="bf16",
                        help="Shortcut for --amp --amp_dtype bf16")
    parser.add_argument("--fp16", dest="amp_dtype", action="store_const", const="fp16",
                        help="Shortcut for --amp --amp_dtype fp16")
    channels_group = parser.add_mutually_exclusive_group()
    channels_group.add_argument("--channels_last", "--channels-last",
                                dest="channels_last", action="store_true", default=cfg.CHANNELS_LAST,
                                help="Use channels-last convolution layout on CUDA (default)")
    channels_group.add_argument("--no_channels_last", "--no-channels-last",
                                dest="channels_last", action="store_false",
                                help="Disable channels-last memory format")
    parser.add_argument("--compile", choices=["off", "auto", "on"], default=cfg.COMPILE,
                        help="Use torch.compile; auto enables it on CUDA/Linux")
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"],
                        default=cfg.COMPILE_MODE, help="torch.compile mode")
    parser.add_argument("--deterministic", action="store_true",
                        help="Prefer deterministic CUDA kernels over fastest cuDNN autotuning")
    parser.add_argument("--require_cuda", "--require-cuda", action="store_true",
                        help="Fail fast when a CUDA PyTorch build/GPU is not available")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Disable saving plots")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip running Patch2Self and MP-PCA baselines")
    p2s_group = parser.add_mutually_exclusive_group()
    p2s_group.add_argument("--patch2self", "--run_patch2self", dest="patch2self",
                           action="store_true", default=True,
                           help="Enable the Patch2Self baseline (default)")
    p2s_group.add_argument("--no-patch2self", "--skip_patch2self", dest="patch2self",
                           action="store_false",
                           help="Disable the Patch2Self baseline")
    parser.add_argument("--p2s_model", "--p2s-model", choices=["ols", "ridge", "lasso"],
                        default=cfg.P2S_MODEL,
                        help="Patch2Self regression model")
    parser.add_argument("--p2s_alpha", "--p2s-alpha", type=float, default=cfg.P2S_ALPHA,
                        help="Patch2Self regularization alpha for ridge/lasso")
    parser.add_argument("--p2s_dti_fit_method", "--p2s-dti-fit-method",
                        choices=["WLS", "OLS", "NLLS"], default=cfg.DTI_FIT_METHOD,
                        help="DTI fit method used after Patch2Self denoising")
    p2s_shift_group = parser.add_mutually_exclusive_group()
    p2s_shift_group.add_argument("--p2s_shift_intensity", "--p2s-shift-intensity",
                                 dest="p2s_shift_intensity", action="store_true",
                                 default=cfg.P2S_SHIFT_INTENSITY,
                                 help="Enable Patch2Self intensity shifting")
    p2s_shift_group.add_argument("--p2s_no_shift_intensity", "--p2s-no-shift-intensity",
                                 dest="p2s_shift_intensity", action="store_false",
                                 help="Disable Patch2Self intensity shifting")
    p2s_clip_group = parser.add_mutually_exclusive_group()
    p2s_clip_group.add_argument("--p2s_clip_negative", "--p2s-clip-negative",
                                dest="p2s_clip_negative", action="store_true",
                                default=cfg.P2S_CLIP_NEGATIVE,
                                help="Enable Patch2Self negative-value clipping")
    p2s_clip_group.add_argument("--p2s_no_clip_negative", "--p2s-no-clip-negative",
                                dest="p2s_clip_negative", action="store_false",
                                help="Disable Patch2Self negative-value clipping")
    p2s_b0_group = parser.add_mutually_exclusive_group()
    p2s_b0_group.add_argument("--p2s_b0_denoising", "--p2s-b0-denoising",
                              dest="p2s_b0_denoising", action="store_true",
                              default=cfg.P2S_B0_DENOISING,
                              help="Enable Patch2Self b0 denoising")
    p2s_b0_group.add_argument("--p2s_no_b0_denoising", "--p2s-no-b0-denoising",
                              dest="p2s_b0_denoising", action="store_false",
                              help="Disable Patch2Self b0 denoising")
    parser.add_argument("--sweep_patch2self", "--sweep-patch2self",
                        action="store_true",
                        help="Run a validation Patch2Self hyperparameter sweep and exit")
    parser.add_argument("--p2s_sweep_metric", "--p2s-sweep-metric",
                        choices=["tensor_rmse", "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                                 "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"],
                        default="fa_rmse",
                        help="Metric used to rank Patch2Self sweep configs")
    parser.add_argument("--p2s_sweep_models", "--p2s-sweep-models",
                        nargs="+", choices=["ols", "ridge", "lasso"],
                        default=["ols", "ridge"],
                        help="Patch2Self models to include in the sweep")
    parser.add_argument("--p2s_sweep_alphas", "--p2s-sweep-alphas",
                        nargs="+", type=float, default=[0.01, 0.1, 1.0],
                        help="Alpha values to sweep for ridge/lasso")
    parser.add_argument("--p2s_sweep_b0_denoising", "--p2s-sweep-b0-denoising",
                        nargs="+", type=str.lower, choices=["true", "false"],
                        default=["false", "true"],
                        help="Patch2Self b0_denoising values to sweep")
    parser.add_argument("--p2s_sweep_clip_negative", "--p2s-sweep-clip-negative",
                        nargs="+", type=str.lower, choices=["true", "false"],
                        default=["true"],
                        help="Patch2Self clip_negative values to sweep")
    parser.add_argument("--p2s_sweep_shift_intensity", "--p2s-sweep-shift-intensity",
                        nargs="+", type=str.lower, choices=["true", "false"],
                        default=["true"],
                        help="Patch2Self shift_intensity values to sweep")
    mppca_group = parser.add_mutually_exclusive_group()
    mppca_group.add_argument("--mppca", "--mp-pca", "--run_mppca", "--run_mp_pca",
                             dest="mppca", action="store_true", default=True,
                             help="Enable the MP-PCA baseline (default)")
    mppca_group.add_argument("--no-mppca", "--no-mp-pca", "--skip_mppca", "--skip_mp_pca",
                             dest="mppca", action="store_false",
                             help="Disable the MP-PCA baseline")
    parser.add_argument("--plot_subject", default=None,
                        help="Subject key to visualize (default: all evaluated subjects)")
    parser.add_argument("--plot_repeat", "--plot-repeat", type=int, default=0,
                        help="Repeat index to visualize when eval_repeats > 1")
    parser.add_argument("--plot_slice_idx", type=int, default=None,
                        help="Axial slice index for visualization (default: auto)")
    parser.add_argument("--plot_volume_idx", type=int, default=None,
                        help="DWI volume index for visualization (default: auto)")

    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
