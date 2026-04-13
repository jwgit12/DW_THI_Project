"""Evaluate a trained QSpaceUNet on test subjects.

Produces a CSV with DTI-level metrics (tensor RMSE, FA, ADC) that is
directly comparable to the baseline CSVs in baselines/*/results/.

When --run_baselines is set (default), Patch2Self and MP-PCA are also
run on the same subjects, and comparison plots + metric tables are saved.

Usage:
    python -m research.evaluate --checkpoint research/runs/run_01/best_model.pt
    python -m research.evaluate --checkpoint research/runs/run_01/best_model.pt --subjects sub-10 sub-11
    python -m research.evaluate --checkpoint research/runs/run_01/best_model.pt --skip_baselines
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.utils import (
    dti6d_to_scalar_maps,
    fit_dti_to_6d,
    scalar_map_metrics,
    save_prediction_slice_plot,
    select_plot_indices,
    _robust_limits,
    _symmetric_limits,
)
from research.model import QSpaceUNet
import config as cfg

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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_subject(
    model: QSpaceUNet,
    zarr_path: str,
    subject_key: str,
    device: torch.device,
    b0_threshold: float = cfg.B0_THRESHOLD,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
) -> np.ndarray:
    """Run inference on a full 3D subject, slice by slice.

    Returns predicted DTI tensor (X, Y, Z, 6) as float32 numpy array.
    """
    store = zarr.open_group(zarr_path, mode="r")
    grp = store[subject_key]

    input_dwi = np.asarray(grp["input_dwi"][:], dtype=np.float32)  # (X, Y, Z, N)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)  # (3, N)

    X, Y, Z, N = input_dwi.shape
    max_n = model.max_n

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
    bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(device)
    bvecs_t = torch.from_numpy(bvecs).unsqueeze(0).to(device)
    vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(device)

    # Compute normalisation factor from mean b0
    b0_idx = bvals[:N] < b0_threshold
    if b0_idx.any():
        mean_b0_vol = input_dwi[:, :, :, :N][..., b0_idx].mean(axis=-1)  # (X, Y, Z)
    else:
        mean_b0_vol = input_dwi[:, :, :, :N].mean(axis=-1)

    pred_dti = np.zeros((X, Y, Z, 6), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for z in range(Z):
            # Extract and normalise slice
            signal = input_dwi[:, :, z, :].transpose(2, 0, 1).astype(np.float32)  # (max_n, H, W)

            b0_slice = mean_b0_vol[:, :, z]
            b0_norm = float(b0_slice[b0_slice > 0.1 * b0_slice.max()].mean()) if (b0_slice > 0).any() else 1.0
            if b0_norm > 0:
                signal = signal / b0_norm

            signal_t = torch.from_numpy(signal).unsqueeze(0).to(device)

            pred = model(signal_t, bvals_t, bvecs_t, vol_mask_t)  # (1, 6, H, W)
            pred_dti[:, :, z, :] = pred[0].permute(1, 2, 0).cpu().numpy()

    # Unscale from training range back to physical units
    pred_dti = pred_dti / dti_scale

    return pred_dti


# ─────────────────────────────────────────────────────────────────────────────
# Baseline runners
# ─────────────────────────────────────────────────────────────────────────────
def _run_patch2self(noisy: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    from dipy.denoise.patch2self import patch2self
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return patch2self(
            noisy, bvals, model=P2S_CFG["model"], alpha=P2S_CFG["alpha"],
            b0_threshold=P2S_CFG["b0_threshold"],
            shift_intensity=P2S_CFG["shift_intensity"],
            clip_negative_vals=P2S_CFG["clip_negative"],
            b0_denoising=P2S_CFG["b0_denoising"], verbose=False,
        ).astype(np.float32)


def _run_mppca(noisy: np.ndarray) -> np.ndarray:
    from dipy.denoise.localpca import mppca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mppca(
            noisy, patch_radius=MPPCA_CFG["patch_radius"],
            pca_method=MPPCA_CFG["pca_method"],
        ).astype(np.float32)


def _baseline_dti_metrics(
    denoised_dwi: np.ndarray,
    target_dti6d: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    b0_threshold: float,
    dti_fit_method: str = "WLS",
) -> tuple[dict, np.ndarray]:
    """Fit DTI from denoised DWI and compute metrics vs target.

    Returns (metrics_dict, fitted_dti6d).
    """
    bvecs_n3 = bvecs.T if bvecs.shape[0] == 3 else bvecs
    fitted_dti6d = fit_dti_to_6d(
        denoised_dwi, bvals, bvecs_n3=bvecs_n3,
        fit_method=dti_fit_method, b0_threshold=b0_threshold,
    )
    diff = fitted_dti6d - target_dti6d
    tensor_rmse = float(np.sqrt(np.mean(diff ** 2)))

    pred_fa, pred_adc = dti6d_to_scalar_maps(fitted_dti6d)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(target_dti6d)
    fa_m = scalar_map_metrics(tgt_fa, pred_fa)
    adc_m = scalar_map_metrics(tgt_adc, pred_adc)

    metrics = {
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
    return metrics, fitted_dti6d


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a single subject (all methods)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subject(
    model: QSpaceUNet,
    zarr_path: str,
    subject_key: str,
    device: torch.device,
    b0_threshold: float = cfg.B0_THRESHOLD,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
    run_baselines: bool = True,
) -> tuple[dict, dict]:
    """Full evaluation pipeline for one subject.

    Returns
    -------
    metrics : dict
        Keys: 'research', and optionally 'patch2self', 'mppca'.
        Each value is a dict of scalar metrics.
    arrays : dict
        Raw arrays needed for visualization.
    """
    t0 = time.time()

    store = zarr.open_group(zarr_path, mode="r")
    grp = store[subject_key]
    target_dti6d = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
    target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
    input_dwi = np.asarray(grp["input_dwi"][:], dtype=np.float32)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)

    # ── Research model ────────────────────────────────────────────────────
    pred_dti6d = predict_subject(model, zarr_path, subject_key, device, b0_threshold, dti_scale, max_bval)

    diff = pred_dti6d - target_dti6d
    tensor_rmse = float(np.sqrt(np.mean(diff ** 2)))
    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_dti6d)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(target_dti6d)
    fa_m = scalar_map_metrics(tgt_fa, pred_fa)
    adc_m = scalar_map_metrics(tgt_adc, pred_adc)

    research_elapsed = time.time() - t0
    research_metrics = {
        "subject": subject_key,
        "elapsed_s": round(research_elapsed, 2),
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

    all_metrics = {"research": research_metrics}
    arrays = {
        "input_dwi": input_dwi,
        "target_dwi": target_dwi,
        "target_dti6d": target_dti6d,
        "bvals": bvals,
        "bvecs": bvecs,
        "research_dti6d": pred_dti6d,
    }

    # ── Baselines ─────────────────────────────────────────────────────────
    if run_baselines:
        # Patch2Self
        t1 = time.time()
        p2s_denoised = _run_patch2self(input_dwi, bvals)
        p2s_metrics, p2s_dti6d = _baseline_dti_metrics(
            p2s_denoised, target_dti6d, bvals, bvecs, b0_threshold,
        )
        p2s_metrics["subject"] = subject_key
        p2s_metrics["elapsed_s"] = round(time.time() - t1, 2)
        all_metrics["patch2self"] = p2s_metrics
        arrays["patch2self_dti6d"] = p2s_dti6d

        # MP-PCA
        t2 = time.time()
        mppca_denoised = _run_mppca(input_dwi)
        mppca_metrics, mppca_dti6d = _baseline_dti_metrics(
            mppca_denoised, target_dti6d, bvals, bvecs, b0_threshold,
        )
        mppca_metrics["subject"] = subject_key
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
    methods.append(("QSpaceUNet", arrays["research_dti6d"]))
    methods.append(("Target", target_dti6d))

    # Derive FA and ADC for all methods
    fa_maps = []
    adc_maps = []
    for label, dti6d in methods:
        fa, adc = dti6d_to_scalar_maps(dti6d)
        fa_maps.append((label, fa[:, :, slice_idx]))
        adc_maps.append((label, adc[:, :, slice_idx]))

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

    # Diff column: show QSpaceUNet diff (the research method)
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

    fig.suptitle(f"Method Comparison  |  {subject_key}  |  z={slice_idx}", fontsize=13)
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
    colors = {"research": "#2196F3", "patch2self": "#FF9800", "mppca": "#4CAF50"}
    bar_colors = [colors.get(m, "#9E9E9E") for m in method_names]

    for i, metric in enumerate(metric_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = [means_df.loc[m, metric] if metric in means_df.columns else 0 for m in method_names]
        bars = ax.bar(method_names, vals, color=bar_colors, edgecolor="black", linewidth=0.5)

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

    fig.suptitle("Metric Comparison (mean over test subjects)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plot_path = out_dir / "comparison_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved metric comparison plot -> %s", plot_path)

    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    log.info("Device: %s", device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    max_n = ckpt["max_n"]
    feat_dim = ckpt.get("feat_dim", 64)
    channels = tuple(ckpt.get("channels", [64, 128, 256, 512]))
    cholesky = ckpt.get("cholesky", False)
    dti_scale = ckpt.get("dti_scale", 1.0)
    max_bval = ckpt.get("max_bval", 1000.0)
    log.info("DTI scale factor: %.4f, max_bval: %.1f, cholesky: %s", dti_scale, max_bval, cholesky)

    model = QSpaceUNet(max_n=max_n, feat_dim=feat_dim, channels=channels, cholesky=cholesky).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint from epoch %d (val_loss=%.6f)", ckpt["epoch"], ckpt["val_loss"])

    # Determine subjects to evaluate
    store = zarr.open_group(args.zarr_path, mode="r")
    all_keys = sorted(store.keys())

    if args.eval_all:
        subjects = all_keys
    elif args.subjects:
        # Expand biological subject IDs to matching Zarr keys
        subjects = []
        for s in args.subjects:
            if s in all_keys:
                subjects.append(s)
            else:
                matches = [k for k in all_keys if k.rsplit("_ses-", 1)[0] == s]
                subjects.extend(matches)
    else:
        subjects = ckpt.get("test_subjects", [])
        if not subjects:
            subjects = all_keys

    run_baselines = not args.skip_baselines
    log.info("Evaluating %d subjects  (baselines=%s)", len(subjects), run_baselines)

    # Run evaluation
    research_rows = []
    p2s_rows = []
    mppca_rows = []
    plot_arrays = {}

    for subj in subjects:
        try:
            all_metrics, arrays = evaluate_subject(
                model, args.zarr_path, subj, device,
                b0_threshold=args.b0_threshold,
                dti_scale=dti_scale,
                max_bval=max_bval,
                run_baselines=run_baselines,
            )
            rm = all_metrics["research"]
            log.info(
                "%-14s  tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  ADC[rmse=%.2e r2=%.3f]  (%.1fs)",
                rm["subject"], rm["tensor_rmse"],
                rm["fa_rmse"], rm["fa_r2"],
                rm["adc_rmse"], rm["adc_r2"],
                rm["elapsed_s"],
            )
            research_rows.append(rm)
            if "patch2self" in all_metrics:
                p2s_rows.append(all_metrics["patch2self"])
                log.info("  Patch2Self    tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  (%.1fs)",
                         all_metrics["patch2self"]["tensor_rmse"],
                         all_metrics["patch2self"]["fa_rmse"],
                         all_metrics["patch2self"]["fa_r2"],
                         all_metrics["patch2self"]["elapsed_s"])
            if "mppca" in all_metrics:
                mppca_rows.append(all_metrics["mppca"])
                log.info("  MP-PCA       tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  (%.1fs)",
                         all_metrics["mppca"]["tensor_rmse"],
                         all_metrics["mppca"]["fa_rmse"],
                         all_metrics["mppca"]["fa_r2"],
                         all_metrics["mppca"]["elapsed_s"])

            plot_arrays[subj] = arrays
        except Exception as exc:
            log.warning("FAIL  %s  —  %s", subj, exc)

    if not research_rows:
        log.error("No subjects evaluated successfully.")
        return

    # ── Save CSVs ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]

    def _save_method_csv(rows: list[dict], name: str) -> pd.DataFrame:
        df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
        cols = [c for c in metric_cols if c in df.columns]
        summary = df[cols].agg(["mean", "std"]).round(6).reset_index()
        summary.columns = ["subject"] + cols
        summary["subject"] = summary["subject"].str.upper()
        df_out = pd.concat([df, summary], ignore_index=True)
        path = out_dir / f"metrics_{name}.csv"
        df_out.to_csv(path, index=False)
        log.info("Saved -> %s", path)
        return df

    df_research = _save_method_csv(research_rows, "research")
    if p2s_rows:
        _save_method_csv(p2s_rows, "patch2self")
    if mppca_rows:
        _save_method_csv(mppca_rows, "mppca")

    # Also save the research-only CSV for backward compatibility
    df_compat = pd.DataFrame(research_rows).sort_values("subject").reset_index(drop=True)
    cols = [c for c in metric_cols if c in df_compat.columns]
    summary = df_compat[cols].agg(["mean", "std"]).round(6).reset_index()
    summary.columns = ["subject"] + cols
    summary["subject"] = summary["subject"].str.upper()
    pd.concat([df_compat, summary], ignore_index=True).to_csv(
        out_dir / "metrics_per_subject.csv", index=False
    )

    # ── Metric comparison ─────────────────────────────────────────────────
    if run_baselines and p2s_rows and mppca_rows:
        comparison_rows = {"research": research_rows, "patch2self": p2s_rows, "mppca": mppca_rows}
        save_metric_comparison(comparison_rows, out_dir)

    # ── Visualization ─────────────────────────────────────────────────────
    if research_rows and not args.skip_plot and plot_arrays:
        for plot_subject, arrs in plot_arrays.items():
            # Original prediction plot
            plot_path = out_dir / f"prediction_example_{plot_subject}.png"
            try:
                plot_meta = save_prediction_slice_plot(
                    input_dwi=arrs["input_dwi"],
                    pred_dti6d=arrs["research_dti6d"],
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
            if run_baselines and "patch2self_dti6d" in arrs:
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
    print(f"  QSpaceUNet  (test subjects)")
    print(f"{'=' * 72}")
    print(df_research[["subject", "tensor_rmse",
              "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
              "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"]].to_string(index=False))
    print(f"\n  MEAN  " + "  ".join(f"{c}={df_research[c].mean():.4f}" for c in metric_cols))

    if run_baselines and p2s_rows and mppca_rows:
        print(f"\n{'─' * 72}")
        print("  Method Comparison (mean over test subjects)")
        print(f"{'─' * 72}")
        comp = {"QSpaceUNet": df_research, "Patch2Self": pd.DataFrame(p2s_rows), "MP-PCA": pd.DataFrame(mppca_rows)}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QSpaceUNet on test subjects")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--zarr_path", default="dataset/default_dataset.zarr")
    parser.add_argument("--out_dir", default="research/results")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Biological subject IDs or Zarr keys to evaluate (default: test subjects from checkpoint)")
    parser.add_argument("--eval_all", action="store_true",
                        help="Evaluate all subjects in the zarr store")
    parser.add_argument("--b0_threshold", type=float, default=cfg.B0_THRESHOLD)
    parser.add_argument("--skip_plot", action="store_true",
                        help="Disable saving plots")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip running Patch2Self and MP-PCA baselines")
    parser.add_argument("--plot_subject", default=None,
                        help="Subject key to visualize (default: first evaluated subject)")
    parser.add_argument("--plot_slice_idx", type=int, default=None,
                        help="Axial slice index for visualization (default: auto)")
    parser.add_argument("--plot_volume_idx", type=int, default=None,
                        help="DWI volume index for visualization (default: auto)")

    main(parser.parse_args())
