"""
Comprehensive baseline visualization pipeline for DW-MRI denoising experiments.

This script can:
1) Run baseline denoisers (MPPCA + trainable Patch2Self) on one or more subjects.
2) Compute comparable metrics against target DWI.
3) Generate a broad suite of qualitative and quantitative plots.
4) Export machine-readable metrics and a markdown summary report.

Example
-------
python baselines/visualize_baselines.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --max_subjects 2 \
  --max_dirs 64 \
  --methods mppca patch2self
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.mppca.mppca_torch import get_best_device, mppca_denoise_chunked
from baselines.patch2self.patch2self_trainable import (
    Patch2SelfConfig,
    denoise_with_model,
    fit_patch2self,
)


@dataclass
class MethodResult:
    name: str
    prediction: np.ndarray
    runtime_total_s: float
    runtime_fit_s: float
    runtime_predict_s: float
    global_metrics: dict[str, float]
    per_direction: dict[str, np.ndarray]
    per_slice: dict[str, np.ndarray]


def _sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return get_best_device()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device option: {device_name}")


def _bound(start: int, end: int, size: int) -> tuple[int, int]:
    s = max(0, start)
    e = size if end <= 0 else min(end, size)
    if e <= s:
        raise ValueError(f"Invalid crop bounds: start={start}, end={end}, size={size}")
    return s, e


def _resolve_subjects(root: zarr.Group, subject: str | None, max_subjects: int) -> list[str]:
    subjects = sorted(root.group_keys())
    if subject is not None:
        if subject not in subjects:
            raise KeyError(f"Subject '{subject}' not found in dataset.")
        subjects = [subject]
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    return subjects


def _compute_global_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    data_range = float(target.max() - target.min())
    if data_range <= 1e-12:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 1e-12:
        data_range = 1.0
    psnr = 10.0 * math.log10((data_range**2) / max(mse, 1e-12))
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def _compute_axis_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    axes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    err = pred - target
    mse = np.mean(err**2, axis=axes)
    mae = np.mean(np.abs(err), axis=axes)

    data_range = np.max(target, axis=axes) - np.min(target, axis=axes)
    fallback = np.max(np.abs(target), axis=axes)
    data_range = np.where(data_range <= 1e-12, fallback, data_range)
    data_range = np.where(data_range <= 1e-12, 1.0, data_range)
    psnr_db = 10.0 * np.log10((data_range**2) / np.maximum(mse, 1e-12))

    return {"mse": mse.astype(np.float64), "mae": mae.astype(np.float64), "psnr_db": psnr_db.astype(np.float64)}


def _evaluate_methods(
    target_dwi: np.ndarray,
    predictions: dict[str, np.ndarray],
    runtimes: dict[str, dict[str, float]],
) -> dict[str, MethodResult]:
    out: dict[str, MethodResult] = {}
    for method, pred in predictions.items():
        global_metrics = _compute_global_metrics(pred, target_dwi)
        per_direction = _compute_axis_metrics(pred, target_dwi, axes=(0, 1, 2))
        per_slice = _compute_axis_metrics(pred, target_dwi, axes=(0, 1, 3))

        rt = runtimes.get(method, {})
        out[method] = MethodResult(
            name=method,
            prediction=pred,
            runtime_total_s=float(rt.get("total", 0.0)),
            runtime_fit_s=float(rt.get("fit", 0.0)),
            runtime_predict_s=float(rt.get("predict", 0.0)),
            global_metrics=global_metrics,
            per_direction=per_direction,
            per_slice=per_slice,
        )
    return out


def _sample_flat(values: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= max_points:
        return flat.astype(np.float32, copy=False)
    idx = rng.choice(flat.size, size=max_points, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def _pick_focus_indices(target: np.ndarray, noisy: np.ndarray) -> tuple[int, int]:
    noisy_abs = np.abs(noisy - target)
    z_idx = int(np.argmax(noisy_abs.mean(axis=(0, 1, 3))))
    d_idx = int(np.argmax(noisy_abs.mean(axis=(0, 1, 2))))
    return z_idx, d_idx


def _get_method_colors(methods: list[str]) -> dict[str, str]:
    palette = {
        "noisy": "#6c757d",
        "mppca": "#0077b6",
        "patch2self": "#2a9d8f",
    }
    out = {}
    for i, m in enumerate(methods):
        out[m] = palette.get(m, plt.cm.tab10(i % 10))
    return out


def _plot_subject_qualitative(
    subject: str,
    method_results: dict[str, MethodResult],
    target_dwi: np.ndarray,
    z_idx: int,
    d_idx: int,
    out_dir: Path,
) -> None:
    methods = list(method_results.keys())
    ncols = len(methods) + 1

    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.8), dpi=180)
    axes = np.asarray(axes)

    target_slice = np.rot90(target_dwi[:, :, z_idx, d_idx], 1)
    vmin = float(np.percentile(target_slice, 1))
    vmax = float(np.percentile(target_slice, 99))

    abs_errors = []
    for method in methods:
        err = np.abs(method_results[method].prediction[:, :, z_idx, d_idx] - target_dwi[:, :, z_idx, d_idx])
        abs_errors.append(err)
    err_vmax = float(np.percentile(np.stack(abs_errors, axis=0), 99))
    err_vmax = max(err_vmax, 1e-8)

    im = axes[0, 0].imshow(target_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Target DWI")
    axes[0, 0].axis("off")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Error maps\n(reference column)", ha="center", va="center")

    for col, method in enumerate(methods, start=1):
        pred_slice = np.rot90(method_results[method].prediction[:, :, z_idx, d_idx], 1)
        abs_err = np.rot90(
            np.abs(method_results[method].prediction[:, :, z_idx, d_idx] - target_dwi[:, :, z_idx, d_idx]),
            1,
        )

        top = axes[0, col].imshow(pred_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(method.upper())
        axes[0, col].axis("off")
        fig.colorbar(top, ax=axes[0, col], fraction=0.046, pad=0.02)

        bottom = axes[1, col].imshow(abs_err, cmap="magma", vmin=0.0, vmax=err_vmax)
        axes[1, col].set_title(f"|{method.upper()} - TARGET|")
        axes[1, col].axis("off")
        fig.colorbar(bottom, ax=axes[1, col], fraction=0.046, pad=0.02)

    fig.suptitle(f"{subject} | qualitative view | z={z_idx}, dir={d_idx}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{subject}_qualitative.png")
    plt.close(fig)


def _plot_subject_histograms(
    subject: str,
    method_results: dict[str, MethodResult],
    target_dwi: np.ndarray,
    out_dir: Path,
    rng: np.random.Generator,
) -> None:
    methods = list(method_results.keys())
    colors = _get_method_colors(methods)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    ax1, ax2, ax3, ax4 = axes.flat

    target_sample = _sample_flat(target_dwi, 200_000, rng)
    ax1.hist(target_sample, bins=80, alpha=0.5, density=True, label="target", color="#444444")

    for method in methods:
        sample = _sample_flat(method_results[method].prediction, 200_000, rng)
        ax1.hist(sample, bins=80, alpha=0.35, density=True, label=method, color=colors[method])
    ax1.set_title("Voxel Value Distribution")
    ax1.set_xlabel("intensity")
    ax1.set_ylabel("density")
    ax1.legend()

    for method in methods:
        residual = method_results[method].prediction - target_dwi
        sample = _sample_flat(residual, 200_000, rng)
        ax2.hist(sample, bins=80, alpha=0.35, density=True, label=method, color=colors[method])
    ax2.set_title("Residual Distribution (prediction - target)")
    ax2.set_xlabel("residual")
    ax2.set_ylabel("density")
    ax2.legend()

    for method in methods:
        abs_err = np.abs(method_results[method].prediction - target_dwi)
        sample = np.sort(_sample_flat(abs_err, 200_000, rng))
        cdf = np.linspace(0.0, 1.0, num=sample.size, dtype=np.float64)
        ax3.plot(sample, cdf, label=method, color=colors[method], lw=2)
    ax3.set_title("Absolute Error CDF")
    ax3.set_xlabel("absolute error")
    ax3.set_ylabel("CDF")
    ax3.grid(alpha=0.25)
    ax3.legend()

    box_data = []
    labels = []
    for method in methods:
        abs_err = np.abs(method_results[method].prediction - target_dwi)
        sample = _sample_flat(abs_err, 80_000, rng)
        box_data.append(sample)
        labels.append(method)
    ax4.boxplot(box_data, tick_labels=labels, showfliers=False)
    ax4.set_title("Absolute Error Boxplot")
    ax4.set_ylabel("absolute error")
    ax4.grid(alpha=0.25)

    fig.suptitle(f"{subject} | distribution diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{subject}_distributions.png")
    plt.close(fig)


def _plot_subject_direction_curves(
    subject: str,
    method_results: dict[str, MethodResult],
    bvals: np.ndarray,
    out_dir: Path,
) -> None:
    methods = list(method_results.keys())
    colors = _get_method_colors(methods)
    x = np.arange(bvals.shape[0], dtype=np.int32)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=180, sharex=True)

    for method in methods:
        axes[0].plot(x, method_results[method].per_direction["psnr_db"], lw=2, label=method, color=colors[method])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Per-direction PSNR")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axb = axes[0].twinx()
    axb.scatter(x, bvals, s=10, color="#888888", alpha=0.35, label="bvals")
    axb.set_ylabel("b-value")

    for method in methods:
        axes[1].plot(x, method_results[method].per_direction["mae"], lw=2, label=method, color=colors[method])
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Per-direction MAE")
    axes[1].grid(alpha=0.25)

    for method in methods:
        axes[2].plot(x, method_results[method].per_direction["mse"], lw=2, label=method, color=colors[method])
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel("direction index")
    axes[2].set_title("Per-direction MSE")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"{subject} | per-direction metrics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{subject}_direction_curves.png")
    plt.close(fig)


def _plot_subject_slice_curves(
    subject: str,
    method_results: dict[str, MethodResult],
    out_dir: Path,
) -> None:
    methods = list(method_results.keys())
    colors = _get_method_colors(methods)

    z_count = next(iter(method_results.values())).per_slice["psnr_db"].shape[0]
    z_axis = np.arange(z_count, dtype=np.int32)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=180, sharex=True)

    for method in methods:
        axes[0].plot(z_axis, method_results[method].per_slice["psnr_db"], lw=2, label=method, color=colors[method])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Per-slice PSNR")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for method in methods:
        axes[1].plot(z_axis, method_results[method].per_slice["mae"], lw=2, label=method, color=colors[method])
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("axial slice index (z)")
    axes[1].set_title("Per-slice MAE")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{subject} | per-slice metrics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{subject}_slice_curves.png")
    plt.close(fig)


def _plot_subject_bval_scatter(
    subject: str,
    method_results: dict[str, MethodResult],
    bvals: np.ndarray,
    out_dir: Path,
) -> None:
    methods = list(method_results.keys())
    colors = _get_method_colors(methods)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    ax1, ax2 = axes

    for method in methods:
        ax1.scatter(
            bvals,
            method_results[method].per_direction["mae"],
            s=20,
            alpha=0.6,
            color=colors[method],
            label=method,
        )
    ax1.set_title("b-value vs MAE")
    ax1.set_xlabel("b-value")
    ax1.set_ylabel("MAE")
    ax1.grid(alpha=0.25)
    ax1.legend()

    for method in methods:
        ax2.scatter(
            bvals,
            method_results[method].per_direction["psnr_db"],
            s=20,
            alpha=0.6,
            color=colors[method],
            label=method,
        )
    ax2.set_title("b-value vs PSNR")
    ax2.set_xlabel("b-value")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.suptitle(f"{subject} | b-value relationship", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{subject}_bval_scatter.png")
    plt.close(fig)


def _plot_subject_runtime_table(
    subject: str,
    method_results: dict[str, MethodResult],
    out_dir: Path,
) -> None:
    methods = list(method_results.keys())
    colors = _get_method_colors(methods)
    runtimes = [method_results[m].runtime_total_s for m in methods]

    fig = plt.figure(figsize=(13, 6), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    ax_bar.bar(methods, runtimes, color=[colors[m] for m in methods], alpha=0.9)
    ax_bar.set_ylabel("seconds")
    ax_bar.set_title("Runtime by method")
    ax_bar.grid(alpha=0.25, axis="y")

    rows = []
    for m in methods:
        g = method_results[m].global_metrics
        rows.append([
            m,
            f"{g['mse']:.6f}",
            f"{g['mae']:.6f}",
            f"{g['psnr_db']:.2f}",
            f"{method_results[m].runtime_total_s:.2f}",
        ])

    table = ax_tbl.table(
        cellText=rows,
        colLabels=["method", "mse", "mae", "psnr_db", "runtime_s"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    ax_tbl.set_title("Global metric summary")
    ax_tbl.axis("off")

    fig.suptitle(f"{subject} | runtime + summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{subject}_runtime_summary.png")
    plt.close(fig)


def _plot_summary_metric_boxplots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return

    methods = sorted({str(r["method"]) for r in rows})
    colors = _get_method_colors(methods)
    metrics = [("mse", "MSE"), ("mae", "MAE"), ("psnr_db", "PSNR (dB)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
    for ax, (key, title) in zip(axes, metrics):
        data = []
        for m in methods:
            vals = [float(r[key]) for r in rows if r["method"] == m]
            data.append(vals)
        bp = ax.boxplot(data, tick_labels=methods, showfliers=False, patch_artist=True)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Cross-subject metric distribution", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "summary_metric_boxplots.png")
    plt.close(fig)


def _plot_summary_runtime(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    methods = sorted({str(r["method"]) for r in rows})
    colors = _get_method_colors(methods)

    means = []
    stds = []
    for m in methods:
        vals = np.array([float(r["runtime_s"]) for r in rows if r["method"] == m], dtype=np.float64)
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], capsize=5, alpha=0.9)
    ax.set_title("Cross-subject runtime (mean ± std)")
    ax.set_ylabel("seconds")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_runtime.png")
    plt.close(fig)


def _plot_summary_psnr_improvement(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return

    subjects = sorted({str(r["subject"]) for r in rows})
    methods = sorted({str(r["method"]) for r in rows})
    if "noisy" not in methods:
        return

    compare_methods = [m for m in methods if m != "noisy"]
    if not compare_methods:
        return

    baseline_psnr = {}
    for r in rows:
        if r["method"] == "noisy":
            baseline_psnr[str(r["subject"])] = float(r["psnr_db"])

    x = np.arange(len(subjects), dtype=np.float64)
    width = 0.8 / max(1, len(compare_methods))
    colors = _get_method_colors(compare_methods)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.6), 5), dpi=180)
    for i, method in enumerate(compare_methods):
        improvements = []
        for subject in subjects:
            row = next((r for r in rows if r["subject"] == subject and r["method"] == method), None)
            if row is None:
                improvements.append(np.nan)
                continue
            improvements.append(float(row["psnr_db"]) - baseline_psnr.get(subject, 0.0))
        offset = (i - (len(compare_methods) - 1) / 2.0) * width
        ax.bar(x + offset, improvements, width=width, label=method, color=colors[method], alpha=0.9)

    ax.axhline(0.0, color="#444444", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("PSNR improvement vs noisy (dB)")
    ax.set_title("Per-subject PSNR gain")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "summary_psnr_improvement.png")
    plt.close(fig)


def _plot_summary_method_heatmap(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    methods = sorted({str(r["method"]) for r in rows})
    metrics = [("mse", False), ("mae", False), ("psnr_db", True), ("runtime_s", False)]

    mean_values = np.zeros((len(methods), len(metrics)), dtype=np.float64)
    for i, method in enumerate(methods):
        method_rows = [r for r in rows if r["method"] == method]
        for j, (metric, _) in enumerate(metrics):
            vals = np.array([float(r[metric]) for r in method_rows], dtype=np.float64)
            mean_values[i, j] = float(vals.mean())

    score = np.zeros_like(mean_values)
    for j, (_, higher_is_better) in enumerate(metrics):
        col = mean_values[:, j]
        lo = float(np.min(col))
        hi = float(np.max(col))
        span = max(hi - lo, 1e-12)
        if higher_is_better:
            score[:, j] = (col - lo) / span
        else:
            score[:, j] = (hi - col) / span

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    im = ax.imshow(score, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Method score heatmap (0=worst, 1=best)")

    for i in range(len(methods)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mean_values[i, j]:.4g}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized score")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_method_heatmap.png")
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("# Baseline Visualization Report\n\nNo rows available.\n", encoding="utf-8")
        return

    methods = sorted({str(r["method"]) for r in rows})
    subjects = sorted({str(r["subject"]) for r in rows})

    lines: list[str] = []
    lines.append("# Baseline Visualization Report")
    lines.append("")
    lines.append(f"- Subjects: {len(subjects)}")
    lines.append(f"- Methods: {', '.join(methods)}")
    lines.append("")
    lines.append("## Mean Metrics Per Method")
    lines.append("")
    lines.append("| method | mse | mae | psnr_db | runtime_s |")
    lines.append("|---|---:|---:|---:|---:|")

    psnr_means: dict[str, float] = {}
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        mse = float(np.mean([float(v["mse"]) for v in vals]))
        mae = float(np.mean([float(v["mae"]) for v in vals]))
        psnr = float(np.mean([float(v["psnr_db"]) for v in vals]))
        rt = float(np.mean([float(v["runtime_s"]) for v in vals]))
        psnr_means[method] = psnr
        lines.append(f"| {method} | {mse:.6f} | {mae:.6f} | {psnr:.3f} | {rt:.3f} |")

    best_method = max(psnr_means, key=psnr_means.get)
    lines.append("")
    lines.append(f"- Best mean PSNR method: `{best_method}` ({psnr_means[best_method]:.3f} dB)")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- Subject-level figures are stored in `subjects/`.")
    lines.append("- Cross-subject summary figures are stored in `summary/`.")
    lines.append("- Full per-subject rows are stored in `metrics.csv`.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _run_mppca(
    input_dwi: np.ndarray,
    *,
    device: torch.device,
    patch_radius: int,
    chunk_size: int,
    progress: bool,
) -> tuple[np.ndarray, float]:
    vol = torch.from_numpy(np.asarray(input_dwi, dtype=np.float32)).to(device)
    _sync_torch_device(device)
    t0 = time.perf_counter()
    denoised_t, _ = mppca_denoise_chunked(
        vol,
        patch_radius=patch_radius,
        chunk_size=chunk_size,
        verbose=False,
        progress=progress,
    )
    _sync_torch_device(device)
    runtime = time.perf_counter() - t0
    denoised = denoised_t.detach().cpu().numpy().astype(np.float32, copy=False)
    return denoised, runtime


def _run_patch2self(
    input_dwi: np.ndarray,
    bvals: np.ndarray,
    *,
    model: str,
    alpha: float,
    b0_threshold: float,
    no_b0_denoising: bool,
    sketch_fraction: float,
    random_state: int,
    sketch_chunk_size: int,
    predict_chunk_size: int,
    clip_negative_vals: bool,
    no_shift_intensity: bool,
) -> tuple[np.ndarray, float, float]:
    cfg = Patch2SelfConfig(
        model=model,
        alpha=alpha,
        b0_threshold=b0_threshold,
        b0_denoising=not no_b0_denoising,
        sketch_fraction=sketch_fraction,
        random_state=random_state,
        sketch_chunk_size=sketch_chunk_size,
        predict_chunk_size=predict_chunk_size,
        dtype=np.float32,
    )

    t0 = time.perf_counter()
    fitted = fit_patch2self(input_dwi, bvals, cfg=cfg, verbose=False)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    denoised = denoise_with_model(
        input_dwi,
        fitted,
        clip_negative_vals=clip_negative_vals,
        shift_intensity=not no_shift_intensity,
    )
    predict_s = time.perf_counter() - t1
    return np.asarray(denoised, dtype=np.float32), fit_s, predict_s


def _save_subject_npz(
    subject: str,
    out_dir: Path,
    method_results: dict[str, MethodResult],
    target_dwi: np.ndarray,
    z_idx: int,
    d_idx: int,
) -> None:
    data: dict[str, Any] = {
        "target_slice": target_dwi[:, :, z_idx, d_idx].astype(np.float32),
        "z_idx": np.array(z_idx, dtype=np.int32),
        "dir_idx": np.array(d_idx, dtype=np.int32),
    }
    for method, result in method_results.items():
        data[f"{method}_slice"] = result.prediction[:, :, z_idx, d_idx].astype(np.float32)
        data[f"{method}_residual_slice"] = (
            (result.prediction[:, :, z_idx, d_idx] - target_dwi[:, :, z_idx, d_idx]).astype(np.float32)
        )
        data[f"{method}_per_direction_psnr"] = result.per_direction["psnr_db"].astype(np.float32)
        data[f"{method}_per_slice_psnr"] = result.per_slice["psnr_db"].astype(np.float32)
    np.savez_compressed(out_dir / f"{subject}_snapshot.npz", **data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline denoising + rich visualization suite.")
    parser.add_argument("--zarr_path", type=str, default="dataset/pretext_dataset.zarr")
    parser.add_argument("--subject", type=str, default=None, help="Single subject, e.g. subject_000")
    parser.add_argument("--max_subjects", type=int, default=0, help="0 = all subjects")
    parser.add_argument("--methods", nargs="+", choices=["mppca", "patch2self"], default=["mppca", "patch2self"])

    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--x1", type=int, default=0)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--y1", type=int, default=0)
    parser.add_argument("--z0", type=int, default=0)
    parser.add_argument("--z1", type=int, default=0)
    parser.add_argument("--max_dirs", type=int, default=64, help="0 = all directions")

    parser.add_argument("--mppca_device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mppca_patch_radius", type=int, default=2)
    parser.add_argument("--mppca_chunk_size", type=int, default=10)
    parser.add_argument("--mppca_progress", action="store_true")

    parser.add_argument("--p2s_model", choices=["ols", "ridge", "lasso"], default="ols")
    parser.add_argument("--p2s_alpha", type=float, default=1.0)
    parser.add_argument("--p2s_b0_threshold", type=float, default=50.0)
    parser.add_argument("--p2s_no_b0_denoising", action="store_true")
    parser.add_argument("--p2s_sketch_fraction", type=float, default=0.30)
    parser.add_argument("--p2s_random_state", type=int, default=42)
    parser.add_argument("--p2s_sketch_chunk_size", type=int, default=200_000)
    parser.add_argument("--p2s_predict_chunk_size", type=int, default=200_000)
    parser.add_argument("--p2s_clip_negative_vals", action="store_true")
    parser.add_argument("--p2s_no_shift_intensity", action="store_true")

    parser.add_argument("--output_dir", type=str, default="", help="Default: baselines/visualizations/run_<timestamp>")
    parser.add_argument("--save_npz", action="store_true", help="Save per-subject snapshot arrays for quick reuse.")
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else Path("baselines/visualizations") / f"run_{timestamp}"
    subjects_dir = out_root / "subjects"
    summary_dir = out_root / "summary"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(args.zarr_path, mode="r")
    subjects = _resolve_subjects(root, args.subject, args.max_subjects)
    if not subjects:
        raise RuntimeError("No subjects selected.")

    methods = list(dict.fromkeys(args.methods))
    if not methods:
        raise RuntimeError("No methods selected.")

    mppca_device = None
    if "mppca" in methods:
        mppca_device = _resolve_torch_device(args.mppca_device)
        print(f"MPPCA device: {mppca_device}")

    print(f"Output directory: {out_root}")
    print(f"Subjects to process: {len(subjects)}")
    print(f"Methods: noisy + {', '.join(methods)}")

    rows: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()

    for i, subject in enumerate(subjects, start=1):
        grp = root[subject]
        shape = grp["input_dwi"].shape
        x0, x1 = _bound(args.x0, args.x1, shape[0])
        y0, y1 = _bound(args.y0, args.y1, shape[1])
        z0, z1 = _bound(args.z0, args.z1, shape[2])
        n_dirs_total = shape[3]
        n_dirs = n_dirs_total if args.max_dirs <= 0 else min(args.max_dirs, n_dirs_total)

        print(
            f"\n[{i}/{len(subjects)}] {subject} | "
            f"block=({x0}:{x1}, {y0}:{y1}, {z0}:{z1}) | dirs={n_dirs}"
        )

        input_dwi = np.asarray(grp["input_dwi"][x0:x1, y0:y1, z0:z1, :n_dirs], dtype=np.float32)
        target_dwi = np.asarray(grp["target_dwi"][x0:x1, y0:y1, z0:z1, :n_dirs], dtype=np.float32)
        bvals = np.asarray(grp["bvals"][:n_dirs], dtype=np.float32)

        predictions: dict[str, np.ndarray] = {"noisy": input_dwi}
        runtimes: dict[str, dict[str, float]] = {
            "noisy": {"total": 0.0, "fit": 0.0, "predict": 0.0}
        }

        if "mppca" in methods:
            print("  running MPPCA...")
            den_mppca, rt_mppca = _run_mppca(
                input_dwi,
                device=mppca_device,
                patch_radius=args.mppca_patch_radius,
                chunk_size=args.mppca_chunk_size,
                progress=args.mppca_progress,
            )
            predictions["mppca"] = den_mppca
            runtimes["mppca"] = {"total": rt_mppca, "fit": rt_mppca, "predict": 0.0}
            print(f"    done in {rt_mppca:.2f}s")

        if "patch2self" in methods:
            print("  running Patch2Self...")
            den_p2s, fit_s, pred_s = _run_patch2self(
                input_dwi,
                bvals,
                model=args.p2s_model,
                alpha=args.p2s_alpha,
                b0_threshold=args.p2s_b0_threshold,
                no_b0_denoising=args.p2s_no_b0_denoising,
                sketch_fraction=args.p2s_sketch_fraction,
                random_state=args.p2s_random_state,
                sketch_chunk_size=args.p2s_sketch_chunk_size,
                predict_chunk_size=args.p2s_predict_chunk_size,
                clip_negative_vals=args.p2s_clip_negative_vals,
                no_shift_intensity=args.p2s_no_shift_intensity,
            )
            predictions["patch2self"] = den_p2s
            runtimes["patch2self"] = {"total": fit_s + pred_s, "fit": fit_s, "predict": pred_s}
            print(f"    fit {fit_s:.2f}s, predict {pred_s:.2f}s")

        method_results = _evaluate_methods(target_dwi, predictions, runtimes)

        z_idx, d_idx = _pick_focus_indices(target_dwi, predictions["noisy"])
        subject_out = subjects_dir / subject
        subject_out.mkdir(parents=True, exist_ok=True)

        _plot_subject_qualitative(subject, method_results, target_dwi, z_idx, d_idx, subject_out)
        _plot_subject_histograms(subject, method_results, target_dwi, subject_out, rng)
        _plot_subject_direction_curves(subject, method_results, bvals, subject_out)
        _plot_subject_slice_curves(subject, method_results, subject_out)
        _plot_subject_bval_scatter(subject, method_results, bvals, subject_out)
        _plot_subject_runtime_table(subject, method_results, subject_out)
        if args.save_npz:
            _save_subject_npz(subject, subject_out, method_results, target_dwi, z_idx, d_idx)

        for method, result in method_results.items():
            row = {
                "subject": subject,
                "method": method,
                "x": x1 - x0,
                "y": y1 - y0,
                "z": z1 - z0,
                "n_dirs": n_dirs,
                "mse": result.global_metrics["mse"],
                "mae": result.global_metrics["mae"],
                "psnr_db": result.global_metrics["psnr_db"],
                "runtime_s": result.runtime_total_s,
                "runtime_fit_s": result.runtime_fit_s,
                "runtime_predict_s": result.runtime_predict_s,
            }
            rows.append(row)

        subject_rows = [r for r in rows if r["subject"] == subject]
        print("  subject metrics:")
        for r in sorted(subject_rows, key=lambda x: x["method"]):
            print(
                f"    {r['method']:>10s} | "
                f"MSE={float(r['mse']):.6f} MAE={float(r['mae']):.6f} "
                f"PSNR={float(r['psnr_db']):.2f} dB runtime={float(r['runtime_s']):.2f}s"
            )

    _write_csv(rows, out_root / "metrics.csv")
    _plot_summary_metric_boxplots(rows, summary_dir)
    _plot_summary_runtime(rows, summary_dir)
    _plot_summary_psnr_improvement(rows, summary_dir)
    _plot_summary_method_heatmap(rows, summary_dir)
    _write_markdown_report(rows, out_root / "report.md")

    total_s = time.perf_counter() - run_t0
    print("\nVisualization run complete")
    print(f"Subjects processed: {len(subjects)}")
    print(f"Total runtime: {total_s:.2f}s")
    print(f"Metrics CSV: {out_root / 'metrics.csv'}")
    print(f"Report: {out_root / 'report.md'}")
    print(f"Subject plots: {subjects_dir}")
    print(f"Summary plots: {summary_dir}")


if __name__ == "__main__":
    main()
