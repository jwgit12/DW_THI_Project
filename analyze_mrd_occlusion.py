#!/usr/bin/env python3
"""Occlusion sensitivity analysis for MRD-CNN.

This is a standalone, non-invasive XAI script. It does not modify the training,
evaluation, dataset, augmentation, or model codepaths. It loads one trained
MRD-CNN checkpoint, selects one degraded DWI slice, and estimates spatial
importance by replacing local patches with local means and measuring the FA
RMSE increase relative to the non-occluded baseline prediction.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

import config_mrd as cfg
from dw_thi.evaluate import _load_input_dwi
from dw_thi.models import MRDCNN
from dw_thi.preprocessing import compute_b0_norm
from dw_thi.runtime import path_str, resolve_project_path
from dw_thi.utils import _symmetric_limits, dti6d_to_scalar_maps, select_plot_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise SystemExit("MPS requested but unavailable.")
    return torch.device(name)


def load_mrd_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[MRDCNN, dict, float, float]:
    ckpt = torch.load(resolve_project_path(checkpoint), map_location=device, weights_only=False)
    model = MRDCNN(
        max_n=int(ckpt["max_n"]),
        feat_dim=int(ckpt.get("feat_dim", cfg.FEAT_DIM)),
        channels=tuple(ckpt.get("channels", cfg.UNET_CHANNELS)),
        cholesky=bool(ckpt.get("cholesky", False)),
        dropout=cfg.DROPOUT,
        denoise_channels=cfg.MRD_DENOISE_CHANNELS,
        denoise_depth=cfg.MRD_DENOISE_DEPTH,
        tensor_channels=cfg.MRD_TENSOR_CHANNELS,
        tensor_depth=cfg.MRD_TENSOR_DEPTH,
        residual_scale=cfg.MRD_RESIDUAL_SCALE,
        grad_hidden=cfg.MRD_GRAD_HIDDEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dti_scale = float(ckpt.get("dti_scale", 1.0))
    max_bval = float(ckpt.get("max_bval", 1000.0))
    return model, ckpt, dti_scale, max_bval


def choose_subject(subject: str | None, store: zarr.Group, ckpt: dict) -> str:
    if subject is not None:
        if subject not in store:
            choices = ", ".join(sorted(store.keys())[:8])
            raise SystemExit(f"Subject {subject!r} not found. First available subjects: {choices}")
        return subject
    for candidate in ckpt.get("test_subjects", []):
        if candidate in store:
            return candidate
    keys = sorted(store.keys())
    if not keys:
        raise SystemExit("No subjects found in the Zarr store.")
    return keys[0]


def normalize_for_model(raw_signal_nhw: np.ndarray, bvals: np.ndarray, b0_threshold: float) -> tuple[np.ndarray, float]:
    signal = np.ascontiguousarray(raw_signal_nhw, dtype=np.float32).copy()
    b0_idx = bvals < b0_threshold
    b0_norm = 1.0
    if b0_idx.any():
        b0_norm = compute_b0_norm(signal[b0_idx].mean(axis=0))
        if b0_norm > 0:
            signal *= np.float32(1.0 / b0_norm)
        else:
            b0_norm = 1.0
    return signal, float(b0_norm)


def pad_directions(
    signal_nhw: np.ndarray,
    bvals_norm: np.ndarray,
    bvecs: np.ndarray,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = signal_nhw.shape[0]
    if n > max_n:
        raise SystemExit(f"Subject has {n} DWI volumes but checkpoint max_n is {max_n}.")
    vol_mask = np.zeros(max_n, dtype=np.float32)
    vol_mask[:n] = 1.0
    if n < max_n:
        pad = max_n - n
        signal_nhw = np.pad(signal_nhw, ((0, pad), (0, 0), (0, 0)))
        bvals_norm = np.pad(bvals_norm, (0, pad))
        bvecs = np.pad(bvecs, ((0, 0), (0, pad)))
    return signal_nhw, bvals_norm, bvecs, vol_mask


def prepare_tensors(
    raw_signal_nhw: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    max_bval: float,
    max_n: int,
    b0_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, float]:
    signal_norm, b0_norm = normalize_for_model(raw_signal_nhw, bvals, b0_threshold)
    bvals_norm = np.asarray(bvals / max_bval, dtype=np.float32)
    signal_pad, bvals_pad, bvecs_pad, vol_mask = pad_directions(
        signal_norm, bvals_norm, np.asarray(bvecs, dtype=np.float32), max_n
    )
    non_blocking = device.type == "cuda"
    signal_t = torch.from_numpy(signal_pad).unsqueeze(0).to(device, non_blocking=non_blocking)
    bvals_t = torch.from_numpy(bvals_pad).unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs_t = torch.from_numpy(bvecs_pad).unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(device, non_blocking=non_blocking)
    return signal_t, bvals_t, bvecs_t, vol_mask_t, signal_norm, b0_norm


def predict_slice(
    model: MRDCNN,
    raw_signal_nhw: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    max_bval: float,
    dti_scale: float,
    b0_threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    signal_t, bvals_t, bvecs_t, vol_mask_t, signal_norm, b0_norm = prepare_tensors(
        raw_signal_nhw=raw_signal_nhw,
        bvals=bvals,
        bvecs=bvecs,
        max_bval=max_bval,
        max_n=model.max_n,
        b0_threshold=b0_threshold,
        device=device,
    )
    with torch.inference_mode():
        denoised_t, _residual_t = model.denoiser(signal_t, bvals_t, bvecs_t, vol_mask_t)
        features = model.q_encoder(denoised_t, bvals_t, bvecs_t, vol_mask_t)
        pred_t = model.tensor_head(features)
        if model.cholesky:
            from dw_thi.model import cholesky_to_tensor6

            pred_t = cholesky_to_tensor6(pred_t)
    pred = pred_t[0].float().cpu().numpy() / dti_scale
    denoised = denoised_t[0, : raw_signal_nhw.shape[0]].float().cpu().numpy()
    return pred, denoised, signal_norm, b0_norm


def fa_from_tensor_slice(tensor_chw: np.ndarray) -> np.ndarray:
    vol = np.asarray(tensor_chw, dtype=np.float32).transpose(1, 2, 0)[:, :, None, :]
    fa, _md = dti6d_to_scalar_maps(vol)
    return np.asarray(fa[:, :, 0], dtype=np.float32)


def fa_rmse(pred_fa: np.ndarray, target_fa: np.ndarray, mask: np.ndarray | None) -> float:
    diff = np.asarray(pred_fa - target_fa, dtype=np.float32)
    if mask is not None:
        valid = np.asarray(mask > 0.5)
        if valid.any():
            diff = diff[valid]
    return float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0


def local_mean_occlude(signal_nhw: np.ndarray, y0: int, x0: int, patch_size: int) -> np.ndarray:
    occluded = np.asarray(signal_nhw, dtype=np.float32).copy()
    _n, h, w = occluded.shape
    y1 = min(y0 + patch_size, h)
    x1 = min(x0 + patch_size, w)

    radius = patch_size
    ly0 = max(0, y0 - radius)
    lx0 = max(0, x0 - radius)
    ly1 = min(h, y1 + radius)
    lx1 = min(w, x1 + radius)

    local = occluded[:, ly0:ly1, lx0:lx1].copy()
    mask = np.ones(local.shape[1:], dtype=bool)
    py0 = y0 - ly0
    px0 = x0 - lx0
    mask[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)] = False

    if mask.any():
        replacement = local[:, mask].mean(axis=1)
    else:
        replacement = occluded.reshape(occluded.shape[0], -1).mean(axis=1)
    occluded[:, y0:y1, x0:x1] = replacement[:, None, None]
    return occluded


def normalize_heatmap_for_display(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(heatmap)
    if not finite.any():
        return np.zeros_like(heatmap, dtype=np.float32)
    lo = float(np.min(heatmap[finite]))
    hi = float(np.max(heatmap[finite]))
    if hi <= lo:
        return np.zeros_like(heatmap, dtype=np.float32)
    return np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)


def normalize01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def imshow_rot(ax, image, title: str, *, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(np.rot90(image, 1), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    return im


def save_figure(
    output: Path,
    *,
    subject: str,
    slice_idx: int,
    volume_idx: int,
    patch_size: int,
    stride: int,
    baseline_rmse: float,
    corrupted_dwi: np.ndarray,
    denoised_dwi: np.ndarray,
    target_dwi: np.ndarray,
    pred_fa: np.ndarray,
    target_fa: np.ndarray,
    raw_heatmap: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    heatmap_vis = normalize_heatmap_for_display(raw_heatmap)
    fa_error = target_fa - pred_fa
    corrupted_vis = normalize01(corrupted_dwi)
    denoised_vis = normalize01(denoised_dwi)
    target_vis = normalize01(target_dwi)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=False)
    for ax in axes.ravel():
        ax.axis("off")

    imshow_rot(axes[0, 0], corrupted_vis, "Corrupted DWI input", cmap="gray", vmin=0.0, vmax=1.0)
    imshow_rot(axes[0, 1], denoised_vis, "MRD denoised DWI", cmap="gray", vmin=0.0, vmax=1.0)
    imshow_rot(axes[0, 2], target_vis, "Target clean DWI", cmap="gray", vmin=0.0, vmax=1.0)
    imshow_rot(axes[0, 3], pred_fa, "Predicted FA", cmap="viridis", vmin=0.0, vmax=1.0)

    imshow_rot(axes[1, 0], target_fa, "Target FA", cmap="viridis", vmin=0.0, vmax=1.0)
    err_lim = _symmetric_limits(fa_error)
    im = imshow_rot(axes[1, 1], fa_error, "FA error (target - pred)", cmap="bwr", vmin=err_lim[0], vmax=err_lim[1])
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    raw_lim = _symmetric_limits(raw_heatmap)
    im = imshow_rot(axes[1, 2], raw_heatmap, "Raw occlusion sensitivity", cmap="bwr", vmin=raw_lim[0], vmax=raw_lim[1])
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Δ FA RMSE")

    axes[1, 3].imshow(np.rot90(normalize01(pred_fa), 1), cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    im = axes[1, 3].imshow(np.rot90(heatmap_vis, 1), cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.48, interpolation="nearest")
    axes[1, 3].set_title("Sensitivity overlay on FA", fontsize=11, pad=8)
    axes[1, 3].axis("off")
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04, label="normalized")

    fig.suptitle(
        f"MRD-CNN Occlusion Sensitivity | {subject} | z={slice_idx} | volume={volume_idx} | "
        f"patch={patch_size} stride={stride} | baseline FA RMSE={baseline_rmse:.5f}",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    log.info("Device: %s", device)
    model, ckpt, dti_scale, max_bval = load_mrd_checkpoint(args.checkpoint, device)

    zarr_path = path_str(args.zarr_path)
    store = zarr.open_group(zarr_path, mode="r")
    subject = choose_subject(args.subject, store, ckpt)
    grp = store[subject]

    target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
    target_dti = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)
    brain_mask = (
        np.asarray(grp["brain_mask"][:], dtype=np.float32)
        if "brain_mask" in set(grp.array_keys())
        else np.ones(target_dwi.shape[:3], dtype=np.float32)
    )

    input_dwi = _load_input_dwi(
        grp,
        target_dwi=target_dwi,
        keep_fraction=cfg.EVAL_KEEP_FRACTION,
        noise_level=cfg.EVAL_NOISE_LEVEL,
        seed=cfg.EVAL_DEGRADE_SEED,
    )
    slice_idx, volume_idx = select_plot_indices(
        dwi_4d=input_dwi,
        bvals=bvals,
        b0_threshold=cfg.B0_THRESHOLD,
        slice_idx=args.slice_idx,
        volume_idx=args.volume_idx,
    )
    log.info("Subject=%s slice=%d volume=%d", subject, slice_idx, volume_idx)

    raw_signal_nhw = np.ascontiguousarray(input_dwi[:, :, slice_idx, :].transpose(2, 0, 1), dtype=np.float32)
    pred_tensor, denoised_nhw, normalized_signal, b0_norm = predict_slice(
        model=model,
        raw_signal_nhw=raw_signal_nhw,
        bvals=bvals,
        bvecs=bvecs,
        max_bval=max_bval,
        dti_scale=dti_scale,
        b0_threshold=cfg.B0_THRESHOLD,
        device=device,
    )
    pred_fa = fa_from_tensor_slice(pred_tensor)
    target_fa = fa_from_tensor_slice(target_dti[:, :, slice_idx, :].transpose(2, 0, 1))
    mask_slice = brain_mask[:, :, slice_idx] > 0.5
    baseline_rmse = fa_rmse(pred_fa, target_fa, mask_slice)
    log.info("Baseline FA RMSE: %.6f", baseline_rmse)

    _, h, w = raw_signal_nhw.shape
    patch_size = max(1, int(args.patch_size))
    stride = max(1, int(args.stride))
    heat_sum = np.zeros((h, w), dtype=np.float32)
    heat_count = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    if ys[-1] != max(h - patch_size, 0):
        ys.append(max(h - patch_size, 0))
    if xs[-1] != max(w - patch_size, 0):
        xs.append(max(w - patch_size, 0))

    total = len(ys) * len(xs)
    log.info("Running %d occlusion windows (patch=%d stride=%d)", total, patch_size, stride)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            occluded = local_mean_occlude(raw_signal_nhw, y0, x0, patch_size)
            occ_tensor, _occ_denoised, _occ_norm, _occ_b0 = predict_slice(
                model=model,
                raw_signal_nhw=occluded,
                bvals=bvals,
                bvecs=bvecs,
                max_bval=max_bval,
                dti_scale=dti_scale,
                b0_threshold=cfg.B0_THRESHOLD,
                device=device,
            )
            occ_fa = fa_from_tensor_slice(occ_tensor)
            delta = float(fa_rmse(occ_fa, target_fa, mask_slice) - baseline_rmse)
            heat_sum[y0:y1, x0:x1] += delta
            heat_count[y0:y1, x0:x1] += 1.0
            done += 1
            if done == 1 or done % max(1, total // 10) == 0 or done == total:
                log.info("Occlusion progress: %d/%d", done, total)

    raw_heatmap = heat_sum / np.maximum(heat_count, 1.0)
    output = resolve_project_path(args.output)
    if output.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output.mkdir(parents=True, exist_ok=True)
        output = output / f"mrd_occlusion_{subject}_z{slice_idx:03d}_v{volume_idx:03d}.png"

    save_figure(
        output,
        subject=subject,
        slice_idx=slice_idx,
        volume_idx=volume_idx,
        patch_size=patch_size,
        stride=stride,
        baseline_rmse=baseline_rmse,
        corrupted_dwi=normalized_signal[volume_idx],
        denoised_dwi=denoised_nhw[volume_idx],
        target_dwi=target_dwi[:, :, slice_idx, volume_idx] * np.float32(1.0 / b0_norm),
        pred_fa=pred_fa,
        target_fa=target_fa,
        raw_heatmap=raw_heatmap,
    )
    log.info("Saved MRD occlusion analysis -> %s", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone MRD-CNN sliding-window occlusion sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=cfg.EVAL_DEFAULT_CHECKPOINT)
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--slice_idx", type=int, default=None)
    parser.add_argument("--volume_idx", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output", default="runs/xai_mrd_occlusion")
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
