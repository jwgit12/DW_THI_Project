#!/usr/bin/env python3
"""Grad-CAM visualization for MRD-CNN DTI prediction.

This script is intentionally MRD-only. It loads one trained MRD-CNN checkpoint,
runs one degraded DWI slice through the residual-denoising path, and saves a
publication-style figure showing DWI inputs, predicted/target scalar maps, the
predicted residual, and a Grad-CAM heatmap over the selected target objective.
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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zarr

import config_mrd as cfg
from dw_thi.evaluate import _load_input_dwi
from dw_thi.loss import tensor6_to_fa_md
from dw_thi.models import MRDCNN
from dw_thi.preprocessing import compute_b0_norm
from dw_thi.runtime import (
    amp_dtype_from_name,
    autocast_context,
    configure_torch_runtime,
    get_device,
    maybe_channels_last,
    path_str,
    require_cuda_if_requested,
    resolve_project_path,
)
from dw_thi.utils import (
    _robust_limits,
    _show_kspace,
    _symmetric_limits,
    dti6d_to_scalar_maps,
    select_plot_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class ActivationGradHook:
    """Capture activations and gradients from one module for Grad-CAM."""

    def __init__(self, module: torch.nn.Module):
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activation = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradient = grad

    def close(self) -> None:
        self._handle.remove()

    def cam(self, out_hw: tuple[int, int]) -> torch.Tensor:
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hook did not capture activation/gradient.")
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min).clamp(min=1e-6)


def _configured_mrd_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[MRDCNN, float, float]:
    max_n = ckpt["max_n"]
    feat_dim = ckpt.get("feat_dim", cfg.FEAT_DIM)
    channels = tuple(ckpt.get("channels", cfg.UNET_CHANNELS))
    cholesky = ckpt.get("cholesky", False)
    model = MRDCNN(
        max_n=max_n,
        feat_dim=feat_dim,
        channels=channels,
        cholesky=cholesky,
        dropout=cfg.DROPOUT,
        denoise_channels=cfg.MRD_DENOISE_CHANNELS,
        denoise_depth=cfg.MRD_DENOISE_DEPTH,
        tensor_channels=cfg.MRD_TENSOR_CHANNELS,
        tensor_depth=cfg.MRD_TENSOR_DEPTH,
        residual_scale=cfg.MRD_RESIDUAL_SCALE,
        grad_hidden=cfg.MRD_GRAD_HIDDEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, float(ckpt.get("dti_scale", 1.0)), float(ckpt.get("max_bval", 1000.0))


def _choose_subject(args: argparse.Namespace, store: zarr.Group, ckpt: dict) -> str:
    if args.subject is not None:
        if args.subject not in store:
            raise ValueError(f"Subject {args.subject!r} not found in {args.zarr_path}")
        return args.subject
    for subject in ckpt.get("test_subjects", []):
        if subject in store:
            return subject
    keys = sorted(store.keys())
    if not keys:
        raise ValueError(f"No subjects found in {args.zarr_path}")
    return keys[0]


def _pad_direction_axis(
    signal_nhw: np.ndarray,
    bvals_norm: np.ndarray,
    bvecs: np.ndarray,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = signal_nhw.shape[0]
    if n > max_n:
        raise ValueError(f"Input has {n} volumes, checkpoint max_n is {max_n}.")
    vol_mask = np.zeros(max_n, dtype=np.float32)
    vol_mask[:n] = 1.0
    if n < max_n:
        pad = max_n - n
        signal_nhw = np.pad(signal_nhw, ((0, pad), (0, 0), (0, 0)))
        bvals_norm = np.pad(bvals_norm, (0, pad))
        bvecs = np.pad(bvecs, ((0, 0), (0, pad)))
    return signal_nhw, bvals_norm, bvecs, vol_mask


def _objective_score(
    pred: torch.Tensor,
    mask: torch.Tensor | None,
    objective: str,
    channel: int,
) -> torch.Tensor:
    if objective == "fa":
        value, _md = tensor6_to_fa_md(pred)
    elif objective == "md":
        _fa, value = tensor6_to_fa_md(pred)
    elif objective == "tensor_norm":
        value = torch.sqrt(pred.square().sum(dim=1).clamp(min=1e-12))
    elif objective == "channel":
        value = pred[:, channel]
    else:  # pragma: no cover - argparse guards this
        raise ValueError(f"Unsupported objective: {objective}")

    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return (value * mask).sum() / denom
    return value.mean()


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x[np.isfinite(x)], [1, 99]) if np.isfinite(x).any() else (0.0, 1.0)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _imshow_panel(ax, image, title, *, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(np.rot90(image, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    return im


def _overlay_cam(ax, base, cam, title, *, base_cmap="gray", alpha=0.45):
    ax.imshow(np.rot90(_normalize01(base), 1), cmap=base_cmap, vmin=0.0, vmax=1.0)
    im = ax.imshow(np.rot90(cam, 1), cmap="inferno", vmin=0.0, vmax=1.0, alpha=alpha)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    return im


def save_gradcam_figure(
    *,
    out_path: Path,
    subject: str,
    slice_idx: int,
    volume_idx: int,
    objective: str,
    score_value: float,
    noisy_slice: np.ndarray,
    denoised_slice: np.ndarray,
    target_slice: np.ndarray,
    residual_slice: np.ndarray,
    pred_fa: np.ndarray,
    target_fa: np.ndarray,
    pred_md: np.ndarray,
    target_md: np.ndarray,
    cam: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    noisy_norm, denoised_norm, target_norm = [_normalize01(x) for x in (noisy_slice, denoised_slice, target_slice)]
    residual_abs = np.abs(residual_slice)
    fa_diff = target_fa - pred_fa
    md_diff = target_md - pred_md
    noisy_k = _show_kspace(noisy_norm)
    denoised_k = _show_kspace(denoised_norm)
    target_k = _show_kspace(target_norm)

    fig, axes = plt.subplots(4, 4, figsize=(18, 17), constrained_layout=False)
    for ax in axes.ravel():
        ax.axis("off")

    _imshow_panel(axes[0, 0], noisy_norm, "Input DWI (corrupted)", cmap="gray", vmin=0, vmax=1)
    _imshow_panel(axes[0, 1], denoised_norm, "MRD denoised DWI", cmap="gray", vmin=0, vmax=1)
    _imshow_panel(axes[0, 2], target_norm, "Target clean DWI", cmap="gray", vmin=0, vmax=1)
    im = _imshow_panel(axes[0, 3], residual_abs, "Predicted residual |r|", cmap="magma")
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    k_vmin, k_vmax = _robust_limits(noisy_k, denoised_k, target_k)
    _imshow_panel(axes[1, 0], noisy_k, "Input k-space", cmap="gray", vmin=k_vmin, vmax=k_vmax)
    _imshow_panel(axes[1, 1], denoised_k, "Denoised k-space", cmap="gray", vmin=k_vmin, vmax=k_vmax)
    _imshow_panel(axes[1, 2], target_k, "Target k-space", cmap="gray", vmin=k_vmin, vmax=k_vmax)
    im = _imshow_panel(axes[1, 3], target_k - denoised_k, "k-space difference", cmap="bwr", vmin=_symmetric_limits(target_k - denoised_k)[0], vmax=_symmetric_limits(target_k - denoised_k)[1])
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    _imshow_panel(axes[2, 0], pred_fa, "Predicted FA", cmap="viridis", vmin=0.0, vmax=1.0)
    _imshow_panel(axes[2, 1], target_fa, "Target FA", cmap="viridis", vmin=0.0, vmax=1.0)
    fa_lim = _symmetric_limits(fa_diff)
    im = _imshow_panel(axes[2, 2], fa_diff, "FA error (target - pred)", cmap="bwr", vmin=fa_lim[0], vmax=fa_lim[1])
    fig.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)
    cam_im = _imshow_panel(axes[2, 3], cam, "Grad-CAM heatmap", cmap="inferno", vmin=0.0, vmax=1.0)
    fig.colorbar(cam_im, ax=axes[2, 3], fraction=0.046, pad=0.04)

    md_vmin, md_vmax = _robust_limits(pred_md, target_md)
    _imshow_panel(axes[3, 0], pred_md, "Predicted MD", cmap="magma", vmin=md_vmin, vmax=md_vmax)
    _imshow_panel(axes[3, 1], target_md, "Target MD", cmap="magma", vmin=md_vmin, vmax=md_vmax)
    md_lim = _symmetric_limits(md_diff)
    im = _imshow_panel(axes[3, 2], md_diff, "MD error (target - pred)", cmap="bwr", vmin=md_lim[0], vmax=md_lim[1])
    fig.colorbar(im, ax=axes[3, 2], fraction=0.046, pad=0.04)
    _overlay_cam(axes[3, 3], pred_fa, cam, "Grad-CAM over predicted FA", base_cmap="viridis")

    fig.suptitle(
        f"MRD-CNN Grad-CAM | {subject} | z={slice_idx} | volume={volume_idx} | "
        f"objective={objective} | score={score_value:.5f}",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    zarr_path = path_str(args.zarr_path)
    checkpoint_path = resolve_project_path(args.checkpoint)
    out_dir = resolve_project_path(args.out_dir)

    device = get_device()
    require_cuda_if_requested(device, args.require_cuda)
    configure_torch_runtime(device, deterministic=args.deterministic)
    amp_dtype = amp_dtype_from_name(device, args.amp_dtype)
    amp_enabled = bool(args.amp and amp_dtype is not None)
    channels_last = bool(args.channels_last and device.type == "cuda")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, dti_scale, max_bval = _configured_mrd_from_checkpoint(ckpt, device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    store = zarr.open_group(zarr_path, mode="r")
    subject = _choose_subject(args, store, ckpt)
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
        keep_fraction=args.keep_fraction,
        noise_level=args.noise_level,
        seed=args.seed,
    )

    slice_idx, volume_idx = select_plot_indices(
        dwi_4d=input_dwi,
        bvals=bvals,
        b0_threshold=args.b0_threshold,
        slice_idx=args.slice_idx,
        volume_idx=args.volume_idx,
    )

    signal_nhw = np.ascontiguousarray(input_dwi[:, :, slice_idx, :].transpose(2, 0, 1), dtype=np.float32)
    b0_idx = bvals < args.b0_threshold
    b0_norm = 1.0
    if b0_idx.any():
        b0_norm = compute_b0_norm(input_dwi[:, :, slice_idx, b0_idx].mean(axis=-1))
        if b0_norm > 0:
            signal_nhw *= np.float32(1.0 / b0_norm)
        else:
            b0_norm = 1.0

    bvals_norm = bvals / max_bval
    signal_nhw, bvals_norm, bvecs_pad, vol_mask = _pad_direction_axis(
        signal_nhw, bvals_norm, bvecs, model.max_n
    )

    non_blocking = device.type == "cuda"
    signal_t = torch.from_numpy(signal_nhw).unsqueeze(0).to(device, non_blocking=non_blocking)
    signal_t = maybe_channels_last(signal_t, channels_last)
    bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs_t = torch.from_numpy(bvecs_pad).unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(device, non_blocking=non_blocking)
    mask_t = torch.from_numpy(brain_mask[:, :, slice_idx]).unsqueeze(0).to(device, non_blocking=non_blocking)

    target_layer = model.tensor_head.blocks[args.layer_index].conv2
    hook = ActivationGradHook(target_layer)
    model.zero_grad(set_to_none=True)
    try:
        with autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
            denoised_t, residual_t = model.denoiser(signal_t, bvals_t, bvecs_t, vol_mask_t)
            features = model.q_encoder(denoised_t, bvals_t, bvecs_t, vol_mask_t)
            pred_t = model.tensor_head(features)
            if model.cholesky:
                from dw_thi.model import cholesky_to_tensor6

                pred_t = cholesky_to_tensor6(pred_t)
            score = _objective_score(pred_t.float(), mask_t, args.objective, args.channel)
        score.backward()
        cam_t = hook.cam(pred_t.shape[-2:])
    finally:
        hook.close()

    pred_np = pred_t.detach().float().cpu().numpy()[0] / dti_scale
    residual_np = residual_t.detach().float().cpu().numpy()[0]
    denoised_np = denoised_t.detach().float().cpu().numpy()[0]
    cam_np = cam_t.detach().float().cpu().numpy()[0, 0]

    pred_vol = pred_np.transpose(1, 2, 0)[:, :, None, :]
    target_slice_dti = target_dti[:, :, slice_idx, :][:, :, None, :]
    pred_fa_vol, pred_md_vol = dti6d_to_scalar_maps(pred_vol)
    target_fa_vol, target_md_vol = dti6d_to_scalar_maps(target_slice_dti)

    noisy_slice = signal_nhw[volume_idx]
    denoised_slice = denoised_np[volume_idx]
    target_slice = target_dwi[:, :, slice_idx, volume_idx] * np.float32(1.0 / b0_norm)
    residual_slice = residual_np[volume_idx]

    out_path = out_dir / f"mrd_gradcam_{subject}_z{slice_idx:03d}_v{volume_idx:03d}_{args.objective}.png"
    save_gradcam_figure(
        out_path=out_path,
        subject=subject,
        slice_idx=slice_idx,
        volume_idx=volume_idx,
        objective=args.objective,
        score_value=float(score.detach().cpu()),
        noisy_slice=noisy_slice,
        denoised_slice=denoised_slice,
        target_slice=target_slice,
        residual_slice=residual_slice,
        pred_fa=pred_fa_vol[:, :, 0],
        target_fa=target_fa_vol[:, :, 0],
        pred_md=pred_md_vol[:, :, 0],
        target_md=target_md_vol[:, :, 0],
        cam=cam_np,
    )
    log.info("Saved MRD Grad-CAM -> %s", out_path)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize MRD-CNN Grad-CAM for one DWI slice")
    parser.add_argument("--checkpoint", default=cfg.EVAL_DEFAULT_CHECKPOINT)
    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default="runs/gradcam_mrd")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--slice_idx", type=int, default=None)
    parser.add_argument("--volume_idx", type=int, default=None)
    parser.add_argument("--objective", choices=["fa", "md", "tensor_norm", "channel"], default="fa")
    parser.add_argument("--channel", type=int, default=0, choices=range(6), metavar="0-5")
    parser.add_argument("--layer_index", type=int, default=-1, help="Residual tensor-head block index to visualize")
    parser.add_argument("--keep_fraction", type=float, default=cfg.EVAL_KEEP_FRACTION)
    parser.add_argument("--noise_level", type=float, default=cfg.EVAL_NOISE_LEVEL)
    parser.add_argument("--seed", type=int, default=cfg.EVAL_DEGRADE_SEED)
    parser.add_argument("--b0_threshold", type=float, default=cfg.B0_THRESHOLD)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.AMP)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default=cfg.AMP_DTYPE)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=cfg.CHANNELS_LAST)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=cfg.DETERMINISTIC)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=cfg.REQUIRE_CUDA)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
