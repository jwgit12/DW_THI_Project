"""
Training script for the channel-invariant DTI pretext task.

Loss: Charbonnier (smooth-L1) + sliding-window SSIM on DWI reconstructions,
      Charbonnier on 6-component DTI tensors.

Metrics (validation):
    DWI domain  – PSNR (dB), SSIM
    DTI derived – FA MAE, MD MAE

Usage:
    python ml/train.py --zarr_path dataset/pretext_dataset.zarr --epochs 100
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.dataset import PretextDWIDataset, pretext_collate_fn
from ml.model import PretextUNet, count_parameters


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 / Charbonnier loss — robust to outliers, differentiable at 0."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# Cache SSIM kernels to avoid repeated CPU/GPU allocations each iteration.
_SSIM_KERNEL_CACHE: dict[tuple[int, float, str, torch.dtype], torch.Tensor] = {}


def _gaussian_kernel_2d(
    size: int = 7,
    sigma: float = 1.5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cache_key = (size, float(sigma), str(device), dtype)
    cached = _SSIM_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
    kernel = kernel.to(dtype=dtype)
    _SSIM_KERNEL_CACHE[cache_key] = kernel
    return kernel


def ssim_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute mean SSIM for (B, 1, H, W) tensors using Gaussian weighting.

    Returns a scalar in [0, 1] (higher = more similar).
    """
    kernel = _gaussian_kernel_2d(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_x = F.conv2d(pred, kernel, padding=pad)
    mu_y = F.conv2d(target, kernel, padding=pad)

    sigma_x2 = F.conv2d(pred ** 2, kernel, padding=pad) - mu_x ** 2
    sigma_y2 = F.conv2d(target ** 2, kernel, padding=pad) - mu_y ** 2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()


def pretext_loss(
    dwi_pred: torch.Tensor,
    dwi_target: torch.Tensor,
    dti_pred: torch.Tensor,
    dti_target: torch.Tensor,
    pad_mask: torch.Tensor,
    alpha_ssim: float = 0.2,
    lambda_dwi: float = 1.0,
    lambda_dti: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined pretext loss.

    DWI: α·(1-SSIM) + (1-α)·Charbonnier  (only on valid, non-padded directions)
    DTI: Charbonnier on 6-component tensor
    """
    # --- DWI loss (only valid directions) ---
    valid = pad_mask > 0.5  # (B, N)
    if not bool(valid.any().item()):
        zero = dwi_pred.new_zeros(())
        loss_dti = charbonnier_loss(dti_pred, dti_target)
        loss_total = lambda_dti * loss_dti
        return {
            "loss": loss_total,
            "loss_dwi": zero,
            "loss_dti": loss_dti,
            "charb_dwi": zero,
            "ssim_dwi": zero,
        }

    pred_valid = dwi_pred[valid].unsqueeze(1)   # (K, 1, H, W)
    tgt_valid = dwi_target[valid].unsqueeze(1)  # (K, 1, H, W)

    loss_charb_dwi = charbonnier_loss(pred_valid, tgt_valid)
    loss_ssim_dwi = 1.0 - ssim_2d(pred_valid, tgt_valid)
    loss_dwi = (1.0 - alpha_ssim) * loss_charb_dwi + alpha_ssim * loss_ssim_dwi

    # --- DTI loss (Charbonnier on all 6 components) ---
    loss_dti = charbonnier_loss(dti_pred, dti_target)

    loss_total = lambda_dwi * loss_dwi + lambda_dti * loss_dti

    return {
        "loss": loss_total,
        "loss_dwi": loss_dwi,
        "loss_dti": loss_dti,
        "charb_dwi": loss_charb_dwi,
        "ssim_dwi": loss_ssim_dwi,
    }


# ---------------------------------------------------------------------------
# Metrics (validation only, on CPU/GPU, no grad)
# ---------------------------------------------------------------------------
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB.  Higher is better. Good DWI ≈ 30-40 dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM index for (K, 1, H, W) image pairs.  Range [0, 1], higher is better."""
    return ssim_2d(pred, target).item()


@torch.no_grad()
def _tensor6_to_fa_md(t6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FA and MD maps from (B, 6, H, W) DTI tensor.

    Returns
    -------
    fa : (B, H, W)  Fractional Anisotropy  [0, 1]
    md : (B, H, W)  Mean Diffusivity
    """
    # Order: Dxx Dxy Dyy Dxz Dyz Dzz
    Dxx, Dxy, Dyy, Dxz, Dyz, Dzz = t6.float().unbind(dim=1)
    md = (Dxx + Dyy + Dzz) / 3.0

    # FA via tensor invariants (equivalent to eigenvalue definition):
    # FA = sqrt(3/2) * ||D - MD*I||_F / ||D||_F
    dxx = Dxx - md
    dyy = Dyy - md
    dzz = Dzz - md

    dev_fro_sq = dxx ** 2 + dyy ** 2 + dzz ** 2 + 2.0 * (Dxy ** 2 + Dxz ** 2 + Dyz ** 2)
    full_fro_sq = Dxx ** 2 + Dyy ** 2 + Dzz ** 2 + 2.0 * (Dxy ** 2 + Dxz ** 2 + Dyz ** 2)
    fa = torch.sqrt((1.5 * dev_fro_sq / (full_fro_sq + 1e-12)).clamp(0.0, 1.0))
    return fa, md


@torch.no_grad()
def compute_fa_md_errors(
    dti_pred: torch.Tensor, dti_target: torch.Tensor,
) -> tuple[float, float]:
    """Mean absolute error of FA and MD maps."""
    fa_pred, md_pred = _tensor6_to_fa_md(dti_pred)
    fa_tgt, md_tgt = _tensor6_to_fa_md(dti_target)
    fa_mae = torch.mean(torch.abs(fa_pred - fa_tgt)).item()
    md_mae = torch.mean(torch.abs(md_pred - md_tgt)).item()
    return fa_mae, md_mae


# ---------------------------------------------------------------------------
# TensorBoard image logging
# ---------------------------------------------------------------------------
def _normalize_for_display(img: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a 2D tensor to [0, 1] for display."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return torch.zeros_like(img)
    return (img - lo) / (hi - lo)


@torch.no_grad()
def _log_images(model, loader, writer, epoch, device, args):
    """Log denoising examples + FA/MD maps to TensorBoard."""
    model.eval()
    batch = next(iter(loader))
    directions = batch["directions"].to(device)
    dir_mask = batch["dir_mask"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    tgt_dwi = batch["target_dwi"].to(device)
    tgt_dti = batch["target_dti"].to(device)
    bvals = batch["bvals"].to(device)
    bvecs = batch["bvecs"].to(device)

    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        dwi_pred, dti_pred = model(
            directions, dir_mask, pad_mask, bvals, bvecs,
            dir_chunk_size=args.dir_chunk_size,
        )

    idx = 0  # first sample in batch

    # --- Pick a *masked* direction (model had to predict it blind) ---
    valid_and_masked = (pad_mask[idx] == 1) & (dir_mask[idx] == 0)
    if valid_and_masked.any():
        d = valid_and_masked.nonzero()[0].item()
    else:
        d = 0

    inp = _normalize_for_display(directions[idx, d].cpu())
    pred = _normalize_for_display(dwi_pred[idx, d].cpu())
    tgt = _normalize_for_display(tgt_dwi[idx, d].cpu())
    err = _normalize_for_display(torch.abs(dwi_pred[idx, d] - tgt_dwi[idx, d]).cpu())

    # DWI row: input | prediction | target | |error|
    row_dwi = torch.cat([inp, pred, tgt, err], dim=1)  # (H, 4W)
    writer.add_image("val/DWI_input_pred_target_error", row_dwi.unsqueeze(0), epoch)

    # --- FA and MD maps ---
    fa_pred, md_pred = _tensor6_to_fa_md(dti_pred[idx : idx + 1].float())
    fa_tgt, md_tgt = _tensor6_to_fa_md(tgt_dti[idx : idx + 1].float())
    fa_err = torch.abs(fa_pred - fa_tgt)
    md_err = torch.abs(md_pred - md_tgt)

    # FA row: predicted | target | |error|
    fa_row = torch.cat([
        _normalize_for_display(fa_pred[0]),
        _normalize_for_display(fa_tgt[0]),
        _normalize_for_display(fa_err[0]),
    ], dim=1)
    writer.add_image("val/FA_pred_target_error", fa_row.unsqueeze(0), epoch)

    # MD row: predicted | target | |error|
    md_row = torch.cat([
        _normalize_for_display(md_pred[0]),
        _normalize_for_display(md_tgt[0]),
        _normalize_for_display(md_err[0]),
    ], dim=1)
    writer.add_image("val/MD_pred_target_error", md_row.unsqueeze(0), epoch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    if args.accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    if args.dir_chunk_size < 1:
        raise ValueError("dir_chunk_size must be >= 1")
    if not (0.0 <= args.mask_fraction <= 1.0):
        raise ValueError("mask_fraction must be in [0, 1]")

    _set_seed(args.seed)
    device = _get_device()
    print(f"Using device: {device}")

    # Speed: device-specific tuning
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        # Metal Performance Shaders benefit from channels-last memory layout
        print("MPS detected (Apple Silicon) — enabling channels_last + bfloat16")

    # --- Data ------------------------------------------------------------------
    import zarr

    store = zarr.open(args.zarr_path, mode="r")
    n_subjects = len(list(store.group_keys()))
    n_val = max(1, int(n_subjects * 0.2))
    n_train = n_subjects - n_val
    all_idx = list(range(n_subjects))
    # Preserve legacy split behavior (and RNG consumption order) for strict reproducibility.
    np.random.seed(args.seed)
    np.random.shuffle(all_idx)
    train_idx = sorted(all_idx[:n_train])
    val_idx = sorted(all_idx[n_train:])
    print(f"Subjects — train: {len(train_idx)}, val: {len(val_idx)}")

    train_ds = PretextDWIDataset(
        args.zarr_path,
        subject_indices=train_idx,
        mask_fraction=args.mask_fraction,
    )
    val_ds = PretextDWIDataset(
        args.zarr_path,
        subject_indices=val_idx,
        mask_fraction=args.mask_fraction,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=pretext_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs,
    )

    # --- Model -----------------------------------------------------------------
    model = PretextUNet(base_features=args.base_features).to(device)

    # Speed: channels_last on MPS (Apple Silicon) — Metal conv kernels are faster in NHWC
    if device.type == "mps":
        model = model.to(memory_format=torch.channels_last)

    # Speed: torch.compile (PyTorch >= 2.0, CUDA only — not yet supported on MPS)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Mixed precision — bfloat16 on MPS (Apple Silicon), float16 on CUDA
    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    # GradScaler is only needed for float16 (narrow exponent range); bfloat16
    # has the same exponent range as float32 so scaling is unnecessary on MPS.
    use_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # --- TensorBoard -----------------------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Checkpointing ---------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    # --- Training loop ---------------------------------------------------------
    accum_steps = args.accum_steps
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            directions = batch["directions"].to(device, non_blocking=True)
            dir_mask = batch["dir_mask"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            tgt_dwi = batch["target_dwi"].to(device, non_blocking=True)
            tgt_dti = batch["target_dti"].to(device, non_blocking=True)
            bvals = batch["bvals"].to(device, non_blocking=True)
            bvecs = batch["bvecs"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                dwi_pred, dti_pred = model(
                    directions, dir_mask, pad_mask, bvals, bvecs,
                    dir_chunk_size=args.dir_chunk_size,
                )
                losses = pretext_loss(
                    dwi_pred, tgt_dwi, dti_pred, tgt_dti, pad_mask,
                    alpha_ssim=args.alpha_ssim,
                    lambda_dwi=args.lambda_dwi,
                    lambda_dti=args.lambda_dti,
                )
                loss = losses["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_val = losses["loss"].item()
            epoch_loss += loss_val
            global_step += 1

            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/loss_dwi", losses["loss_dwi"].item(), global_step)
            writer.add_scalar("train/loss_dti", losses["loss_dti"].item(), global_step)
            writer.add_scalar("train/charb_dwi", losses["charb_dwi"].item(), global_step)
            writer.add_scalar("train/ssim_loss", losses["ssim_dwi"].item(), global_step)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Free cached memory between epochs
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        writer.add_scalar("train/epoch_loss", avg_train, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # --- Validation --------------------------------------------------------
        metrics = _validate(model, val_loader, device, args)
        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/psnr_dB", metrics["psnr"], epoch)
        writer.add_scalar("val/ssim", metrics["ssim"], epoch)
        writer.add_scalar("val/fa_mae", metrics["fa_mae"], epoch)
        writer.add_scalar("val/md_mae", metrics["md_mae"], epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | "
            f"train {avg_train:.4f} | "
            f"val {metrics['loss']:.4f} | "
            f"PSNR {metrics['psnr']:.1f} dB | "
            f"SSIM {metrics['ssim']:.3f} | "
            f"FA-MAE {metrics['fa_mae']:.4f} | "
            f"MD-MAE {metrics['md_mae']:.2e} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Log sample images periodically
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            _log_images(model, val_loader, writer, epoch, device, args)

        # Checkpoint
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            path = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": metrics["loss"],
                    "val_psnr": metrics["psnr"],
                    "val_ssim": metrics["ssim"],
                    "val_fa_mae": metrics["fa_mae"],
                    "val_md_mae": metrics["md_mae"],
                },
                path,
            )

    writer.close()
    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints : {args.ckpt_dir}/")
    print(f"  TensorBoard : {args.log_dir}/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _validate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_mse_sum = 0.0
    total_mse_count = 0
    total_ssim = 0.0
    total_fa_mae = 0.0
    total_md_mae = 0.0
    n_subjects = 0
    n_dwi_dirs = 0

    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16

    for batch in loader:
        directions = batch["directions"].to(device, non_blocking=True)
        dir_mask = batch["dir_mask"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        tgt_dwi = batch["target_dwi"].to(device, non_blocking=True)
        tgt_dti = batch["target_dti"].to(device, non_blocking=True)
        bvals = batch["bvals"].to(device, non_blocking=True)
        bvecs = batch["bvecs"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            dwi_pred, dti_pred = model(
                directions, dir_mask, pad_mask, bvals, bvecs,
                dir_chunk_size=args.dir_chunk_size,
            )
            losses = pretext_loss(
                dwi_pred, tgt_dwi, dti_pred, tgt_dti, pad_mask,
                alpha_ssim=args.alpha_ssim,
                lambda_dwi=args.lambda_dwi,
                lambda_dti=args.lambda_dti,
            )

        batch_size = directions.shape[0]
        total_loss += losses["loss"].item() * batch_size

        # DWI metrics on valid directions
        valid = pad_mask > 0.5
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            pred_valid = dwi_pred.float()[valid].unsqueeze(1)
            tgt_valid = tgt_dwi.float()[valid].unsqueeze(1)
            sq_err = (pred_valid - tgt_valid) ** 2
            total_mse_sum += sq_err.sum().item()
            total_mse_count += sq_err.numel()
            total_ssim += compute_ssim(pred_valid, tgt_valid) * n_valid
            n_dwi_dirs += n_valid

        # DTI-derived metrics
        fa_mae, md_mae = compute_fa_md_errors(dti_pred.float(), tgt_dti.float())
        total_fa_mae += fa_mae * batch_size
        total_md_mae += md_mae * batch_size

        n_subjects += batch_size

    n_subjects = max(n_subjects, 1)
    n_dwi_dirs = max(n_dwi_dirs, 1)
    if total_mse_count > 0:
        mse = total_mse_sum / total_mse_count
        psnr = 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)
    else:
        psnr = 0.0
    return {
        "loss": total_loss / n_subjects,
        "psnr": psnr,
        "ssim": total_ssim / n_dwi_dirs,
        "fa_mae": total_fa_mae / n_subjects,
        "md_mae": total_md_mae / n_subjects,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the DTI pretext model.")
    parser.add_argument("--zarr_path", type=str, default="dataset/pretext_dataset.zarr")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * accum_steps).")
    parser.add_argument("--dir_chunk_size", type=int, default=16,
                        help="Max directions through encoder/decoder at once.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_fraction", type=float, default=0.4)
    parser.add_argument("--alpha_ssim", type=float, default=0.2,
                        help="Weight of SSIM term in DWI loss (0 = pure Charbonnier).")
    parser.add_argument("--lambda_dwi", type=float, default=1.0)
    parser.add_argument("--lambda_dti", type=float, default=1.0)
    parser.add_argument("--base_features", type=int, default=32)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="ml/runs/pretext")
    parser.add_argument("--ckpt_dir", type=str, default="ml/checkpoints")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
