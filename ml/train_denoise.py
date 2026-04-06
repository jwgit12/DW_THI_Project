"""
Training script for the direction-invariant DWI denoising model.

Loss: Charbonnier + MS-SSIM + Edge (spatial gradients).
Training: EMA weights, linear warmup + cosine LR, gradient accumulation.

Usage:
    # Fast prototyping with tiny model
    python ml/train_denoise.py --zarr_path dataset/pretext_dataset.zarr \\
        --model_size tiny --epochs 20 --augment

    # Full training
    python ml/train_denoise.py --zarr_path dataset/pretext_dataset.zarr \\
        --model_size base --epochs 100 --augment --alpha_edge 0.1
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.denoise_dataset import DWIDenoiseDataset, denoise_collate_fn
from ml.denoise_model import DenoiseUNet, MODEL_PRESETS, count_parameters


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model parameters.

    Smooths parameter updates for better generalization (~0.2-0.5 dB PSNR
    improvement, Zamir et al., Restormer 2022).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(p.data, 1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """Replace model params with EMA values (call restore() after)."""
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """Restore original model params after apply_to()."""
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return dict(self.shadow)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 / Charbonnier — robust to outliers, differentiable at 0."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


_SSIM_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}


def _gaussian_kernel_2d(
    size: int, sigma: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    key = (size, float(sigma), str(device), dtype)
    cached = _SSIM_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    _SSIM_KERNEL_CACHE[key] = kernel
    return kernel


def _ssim_components(
    pred: torch.Tensor, target: torch.Tensor,
    kernel: torch.Tensor, pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSIM luminance and contrast-structure components."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.conv2d(pred, kernel, padding=pad)
    mu_y = F.conv2d(target, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_x2 = F.conv2d(pred ** 2, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(target ** 2, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad) - mu_xy
    luminance = (2 * mu_xy + C1) / (mu_x2 + mu_y2 + C1)
    cs = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)
    return luminance, cs


def ssim_2d(pred: torch.Tensor, target: torch.Tensor,
            window_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for (B, 1, H, W) tensors. Returns scalar in [0, 1]."""
    kernel = _gaussian_kernel_2d(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2
    lum, cs = _ssim_components(pred, target, kernel, pad)
    return (lum * cs).mean()


def ms_ssim_2d(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Multi-scale SSIM for (B, 1, H, W) tensors.

    Computes contrast-structure at multiple downsampled scales and full SSIM
    at the coarsest scale (Wang et al., 2003).  Falls back to single-scale
    SSIM when the image is too small for multi-scale computation.
    """
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    # Determine feasible scales
    min_dim = min(pred.shape[2], pred.shape[3])
    n_scales = 1
    dim = min_dim
    while dim // 2 >= window_size and n_scales < len(weights):
        dim //= 2
        n_scales += 1

    if n_scales == 1:
        return ssim_2d(pred, target, window_size, sigma)

    weights = weights[:n_scales]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    cs_values: list[torch.Tensor] = []
    for i in range(n_scales):
        kernel = _gaussian_kernel_2d(window_size, sigma, pred.device, pred.dtype)
        pad = window_size // 2
        lum, cs = _ssim_components(pred, target, kernel, pad)

        if i == n_scales - 1:
            cs_values.append((lum * cs).mean())
        else:
            cs_values.append(cs.mean())
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)

    # Weighted geometric mean in log-space for numerical stability
    log_result = sum(
        w * torch.log(val.clamp(min=1e-4))
        for val, w in zip(cs_values, weights)
    )
    return torch.exp(log_result)


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on spatial gradients — preserves edges and fine detail.

    Achieves the same high-frequency regularisation goal as FFT magnitude
    loss but works on all devices (including MPS) without complex numbers.
    """
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    return (F.l1_loss(dy_pred, dy_tgt) + F.l1_loss(dx_pred, dx_tgt)) / 2


def denoise_loss(
    pred: torch.Tensor, target: torch.Tensor,
    pad_mask: torch.Tensor,
    alpha_ssim: float = 0.2,
    alpha_edge: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Combined loss: Charbonnier + MS-SSIM + Edge (valid directions only).

    Default weights: 70% Charbonnier + 20% MS-SSIM + 10% Edge.
    """
    valid = pad_mask > 0.5
    if not bool(valid.any().item()):
        zero = pred.new_zeros(())
        return {"loss": zero, "charb": zero, "ssim_loss": zero, "edge_loss": zero}

    pred_valid = pred[valid].unsqueeze(1)    # (K, 1, H, W)
    tgt_valid = target[valid].unsqueeze(1)

    loss_charb = charbonnier_loss(pred_valid, tgt_valid)
    loss_ssim = 1.0 - ms_ssim_2d(pred_valid, tgt_valid)
    loss_edge = edge_loss(pred_valid, tgt_valid)

    alpha_charb = 1.0 - alpha_ssim - alpha_edge
    loss = alpha_charb * loss_charb + alpha_ssim * loss_ssim + alpha_edge * loss_edge

    return {"loss": loss, "charb": loss_charb, "ssim_loss": loss_ssim,
            "edge_loss": loss_edge}


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_val_metrics(pred: torch.Tensor, target: torch.Tensor,
                        pad_mask: torch.Tensor) -> dict[str, float]:
    """DWI-space metrics on valid directions: PSNR, SSIM, RMSE, MAE, NRMSE."""
    valid = pad_mask > 0.5
    if not bool(valid.any().item()):
        return {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "mae": 0.0, "nrmse": 0.0}

    p = pred.float()[valid]
    t = target.float()[valid]

    psnr_list, ssim_list, rmse_list, mae_list, nrmse_list = [], [], [], [], []

    for i in range(p.shape[0]):
        pi, ti = p[i], t[i]
        dr = float(ti.max() - ti.min())
        if dr < 1e-10:
            continue
        mse_val = torch.mean((pi - ti) ** 2).item()
        rmse_val = math.sqrt(mse_val)
        mae_val = torch.mean(torch.abs(pi - ti)).item()
        psnr_list.append(10.0 * math.log10(dr ** 2 / max(mse_val, 1e-10)))
        ssim_list.append(ssim_2d(
            pi.unsqueeze(0).unsqueeze(0),
            ti.unsqueeze(0).unsqueeze(0),
        ).item())
        rmse_list.append(rmse_val)
        mae_list.append(mae_val)
        nrmse_list.append(rmse_val / dr)

    if not psnr_list:
        return {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "mae": 0.0, "nrmse": 0.0}

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "nrmse": float(np.mean(nrmse_list)),
    }


# ---------------------------------------------------------------------------
# TensorBoard helpers
# ---------------------------------------------------------------------------
def _normalize_for_display(img: torch.Tensor) -> torch.Tensor:
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return torch.zeros_like(img)
    return (img - lo) / (hi - lo)


@torch.no_grad()
def _log_images(model, loader, writer, epoch, device, args):
    model.eval()
    batch = next(iter(loader))
    noisy = batch["noisy_dwi"].to(device)
    clean = batch["clean_dwi"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    bvals = batch["bvals"].to(device)
    bvecs = batch["bvecs"].to(device)

    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        denoised = model(noisy, pad_mask, bvals, bvecs,
                         dir_chunk_size=args.dir_chunk_size)

    idx = 0
    valid = pad_mask[idx] > 0.5
    d = valid.nonzero()[0].item() if valid.any() else 0

    inp = _normalize_for_display(noisy[idx, d].cpu())
    pred = _normalize_for_display(denoised[idx, d].cpu())
    tgt = _normalize_for_display(clean[idx, d].cpu())
    err = _normalize_for_display(torch.abs(denoised[idx, d] - clean[idx, d]).cpu())

    row = torch.cat([inp, pred, tgt, err], dim=1)
    writer.add_image("val/noisy_denoised_target_error", row.unsqueeze(0), epoch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    if args.accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")

    _set_seed(args.seed)
    device = _get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Data -----------------------------------------------------------------
    import zarr
    store = zarr.open(args.zarr_path, mode="r")
    n_subjects = len(list(store.group_keys()))
    n_val = max(1, int(n_subjects * 0.2))
    n_train = n_subjects - n_val
    all_idx = list(range(n_subjects))
    np.random.seed(args.seed)
    np.random.shuffle(all_idx)
    train_idx = sorted(all_idx[:n_train])
    val_idx = sorted(all_idx[n_train:])
    print(f"Subjects — train: {len(train_idx)}, val: {len(val_idx)}")

    train_ds = DWIDenoiseDataset(args.zarr_path, subject_indices=train_idx,
                                 augment=args.augment)
    val_ds = DWIDenoiseDataset(args.zarr_path, subject_indices=val_idx)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=denoise_collate_fn,
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

    # --- Model ----------------------------------------------------------------
    model = DenoiseUNet(
        base_features=args.base_features,
        drop_path=args.drop_path,
    ).to(device)
    if device.type == "mps":
        model = model.to(memory_format=torch.channels_last)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass

    print(f"Model: {args.model_size} (f={args.base_features}), "
          f"{count_parameters(model):,} params")

    # --- Optimizer & scheduler ------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = max(1, args.epochs // 20)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # --- EMA ------------------------------------------------------------------
    ema = EMA(model, decay=args.ema_decay)

    # --- Mixed precision ------------------------------------------------------
    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    use_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # --- Logging & checkpointing ----------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_loss = float("inf")

    # --- Training loop --------------------------------------------------------
    accum_steps = args.accum_steps
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            noisy = batch["noisy_dwi"].to(device, non_blocking=True)
            clean = batch["clean_dwi"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            bvals = batch["bvals"].to(device, non_blocking=True)
            bvecs = batch["bvecs"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                denoised = model(noisy, pad_mask, bvals, bvecs,
                                 dir_chunk_size=args.dir_chunk_size)
                losses = denoise_loss(denoised, clean, pad_mask,
                                      alpha_ssim=args.alpha_ssim,
                                      alpha_edge=args.alpha_edge)
                loss = losses["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            loss_val = losses["loss"].item()
            epoch_loss += loss_val
            global_step += 1

            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/charb", losses["charb"].item(), global_step)
            writer.add_scalar("train/ssim_loss", losses["ssim_loss"].item(),
                              global_step)
            writer.add_scalar("train/edge_loss", losses["edge_loss"].item(),
                              global_step)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Free cached memory
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        writer.add_scalar("train/epoch_loss", avg_train, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # --- Validation (with EMA weights) ------------------------------------
        ema.apply_to(model)
        metrics = _validate(model, val_loader, device, args)
        ema.restore(model)

        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/psnr_dB", metrics["psnr"], epoch)
        writer.add_scalar("val/ssim", metrics["ssim"], epoch)
        writer.add_scalar("val/rmse", metrics["rmse"], epoch)
        writer.add_scalar("val/mae", metrics["mae"], epoch)
        writer.add_scalar("val/nrmse", metrics["nrmse"], epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | "
            f"train {avg_train:.4f} | "
            f"val {metrics['loss']:.4f} | "
            f"PSNR {metrics['psnr']:.1f} dB | "
            f"SSIM {metrics['ssim']:.3f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Log sample images periodically
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            ema.apply_to(model)
            _log_images(model, val_loader, writer, epoch, device, args)
            ema.restore(model)

        # Checkpoint (EMA weights)
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            ema.apply_to(model)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_features": args.base_features,
                "drop_path": args.drop_path,
                "val_loss": metrics["loss"],
                "val_psnr": metrics["psnr"],
                "val_ssim": metrics["ssim"],
            }
            ema.restore(model)
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best_denoise.pt"))

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
    total_metrics = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0,
                     "mae": 0.0, "nrmse": 0.0}
    n_batches = 0

    use_amp = device.type in ["cuda", "mps"]
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16

    for batch in loader:
        noisy = batch["noisy_dwi"].to(device, non_blocking=True)
        clean = batch["clean_dwi"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        bvals = batch["bvals"].to(device, non_blocking=True)
        bvecs = batch["bvecs"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            denoised = model(noisy, pad_mask, bvals, bvecs,
                             dir_chunk_size=args.dir_chunk_size)
            losses = denoise_loss(denoised, clean, pad_mask,
                                  alpha_ssim=args.alpha_ssim,
                                  alpha_edge=args.alpha_edge)

        total_loss += losses["loss"].item()
        batch_metrics = compute_val_metrics(denoised.float(), clean.float(),
                                            pad_mask)
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k]
        n_batches += 1

    n_batches = max(n_batches, 1)
    result = {"loss": total_loss / n_batches}
    for k in total_metrics:
        result[k] = total_metrics[k] / n_batches
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the DWI denoising model.")

    # Data
    parser.add_argument("--zarr_path", type=str,
                        default="dataset/pretext_dataset.zarr")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (flips + rotations).")

    # Model
    parser.add_argument("--model_size", type=str, default="base",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model size preset (overrides --base_features).")
    parser.add_argument("--base_features", type=int, default=None,
                        help="Override model preset feature width.")
    parser.add_argument("--drop_path", type=float, default=0.1,
                        help="DropPath rate for stochastic depth.")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--dir_chunk_size", type=int, default=16)

    # Loss weights
    parser.add_argument("--alpha_ssim", type=float, default=0.2,
                        help="MS-SSIM loss weight.")
    parser.add_argument("--alpha_edge", type=float, default=0.1,
                        help="Edge (spatial gradient) loss weight.")

    # Infrastructure
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="ml/runs/denoise")
    parser.add_argument("--ckpt_dir", type=str, default="ml/checkpoints")

    args = parser.parse_args()

    # Resolve model preset
    if args.base_features is None:
        args.base_features = MODEL_PRESETS[args.model_size]["base_features"]

    train(args)


if __name__ == "__main__":
    main()
