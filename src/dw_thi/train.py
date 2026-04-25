"""Train QSpaceUNet with TensorBoard tracking.

The production defaults live in config.py. CLI flags only override that single
config source when you need an ad-hoc run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    SummaryWriter = None

import config as cfg
from .augment import gpu_b0_normalize_batch, gpu_degrade_dwi_batch
from .dataset import DWISliceDataset, dwi_worker_init
from .loss import DTILoss
from .model import QSpaceUNet
from .runtime import (
    amp_dtype_from_name,
    autocast_context,
    configure_torch_runtime,
    default_num_workers,
    get_device,
    make_grad_scaler,
    maybe_channels_last,
    maybe_compile_model,
    path_str,
    require_cuda_if_requested,
    resolve_project_path,
)
from .utils import dti6d_to_scalar_maps, scalar_map_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def biological_subject(key: str) -> str:
    return key.split("_ses-", 1)[0]


def split_subjects(
    all_keys: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> tuple[list[str], list[str], list[str]]:
    train, val, test = [], [], []
    for key in all_keys:
        bio = biological_subject(key)
        if bio in test_subjects:
            test.append(key)
        elif bio in val_subjects:
            val.append(key)
        else:
            train.append(key)
    if not train:
        raise ValueError("No training subjects selected.")
    if not val:
        raise ValueError("No validation subjects selected.")
    return train, val, test


def load_baseline_metrics(csv_paths: list[Path]) -> dict[str, dict[str, float]]:
    """Load optional baseline CSVs and return {baseline_name: metric_means}."""
    baselines: dict[str, dict[str, float]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        mean_row = df[df["subject"] == "MEAN"]
        if mean_row.empty:
            continue
        name = csv_path.stem.replace("metrics_", "")
        metrics: dict[str, float] = {}
        for col in [
            "fa_rmse",
            "fa_mae",
            "fa_nrmse",
            "fa_r2",
            "adc_rmse",
            "adc_mae",
            "adc_nrmse",
            "adc_r2",
        ]:
            if col in mean_row.columns:
                val = mean_row[col].values[0]
                if pd.notna(val):
                    metrics[col] = float(val)
        baselines[name] = metrics
    return baselines


def make_val_figure(
    model: QSpaceUNet,
    val_ds: DWISliceDataset,
    device: torch.device,
    dti_scale: float,
    slice_idx: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
) -> plt.Figure:
    """Generate a prediction-vs-target figure for TensorBoard."""
    if slice_idx is None:
        slice_idx = len(val_ds) // 2

    sample = val_ds[slice_idx]
    non_blocking = device.type == "cuda"
    signal = sample["input"].unsqueeze(0).to(device, non_blocking=non_blocking)
    signal = maybe_channels_last(signal, channels_last)
    bvals = sample["bvals"].unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs = sample["bvecs"].unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask = sample["vol_mask"].unsqueeze(0).to(device, non_blocking=non_blocking)
    target = sample["target"].numpy()
    bmask = sample["brain_mask"].numpy()

    model.eval()
    with torch.inference_mode(), autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
        pred = model(signal, bvals, bvecs, vol_mask)
    pred_np = pred[0].float().cpu().numpy()

    pred_np = pred_np / dti_scale
    target = target / dti_scale

    pred_vol = pred_np.transpose(1, 2, 0)[..., np.newaxis, :]
    tgt_vol = target.transpose(1, 2, 0)[..., np.newaxis, :]
    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_vol)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(tgt_vol)

    pred_fa = pred_fa[:, :, 0]
    pred_adc = pred_adc[:, :, 0]
    tgt_fa = tgt_fa[:, :, 0]
    tgt_adc = tgt_adc[:, :, 0]
    bmask_bool = bmask > 0.5

    fa_diff = (tgt_fa - pred_fa) * bmask
    adc_diff = (tgt_adc - pred_adc) * bmask

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(np.rot90(tgt_fa * bmask), cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("Target FA")
    axes[0, 1].imshow(np.rot90(pred_fa * bmask), cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted FA")
    fa_abs = max(float(np.max(np.abs(fa_diff))), 1e-6)
    im_fa = axes[0, 2].imshow(np.rot90(fa_diff), cmap="bwr", vmin=-fa_abs, vmax=fa_abs)
    axes[0, 2].set_title("FA error")
    fig.colorbar(im_fa, ax=axes[0, 2], fraction=0.046, pad=0.04)

    fa_tgt_brain = tgt_fa[bmask_bool]
    fa_pred_brain = pred_fa[bmask_bool]
    axes[0, 3].scatter(fa_tgt_brain, fa_pred_brain, s=1, alpha=0.3)
    axes[0, 3].plot([0, 1], [0, 1], "r--", lw=1)
    fa_metrics = scalar_map_metrics(tgt_fa, pred_fa, mask=bmask_bool)
    axes[0, 3].set_title(f"FA RMSE={fa_metrics['rmse']:.4f} R2={fa_metrics['r2']:.3f}")
    axes[0, 3].set_xlabel("Target")
    axes[0, 3].set_ylabel("Predicted")
    axes[0, 3].set_aspect("equal")

    adc_brain = tgt_adc[bmask_bool]
    adc_hi = max(float(np.percentile(adc_brain, 99)), 1e-6) if adc_brain.size else 1e-6
    axes[1, 0].imshow(np.rot90(tgt_adc * bmask), cmap="magma", vmin=0, vmax=adc_hi)
    axes[1, 0].set_title("Target ADC")
    axes[1, 1].imshow(np.rot90(pred_adc * bmask), cmap="magma", vmin=0, vmax=adc_hi)
    axes[1, 1].set_title("Predicted ADC")
    adc_abs = max(float(np.max(np.abs(adc_diff))), 1e-6)
    im_adc = axes[1, 2].imshow(np.rot90(adc_diff), cmap="bwr", vmin=-adc_abs, vmax=adc_abs)
    axes[1, 2].set_title("ADC error")
    fig.colorbar(im_adc, ax=axes[1, 2], fraction=0.046, pad=0.04)

    adc_pred_brain = pred_adc[bmask_bool]
    axes[1, 3].scatter(adc_brain, adc_pred_brain, s=1, alpha=0.3)
    axes[1, 3].plot([0, adc_hi], [0, adc_hi], "r--", lw=1)
    adc_metrics = scalar_map_metrics(tgt_adc, pred_adc, mask=bmask_bool)
    axes[1, 3].set_title(f"ADC RMSE={adc_metrics['rmse']:.2e} R2={adc_metrics['r2']:.3f}")
    axes[1, 3].set_xlabel("Target")
    axes[1, 3].set_ylabel("Predicted")
    axes[1, 3].set_aspect("equal")

    for ax in axes.ravel():
        if ax not in [axes[0, 3], axes[1, 3]]:
            ax.axis("off")

    fig.tight_layout()
    return fig


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DTILoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_brain_mask: bool = True,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
    channels_last: bool = False,
) -> dict[str, float]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = torch.zeros((), device=device)
    total_tensor = torch.zeros((), device=device)
    total_fa = torch.zeros((), device=device)
    total_md = torch.zeros((), device=device)
    n_batches = 0
    non_blocking = device.type == "cuda"

    def metric_tensor(metrics: dict, key: str) -> torch.Tensor:
        value = metrics.get(key)
        if value is None:
            return torch.zeros((), device=device)
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(float(value), device=device)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            signal = batch["input"].to(device, non_blocking=non_blocking)

            if "degrade_kf" in batch:
                degrade_kf = batch["degrade_kf"].to(device, non_blocking=non_blocking)
                degrade_nl = batch["degrade_nl"].to(device, non_blocking=non_blocking)
                b0_mask = batch["b0_mask"].to(device, non_blocking=non_blocking)
                signal = gpu_degrade_dwi_batch(signal, degrade_kf, degrade_nl)
                signal = gpu_b0_normalize_batch(signal, b0_mask)

            signal = maybe_channels_last(signal, channels_last)
            target = batch["target"].to(device, non_blocking=non_blocking)
            target = maybe_channels_last(target, channels_last)
            bvals = batch["bvals"].to(device, non_blocking=non_blocking)
            bvecs = batch["bvecs"].to(device, non_blocking=non_blocking)
            vol_mask = batch["vol_mask"].to(device, non_blocking=non_blocking)
            brain_mask = (
                batch["brain_mask"].to(device, non_blocking=non_blocking)
                if use_brain_mask
                else None
            )

            with autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                pred = model(signal, bvals, bvecs, vol_mask)
            loss, metrics = criterion(
                pred.float(),
                target,
                mask=brain_mask,
                return_tensor_metrics=True,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                    optimizer.step()

            total_loss += loss.detach()
            total_tensor += metric_tensor(metrics, "tensor_mse")
            total_fa += metric_tensor(metrics, "fa_mae")
            total_md += metric_tensor(metrics, "md_mae")
            n_batches += 1

    n = max(n_batches, 1)

    def mean_float(value: torch.Tensor) -> float:
        return float((value / n).detach().cpu())

    return {
        "loss": mean_float(total_loss),
        "tensor_mse": mean_float(total_tensor),
        "fa_mae": mean_float(total_fa),
        "md_mae": mean_float(total_md),
    }


def build_run_config(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    zarr_path: str,
    device: torch.device,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    train_ds: DWISliceDataset,
    val_ds: DWISliceDataset,
    num_workers: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    fused_adamw: bool,
    n_params: int,
    is_compiled: bool,
) -> dict[str, object]:
    return {
        "out_dir": str(out_dir),
        "zarr_path": zarr_path,
        "device": str(device),
        "seed": args.seed,
        "max_n": train_ds.max_n,
        "max_bval": train_ds.max_bval,
        "dti_scale": train_ds.dti_scale,
        "canonical_hw": list(train_ds.canonical_hw),
        "feat_dim": args.feat_dim,
        "channels": list(args.channels),
        "cholesky": args.cholesky,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_scalar": args.lambda_scalar,
        "lambda_edge": args.lambda_edge,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "use_brain_mask": not args.no_brain_mask,
        "amp": amp_enabled,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "channels_last": channels_last,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "compile_enabled": is_compiled,
        "num_workers": num_workers,
        "prefetch_factor": args.prefetch_factor if num_workers > 0 else None,
        "fused_adamw": fused_adamw,
        "n_params": n_params,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }


def log_scalars(
    writer: SummaryWriter,
    prefix: str,
    metrics: dict[str, float],
    epoch: int,
) -> None:
    writer.add_scalar(f"{prefix}/loss", metrics["loss"], epoch)
    writer.add_scalar(f"{prefix}/tensor_mse", metrics["tensor_mse"], epoch)
    writer.add_scalar(f"{prefix}/fa_mae", metrics["fa_mae"], epoch)
    writer.add_scalar(f"{prefix}/md_mae", metrics["md_mae"], epoch)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    raw_model: QSpaceUNet,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    args: argparse.Namespace,
    train_ds: DWISliceDataset,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    use_brain_mask: bool,
    channels_last: bool,
    run_config: dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "max_n": train_ds.max_n,
            "feat_dim": args.feat_dim,
            "channels": list(args.channels),
            "cholesky": args.cholesky,
            "dti_scale": train_ds.dti_scale,
            "max_bval": train_ds.max_bval,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "use_brain_mask": use_brain_mask,
            "amp_dtype": args.amp_dtype,
            "channels_last": channels_last,
            "run_config": run_config,
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    import zarr

    seed_everything(args.seed)

    out_dir = resolve_project_path(args.out_dir)
    zarr_path = path_str(args.zarr_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log.info("Device: %s", device)
    require_cuda_if_requested(device, args.require_cuda)
    configure_torch_runtime(device, deterministic=args.deterministic)
    amp_dtype = amp_dtype_from_name(device, args.amp_dtype)
    amp_enabled = bool(args.amp and amp_dtype is not None)
    channels_last = bool(args.channels_last and device.type == "cuda")
    num_workers = default_num_workers(args.num_workers)
    non_blocking_cuda = device.type == "cuda"

    log.info("AMP: %s", str(amp_dtype).replace("torch.", "") if amp_enabled else "disabled")
    if channels_last:
        log.info("Memory format: channels_last")

    store = zarr.open_group(zarr_path, mode="r")
    all_keys = sorted(store.keys())
    log.info("Found %d entries in %s", len(all_keys), zarr_path)

    val_bio = args.val_subjects if args.val_subjects is not None else cfg.VAL_SUBJECTS
    test_bio = args.test_subjects if args.test_subjects is not None else cfg.TEST_SUBJECTS
    train_subjects, val_subjects, test_subjects = split_subjects(all_keys, val_bio, test_bio)
    log.info(
        "Train: %d Val: %d Test: %d",
        len(train_subjects),
        len(val_subjects),
        len(test_subjects),
    )

    use_brain_mask = not args.no_brain_mask
    train_ds = DWISliceDataset(
        zarr_path,
        train_subjects,
        augment=True,
        use_brain_mask=use_brain_mask,
        keep_fraction_range=(args.keep_fraction_min, args.keep_fraction_max),
        noise_range=(args.noise_min, args.noise_max),
        random_axis=args.random_slice_axis,
        slice_axes=tuple(args.slice_axes),
        aug_flip=args.aug_flip,
        aug_intensity=args.aug_intensity,
        aug_volume_dropout=args.aug_volume_dropout,
        gpu_degrade=device.type == "cuda",
    )
    val_ds = DWISliceDataset(
        zarr_path,
        val_subjects,
        augment=False,
        use_brain_mask=use_brain_mask,
        random_axis=False,
        eval_mode=True,
        eval_keep_fraction=args.eval_keep_fraction,
        eval_noise_level=args.eval_noise_level,
        eval_seed=args.eval_seed,
    )

    global_max_n = max(train_ds.max_n, val_ds.max_n)
    train_ds.max_n = global_max_n
    val_ds.max_n = global_max_n

    canonical_hw = (
        max(train_ds.canonical_hw[0], val_ds.canonical_hw[0]),
        max(train_ds.canonical_hw[1], val_ds.canonical_hw[1]),
    )
    train_ds.canonical_hw = canonical_hw
    val_ds.canonical_hw = canonical_hw
    val_ds.max_bval = train_ds.max_bval
    val_ds.dti_scale = train_ds.dti_scale

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": non_blocking_cuda,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": dwi_worker_init if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    log.info(
        "DataLoader: batch_size=%d workers=%d pin_memory=%s gpu_degrade=%s",
        args.batch_size,
        num_workers,
        non_blocking_cuda,
        train_ds.gpu_degrade,
    )
    log.info(
        "Train slices: %d Val slices: %d max_n: %d max_bval: %.0f dti_scale: %.4f",
        len(train_ds),
        len(val_ds),
        global_max_n,
        train_ds.max_bval,
        train_ds.dti_scale,
    )

    raw_model = QSpaceUNet(
        max_n=global_max_n,
        feat_dim=args.feat_dim,
        channels=tuple(args.channels),
        cholesky=args.cholesky,
        dropout=args.dropout,
    ).to(device)
    if channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)
    model, is_compiled = maybe_compile_model(
        raw_model,
        setting=args.compile,
        device=device,
        mode=args.compile_mode,
    )
    log.info("torch.compile: %s", "enabled" if is_compiled else "disabled")

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    criterion = DTILoss(lambda_scalar=args.lambda_scalar, lambda_edge=args.lambda_edge).to(device)
    fused_adamw = bool(args.fused_adamw and device.type == "cuda")
    try:
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
            fused=fused_adamw,
        )
    except TypeError:
        if fused_adamw:
            log.warning("Fused AdamW unavailable in this PyTorch build; using regular AdamW")
        fused_adamw = False
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
        )
    scaler = make_grad_scaler(device, enabled=amp_enabled, dtype=amp_dtype)

    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - warmup_epochs, 1),
        eta_min=args.lr * 0.01,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    run_config = build_run_config(
        args,
        out_dir=out_dir,
        zarr_path=zarr_path,
        device=device,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_ds=train_ds,
        val_ds=val_ds,
        num_workers=num_workers,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        fused_adamw=fused_adamw,
        n_params=n_params,
        is_compiled=is_compiled,
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    baseline_csvs = [
        out_dir / "metrics_patch2self.csv",
        out_dir / "metrics_mppca.csv",
    ]
    baselines = load_baseline_metrics(baseline_csvs)

    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is required for production training. "
            "Install dependencies with `pip install -r requirements.txt`."
        )
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    writer.add_text("config/json", f"```json\n{json.dumps(run_config, indent=2)}\n```", 0)
    for name, metrics in baselines.items():
        for metric, value in metrics.items():
            writer.add_scalar(f"baseline/{name}/{metric}", value, 0)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, object]] = []
    vis_slice_idx = -1
    completed = False

    log.info("Starting training for %d epochs (patience=%d)", args.epochs, args.patience)
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                use_brain_mask=use_brain_mask,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                channels_last=channels_last,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                use_brain_mask=use_brain_mask,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_tensor_mse": train_metrics["tensor_mse"],
                "train_fa_mae": train_metrics["fa_mae"],
                "train_md_mae": train_metrics["md_mae"],
                "val_loss": val_metrics["loss"],
                "val_tensor_mse": val_metrics["tensor_mse"],
                "val_fa_mae": val_metrics["fa_mae"],
                "val_md_mae": val_metrics["md_mae"],
                "elapsed_s": round(elapsed, 1),
            }
            history.append(record)

            log_scalars(writer, "train", train_metrics, epoch)
            log_scalars(writer, "val", val_metrics, epoch)
            writer.add_scalar("train/learning_rate", lr, epoch)
            writer.add_scalar("train/epoch_time_s", elapsed, epoch)

            if epoch % args.vis_every == 0 or epoch == 1:
                fig = make_val_figure(
                    model,
                    val_ds,
                    device,
                    dti_scale=train_ds.dti_scale,
                    slice_idx=vis_slice_idx,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                )
                writer.add_figure("val/prediction", fig, epoch)
                plt.close(fig)

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    out_dir / "best_model.pt",
                    epoch=epoch,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    val_loss=best_val_loss,
                    args=args,
                    train_ds=train_ds,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    test_subjects=test_subjects,
                    use_brain_mask=use_brain_mask,
                    channels_last=channels_last,
                    run_config=run_config,
                )
            else:
                patience_counter += 1

            marker = "*" if improved else ""
            log.info(
                "Epoch %3d/%d train=%.6f val=%.6f t_mse=%.6f fa=%.4f md=%.6f lr=%.2e %.1fs %s",
                epoch,
                args.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["tensor_mse"],
                val_metrics["fa_mae"],
                val_metrics["md_mae"],
                lr,
                elapsed,
                marker,
            )

            if patience_counter >= args.patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(
            out_dir / "last_model.pt",
            epoch=history[-1]["epoch"],
            raw_model=raw_model,
            optimizer=optimizer,
            val_loss=history[-1]["val_loss"],
            args=args,
            train_ds=train_ds,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            use_brain_mask=use_brain_mask,
            channels_last=channels_last,
            run_config=run_config,
        )

        writer.add_scalar("summary/best_val_loss", best_val_loss, best_epoch)
        writer.add_scalar("summary/final_val_loss", history[-1]["val_loss"], history[-1]["epoch"])
        completed = True
        log.info("Done. Best val loss: %.6f at epoch %d. Saved to %s", best_val_loss, best_epoch, out_dir)
    finally:
        writer.flush()
        writer.close()
        if not completed:
            log.info("Training stopped before normal completion; partial outputs remain in %s", out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train QSpaceUNet for DWI -> DTI prediction")

    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default=cfg.TRAIN_OUT_DIR)
    parser.add_argument("--test_subjects", nargs="*", default=None)
    parser.add_argument("--val_subjects", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)

    parser.add_argument("--feat_dim", type=int, default=cfg.FEAT_DIM)
    parser.add_argument("--channels", type=int, nargs="+", default=cfg.UNET_CHANNELS)
    parser.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    parser.add_argument("--cholesky", action="store_true")

    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--lambda_scalar", type=float, default=cfg.LAMBDA_SCALAR)
    parser.add_argument("--lambda_edge", type=float, default=cfg.LAMBDA_EDGE)
    parser.add_argument("--warmup_epochs", type=int, default=cfg.WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--vis_every", type=int, default=cfg.VIS_EVERY)

    parser.add_argument("--keep_fraction_min", type=float, default=cfg.KEEP_FRACTION_MIN)
    parser.add_argument("--keep_fraction_max", type=float, default=cfg.KEEP_FRACTION_MAX)
    parser.add_argument("--noise_min", type=float, default=cfg.NOISE_MIN)
    parser.add_argument("--noise_max", type=float, default=cfg.NOISE_MAX)
    parser.add_argument("--eval_keep_fraction", type=float, default=cfg.EVAL_KEEP_FRACTION)
    parser.add_argument("--eval_noise_level", type=float, default=cfg.EVAL_NOISE_LEVEL)
    parser.add_argument("--eval_seed", type=int, default=cfg.EVAL_DEGRADE_SEED)
    parser.add_argument("--slice_axes", type=int, nargs="+", default=list(cfg.SLICE_AXES))
    parser.add_argument("--random_slice_axis", action=argparse.BooleanOptionalAction, default=cfg.RANDOM_SLICE_AXIS)
    parser.add_argument("--aug_flip", action=argparse.BooleanOptionalAction, default=cfg.AUG_FLIP)
    parser.add_argument("--aug_intensity", type=float, default=cfg.AUG_INTENSITY)
    parser.add_argument("--aug_volume_dropout", type=float, default=cfg.AUG_VOLUME_DROPOUT)

    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=cfg.PREFETCH_FACTOR)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.AMP)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default=cfg.AMP_DTYPE)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=cfg.CHANNELS_LAST)
    parser.add_argument("--compile", choices=["off", "auto", "on"], default=cfg.COMPILE)
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default=cfg.COMPILE_MODE)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=cfg.DETERMINISTIC)
    parser.add_argument("--fused_adamw", action=argparse.BooleanOptionalAction, default=cfg.FUSED_ADAMW)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=cfg.REQUIRE_CUDA)
    parser.add_argument("--no_brain_mask", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
