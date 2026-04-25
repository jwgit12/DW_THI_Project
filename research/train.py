"""Training script for the QSpaceUNet DTI prediction model.

Usage:
    python -m research.train --zarr_path dataset/default_dataset.zarr --cholesky --out_dir research/runs/run_small
    python -m research.train --zarr_path dataset/pretext_dataset_new.zarr --epochs 200 --batch_size 8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    import wandb
except ImportError:
    wandb = None

from research.utils import dti6d_to_scalar_maps, scalar_map_metrics
from research.dataset import DWISliceDataset, dwi_worker_init
from research.augment import gpu_degrade_dwi_batch, gpu_b0_normalize_batch
from research.loss import DTILoss
from research.model import QSpaceUNet
from research.runtime import (
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
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# ── Default subject split (biological subject IDs) ──────────────────────────
# 11 biological subjects → 7 train / 2 val / 2 test
# All sessions of a subject stay in the same split to prevent data leakage.
DEFAULT_TEST_SUBJECTS = cfg.TEST_SUBJECTS
DEFAULT_VAL_SUBJECTS = cfg.VAL_SUBJECTS


def load_baseline_metrics(csv_paths: list[Path]) -> dict[str, dict[str, float]]:
    """Load baseline CSV files and return {name: {metric: mean_value}}."""
    baselines = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        # The MEAN row has subject == "MEAN"
        mean_row = df[df["subject"] == "MEAN"]
        if mean_row.empty:
            continue
        name = csv_path.stem.replace("metrics_", "")  # e.g. "patch2self" or "mppca"
        metrics = {}
        for col in ["fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
                     "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"]:
            if col in mean_row.columns:
                val = mean_row[col].values[0]
                if pd.notna(val):
                    metrics[col] = float(val)
        baselines[name] = metrics
    return baselines


def flatten_baseline_metrics(baselines: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten baseline metrics into W&B-safe metric names."""
    flat_metrics = {}
    for name, metrics in baselines.items():
        safe_name = name.replace("-", "_")
        for metric, value in metrics.items():
            flat_metrics[f"baseline_{safe_name}_{metric}"] = value
    return flat_metrics


def build_wandb_config(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    zarr_path: str,
    device: torch.device,
    global_max_n: int,
    global_max_bval: float,
    global_dti_scale: float,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    train_slices: int,
    val_slices: int,
    use_brain_mask: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    num_workers: int,
    fused_adamw: bool,
    n_params: int,
    is_compiled: bool,
) -> dict[str, object]:
    return {
        "out_dir": str(out_dir),
        "zarr_path": zarr_path,
        "device": str(device),
        "max_n": global_max_n,
        "max_bval": global_max_bval,
        "dti_scale": global_dti_scale,
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
        "vis_every": args.vis_every,
        "use_brain_mask": use_brain_mask,
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
        "train_slices": train_slices,
        "val_slices": val_slices,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }


def init_wandb_run(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    config: dict[str, object],
    baseline_metrics: dict[str, float],
):
    if wandb is None:
        raise RuntimeError(
            "wandb is required for training tracking but is not installed. "
            "Install dependencies from requirements.txt before running research.train."
        )

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name or out_dir.name,
        "job_type": "train",
        "dir": str(out_dir),
        "config": config,
        "mode": args.wandb_mode,
    }
    try:
        run = wandb.init(**init_kwargs)
    except wandb.errors.UsageError as exc:
        if args.wandb_mode != "online":
            raise
        log.warning(
            "wandb online init failed (%s). Falling back to offline mode; run `wandb login` "
            "to enable cloud sync.",
            exc,
        )
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "offline"
        run = wandb.init(**fallback_kwargs)

    run.define_metric("epoch")
    tracked_metrics = [
        "train_loss",
        "val_loss",
        "train_tensor_mse",
        "val_tensor_mse",
        "train_fa_mae",
        "val_fa_mae",
        "train_md_mae",
        "val_md_mae",
        "learning_rate",
        "epoch_time_s",
        *baseline_metrics.keys(),
    ]
    for metric_name in tracked_metrics:
        run.define_metric(metric_name, step_metric="epoch")
    run.define_metric("val_loss", step_metric="epoch", summary="min")
    run.define_metric("val_tensor_mse", step_metric="epoch", summary="min")
    run.define_metric("val_fa_mae", step_metric="epoch", summary="min")
    run.define_metric("val_md_mae", step_metric="epoch", summary="min")
    return run


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
    """Generate a prediction vs target figure for one validation slice."""
    if slice_idx is None:
        slice_idx = len(val_ds) // 2

    sample = val_ds[slice_idx]
    non_blocking = device.type == "cuda"
    signal = sample["input"].unsqueeze(0).to(device, non_blocking=non_blocking)
    signal = maybe_channels_last(signal, channels_last)
    bvals = sample["bvals"].unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs = sample["bvecs"].unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask = sample["vol_mask"].unsqueeze(0).to(device, non_blocking=non_blocking)
    target = sample["target"].numpy()  # (6, H, W)
    bmask = sample["brain_mask"].numpy()  # (H, W) float32

    model.eval()
    with torch.inference_mode(), autocast_context(
        device, enabled=amp_enabled, dtype=amp_dtype,
    ):
        pred = model(signal, bvals, bvecs, vol_mask)  # (1, 6, H, W)
    pred_np = pred[0].float().cpu().numpy()  # (6, H, W)

    # Unscale to physical units before computing FA / ADC
    pred_np = pred_np / dti_scale
    target = target / dti_scale

    # Compute FA / ADC from 6-channel tensors → need (X, Y, Z, 6) shape
    pred_vol = pred_np.transpose(1, 2, 0)[..., np.newaxis, :]  # (H, W, 1, 6)
    tgt_vol = target.transpose(1, 2, 0)[..., np.newaxis, :]

    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_vol)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(tgt_vol)

    # Squeeze Z dim
    pred_fa = pred_fa[:, :, 0]
    pred_adc = pred_adc[:, :, 0]
    tgt_fa = tgt_fa[:, :, 0]
    tgt_adc = tgt_adc[:, :, 0]

    bmask_bool = bmask > 0.5

    fa_diff = (tgt_fa - pred_fa) * bmask
    adc_diff = (tgt_adc - pred_adc) * bmask

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: FA
    axes[0, 0].imshow(np.rot90(tgt_fa * bmask), cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("Target FA")
    axes[0, 1].imshow(np.rot90(pred_fa * bmask), cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted FA")
    fa_abs = max(float(np.max(np.abs(fa_diff))), 1e-6)
    im_fa = axes[0, 2].imshow(np.rot90(fa_diff), cmap="bwr", vmin=-fa_abs, vmax=fa_abs)
    axes[0, 2].set_title("FA Error (tgt - pred)")
    fig.colorbar(im_fa, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # FA scatter (brain voxels only)
    fa_tgt_brain = tgt_fa[bmask_bool]
    fa_pred_brain = pred_fa[bmask_bool]
    axes[0, 3].scatter(fa_tgt_brain, fa_pred_brain, s=1, alpha=0.3)
    axes[0, 3].plot([0, 1], [0, 1], "r--", lw=1)
    m = scalar_map_metrics(tgt_fa, pred_fa, mask=bmask_bool)
    axes[0, 3].set_title(f"FA: RMSE={m['rmse']:.4f}  R²={m['r2']:.3f}")
    axes[0, 3].set_xlabel("Target")
    axes[0, 3].set_ylabel("Predicted")
    axes[0, 3].set_aspect("equal")

    # Row 2: ADC
    adc_brain = tgt_adc[bmask_bool]
    adc_lo, adc_hi = 0, max(float(np.percentile(adc_brain, 99)), 1e-6) if adc_brain.size > 0 else 1e-6
    axes[1, 0].imshow(np.rot90(tgt_adc * bmask), cmap="magma", vmin=adc_lo, vmax=adc_hi)
    axes[1, 0].set_title("Target ADC")
    axes[1, 1].imshow(np.rot90(pred_adc * bmask), cmap="magma", vmin=adc_lo, vmax=adc_hi)
    axes[1, 1].set_title("Predicted ADC")
    adc_abs = max(float(np.max(np.abs(adc_diff))), 1e-6)
    im_adc = axes[1, 2].imshow(np.rot90(adc_diff), cmap="bwr", vmin=-adc_abs, vmax=adc_abs)
    axes[1, 2].set_title("ADC Error (tgt - pred)")
    fig.colorbar(im_adc, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # ADC scatter (brain voxels only)
    adc_pred_brain = pred_adc[bmask_bool]
    axes[1, 3].scatter(adc_brain, adc_pred_brain, s=1, alpha=0.3)
    lim = adc_hi
    axes[1, 3].plot([0, lim], [0, lim], "r--", lw=1)
    m = scalar_map_metrics(tgt_adc, pred_adc, mask=bmask_bool)
    axes[1, 3].set_title(f"ADC: RMSE={m['rmse']:.2e}  R²={m['r2']:.3f}")
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
    """Run one train or validation epoch. Pass optimizer=None for val."""
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

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch in loader:
            signal = batch["input"].to(device, non_blocking=non_blocking)

            # GPU degradation path: apply cuFFT k-space cutout + noise + b0
            # normalization on the full batch in one shot (~50x faster than the
            # per-sample CPU scipy path that runs inside DataLoader workers).
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
                if use_brain_mask else None
            )

            with autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                pred = model(signal, bvals, bvecs, vol_mask)
            loss, metrics = criterion(
                pred.float(), target, mask=brain_mask, return_tensor_metrics=True,
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


def main(args):
    wandb_run = None
    completed = False

    try:
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
        if amp_enabled:
            log.info("AMP: enabled (%s)", str(amp_dtype).replace("torch.", ""))
        else:
            log.info("AMP: disabled")
        if channels_last:
            log.info("Memory format: channels_last")

        # ── Load baseline metrics for flat reference lines ──────────────────────
        # Look for baseline CSVs produced by research.evaluate in the output dir.
        baseline_csvs = [
            out_dir / "metrics_patch2self.csv",
            out_dir / "metrics_mppca.csv",
        ]
        baselines = load_baseline_metrics(baseline_csvs)
        baseline_log_metrics = flatten_baseline_metrics(baselines)
        if baselines:
            log.info("Loaded baseline references: %s", list(baselines.keys()))

        # ── Subject split (by biological subject to prevent leakage) ────────────
        import zarr

        store = zarr.open_group(zarr_path, mode="r")
        all_keys = sorted(store.keys())
        log.info("Found %d entries in %s", len(all_keys), zarr_path)

        test_bio = args.test_subjects or DEFAULT_TEST_SUBJECTS
        val_bio = args.val_subjects or DEFAULT_VAL_SUBJECTS

        train_subjects, val_subjects, test_subjects = [], [], []
        for key in all_keys:
            bio_subject = key.rsplit("_ses-", 1)[0]
            if bio_subject in test_bio:
                test_subjects.append(key)
            elif bio_subject in val_bio:
                val_subjects.append(key)
            else:
                train_subjects.append(key)

        log.info("Train: %d  Val: %d  Test: %d (from %d/%d/%d biological subjects)",
                 len(train_subjects), len(val_subjects), len(test_subjects),
                 len({k.rsplit("_ses-", 1)[0] for k in train_subjects}),
                 len({k.rsplit("_ses-", 1)[0] for k in val_subjects}),
                 len({k.rsplit("_ses-", 1)[0] for k in test_subjects}))

        # ── Datasets & loaders ──────────────────────────────────────────────────
        use_brain_mask = not args.no_brain_mask
        train_ds = DWISliceDataset(
            zarr_path, train_subjects,
            augment=True,
            use_brain_mask=use_brain_mask,
            random_axis=cfg.RANDOM_SLICE_AXIS,
            slice_axes=cfg.SLICE_AXES,
            gpu_degrade=device.type == "cuda",
        )
        # Validation uses axial-only slicing + deterministic degradation so the
        # reported val loss is comparable across epochs.
        val_ds = DWISliceDataset(
            zarr_path, val_subjects,
            augment=False,
            use_brain_mask=use_brain_mask,
            random_axis=False,
            eval_mode=True,
        )

        # Derive all normalisation constants from training data only to prevent
        # information leakage from val/test into training.
        # max_n must accommodate the largest subject across all splits (structural
        # padding requirement), but max_bval, dti_scale and canonical_hw are true
        # normalisation scales and must come from training data exclusively.
        global_max_n = max(train_ds.max_n, val_ds.max_n)
        train_ds.max_n = global_max_n
        val_ds.max_n = global_max_n

        # Keep explicit local names for logging and checkpoint metadata.
        global_max_bval = train_ds.max_bval
        global_dti_scale = train_ds.dti_scale
        canonical_hw = (
            max(train_ds.canonical_hw[0], val_ds.canonical_hw[0]),
            max(train_ds.canonical_hw[1], val_ds.canonical_hw[1]),
        )
        train_ds.canonical_hw = canonical_hw
        val_ds.canonical_hw = canonical_hw

        val_ds.max_bval = global_max_bval
        val_ds.dti_scale = global_dti_scale

        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": num_workers > 0,
            # Each worker preloads all subject data into its own RAM on startup,
            # eliminating zarr I/O overhead during training (~25x faster per sample).
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
            "DataLoader: batch_size=%d workers=%d prefetch=%s pin_memory=%s "
            "preload_in_workers=%s gpu_degrade=%s",
            args.batch_size,
            num_workers,
            args.prefetch_factor if num_workers > 0 else "off",
            device.type == "cuda",
            num_workers > 0,
            train_ds.gpu_degrade,
        )

        log.info("Train slices: %d  Val slices: %d  max_n: %d  max_bval: %.0f  dti_scale: %.4f",
                 len(train_ds), len(val_ds), global_max_n, global_max_bval, global_dti_scale)

        # ── Model ───────────────────────────────────────────────────────────────
        raw_model = QSpaceUNet(
            max_n=global_max_n,
            feat_dim=args.feat_dim,
            channels=tuple(args.channels),
            cholesky=args.cholesky,
        ).to(device)
        if channels_last:
            raw_model = raw_model.to(memory_format=torch.channels_last)
        model, is_compiled = maybe_compile_model(
            raw_model,
            setting=args.compile,
            device=device,
            mode=args.compile_mode,
        )
        if is_compiled:
            log.info("torch.compile: enabled (mode=%s)", args.compile_mode)
        else:
            log.info("torch.compile: disabled")

        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        log.info("Model parameters: %s", f"{n_params:,}")

        # ── Optimiser & scheduler ───────────────────────────────────────────────
        criterion = DTILoss(
            lambda_scalar=args.lambda_scalar,
            lambda_edge=args.lambda_edge,
        ).to(device)
        fused_adamw = args.fused_adamw and device.type == "cuda"
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
                log.warning("Fused AdamW is unavailable in this PyTorch build; using regular AdamW")
            fused_adamw = False
            optimizer = torch.optim.AdamW(
                raw_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.99),
            )
        log.info("Optimizer: AdamW%s", " (fused)" if fused_adamw else "")
        scaler = make_grad_scaler(device, enabled=amp_enabled, dtype=amp_dtype)

        wandb_config = build_wandb_config(
            args,
            out_dir=out_dir,
            zarr_path=zarr_path,
            device=device,
            global_max_n=global_max_n,
            global_max_bval=global_max_bval,
            global_dti_scale=global_dti_scale,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            train_slices=len(train_ds),
            val_slices=len(val_ds),
            use_brain_mask=use_brain_mask,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            num_workers=num_workers,
            fused_adamw=fused_adamw,
            n_params=n_params,
            is_compiled=is_compiled,
        )
        wandb_run = init_wandb_run(
            args,
            out_dir=out_dir,
            config=wandb_config,
            baseline_metrics=baseline_log_metrics,
        )
        log.info(
            "W&B tracking enabled for run '%s'; GPU/system utilization is collected automatically",
            wandb_run.name,
        )

        # Linear warmup -> cosine annealing. Warmup stabilises early epochs when
        # the encoder embeddings are still random and gradients can spike.
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
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine

        # ── Pick a fixed validation slice for visualisation ───────────────────
        vis_slice_idx = -1

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history: list[dict] = []

        log.info("Starting training for %d epochs (patience=%d)", args.epochs, args.patience)

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

            epoch_log = {
                "epoch": epoch,
                "learning_rate": lr,
                "epoch_time_s": elapsed,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_tensor_mse": train_metrics["tensor_mse"],
                "val_tensor_mse": val_metrics["tensor_mse"],
                "train_fa_mae": train_metrics["fa_mae"],
                "val_fa_mae": val_metrics["fa_mae"],
                "train_md_mae": train_metrics["md_mae"],
                "val_md_mae": val_metrics["md_mae"],
            }
            epoch_log.update(baseline_log_metrics)

            # ── W&B: validation visualisation ────────────────────────────────
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
                epoch_log["val_prediction"] = wandb.Image(fig, caption=f"Epoch {epoch}")
                plt.close(fig)

            wandb_run.log(epoch_log)

            # ── Checkpoint ───────────────────────────────────────────────────
            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "max_n": global_max_n,
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
                    },
                    out_dir / "best_model.pt",
                )
                wandb_run.summary["best_epoch"] = epoch
                wandb_run.summary["best_val_loss"] = best_val_loss
                wandb_run.summary["best_model_path"] = str(out_dir / "best_model.pt")
            else:
                patience_counter += 1

            marker = "*" if improved else ""
            log.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  "
                "t_mse=%.6f  fa=%.4f  md=%.6f  "
                "lr=%.2e  %.1fs %s",
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

        # ── Save training history ─────────────────────────────────────────────
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save final model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "max_n": global_max_n,
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
            },
            out_dir / "last_model.pt",
        )

        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["final_epoch"] = epoch
        wandb_run.summary["final_val_loss"] = val_metrics["loss"]
        wandb_run.summary["history_path"] = str(out_dir / "history.json")
        wandb_run.summary["last_model_path"] = str(out_dir / "last_model.pt")

        completed = True
        log.info("Done. Best val loss: %.6f  Saved to %s", best_val_loss, out_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QSpaceUNet for DWI -> DTI prediction")

    # Data
    parser.add_argument("--zarr_path", default="dataset/pretext_dataset_new.zarr")
    parser.add_argument("--out_dir", default="research/runs/run_01")
    parser.add_argument("--test_subjects", nargs="*", default=None,
                        help="Biological subject IDs for test (default: sub-10 sub-11)")
    parser.add_argument("--val_subjects", nargs="*", default=None,
                        help="Biological subject IDs for validation (default: sub-08 sub-09)")
    parser.add_argument("--wandb_project", "--wandb-project", default="DW_THI_Project",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", "--wandb-entity", default=None,
                        help="Optional Weights & Biases team/user entity")
    parser.add_argument("--wandb_name", "--wandb-name", default=None,
                        help="Optional Weights & Biases run name (defaults to out_dir name)")
    parser.add_argument("--wandb_mode", "--wandb-mode",
                        choices=["online", "offline", "disabled"], default="online",
                        help="Weights & Biases sync mode")

    # Model
    parser.add_argument("--feat_dim", type=int, default=cfg.FEAT_DIM)
    parser.add_argument("--channels", type=int, nargs="+", default=cfg.UNET_CHANNELS)
    parser.add_argument("--cholesky", action="store_true",
                        help="Use Cholesky parameterization to guarantee positive semi-definite tensors")

    # Training
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--lambda_scalar", type=float, default=cfg.LAMBDA_SCALAR,
                        help="Weight for FA/MD auxiliary loss (0 = tensor term only)")
    parser.add_argument("--lambda_edge", type=float, default=cfg.LAMBDA_EDGE,
                        help="Weight for FA spatial-gradient (edge) loss (0 disables)")
    parser.add_argument("--warmup_epochs", type=int, default=cfg.WARMUP_EPOCHS,
                        help="Linear LR warmup length before cosine annealing")
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--vis_every", type=int, default=1,
                        help="Generate validation visualisation every N epochs (default: 1)")
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="DataLoader worker processes (-1 = OS-aware auto)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="Batches prefetched per DataLoader worker")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=True,
                           help="Enable CUDA automatic mixed precision (default)")
    amp_group.add_argument("--no_amp", "--no-amp", dest="amp", action="store_false",
                           help="Disable CUDA automatic mixed precision")
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                        help="AMP dtype; auto prefers bf16 on RTX 40-series")
    parser.add_argument("--bf16", dest="amp_dtype", action="store_const", const="bf16",
                        help="Shortcut for --amp --amp_dtype bf16")
    parser.add_argument("--fp16", dest="amp_dtype", action="store_const", const="fp16",
                        help="Shortcut for --amp --amp_dtype fp16")
    channels_group = parser.add_mutually_exclusive_group()
    channels_group.add_argument("--channels_last", "--channels-last",
                                dest="channels_last", action="store_true", default=True,
                                help="Use channels-last convolution layout on CUDA (default)")
    channels_group.add_argument("--no_channels_last", "--no-channels-last",
                                dest="channels_last", action="store_false",
                                help="Disable channels-last memory format")
    parser.add_argument("--compile", choices=["off", "auto", "on"], default="auto",
                        help="Use torch.compile; auto enables it on CUDA/Linux")
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"],
                        default="max-autotune", help="torch.compile mode")
    parser.add_argument("--deterministic", action="store_true",
                        help="Prefer deterministic CUDA kernels over fastest cuDNN autotuning")
    fused_group = parser.add_mutually_exclusive_group()
    fused_group.add_argument("--fused_adamw", "--fused-adamw",
                             dest="fused_adamw", action="store_true", default=True,
                             help="Use fused CUDA AdamW when available (default)")
    fused_group.add_argument("--no_fused_adamw", "--no-fused-adamw",
                             dest="fused_adamw", action="store_false",
                             help="Disable fused CUDA AdamW")
    parser.add_argument("--require_cuda", "--require-cuda", action="store_true",
                        help="Fail fast when a CUDA PyTorch build/GPU is not available")
    parser.add_argument("--no_brain_mask", action="store_true",
                        help="Train and validate losses over the full image instead of brain-mask voxels only")

    main(parser.parse_args())
