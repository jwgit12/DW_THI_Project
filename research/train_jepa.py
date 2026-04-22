"""Training script for the JEPA-augmented QSpaceUNet.

Adds a self-distillation (JEPA) auxiliary loss on top of the regular
DTI supervised objective. Checkpoints are saved in pure ``QSpaceUNet``
format so ``research.evaluate`` can load them without any changes.

Usage:
    python -m research.train_jepa --zarr_path dataset/default_clean.zarr --out_dir research/runs/jepa_01
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from research.dataset import DWISliceDataset
from research.loss import DTILoss
from research.model_jepa import JEPAQSpaceUNet, jepa_feature_loss
from research.train import (
    DEFAULT_TEST_SUBJECTS,
    DEFAULT_VAL_SUBJECTS,
    get_device,
    load_baseline_metrics,
    log_baseline_references,
    make_val_figure,
)
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_epoch(
    model: JEPAQSpaceUNet,
    loader: DataLoader,
    criterion: DTILoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_brain_mask: bool = True,
    jepa_lambda: float = cfg.JEPA_LAMBDA,
) -> dict[str, float]:
    """Run one train or validation epoch. Pass optimizer=None for val."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_task = 0.0
    total_jepa = 0.0
    total_tensor = 0.0
    total_fa = 0.0
    total_md = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch in loader:
            noisy = batch["input"].to(device)
            clean = batch["clean_input"].to(device)
            target = batch["target"].to(device)
            bvals = batch["bvals"].to(device)
            bvecs = batch["bvecs"].to(device)
            vol_mask = batch["vol_mask"].to(device)
            brain_mask = batch["brain_mask"].to(device) if use_brain_mask else None

            pred_dti, pred_feats, tgt_feats = model.forward_jepa(
                noisy, clean, bvals, bvecs, vol_mask,
            )
            task_loss, metrics = criterion(pred_dti, target, mask=brain_mask)
            j_loss = jepa_feature_loss(pred_feats, tgt_feats, mask=brain_mask)
            loss = task_loss + jepa_lambda * j_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.GRAD_CLIP,
                )
                optimizer.step()
                model.ema_update()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_jepa += j_loss.item()
            total_tensor += metrics.get("tensor_mse", 0.0)
            total_fa += metrics.get("fa_mae", 0.0)
            total_md += metrics.get("md_mae", 0.0)
            n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "task_loss": total_task / n,
        "jepa_loss": total_jepa / n,
        "tensor_mse": total_tensor / n,
        "fa_mae": total_fa / n,
        "md_mae": total_md / n,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: JEPAQSpaceUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    global_max_n: int,
    args,
    train_ds: DWISliceDataset,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    use_brain_mask: bool,
) -> None:
    """Save a checkpoint in pure QSpaceUNet format (student weights only).

    The teacher encoder + predictor are dropped so that ``research.evaluate``
    can load this file with the unchanged ``QSpaceUNet`` class.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.student_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
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
            # JEPA-specific metadata (non-breaking for evaluate.py).
            "training_objective": "jepa",
            "jepa_lambda": args.jepa_lambda,
            "jepa_ema": args.jepa_ema,
            "jepa_pred_hidden": args.jepa_pred_hidden,
        },
        path,
    )


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log.info("Device: %s", device)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    baseline_csvs = [
        out_dir / "metrics_patch2self.csv",
        out_dir / "metrics_mppca.csv",
    ]
    baselines = load_baseline_metrics(baseline_csvs)
    if baselines:
        log.info("Loaded baseline references: %s", list(baselines.keys()))

    # ── Subject split ────────────────────────────────────────────────────────
    import zarr
    store = zarr.open_group(args.zarr_path, mode="r")
    all_keys = sorted(store.keys())
    log.info("Found %d entries in %s", len(all_keys), args.zarr_path)

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

    log.info("Train: %d  Val: %d  Test: %d",
             len(train_subjects), len(val_subjects), len(test_subjects))

    # ── Datasets ─────────────────────────────────────────────────────────────
    use_brain_mask = not args.no_brain_mask
    train_ds = DWISliceDataset(
        args.zarr_path, train_subjects,
        augment=True,
        use_brain_mask=use_brain_mask,
        random_axis=cfg.RANDOM_SLICE_AXIS,
        slice_axes=cfg.SLICE_AXES,
        return_clean_input=True,
    )
    val_ds = DWISliceDataset(
        args.zarr_path, val_subjects,
        augment=False,
        use_brain_mask=use_brain_mask,
        random_axis=False,
        eval_mode=True,
        return_clean_input=True,
    )

    global_max_n = max(train_ds.max_n, val_ds.max_n)
    train_ds.max_n = global_max_n
    val_ds.max_n = global_max_n

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    log.info("Train slices: %d  Val slices: %d  max_n: %d  max_bval: %.0f  dti_scale: %.4f",
             len(train_ds), len(val_ds), global_max_n, global_max_bval, global_dti_scale)

    # ── Model ────────────────────────────────────────────────────────────────
    model = JEPAQSpaceUNet(
        max_n=global_max_n,
        feat_dim=args.feat_dim,
        channels=tuple(args.channels),
        cholesky=args.cholesky,
        ema_momentum=args.jepa_ema,
        pred_hidden=args.jepa_pred_hidden,
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_student = sum(v.numel() for v in model.student_state_dict().values())
    log.info("Params  trainable=%s  total=%s  student_only=%s",
             f"{n_params_train:,}", f"{n_params_total:,}", f"{n_params_student:,}")

    writer.add_text("hparams", json.dumps({
        "objective": "jepa",
        "max_n": global_max_n, "feat_dim": args.feat_dim,
        "channels": list(args.channels), "cholesky": args.cholesky,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_scalar": args.lambda_scalar, "lambda_edge": args.lambda_edge,
        "jepa_lambda": args.jepa_lambda, "jepa_ema": args.jepa_ema,
        "jepa_pred_hidden": args.jepa_pred_hidden,
        "warmup_epochs": args.warmup_epochs, "patience": args.patience,
        "use_brain_mask": use_brain_mask,
        "n_params_trainable": n_params_train,
        "n_params_student": n_params_student,
        "train_subjects": train_subjects, "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }, indent=2))

    # ── Loss & optim ─────────────────────────────────────────────────────────
    criterion = DTILoss(
        lambda_scalar=args.lambda_scalar,
        lambda_edge=args.lambda_edge,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )

    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - warmup_epochs, 1),
        eta_min=args.lr * 0.01,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    vis_slice_idx = -1
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    log.info("Starting JEPA training for %d epochs  (jepa_lambda=%.3f  ema=%.4f)",
             args.epochs, args.jepa_lambda, args.jepa_ema)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer,
            use_brain_mask=use_brain_mask, jepa_lambda=args.jepa_lambda,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None,
            use_brain_mask=use_brain_mask, jepa_lambda=args.jepa_lambda,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_task_loss": train_metrics["task_loss"],
            "train_jepa_loss": train_metrics["jepa_loss"],
            "train_tensor_mse": train_metrics["tensor_mse"],
            "train_fa_mae": train_metrics["fa_mae"],
            "train_md_mae": train_metrics["md_mae"],
            "val_loss": val_metrics["loss"],
            "val_task_loss": val_metrics["task_loss"],
            "val_jepa_loss": val_metrics["jepa_loss"],
            "val_tensor_mse": val_metrics["tensor_mse"],
            "val_fa_mae": val_metrics["fa_mae"],
            "val_md_mae": val_metrics["md_mae"],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)

        # TensorBoard scalars — mirror train.py so comparison is direct.
        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("task_loss/train", train_metrics["task_loss"], epoch)
        writer.add_scalar("task_loss/val", val_metrics["task_loss"], epoch)
        writer.add_scalar("jepa_loss/train", train_metrics["jepa_loss"], epoch)
        writer.add_scalar("jepa_loss/val", val_metrics["jepa_loss"], epoch)
        writer.add_scalar("tensor_mse/train", train_metrics["tensor_mse"], epoch)
        writer.add_scalar("tensor_mse/val", val_metrics["tensor_mse"], epoch)
        writer.add_scalar("fa_mae/train", train_metrics["fa_mae"], epoch)
        writer.add_scalar("fa_mae/val", val_metrics["fa_mae"], epoch)
        writer.add_scalar("md_mae/train", train_metrics["md_mae"], epoch)
        writer.add_scalar("md_mae/val", val_metrics["md_mae"], epoch)
        writer.add_scalar("lr", lr, epoch)

        log_baseline_references(writer, baselines, epoch)

        if epoch % args.vis_every == 0 or epoch == 1:
            fig = make_val_figure(
                model, val_ds, device, dti_scale=train_ds.dti_scale,
                slice_idx=vis_slice_idx,
            )
            writer.add_figure("val_prediction", fig, epoch)
            plt.close(fig)

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            _save_checkpoint(
                out_dir / "best_model.pt",
                model=model, optimizer=optimizer, epoch=epoch,
                val_loss=best_val_loss, global_max_n=global_max_n,
                args=args, train_ds=train_ds,
                train_subjects=train_subjects, val_subjects=val_subjects,
                test_subjects=test_subjects, use_brain_mask=use_brain_mask,
            )
        else:
            patience_counter += 1

        marker = "*" if improved else ""
        log.info(
            "Epoch %3d/%d  train=%.6f (task=%.6f jepa=%.6f)  val=%.6f  "
            "fa=%.4f  md=%.6f  lr=%.2e  %.1fs %s",
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["task_loss"], train_metrics["jepa_loss"],
            val_metrics["loss"], val_metrics["fa_mae"], val_metrics["md_mae"],
            lr, elapsed, marker,
        )

        if patience_counter >= args.patience:
            log.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    _save_checkpoint(
        out_dir / "last_model.pt",
        model=model, optimizer=optimizer, epoch=epoch,
        val_loss=val_metrics["loss"], global_max_n=global_max_n,
        args=args, train_ds=train_ds,
        train_subjects=train_subjects, val_subjects=val_subjects,
        test_subjects=test_subjects, use_brain_mask=use_brain_mask,
    )

    writer.close()
    log.info("Done. Best val loss: %.6f  Saved to %s", best_val_loss, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEPA-augmented QSpaceUNet")

    parser.add_argument("--zarr_path", default="dataset/default_clean.zarr")
    parser.add_argument("--out_dir", default="research/runs/jepa_01")
    parser.add_argument("--test_subjects", nargs="*", default=None)
    parser.add_argument("--val_subjects", nargs="*", default=None)

    parser.add_argument("--feat_dim", type=int, default=cfg.FEAT_DIM)
    parser.add_argument("--channels", type=int, nargs="+", default=cfg.UNET_CHANNELS)
    parser.add_argument("--cholesky", action="store_true")

    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--lambda_scalar", type=float, default=cfg.LAMBDA_SCALAR)
    parser.add_argument("--lambda_edge", type=float, default=cfg.LAMBDA_EDGE)
    parser.add_argument("--warmup_epochs", type=int, default=cfg.WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_brain_mask", action="store_true")

    # JEPA-specific args
    parser.add_argument("--jepa_lambda", type=float, default=cfg.JEPA_LAMBDA)
    parser.add_argument("--jepa_ema", type=float, default=cfg.JEPA_EMA)
    parser.add_argument("--jepa_pred_hidden", type=int, default=cfg.JEPA_PRED_HIDDEN)

    main(parser.parse_args())
