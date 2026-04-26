"""Train QSpaceUNet for fODF SH prediction with TensorBoard tracking.

The production defaults live in config.py. CLI flags only override that single
config source when you need an ad-hoc run. This branch trains the fODF head
only — no DTI supervision.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from torch.profiler import (
        ProfilerActivity,
        profile as torch_profile,
        record_function,
        schedule as profiler_schedule,
        tensorboard_trace_handler,
    )
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    ProfilerActivity = None
    torch_profile = None
    record_function = None
    profiler_schedule = None
    tensorboard_trace_handler = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    SummaryWriter = None

import config as cfg
from .augment import gpu_b0_normalize_batch, gpu_degrade_dwi_batch
from .dataset import DWISliceDataset, dwi_worker_init
from .loss import FodfLoss
from .model import QSpaceUNet
from .runtime import (
    amp_dtype_from_name,
    autocast_context,
    configure_torch_runtime,
    default_num_workers,
    get_device,
    make_grad_scaler,
    maybe_compile_model,
    path_str,
    require_cuda_if_requested,
    resolve_project_path,
    should_pin_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


FODF_METRIC_KEYS = (
    "loss",
    "fodf_loss",
    "fodf_mse",
    "fodf_band_loss",
    "fodf_acc",
    "fodf_corr_loss",
    "fodf_aniso_corr_loss",
    "fodf_sf_loss",
    "fodf_peak_loss",
    "fodf_peak_mae",
    "fodf_peak_ratio",
    "fodf_nonneg_loss",
    "fodf_power_loss",
)


@dataclass
class ProfilerController:
    enabled: bool = False
    output_dir: Path | None = None
    tensorboard_dir: Path | None = None
    row_limit: int = cfg.PROFILE_ROW_LIMIT
    sort_key: str = "self_device_time_total"
    wait_steps: int = cfg.PROFILE_WAIT
    warmup_steps: int = cfg.PROFILE_WARMUP
    active_steps: int = cfg.PROFILE_ACTIVE
    repeat: int = cfg.PROFILE_REPEAT
    trace_count: int = 0
    train_steps: int = 0
    capture_steps: int = 0
    exit_after_capture: bool = False
    stop_requested: bool = False
    summary_paths: list[Path] = field(default_factory=list)
    _profiler: object | None = None
    _trace_handler: object | None = None

    @classmethod
    def create(
        cls,
        args: argparse.Namespace,
        *,
        device: torch.device,
        out_dir: Path,
    ) -> ProfilerController:
        if not args.profile:
            return cls(enabled=False)
        if (
            torch_profile is None
            or profiler_schedule is None
            or tensorboard_trace_handler is None
            or record_function is None
            or ProfilerActivity is None
        ):
            raise RuntimeError(
                "PyTorch profiler is unavailable in this environment. "
                "Install a complete PyTorch build before using --profile."
            )

        output_dir = out_dir / "profiler"
        tensorboard_dir = out_dir / "tb"
        output_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        controller = cls(
            enabled=True,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
            row_limit=args.profile_row_limit,
            sort_key="self_device_time_total" if device.type == "cuda" else "self_cpu_time_total",
            wait_steps=args.profile_wait,
            warmup_steps=args.profile_warmup,
            active_steps=args.profile_active,
            repeat=args.profile_repeat,
            capture_steps=(args.profile_wait + args.profile_warmup + args.profile_active)
            * args.profile_repeat,
            exit_after_capture=args.profile_exit_after_capture,
        )
        controller._trace_handler = tensorboard_trace_handler(str(tensorboard_dir))
        controller._profiler = torch_profile(
            activities=activities,
            schedule=profiler_schedule(
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=args.profile_repeat,
            ),
            on_trace_ready=controller._on_trace_ready,
            record_shapes=args.profile_record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_with_stack,
            with_flops=args.profile_with_flops,
        )
        return controller

    def __enter__(self) -> ProfilerController:
        if self._profiler is not None:
            self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        if self._profiler is None:
            return None
        return self._profiler.__exit__(exc_type, exc, tb)

    def record(self, name: str):
        if not self.enabled or record_function is None:
            return nullcontext()
        return record_function(name)

    def step(self, *, is_train: bool) -> None:
        if not self.enabled or self._profiler is None or not is_train:
            return
        self.train_steps += 1
        self._profiler.step()
        if self.exit_after_capture and self.train_steps >= self.capture_steps:
            self.stop_requested = True

    def log_enabled(self) -> None:
        if not self.enabled:
            return
        log.info(
            "Profiler enabled: wait=%d warmup=%d active=%d repeat=%d traces -> %s",
            self.wait_steps,
            self.warmup_steps,
            self.active_steps,
            self.repeat,
            self.tensorboard_dir,
        )

    def _on_trace_ready(self, prof) -> None:
        self.trace_count += 1
        if self._trace_handler is not None:
            self._trace_handler(prof)
        self._write_summary(prof, trace_idx=self.trace_count)

    def _write_summary(self, prof, *, trace_idx: int) -> None:
        if self.output_dir is None:
            return

        key_averages = prof.key_averages()
        text_path = self.output_dir / f"summary_trace_{trace_idx:02d}.txt"
        json_path = self.output_dir / f"summary_trace_{trace_idx:02d}.json"

        sort_candidates = [
            self.sort_key,
            "self_cuda_time_total",
            "device_time_total",
            "cuda_time_total",
            "self_cpu_time_total",
            "cpu_time_total",
        ]
        written_tables: list[tuple[str, str]] = []
        for sort_key in dict.fromkeys(sort_candidates):
            try:
                table = key_averages.table(sort_by=sort_key, row_limit=self.row_limit)
            except Exception:
                continue
            written_tables.append((sort_key, table))

        with text_path.open("w", encoding="utf-8") as f:
            f.write("PyTorch profiler summary\n")
            f.write(f"trace_index: {trace_idx}\n")
            f.write(f"recorded_train_steps: {self.train_steps}\n")
            f.write(f"tensorboard_logdir: {self.tensorboard_dir}\n")
            for sort_key, table in written_tables:
                f.write(f"\n=== Sorted by {sort_key} ===\n")
                f.write(table)
                f.write("\n")

        top_ops: list[dict[str, float | int | str]] = []
        for evt in sorted(
            key_averages,
            key=lambda item: self._metric_value(item, self.sort_key),
            reverse=True,
        )[: self.row_limit]:
            top_ops.append(
                {
                    "name": evt.key,
                    "count": int(getattr(evt, "count", 0)),
                    "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0)),
                    "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
                    "self_device_time_total_us": self._metric_value(
                        evt,
                        "self_device_time_total",
                        "self_cuda_time_total",
                    ),
                    "device_time_total_us": self._metric_value(
                        evt,
                        "device_time_total",
                        "cuda_time_total",
                    ),
                    "self_cpu_memory_usage_bytes": float(
                        getattr(evt, "self_cpu_memory_usage", 0.0)
                    ),
                    "self_device_memory_usage_bytes": float(
                        getattr(evt, "self_device_memory_usage", getattr(evt, "self_cuda_memory_usage", 0.0))
                    ),
                }
            )

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "trace_index": trace_idx,
                    "recorded_train_steps": self.train_steps,
                    "sort_key": self.sort_key,
                    "tensorboard_logdir": str(self.tensorboard_dir),
                    "top_ops": top_ops,
                },
                f,
                indent=2,
            )

        self.summary_paths.extend([text_path, json_path])
        log.info("Profiler trace %d written to %s", trace_idx, text_path)

    @staticmethod
    def _metric_value(evt, *names: str) -> float:
        for name in names:
            if hasattr(evt, name):
                return float(getattr(evt, name))
        return 0.0


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


def move_batch_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool,
    channels_last: bool = False,
) -> torch.Tensor:
    """Move a batch tensor to device, optionally fusing the NHWC relayout.

    When ``channels_last`` is enabled we ask ``Tensor.to`` to apply the memory
    format during the host->device copy instead of copying once to CUDA and
    then doing a second device-side ``contiguous(memory_format=...)`` pass.
    """
    if channels_last and tensor.ndim == 4:
        return tensor.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return tensor.to(device, non_blocking=non_blocking)


def make_val_figure(
    model: QSpaceUNet,
    val_ds: DWISliceDataset,
    device: torch.device,
    n_bands: int,
    band_slices: list[tuple[int, int]],
    slice_idx: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
) -> plt.Figure:
    """Generate an fODF prediction-vs-target figure for TensorBoard.

    Plots SH-energy maps (rotation-invariant), the per-voxel cosine
    similarity, and a target-vs-pred per-ℓ-band power-spectrum bar plot
    averaged over brain voxels. No DTI scalars.
    """
    if slice_idx is None:
        slice_idx = len(val_ds) // 2

    sample = val_ds[slice_idx]
    non_blocking = device.type == "cuda"
    signal = move_batch_tensor(
        sample["input"].unsqueeze(0),
        device,
        non_blocking=non_blocking,
        channels_last=channels_last,
    )
    bvals = sample["bvals"].unsqueeze(0).to(device, non_blocking=non_blocking)
    bvecs = sample["bvecs"].unsqueeze(0).to(device, non_blocking=non_blocking)
    vol_mask = sample["vol_mask"].unsqueeze(0).to(device, non_blocking=non_blocking)

    target_fodf = sample["target_fodf"].numpy()  # (C, H, W)
    bmask = sample["brain_mask"].numpy() if "brain_mask" in sample else None

    model.eval()
    with torch.inference_mode(), autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
        pred = model(signal, bvals, bvecs, vol_mask)
    _, pred_fodf_t = model.split_outputs(pred.float())
    pred_fodf = pred_fodf_t[0].cpu().numpy()  # (C, H, W)

    tgt_energy = np.sqrt((target_fodf ** 2).sum(axis=0))
    pred_energy = np.sqrt((pred_fodf ** 2).sum(axis=0))
    energy_diff = pred_energy - tgt_energy

    pred_flat = pred_fodf.reshape(pred_fodf.shape[0], -1)
    tgt_flat = target_fodf.reshape(target_fodf.shape[0], -1)
    norm_pred = np.linalg.norm(pred_flat, axis=0) + 1e-8
    norm_tgt = np.linalg.norm(tgt_flat, axis=0) + 1e-8
    cosine = (pred_flat * tgt_flat).sum(axis=0) / (norm_pred * norm_tgt)
    cosine_map = cosine.reshape(target_fodf.shape[1], target_fodf.shape[2])

    if bmask is not None:
        bmask_bool = bmask > 0.5
        tgt_energy_disp = tgt_energy * bmask
        pred_energy_disp = pred_energy * bmask
        energy_diff_disp = energy_diff * bmask
        cosine_disp = np.where(bmask_bool, cosine_map, np.nan)
    else:
        bmask_bool = np.ones_like(tgt_energy, dtype=bool)
        tgt_energy_disp = tgt_energy
        pred_energy_disp = pred_energy
        energy_diff_disp = energy_diff
        cosine_disp = cosine_map

    if bmask_bool.any():
        tgt_band_power = np.array(
            [(target_fodf[s:e][:, bmask_bool] ** 2).sum(axis=0).mean()
             for s, e in band_slices]
        )
        pred_band_power = np.array(
            [(pred_fodf[s:e][:, bmask_bool] ** 2).sum(axis=0).mean()
             for s, e in band_slices]
        )
    else:
        tgt_band_power = np.zeros(n_bands)
        pred_band_power = np.zeros(n_bands)

    energy_max = max(float(np.nanmax(tgt_energy_disp)), float(np.nanmax(pred_energy_disp)), 1e-6)
    diff_abs = max(float(np.nanmax(np.abs(energy_diff_disp))), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(np.rot90(tgt_energy_disp), cmap="viridis", vmin=0, vmax=energy_max)
    axes[0, 0].set_title("Target SH energy ‖c‖")
    axes[0, 1].imshow(np.rot90(pred_energy_disp), cmap="viridis", vmin=0, vmax=energy_max)
    axes[0, 1].set_title("Predicted SH energy ‖c‖")
    im_diff = axes[0, 2].imshow(np.rot90(energy_diff_disp), cmap="bwr", vmin=-diff_abs, vmax=diff_abs)
    axes[0, 2].set_title("Energy residual (pred − target)")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_cos = axes[1, 0].imshow(np.rot90(cosine_disp), cmap="RdYlBu", vmin=-1, vmax=1)
    cosine_brain = cosine_map[bmask_bool] if bmask_bool.any() else np.array([0.0])
    axes[1, 0].set_title(f"Cosine similarity (mean={cosine_brain.mean():.3f})")
    fig.colorbar(im_cos, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].hist(cosine_brain, bins=40, range=(-1.0, 1.0), color="steelblue")
    axes[1, 1].axvline(float(cosine_brain.mean()), color="red", lw=1)
    axes[1, 1].set_title("Cosine sim histogram (brain)")
    axes[1, 1].set_xlabel("cosine")
    axes[1, 1].set_ylabel("voxels")

    band_idx = np.arange(n_bands)
    width = 0.4
    axes[1, 2].bar(band_idx - width / 2, tgt_band_power, width, label="target", color="black", alpha=0.7)
    axes[1, 2].bar(band_idx + width / 2, pred_band_power, width, label="pred", color="orange", alpha=0.8)
    axes[1, 2].set_xticks(band_idx)
    axes[1, 2].set_xticklabels([f"ℓ={2 * i}" for i in band_idx])
    axes[1, 2].set_title("Mean per-ℓ-band power (brain)")
    axes[1, 2].set_yscale("log")
    axes[1, 2].legend()

    for r, c in [(0, 0), (0, 1), (0, 2), (1, 0)]:
        axes[r, c].axis("off")

    fig.tight_layout()
    return fig


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: FodfLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_brain_mask: bool = True,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
    non_blocking: bool = False,
    channels_last: bool = False,
    profiler: ProfilerController | None = None,
) -> dict[str, float]:
    """Run one train or validation epoch (fODF-only)."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {key: torch.zeros((), device=device) for key in FODF_METRIC_KEYS}
    n_batches = 0

    def metric_tensor(metrics: dict, key: str) -> torch.Tensor:
        value = metrics.get(key)
        if value is None:
            return torch.zeros((), device=device)
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(float(value), device=device)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    try:
        with ctx:
            for batch in loader:
                if profiler is not None and profiler.stop_requested:
                    break

                record = profiler.record if profiler is not None else (lambda _name: nullcontext())

                with record("train/data_to_device" if is_train else "val/data_to_device"):
                    signal = move_batch_tensor(
                        batch["input"],
                        device,
                        non_blocking=non_blocking,
                        channels_last=channels_last,
                    )
                    target_fodf = batch["target_fodf"].to(device, non_blocking=non_blocking)
                    bvals = batch["bvals"].to(device, non_blocking=non_blocking)
                    bvecs = batch["bvecs"].to(device, non_blocking=non_blocking)
                    vol_mask = batch["vol_mask"].to(device, non_blocking=non_blocking)
                    brain_mask = (
                        batch["brain_mask"].to(device, non_blocking=non_blocking)
                        if use_brain_mask
                        else None
                    )

                if "degrade_kf" in batch:
                    with record("train/gpu_degrade" if is_train else "val/gpu_degrade"):
                        degrade_kf = batch["degrade_kf"].to(device, non_blocking=non_blocking)
                        degrade_nl = batch["degrade_nl"].to(device, non_blocking=non_blocking)
                        b0_mask = batch["b0_mask"].to(device, non_blocking=non_blocking)
                        signal = gpu_degrade_dwi_batch(signal, degrade_kf, degrade_nl)
                        signal = gpu_b0_normalize_batch(signal, b0_mask)

                with record("train/forward" if is_train else "val/forward"):
                    with autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                        pred = model(signal, bvals, bvecs, vol_mask)
                pred_fodf = pred.float()
                with record("train/loss" if is_train else "val/loss"):
                    loss, metrics = criterion(
                        pred_fodf,
                        target_fodf,
                        mask=brain_mask,
                        return_tensor_metrics=True,
                    )

                if is_train:
                    with record("train/optimizer_zero_grad"):
                        optimizer.zero_grad(set_to_none=True)
                    with record("train/backward"):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                    with record("train/optimizer_step"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                totals["loss"] += loss.detach()
                for key in FODF_METRIC_KEYS:
                    if key == "loss":
                        continue
                    totals[key] += metric_tensor(metrics, key)
                n_batches += 1

                if profiler is not None:
                    profiler.step(is_train=is_train)
    except torch.AcceleratorError as exc:
        if "resource already mapped" in str(exc):
            raise RuntimeError(
                "CUDA pinned host memory failed with 'resource already mapped'. "
                "Disable DataLoader pinning with --no-pin_memory, or use --num_workers 0 "
                "if you want to retry pinned transfers on Windows."
            ) from exc
        raise

    n = max(n_batches, 1)
    return {key: float((totals[key] / n).detach().cpu()) for key in FODF_METRIC_KEYS}


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
    pin_memory: bool,
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
        "fodf_channels": train_ds.fodf_n_coeffs,
        "stored_fodf_channels": train_ds.stored_fodf_n_coeffs,
        "train_fodf_sh_order": args.train_fodf_sh_order,
        "dti_channels": 0,
        "canonical_hw": list(train_ds.canonical_hw),
        "feat_dim": args.feat_dim,
        "channels": list(args.channels),
        "context_slices": args.context_slices,
        "context_fusion_layers": args.context_fusion_layers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_fodf": args.lambda_fodf,
        "lambda_fodf_band": args.lambda_fodf_band,
        "lambda_fodf_corr": args.lambda_fodf_corr,
        "lambda_fodf_aniso_corr": args.lambda_fodf_aniso_corr,
        "lambda_fodf_sf": args.lambda_fodf_sf,
        "lambda_fodf_peak": args.lambda_fodf_peak,
        "lambda_fodf_nonneg": args.lambda_fodf_nonneg,
        "lambda_fodf_power": args.lambda_fodf_power,
        "fodf_loss_sphere": args.fodf_loss_sphere,
        "fodf_sf_chunk_size": args.fodf_sf_chunk_size,
        "fodf_peak_topk": args.fodf_peak_topk,
        "fodf_peak_weight": args.fodf_peak_weight,
        "fodf_peak_gamma": args.fodf_peak_gamma,
        "fodf_peak_rel_threshold": args.fodf_peak_rel_threshold,
        "fodf_band_weight_gamma": args.fodf_band_weight_gamma,
        "fodf_power_weight_gamma": args.fodf_power_weight_gamma,
        "fodf_band_scale_floor": args.fodf_band_scale_floor,
        "fodf_power_scale_floor": args.fodf_power_scale_floor,
        "fodf_aniso_min_l": args.fodf_aniso_min_l,
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
        "pin_memory": pin_memory,
        "fused_adamw": fused_adamw,
        "n_params": n_params,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "profile": {
            "enabled": args.profile,
            "wait": args.profile_wait,
            "warmup": args.profile_warmup,
            "active": args.profile_active,
            "repeat": args.profile_repeat,
            "record_shapes": args.profile_record_shapes,
            "memory": args.profile_memory,
            "with_stack": args.profile_with_stack,
            "with_flops": args.profile_with_flops,
            "row_limit": args.profile_row_limit,
            "exit_after_capture": args.profile_exit_after_capture,
            "summary_dir": str(out_dir / "profiler") if args.profile else None,
            "tensorboard_logdir": str(out_dir / "tb") if args.profile else None,
        },
    }


def log_scalars(
    writer: SummaryWriter,
    prefix: str,
    metrics: dict[str, float],
    epoch: int,
) -> None:
    for key in FODF_METRIC_KEYS:
        if key in metrics:
            writer.add_scalar(f"{prefix}/{key}", metrics[key], epoch)


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
            "context_slices": args.context_slices,
            "context_fusion_layers": args.context_fusion_layers,
            "train_fodf_sh_order": args.train_fodf_sh_order,
            "stored_fodf_channels": train_ds.stored_fodf_n_coeffs,
            "cholesky": False,
            "fodf_channels": train_ds.fodf_n_coeffs,
            "dti_channels": 0,
            # ``dti_scale`` is preserved so the visualizer can fall back to a
            # sensible default when this checkpoint is reused as a DTI source.
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
    if channels_last and args.context_slices > 1:
        log.info("Disabling channels_last for %d-slice context model.", args.context_slices)
        channels_last = False
    num_workers = default_num_workers(args.num_workers)
    pin_memory = should_pin_memory(device, requested=args.pin_memory, num_workers=num_workers)
    non_blocking_cuda = pin_memory

    log.info("AMP: %s", str(amp_dtype).replace("torch.", "") if amp_enabled else "disabled")
    if channels_last:
        log.info("Memory format: channels_last")
    profiler = ProfilerController.create(args, device=device, out_dir=out_dir)
    profiler.log_enabled()

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
        context_slices=args.context_slices,
        target_fodf_sh_order=args.train_fodf_sh_order,
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
        context_slices=args.context_slices,
        target_fodf_sh_order=args.train_fodf_sh_order,
    )

    if train_ds.fodf_n_coeffs <= 0:
        raise ValueError(
            f"This branch trains the fODF head only, but the dataset at "
            f"{zarr_path} has no target_fodf_sh coefficients."
        )
    if not val_ds.has_fodf:
        raise ValueError(
            "Validation dataset has no target_fodf_sh coefficients; cannot validate fODF training."
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
    val_ds.fodf_n_coeffs = train_ds.fodf_n_coeffs

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
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
        pin_memory,
        train_ds.gpu_degrade,
    )
    log.info(
        "Train slices: %d Val slices: %d max_n: %d max_bval: %.0f fodf_channels: %d/%d context_slices: %d",
        len(train_ds),
        len(val_ds),
        global_max_n,
        train_ds.max_bval,
        train_ds.fodf_n_coeffs,
        train_ds.stored_fodf_n_coeffs,
        args.context_slices,
    )

    raw_model = QSpaceUNet(
        max_n=global_max_n,
        feat_dim=args.feat_dim,
        channels=tuple(args.channels),
        cholesky=False,
        fodf_channels=train_ds.fodf_n_coeffs,
        dti_channels=0,
        dropout=args.dropout,
        context_slices=args.context_slices,
        context_fusion_layers=args.context_fusion_layers,
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

    criterion = FodfLoss(
        lambda_fodf=args.lambda_fodf,
        lambda_fodf_band=args.lambda_fodf_band,
        lambda_fodf_corr=args.lambda_fodf_corr,
        lambda_fodf_aniso_corr=args.lambda_fodf_aniso_corr,
        lambda_fodf_sf=args.lambda_fodf_sf,
        lambda_fodf_peak=args.lambda_fodf_peak,
        lambda_fodf_nonneg=args.lambda_fodf_nonneg,
        lambda_fodf_power=args.lambda_fodf_power,
        fodf_loss_sphere=args.fodf_loss_sphere,
        fodf_sf_chunk_size=args.fodf_sf_chunk_size,
        fodf_peak_topk=args.fodf_peak_topk,
        fodf_peak_weight=args.fodf_peak_weight,
        fodf_peak_gamma=args.fodf_peak_gamma,
        fodf_peak_rel_threshold=args.fodf_peak_rel_threshold,
        fodf_band_weight_gamma=args.fodf_band_weight_gamma,
        fodf_power_weight_gamma=args.fodf_power_weight_gamma,
        fodf_band_scale_floor=args.fodf_band_scale_floor,
        fodf_power_scale_floor=args.fodf_power_scale_floor,
        fodf_aniso_min_l=args.fodf_aniso_min_l,
    ).to(device)

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
        pin_memory=pin_memory,
        fused_adamw=fused_adamw,
        n_params=n_params,
        is_compiled=is_compiled,
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is required for production training. "
            "Install dependencies with `pip install -r requirements.txt`."
        )
    writer: SummaryWriter | None = SummaryWriter(log_dir=str(out_dir / "tb"))
    writer.add_text("config/json", f"```json\n{json.dumps(run_config, indent=2)}\n```", 0)

    from .loss import _sh_band_slices  # local import to avoid cycle at top-level
    band_slices = _sh_band_slices(train_ds.fodf_n_coeffs)
    n_bands = len(band_slices)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, object]] = []
    completed = False

    log.info("Starting training for %d epochs (patience=%d)", args.epochs, args.patience)
    try:
        with profiler:
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
                    non_blocking=non_blocking_cuda,
                    channels_last=channels_last,
                    profiler=profiler,
                )
                if profiler.stop_requested:
                    log.info(
                        "Profiler capture complete after %d training batches; "
                        "stopping early because --profile_exit_after_capture was set.",
                        profiler.train_steps,
                    )
                    completed = True
                    break

                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    use_brain_mask=use_brain_mask,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    non_blocking=non_blocking_cuda,
                    channels_last=channels_last,
                )
                scheduler.step()

                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                record = {"epoch": epoch, "lr": lr, "elapsed_s": round(elapsed, 1)}
                for key in FODF_METRIC_KEYS:
                    record[f"train_{key}"] = train_metrics[key]
                    record[f"val_{key}"] = val_metrics[key]
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
                        n_bands=n_bands,
                        band_slices=band_slices,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        channels_last=channels_last,
                    )
                    writer.add_figure("val/fodf", fig, epoch)
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
                    "Epoch %3d/%d train=%.6f val=%.6f fodf=%.6f band=%.6f "
                    "sf=%.6f peak=%.6f p_ratio=%.3f acc=%.4f aniso=%.6f "
                    "power=%.6f lr=%.2e %.1fs %s",
                    epoch,
                    args.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["fodf_loss"],
                    val_metrics["fodf_band_loss"],
                    val_metrics["fodf_sf_loss"],
                    val_metrics["fodf_peak_loss"],
                    val_metrics["fodf_peak_ratio"],
                    val_metrics["fodf_acc"],
                    val_metrics["fodf_aniso_corr_loss"],
                    val_metrics["fodf_power_loss"],
                    lr,
                    elapsed,
                    marker,
                )
                if patience_counter >= args.patience:
                    log.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                    break

        if history:
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
            log.info(
                "Done. Best val loss: %.6f at epoch %d. Saved to %s",
                best_val_loss,
                best_epoch,
                out_dir,
            )
        elif profiler.enabled and profiler.trace_count > 0:
            completed = True
            log.info("Profiler-only run complete. Trace artifacts saved to %s", profiler.output_dir)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if not completed:
            log.info("Training stopped before normal completion; partial outputs remain in %s", out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QSpaceUNet for DWI -> fODF SH coefficient prediction"
    )

    parser.add_argument("--zarr_path", default=cfg.DATASET_ZARR_PATH)
    parser.add_argument("--out_dir", default=cfg.TRAIN_OUT_DIR)
    parser.add_argument("--test_subjects", nargs="*", default=None)
    parser.add_argument("--val_subjects", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)

    parser.add_argument("--feat_dim", type=int, default=cfg.FEAT_DIM)
    parser.add_argument("--channels", type=int, nargs="+", default=cfg.UNET_CHANNELS)
    parser.add_argument("--context_slices", type=int, default=cfg.CONTEXT_SLICES)
    parser.add_argument("--context_fusion_layers", type=int, default=cfg.CONTEXT_FUSION_LAYERS)
    parser.add_argument("--train_fodf_sh_order", type=int, default=cfg.TRAIN_FODF_SH_ORDER)
    parser.add_argument("--dropout", type=float, default=cfg.DROPOUT)

    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--lambda_fodf", type=float, default=cfg.LAMBDA_FODF)
    parser.add_argument("--lambda_fodf_band", type=float, default=cfg.LAMBDA_FODF_BAND)
    parser.add_argument("--lambda_fodf_corr", type=float, default=cfg.LAMBDA_FODF_CORR)
    parser.add_argument("--lambda_fodf_aniso_corr", type=float, default=cfg.LAMBDA_FODF_ANISO_CORR)
    parser.add_argument("--lambda_fodf_sf", type=float, default=cfg.LAMBDA_FODF_SF)
    parser.add_argument("--lambda_fodf_peak", type=float, default=cfg.LAMBDA_FODF_PEAK)
    parser.add_argument("--lambda_fodf_nonneg", type=float, default=cfg.LAMBDA_FODF_NONNEG)
    parser.add_argument("--lambda_fodf_power", type=float, default=cfg.LAMBDA_FODF_POWER)
    parser.add_argument("--fodf_loss_sphere", default=cfg.FODF_LOSS_SPHERE)
    parser.add_argument("--fodf_sf_chunk_size", type=int, default=cfg.FODF_SF_CHUNK_SIZE)
    parser.add_argument("--fodf_peak_topk", type=int, default=cfg.FODF_PEAK_TOPK)
    parser.add_argument("--fodf_peak_weight", type=float, default=cfg.FODF_PEAK_WEIGHT)
    parser.add_argument("--fodf_peak_gamma", type=float, default=cfg.FODF_PEAK_GAMMA)
    parser.add_argument(
        "--fodf_peak_rel_threshold",
        type=float,
        default=cfg.FODF_PEAK_REL_THRESHOLD,
    )
    parser.add_argument("--fodf_band_weight_gamma", type=float, default=cfg.FODF_BAND_WEIGHT_GAMMA)
    parser.add_argument("--fodf_power_weight_gamma", type=float, default=cfg.FODF_POWER_WEIGHT_GAMMA)
    parser.add_argument("--fodf_band_scale_floor", type=float, default=cfg.FODF_BAND_SCALE_FLOOR)
    parser.add_argument("--fodf_power_scale_floor", type=float, default=cfg.FODF_POWER_SCALE_FLOOR)
    parser.add_argument("--fodf_aniso_min_l", type=int, default=cfg.FODF_ANISO_MIN_L)
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
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=cfg.PIN_MEMORY)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.AMP)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default=cfg.AMP_DTYPE)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=cfg.CHANNELS_LAST)
    parser.add_argument("--compile", choices=["off", "auto", "on"], default=cfg.COMPILE)
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default=cfg.COMPILE_MODE)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=cfg.DETERMINISTIC)
    parser.add_argument("--fused_adamw", action=argparse.BooleanOptionalAction, default=cfg.FUSED_ADAMW)
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=cfg.REQUIRE_CUDA)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=cfg.PROFILE)
    parser.add_argument("--profile_wait", type=int, default=cfg.PROFILE_WAIT)
    parser.add_argument("--profile_warmup", type=int, default=cfg.PROFILE_WARMUP)
    parser.add_argument("--profile_active", type=int, default=cfg.PROFILE_ACTIVE)
    parser.add_argument("--profile_repeat", type=int, default=cfg.PROFILE_REPEAT)
    parser.add_argument(
        "--profile_record_shapes",
        action=argparse.BooleanOptionalAction,
        default=cfg.PROFILE_RECORD_SHAPES,
    )
    parser.add_argument(
        "--profile_memory",
        action=argparse.BooleanOptionalAction,
        default=cfg.PROFILE_MEMORY,
    )
    parser.add_argument(
        "--profile_with_stack",
        action=argparse.BooleanOptionalAction,
        default=cfg.PROFILE_WITH_STACK,
    )
    parser.add_argument(
        "--profile_with_flops",
        action=argparse.BooleanOptionalAction,
        default=cfg.PROFILE_WITH_FLOPS,
    )
    parser.add_argument("--profile_row_limit", type=int, default=cfg.PROFILE_ROW_LIMIT)
    parser.add_argument(
        "--profile_exit_after_capture",
        action=argparse.BooleanOptionalAction,
        default=cfg.PROFILE_EXIT_AFTER_CAPTURE,
    )
    parser.add_argument("--no_brain_mask", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
