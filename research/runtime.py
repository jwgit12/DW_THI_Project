"""Runtime helpers shared by research training and evaluation scripts."""

from __future__ import annotations

import logging
import os
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger(__name__)


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    """Resolve user paths consistently across OSes and launch directories."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve(strict=False)


def path_str(path: str | os.PathLike[str]) -> str:
    return os.fspath(resolve_project_path(path))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_runtime(
    device: torch.device,
    *,
    matmul_precision: str = "high",
    deterministic: bool = False,
) -> None:
    """Enable the fast CUDA defaults that matter for RTX 40-series GPUs."""
    if device.type != "cuda":
        return

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        log.debug("Could not set float32 matmul precision: %s", exc)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    log.info(
        "CUDA: %s  capability=%d.%d  total_vram=%.1f GiB  cuda=%s  cudnn=%s",
        props.name,
        props.major,
        props.minor,
        props.total_memory / (1024**3),
        torch.version.cuda,
        torch.backends.cudnn.version(),
    )


def require_cuda_if_requested(device: torch.device, enabled: bool) -> None:
    if enabled and device.type != "cuda":
        raise RuntimeError(
            "CUDA was required but is not available. "
            f"Current torch build: {torch.__version__}. "
            "Install a CUDA-enabled PyTorch build for this OS before using the RTX GPU."
        )


def default_num_workers(requested: int | None) -> int:
    if requested is not None and requested >= 0:
        return requested

    cpu_count = os.cpu_count() or 1
    # This dataset preloads full subject arrays into RAM. Windows uses spawn
    # workers, which must pickle that dataset state and can fail before the
    # first batch; keep auto mode single-process there.
    if platform.system() == "Windows":
        return 0
    # macOS also uses spawn, so keep the automatic default conservative.
    if platform.system() == "Darwin":
        return min(2, max(1, cpu_count // 2))
    if platform.system() == "Linux":
        return min(12, max(1, cpu_count - 2))
    return min(4, max(1, cpu_count // 2))


def amp_dtype_from_name(device: torch.device, name: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    normalized = name.lower()
    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if normalized == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("bf16 AMP requested but unsupported; falling back to fp16")
        return torch.float16
    if normalized == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype | None,
) -> Any:
    if not enabled or dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype | None,
):
    use_scaler = enabled and device.type == "cuda" and dtype == torch.float16
    if not use_scaler:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def maybe_channels_last(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def should_compile_model(setting: str, device: torch.device) -> bool:
    if setting == "off" or device.type != "cuda" or not hasattr(torch, "compile"):
        return False
    if setting == "on":
        return True
    # Inductor support is most reliable on Linux. Users can still force it with
    # ``--compile on`` on other platforms when their PyTorch install supports it.
    return platform.system() == "Linux"


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    setting: str,
    device: torch.device,
    mode: str,
) -> tuple[torch.nn.Module, bool]:
    if not should_compile_model(setting, device):
        return model, False
    try:
        return torch.compile(model, mode=mode), True
    except Exception as exc:
        if setting == "on":
            raise
        log.warning("torch.compile unavailable; continuing eager: %s", exc)
        return model, False
