#!/usr/bin/env python3
"""One-off checkpoint -> ONNX exporter for DW_THI models.

Examples
--------
python src/export_pt_to_onnx.py \
  --checkpoint runs/qspace_attention/best_model.pt \
  --output runs/qspace_attention/model.onnx

python src/export_pt_to_onnx.py \
  --checkpoint runs/mrd_cnn/best_model.pt \
  --output runs/mrd_cnn/model.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("MRD_DEBUG_EVERY", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dw_thi.models import build_model, model_names


class ONNXModelWrapper(torch.nn.Module):
    """Expose the production forward signature with a stable output name."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(signal, bvals, bvecs, vol_mask)


def _checkpoint_value(ckpt: dict, key: str, default=None):
    value = ckpt.get(key, default)
    if isinstance(value, torch.Tensor):
        value = value.item()
    return value


def _canonical_hw_from_checkpoint(ckpt: dict) -> tuple[int, int]:
    run_config = ckpt.get("run_config") or {}
    hw = run_config.get("canonical_hw")
    if hw is None:
        return (128, 128)
    if len(hw) != 2:
        raise ValueError(f"Invalid run_config.canonical_hw in checkpoint: {hw!r}")
    return int(hw[0]), int(hw[1])


def load_checkpoint_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_name = ckpt.get("model_name")
    if not model_name:
        options = ", ".join(model_names())
        raise ValueError(
            "Checkpoint does not contain 'model_name'. "
            f"Expected one of: {options}. Re-save or patch the checkpoint metadata "
            "before exporting so the correct architecture is used."
        )

    max_n = int(_checkpoint_value(ckpt, "max_n"))
    feat_dim = int(_checkpoint_value(ckpt, "feat_dim", 64))
    channels = tuple(int(c) for c in ckpt.get("channels", [64, 128, 256, 512]))
    cholesky = bool(ckpt.get("cholesky", False))

    model = build_model(
        model_name,
        max_n=max_n,
        feat_dim=feat_dim,
        channels=channels,
        cholesky=cholesky,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metadata = {
        "model_name": model_name,
        "max_n": max_n,
        "feat_dim": feat_dim,
        "channels": list(channels),
        "cholesky": cholesky,
        "canonical_hw": list(_canonical_hw_from_checkpoint(ckpt)),
        "dti_scale": ckpt.get("dti_scale"),
        "max_bval": ckpt.get("max_bval"),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
    }
    return model, metadata


def build_dummy_inputs(
    *,
    batch_size: int,
    max_n: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    signal = torch.randn(batch_size, max_n, height, width, dtype=torch.float32)
    bvals = torch.zeros(batch_size, max_n, dtype=torch.float32)
    bvecs = torch.zeros(batch_size, 3, max_n, dtype=torch.float32)
    vol_mask = torch.ones(batch_size, max_n, dtype=torch.float32)
    return signal, bvals, bvecs, vol_mask


def export_onnx(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, metadata = load_checkpoint_model(checkpoint_path)
    height, width = metadata["canonical_hw"]
    if args.height is not None:
        height = args.height
    if args.width is not None:
        width = args.width

    wrapper = ONNXModelWrapper(model).eval()
    dummy_inputs = build_dummy_inputs(
        batch_size=args.batch_size,
        max_n=metadata["max_n"],
        height=int(height),
        width=int(width),
    )

    print("Exporting checkpoint:")
    print(json.dumps({**metadata, "height": int(height), "width": int(width)}, indent=2))
    print(f"ONNX output: {output_path}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "signal": {0: "batch"},
            "bvals": {0: "batch"},
            "bvecs": {0: "batch"},
            "vol_mask": {0: "batch"},
            "dti6d": {0: "batch"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=["signal", "bvals", "bvecs", "vol_mask"],
            output_names=["dti6d"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    if args.check:
        try:
            import onnx
        except ImportError:
            print("ONNX checker skipped: onnx is not installed.")
        else:
            onnx.checker.check_model(str(output_path))
            print("ONNX checker passed.")

    if args.metadata:
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps({**metadata, "height": int(height), "width": int(width)}, indent=2),
            encoding="utf-8",
        )
        print(f"Metadata output: {metadata_path}")

    print("Export complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DW_THI .pt checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt / last_model.pt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy export batch size")
    parser.add_argument("--height", type=int, default=None, help="Override export image height")
    parser.add_argument("--width", type=int, default=None, help="Override export image width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Export dynamic batch axis. Spatial size and max_n remain fixed.",
    )
    parser.add_argument(
        "--no_check",
        dest="check",
        action="store_false",
        help="Do not run onnx.checker after export.",
    )
    parser.add_argument(
        "--no_metadata",
        dest="metadata",
        action="store_false",
        help="Do not write the companion .metadata.json file.",
    )
    parser.set_defaults(check=True, metadata=True)
    return parser


if __name__ == "__main__":
    export_onnx(build_arg_parser().parse_args())
