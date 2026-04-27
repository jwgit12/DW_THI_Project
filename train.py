#!/usr/bin/env python3
"""Training entry point for standard FA/MD or fODF training."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_MODULES = {
    "standard": "dw_thi.train",
    "fa-md": "dw_thi.train",
    "fa_md": "dw_thi.train",
    "famd": "dw_thi.train",
    "f-odf": "dw_thi.f_odf.train",
    "f_odf": "dw_thi.f_odf.train",
    "fodf": "dw_thi.f_odf.train",
}


def _normalize_training_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized not in TRAINING_MODULES:
        choices = ", ".join(sorted(TRAINING_MODULES))
        raise SystemExit(f"Unsupported training mode {mode!r}. Choose one of: {choices}")
    if normalized in {"fodf", "f_odf"}:
        return "f-odf"
    if normalized in {"fa_md", "famd"}:
        return "fa-md"
    return normalized


def _load_training_module(mode: str):
    return importlib.import_module(TRAINING_MODULES[mode])


def build_arg_parser(training: str = "standard") -> argparse.ArgumentParser:
    """Return the mode-specific training parser."""
    mode = _normalize_training_mode(training)
    return _load_training_module(mode).build_arg_parser()


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    dispatch_parser = argparse.ArgumentParser(add_help=False)
    dispatch_parser.add_argument(
        "--training",
        "--mode",
        choices=sorted(TRAINING_MODULES),
        default="standard",
        help="Training pipeline to run: standard/fa-md or f-odf.",
    )
    dispatch_args, remaining = dispatch_parser.parse_known_args(raw_args)
    mode = _normalize_training_mode(dispatch_args.training)

    module = _load_training_module(mode)
    parser = module.build_arg_parser()
    parser.prog = f"{Path(sys.argv[0]).name} --training {mode}"
    module.main(parser.parse_args(remaining))


if __name__ == "__main__":
    main()
