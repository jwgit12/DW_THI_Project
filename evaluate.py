#!/usr/bin/env python3
"""Evaluation entry point for the standard FA/MD or fODF pipelines."""

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

EVAL_MODULES = {
    "standard": "dw_thi.evaluate",
    "fa-md": "dw_thi.evaluate",
    "fa_md": "dw_thi.evaluate",
    "famd": "dw_thi.evaluate",
    "attention": "evaluate_attention",
    "qspace-attention": "evaluate_attention",
    "qspace_attention": "evaluate_attention",
    "f-odf": "fodf.evaluate",
    "f_odf": "fodf.evaluate",
    "fodf": "fodf.evaluate",
}


def _normalize_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized not in EVAL_MODULES:
        choices = ", ".join(sorted(EVAL_MODULES))
        raise SystemExit(f"Unsupported evaluation mode {mode!r}. Choose one of: {choices}")
    if normalized in {"qspace-attention", "qspace_attention"}:
        return "attention"
    if normalized in {"fodf", "f_odf"}:
        return "f-odf"
    if normalized in {"fa_md", "famd"}:
        return "fa-md"
    return normalized


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    dispatch_parser = argparse.ArgumentParser(add_help=False)
    dispatch_parser.add_argument(
        "--training",
        "--mode",
        choices=sorted(EVAL_MODULES),
        default="standard",
        help="Evaluation pipeline to run: standard/fa-md or f-odf.",
    )
    dispatch_args, remaining = dispatch_parser.parse_known_args(raw_args)
    mode = _normalize_mode(dispatch_args.training)

    module = importlib.import_module(EVAL_MODULES[mode])
    parser = module.build_arg_parser()
    parser.prog = f"{Path(sys.argv[0]).name} --training {mode}"
    module.main(parser.parse_args(remaining))


if __name__ == "__main__":
    main()
