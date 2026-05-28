#!/usr/bin/env python3
"""Evaluation entry point for the DWI -> DTI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dw_thi.evaluate import build_arg_parser, main

if __name__ == "__main__":
    main(build_arg_parser().parse_args())
