#!/usr/bin/env python3
"""Build the fODF Zarr dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dw_thi.f_odf import defaults as cfg
from dw_thi.preprocessing import main


if __name__ == "__main__":
    main(settings=cfg, include_fodf=True)
