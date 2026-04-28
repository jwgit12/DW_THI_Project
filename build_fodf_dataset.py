#!/usr/bin/env python3
"""Build the fODF Zarr dataset.

Same preprocessing pipeline as ``build_dataset.py`` but additionally fits a
single-shell CSD ODF and stores the SH coefficients per subject. fODF defaults
come from ``src/fodf/defaults.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dw_thi.preprocessing import main
from fodf import defaults as cfg


if __name__ == "__main__":
    main(settings=cfg, include_fodf=True)
