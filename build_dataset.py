#!/usr/bin/env python3
"""Build the clean production Zarr dataset.

This entry point writes target_dwi, target_dti_6d, bvals, bvecs, and the
precomputed brain_mask. The brain mask is computed in src/dw_thi/preprocessing.py
with DIPY's median_otsu on the mean b0 volume.
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


if __name__ == "__main__":
    main()
