#!/usr/bin/env python3
"""Evaluation entry point for MRD-CNN checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_mrd as cfg_mrd
import dw_thi.evaluate as base_evaluate
from dw_thi.models import MRDCNN


class _ConfiguredMRDCNN(MRDCNN):
    def __init__(
        self,
        max_n: int,
        feat_dim: int = cfg_mrd.FEAT_DIM,
        channels: tuple[int, ...] = tuple(cfg_mrd.UNET_CHANNELS),
        cholesky: bool = False,
        dropout: float = cfg_mrd.DROPOUT,
    ):
        super().__init__(
            max_n=max_n,
            feat_dim=feat_dim,
            channels=channels,
            cholesky=cholesky,
            dropout=dropout,
            denoise_channels=cfg_mrd.MRD_DENOISE_CHANNELS,
            denoise_depth=cfg_mrd.MRD_DENOISE_DEPTH,
            tensor_channels=cfg_mrd.MRD_TENSOR_CHANNELS,
            tensor_depth=cfg_mrd.MRD_TENSOR_DEPTH,
            residual_scale=cfg_mrd.MRD_RESIDUAL_SCALE,
            grad_hidden=cfg_mrd.MRD_GRAD_HIDDEN,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    original_cfg = base_evaluate.cfg
    try:
        base_evaluate.cfg = cfg_mrd
        parser = base_evaluate.build_arg_parser()
    finally:
        base_evaluate.cfg = original_cfg

    parser.description = "Evaluate MRD-CNN on test subjects"
    parser.set_defaults(
        checkpoint=cfg_mrd.EVAL_DEFAULT_CHECKPOINT,
        out_dir=cfg_mrd.EVAL_OUT_DIR,
    )
    return parser


def main(args: argparse.Namespace) -> None:
    original_cfg = base_evaluate.cfg
    original_model = base_evaluate.QSpaceUNet
    try:
        base_evaluate.cfg = cfg_mrd
        base_evaluate.QSpaceUNet = _ConfiguredMRDCNN
        base_evaluate.main(args)
    finally:
        base_evaluate.QSpaceUNet = original_model
        base_evaluate.cfg = original_cfg


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
