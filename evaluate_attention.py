#!/usr/bin/env python3
"""Evaluation entry point for QSpaceAttentionNet checkpoints."""

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

import config_attention as cfg_attention
import dw_thi.evaluate as base_evaluate
from dw_thi.models import QSpaceAttentionNet


class _ConfiguredQSpaceAttentionNet(QSpaceAttentionNet):
    def __init__(
        self,
        max_n: int,
        feat_dim: int = cfg_attention.FEAT_DIM,
        channels: tuple[int, ...] = tuple(cfg_attention.UNET_CHANNELS),
        cholesky: bool = False,
        dropout: float = cfg_attention.DROPOUT,
    ):
        super().__init__(
            max_n=max_n,
            feat_dim=feat_dim,
            channels=channels,
            cholesky=cholesky,
            dropout=dropout,
            attention_heads=cfg_attention.ATTENTION_HEADS,
            attention_depth=cfg_attention.ATTENTION_DEPTH,
            attention_dropout=cfg_attention.ATTENTION_DROPOUT,
            grad_hidden=cfg_attention.ATTENTION_GRAD_HIDDEN,
            use_signal_stats=cfg_attention.ATTENTION_USE_SIGNAL_STATS,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    original_cfg = base_evaluate.cfg
    try:
        base_evaluate.cfg = cfg_attention
        parser = base_evaluate.build_arg_parser()
    finally:
        base_evaluate.cfg = original_cfg

    parser.description = "Evaluate QSpaceAttentionNet on test subjects"
    parser.set_defaults(
        checkpoint=cfg_attention.EVAL_DEFAULT_CHECKPOINT,
        out_dir=cfg_attention.EVAL_OUT_DIR,
    )
    return parser


def main(args: argparse.Namespace) -> None:
    original_cfg = base_evaluate.cfg
    original_model = base_evaluate.QSpaceUNet
    try:
        base_evaluate.cfg = cfg_attention
        base_evaluate.QSpaceUNet = _ConfiguredQSpaceAttentionNet
        base_evaluate.main(args)
    finally:
        base_evaluate.QSpaceUNet = original_model
        base_evaluate.cfg = original_cfg


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
