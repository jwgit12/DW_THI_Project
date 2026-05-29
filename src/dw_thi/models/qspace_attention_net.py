"""Q-space attention U-Net for DWI -> 6D DTI prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg
from ..model import UNet2D, cholesky_to_tensor6
from .qspace_attention import AttentionAwareQSpaceFusion


class QSpaceAttentionNet(nn.Module):
    """Direction-attention encoder + 2D U-Net: DWI volumes -> DTI tensor."""

    def __init__(
        self,
        max_n: int,
        feat_dim: int = 64,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        cholesky: bool = False,
        dropout: float = cfg.DROPOUT,
        attention_heads: int = 4,
        attention_depth: int = 2,
        attention_dropout: float | None = None,
        grad_hidden: int = 128,
        use_signal_stats: bool = True,
    ):
        super().__init__()
        self.max_n = max_n
        self.cholesky = cholesky
        self.attention_heads = attention_heads
        self.attention_depth = attention_depth
        self.q_encoder = AttentionAwareQSpaceFusion(
            feat_dim=feat_dim,
            num_heads=attention_heads,
            depth=attention_depth,
            dropout=dropout if attention_dropout is None else attention_dropout,
            grad_hidden=grad_hidden,
            use_signal_stats=use_signal_stats,
        )
        self.unet = UNet2D(feat_dim, out_ch=6, channels=channels, dropout=dropout)

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.q_encoder(signal, bvals, bvecs, vol_mask)
        out = self.unet(features)
        if self.cholesky:
            out = cholesky_to_tensor6(out)
        return out
