"""MRD-CNN: residual DWI denoising followed by residual CNN tensor estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import config as cfg
from ..model import QSpaceEncoder, cholesky_to_tensor6


class FiLMResidualBlock(nn.Module):
    """Residual conv block with optional feature-wise affine conditioning."""

    def __init__(self, channels: int, dropout: float = 0.0, film_dim: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.film = nn.Linear(film_dim, channels * 2) if film_dim is not None else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        y = self.act(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            y = y * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        return self.act(y + residual)


class DirectionConditionedResidualDenoiser(nn.Module):
    """Predict a residual corruption map for each DWI direction."""

    def __init__(
        self,
        channels: int = 16,
        depth: int = 4,
        dropout: float = 0.0,
        grad_hidden: int = 64,
        residual_scale: float = 0.5,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.grad_mlp = nn.Sequential(
            nn.Linear(4, grad_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(grad_hidden, grad_hidden),
            nn.SiLU(inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(channels, dropout=dropout, film_dim=grad_hidden)
                for _ in range(depth)
            ]
        )
        self.head = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, h, w = signal.shape
        grad_info = torch.cat([bvals.unsqueeze(-1), bvecs.permute(0, 2, 1)], dim=-1)
        cond = self.grad_mlp(grad_info.reshape(b * n, 4))

        x = signal.reshape(b * n, 1, h, w)
        y = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                y = checkpoint(block, y, cond, use_reentrant=False)
            else:
                y = block(y, cond)
        residual = torch.tanh(self.head(y)).reshape(b, n, h, w)
        residual = residual * self.residual_scale * vol_mask[:, :, None, None]
        denoised = (signal - residual) * vol_mask[:, :, None, None]
        return denoised, residual


class ResidualTensorCNN(nn.Module):
    """Stacked residual CNN tensor head, deliberately not a U-Net."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 64,
        depth: int = 8,
        dropout: float = 0.0,
        out_ch: int = 6,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_ch, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


class MRDCNN(nn.Module):
    """Residual DWI denoising CNN + residual CNN tensor estimator."""

    def __init__(
        self,
        max_n: int,
        feat_dim: int = 64,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        cholesky: bool = False,
        dropout: float = cfg.DROPOUT,
        denoise_channels: int = 16,
        denoise_depth: int = 4,
        tensor_channels: int | None = None,
        tensor_depth: int = 8,
        residual_scale: float = 0.5,
        grad_hidden: int = 64,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        self.max_n = max_n
        self.cholesky = cholesky
        self.denoiser = DirectionConditionedResidualDenoiser(
            channels=denoise_channels,
            depth=denoise_depth,
            dropout=dropout,
            grad_hidden=grad_hidden,
            residual_scale=residual_scale,
            checkpoint_blocks=checkpoint_blocks,
        )
        self.q_encoder = QSpaceEncoder(feat_dim=feat_dim, grad_hidden=max(128, grad_hidden))
        hidden = int(tensor_channels if tensor_channels is not None else (channels[0] if channels else feat_dim))
        self.tensor_head = ResidualTensorCNN(
            in_ch=feat_dim,
            hidden_ch=hidden,
            depth=tensor_depth,
            dropout=dropout,
            out_ch=6,
            checkpoint_blocks=checkpoint_blocks,
        )

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        denoised, _residual = self.denoiser(signal, bvals, bvecs, vol_mask)
        features = self.q_encoder(denoised, bvals, bvecs, vol_mask)
        out = self.tensor_head(features)
        if self.cholesky:
            out = cholesky_to_tensor6(out)
        return out
