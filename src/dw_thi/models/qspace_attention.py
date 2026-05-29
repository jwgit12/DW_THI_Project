"""Lightweight q-space attention blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class QSpaceSelfAttention(nn.Module):
    """Self-attention over diffusion directions for each slice in a batch."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        grad_hidden: int = 128,
        use_signal_stats: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.use_signal_stats = use_signal_stats
        self.grad_mlp = nn.Sequential(
            nn.Linear(4, grad_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(grad_hidden, embed_dim),
        )
        self.stat_mlp = (
            nn.Sequential(
                nn.Linear(2, grad_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(grad_hidden, embed_dim),
            )
            if use_signal_stats
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        grad_info = torch.cat(
            [bvals.unsqueeze(-1), bvecs.permute(0, 2, 1)], dim=-1
        )
        tokens = self.grad_mlp(grad_info)

        if self.stat_mlp is not None:
            valid = vol_mask.view(vol_mask.shape[0], vol_mask.shape[1], 1, 1)
            masked_signal = signal * valid
            mean = masked_signal.mean(dim=(2, 3))
            centered = (signal - mean[..., None, None]) * valid
            std = torch.sqrt(centered.square().mean(dim=(2, 3)).clamp(min=1e-6))
            stats = torch.stack([mean, std], dim=-1)
            tokens = tokens + self.stat_mlp(stats) * vol_mask.unsqueeze(-1)

        key_padding_mask = vol_mask <= 0
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        tokens = tokens * vol_mask.unsqueeze(-1)
        attended = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        attended = self.norm(attended)
        return attended * vol_mask.unsqueeze(-1)


class AttentionAwareQSpaceFusion(nn.Module):
    """Fuse DWI slices with attention-contextualized direction embeddings."""

    def __init__(
        self,
        feat_dim: int = 64,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        grad_hidden: int = 128,
        use_signal_stats: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.attention = QSpaceSelfAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            grad_hidden=grad_hidden,
            use_signal_stats=use_signal_stats,
        )
        self.post = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        direction_features = self.attention(signal, bvals, bvecs, vol_mask)
        features = torch.einsum("bnhw,bnf->bfhw", signal, direction_features)
        n_eff = vol_mask.sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
        features = features / n_eff
        if signal.is_contiguous(memory_format=torch.channels_last):
            features = features.contiguous(memory_format=torch.channels_last)
        return self.post(features)
