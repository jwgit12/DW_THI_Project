"""U-Net with q-space encoder for DWI -> DTI (+ optional fODF SH) prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

DTI_CHANNELS = 6


# ---------------------------------------------------------------------------
# Differentiable DTI scalar maps (no eigendecomposition — MPS-safe)
# ---------------------------------------------------------------------------

def cholesky_to_tensor6(chol6: torch.Tensor) -> torch.Tensor:
    """Convert 6 Cholesky factor channels to 6 symmetric PSD tensor channels.

    Input order:  l11, l21, l22, l31, l32, l33  (lower-triangular L)
    Output order:  Dxx, Dxy, Dyy, Dxz, Dyz, Dzz  (D = L @ L^T)

    Diagonal elements are passed through softplus to guarantee positivity.
    """
    l11 = F.softplus(chol6[:, 0])
    l21 = chol6[:, 1]
    l22 = F.softplus(chol6[:, 2])
    l31 = chol6[:, 3]
    l32 = chol6[:, 4]
    l33 = F.softplus(chol6[:, 5])

    dxx = l11 * l11
    dxy = l11 * l21
    dyy = l21 * l21 + l22 * l22
    dxz = l11 * l31
    dyz = l21 * l31 + l22 * l32
    dzz = l31 * l31 + l32 * l32 + l33 * l33

    return torch.stack([dxx, dxy, dyy, dxz, dyz, dzz], dim=1)


# ---------------------------------------------------------------------------
# Q-space encoder
# ---------------------------------------------------------------------------

class QSpaceEncoder(nn.Module):
    """Permutation-invariant q-space encoder.

    Each diffusion-weighted volume is aggregated into ``feat_dim`` feature
    channels via a learned, gradient-conditioned weighted sum. The encoder
    accepts either a single 2D slice ``(B, N, H, W)`` or a small 2.5D stack
    ``(B, N, D, H, W)``; output is independent of volume ordering and of
    ``max_n`` padding:

        features[b, f, h, w] = sum_n signal[b, n, h, w] * e[b, n, f]

    where ``e = MLP(bval, bvec)`` is a per-volume embedding. A post 3x3
    conv block mixes features spatially.
    """

    def __init__(self, feat_dim: int = 64, grad_hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.grad_mlp = nn.Sequential(
            nn.Linear(4, grad_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(grad_hidden, grad_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(grad_hidden, feat_dim),
        )
        self.post = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_dim),
            nn.SiLU(inplace=True),
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
        # signal: (B, N, H, W), bvals: (B, N), bvecs: (B, 3, N), vol_mask: (B, N)
        grad_info = torch.cat(
            [bvals.unsqueeze(-1), bvecs.permute(0, 2, 1)], dim=-1
        )  # (B, N, 4)
        e = self.grad_mlp(grad_info)  # (B, N, F)
        e = e * vol_mask.unsqueeze(-1)  # zero padded / dropped volumes

        if signal.ndim == 4:
            # Permutation-invariant aggregation across the N diffusion axis.
            features = torch.einsum("bnhw,bnf->bfhw", signal, e)

            # Normalize by effective number of valid volumes to decouple feature
            # magnitude from N (subjects have varying numbers of volumes).
            n_eff = vol_mask.sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
            features = features / n_eff
            if signal.is_contiguous(memory_format=torch.channels_last):
                features = features.contiguous(memory_format=torch.channels_last)

            return self.post(features)

        if signal.ndim == 5:
            # signal: (B, N, D, H, W). Reuse the 2D spatial post-encoder for
            # each neighboring slice, then hand the depth stack to context
            # fusion in QSpaceUNet.
            features = torch.einsum("bndhw,bnf->bfdhw", signal, e)
            n_eff = vol_mask.sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1, 1)
            features = features / n_eff

            b, f, d, h, w = features.shape
            features_2d = features.permute(0, 2, 1, 3, 4).reshape(b * d, f, h, w)
            features_2d = self.post(features_2d.contiguous())
            return (
                features_2d.reshape(b, d, f, h, w)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )

        raise ValueError(
            f"QSpaceEncoder expected signal with 4 or 5 dims, got shape {tuple(signal.shape)}."
        )


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ContextFusionBlock(nn.Module):
    """Residual 3D block over a short through-plane feature stack."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ContextFusion2p5D(nn.Module):
    """Fuse neighboring slice features and return the central feature map."""

    def __init__(
        self,
        channels: int,
        *,
        context_slices: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd integer.")
        self.context_slices = int(context_slices)
        self.blocks = nn.Sequential(
            *[ContextFusionBlock(channels, dropout=dropout) for _ in range(max(1, n_layers))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W). Dynamic center keeps older/smaller context calls valid.
        x = self.blocks(x)
        return x[:, :, x.shape[2] // 2]


class UNet2D(nn.Module):
    """Standard 2D U-Net with automatic padding for arbitrary spatial dims."""

    def __init__(
        self,
        in_ch: int = 64,
        out_ch: int = 6,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = len(channels)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for c in channels:
            self.encoders.append(ConvBlock(ch, c, dropout=dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = c

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = channels[-1] * 2
        for c in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(ch, c, 2, stride=2))
            self.decoders.append(ConvBlock(c * 2, c, dropout=dropout))
            ch = c

        self.head = nn.Conv2d(channels[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**self.depth
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh or dw:
                x = F.pad(x, (0, dw, 0, dh))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)

        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class QSpaceUNet(nn.Module):
    """Q-space encoder + 2D/2.5D U-Net: DWI volumes -> DTI (+ optional fODF SH)."""

    def __init__(
        self,
        max_n: int,
        feat_dim: int = 64,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        cholesky: bool = False,
        fodf_channels: int = 0,
        dti_channels: int = DTI_CHANNELS,
        dropout: float = cfg.DROPOUT,
        context_slices: int = 1,
        context_fusion_layers: int = getattr(cfg, "CONTEXT_FUSION_LAYERS", 2),
    ):
        super().__init__()
        # ``max_n`` is kept in the signature for checkpoint metadata and so
        # the dataset/evaluator can continue to pad to a shared length. The
        # encoder itself is now permutation-invariant and does not need it.
        self.max_n = max_n
        self.cholesky = cholesky
        self.dti_channels = int(dti_channels)
        self.fodf_channels = int(fodf_channels)
        self.context_slices = int(context_slices)
        self.context_fusion_layers = int(context_fusion_layers)
        if self.context_slices < 1 or self.context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd integer.")
        out_ch = self.dti_channels + self.fodf_channels
        if out_ch <= 0:
            raise ValueError("QSpaceUNet must predict at least one DTI or fODF channel.")
        if self.cholesky and self.dti_channels not in (0, DTI_CHANNELS):
            raise ValueError(
                f"cholesky=True requires dti_channels={DTI_CHANNELS}, got {self.dti_channels}."
            )
        self.q_encoder = QSpaceEncoder(feat_dim=feat_dim)
        self.context_fusion = (
            ContextFusion2p5D(
                feat_dim,
                context_slices=self.context_slices,
                n_layers=self.context_fusion_layers,
                dropout=dropout,
            )
            if self.context_slices > 1
            else None
        )
        self.unet = UNet2D(
            feat_dim,
            out_ch=out_ch,
            channels=channels,
            dropout=dropout,
        )

    def split_outputs(
        self,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dti = output[:, : self.dti_channels] if self.dti_channels > 0 else None
        fodf = output[:, self.dti_channels :] if self.fodf_channels > 0 else None
        return dti, fodf

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.q_encoder(signal, bvals, bvecs, vol_mask)
        if features.ndim == 5:
            if self.context_fusion is not None:
                features = self.context_fusion(features)
            else:
                features = features[:, :, features.shape[2] // 2]
        out = self.unet(features)
        if self.cholesky and self.dti_channels == DTI_CHANNELS:
            dti, fodf = self.split_outputs(out)
            dti = cholesky_to_tensor6(dti)
            out = torch.cat([dti, fodf], dim=1) if fodf is not None else dti
        return out
