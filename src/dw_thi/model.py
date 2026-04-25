"""U-Net with q-space encoder for DWI -> 6D DTI prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


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
    channels via a learned, gradient-conditioned weighted sum — output is
    independent of volume ordering and of ``max_n`` padding:

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

        # Permutation-invariant aggregation across the N diffusion axis.
        features = torch.einsum("bnhw,bnf->bfhw", signal, e)

        # Normalize by effective number of valid volumes to decouple feature
        # magnitude from N (subjects have varying numbers of volumes).
        n_eff = vol_mask.sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
        features = features / n_eff
        if signal.is_contiguous(memory_format=torch.channels_last):
            features = features.contiguous(memory_format=torch.channels_last)

        return self.post(features)


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
    """Q-space encoder + 2D U-Net: DWI volumes -> 6D DTI tensor."""

    def __init__(
        self,
        max_n: int,
        feat_dim: int = 64,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        cholesky: bool = False,
        dropout: float = cfg.DROPOUT,
    ):
        super().__init__()
        # ``max_n`` is kept in the signature for checkpoint metadata and so
        # the dataset/evaluator can continue to pad to a shared length. The
        # encoder itself is now permutation-invariant and does not need it.
        self.max_n = max_n
        self.cholesky = cholesky
        self.q_encoder = QSpaceEncoder(feat_dim=feat_dim)
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
