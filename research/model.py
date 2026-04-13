"""U-Net with q-space encoder for DWI -> 6D DTI prediction."""

import math

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


def tensor6_to_fa_md(tensor6: torch.Tensor):
    """Compute FA and MD from a (B, 6, H, W) tensor using Frobenius norms.

    Order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    """
    dxx = tensor6[:, 0]
    dxy = tensor6[:, 1]
    dyy = tensor6[:, 2]
    dxz = tensor6[:, 3]
    dyz = tensor6[:, 4]
    dzz = tensor6[:, 5]

    md = (dxx + dyy + dzz) / 3.0

    # ||D - MD·I||_F²
    dev_sq = (dxx - md) ** 2 + (dyy - md) ** 2 + (dzz - md) ** 2 + 2 * (
        dxy**2 + dxz**2 + dyz**2
    )
    # ||D||_F²
    d_sq = dxx**2 + dyy**2 + dzz**2 + 2 * (dxy**2 + dxz**2 + dyz**2)

    fa = (math.sqrt(1.5) * torch.sqrt(dev_sq + 1e-12) / torch.sqrt(d_sq + 1e-12)).clamp(
        0.0, 1.0
    )
    return fa, md


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DTILoss(nn.Module):
    """MSE on 6D tensor + optional FA/MD MAE regularisation."""

    def __init__(self, lambda_scalar: float = cfg.LAMBDA_SCALAR):
        super().__init__()
        self.lambda_scalar = lambda_scalar

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if brain_mask is not None:
            mask_4d = brain_mask.unsqueeze(1)  # (B, 1, H, W)
            n_voxels = mask_4d.sum().clamp(min=1)

            tensor_loss = ((pred - target) ** 2 * mask_4d).sum() / (n_voxels * 6)

            if self.lambda_scalar > 0:
                pred_fa, pred_md = tensor6_to_fa_md(pred)
                tgt_fa, tgt_md = tensor6_to_fa_md(target)
                mask_2d = brain_mask
                n_2d = mask_2d.sum().clamp(min=1)
                fa_loss = (torch.abs(pred_fa - tgt_fa) * mask_2d).sum() / n_2d
                md_loss = (torch.abs(pred_md - tgt_md) * mask_2d).sum() / n_2d
                total = tensor_loss + self.lambda_scalar * (fa_loss + md_loss)
                return total, {
                    "tensor_mse": tensor_loss.item(),
                    "fa_mae": fa_loss.item(),
                    "md_mae": md_loss.item(),
                }
        else:
            tensor_loss = F.mse_loss(pred, target)
            if self.lambda_scalar > 0:
                pred_fa, pred_md = tensor6_to_fa_md(pred)
                tgt_fa, tgt_md = tensor6_to_fa_md(target)
                fa_loss = F.l1_loss(pred_fa, tgt_fa)
                md_loss = F.l1_loss(pred_md, tgt_md)
                total = tensor_loss + self.lambda_scalar * (fa_loss + md_loss)
                return total, {
                    "tensor_mse": tensor_loss.item(),
                    "fa_mae": fa_loss.item(),
                    "md_mae": md_loss.item(),
                }

        return tensor_loss, {"tensor_mse": tensor_loss.item()}


# ---------------------------------------------------------------------------
# Q-space encoder
# ---------------------------------------------------------------------------

class QSpaceEncoder(nn.Module):
    """Compress N DWI volumes into C feature channels.

    1×1 conv collapses the diffusion dimension; a small MLP embeds the
    gradient table (bvals/bvecs) into FiLM parameters (scale + shift)
    that modulate the spatial features.
    """

    def __init__(self, max_n: int, feat_dim: int = 64):
        super().__init__()
        self.signal_conv = nn.Sequential(
            nn.Conv2d(max_n, feat_dim, 1, bias=False),
            nn.GroupNorm(8, feat_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.grad_mlp = nn.Sequential(
            nn.Linear(4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim * 2),
        )

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.signal_conv(signal)  # (B, F, H, W)

        grad_info = torch.cat(
            [bvals.unsqueeze(-1), bvecs.permute(0, 2, 1)], dim=-1
        )  # (B, N, 4)
        grad_feat = self.grad_mlp(grad_info)  # (B, N, 2F)

        m = vol_mask.unsqueeze(-1)
        grad_feat = (grad_feat * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)  # (B, 2F)
        gamma, beta = grad_feat.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
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
    ):
        super().__init__()
        self.depth = len(channels)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for c in channels:
            self.encoders.append(ConvBlock(ch, c))
            self.pools.append(nn.MaxPool2d(2))
            ch = c

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = channels[-1] * 2
        for c in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(ch, c, 2, stride=2))
            self.decoders.append(ConvBlock(c * 2, c))
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
    ):
        super().__init__()
        self.max_n = max_n
        self.cholesky = cholesky
        self.q_encoder = QSpaceEncoder(max_n, feat_dim)
        self.unet = UNet2D(feat_dim, out_ch=6, channels=channels)

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
