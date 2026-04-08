"""
Direction-invariant Residual U-Net for DWI denoising.

Architecture
------------
Each DWI direction is processed independently by a **shared** spatial encoder.
Features from all valid directions are aggregated via masked mean to form a
global context.  A shared decoder reconstructs each direction using both
per-direction and global features.  Output uses residual learning.

Model presets
-------------
    tiny  : base_features=16  (~1.1M params) — fast prototyping
    small : base_features=24  (~2.6M params) — quick experiments
    base  : base_features=48  (~10M params)  — full training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PRESETS = {
    "tiny":  {"base_features": 16},
    "small": {"base_features": 24},
    "base":  {"base_features": 48},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gn_groups(ch: int, max_groups: int = 32) -> int:
    """Find largest group count <= max_groups that divides ch."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branch during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class ChannelAttention(nn.Module):
    """Channel attention (CBAM component)."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention (CBAM component)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel + spatial attention.

    Stronger than SE blocks: captures both channel interdependencies and
    spatial saliency (Woo et al., ECCV 2018).
    """

    def __init__(self, ch: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(ch, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ResConvBlock(nn.Module):
    """Two-conv residual block with GroupNorm, LeakyReLU, and DropPath."""

    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop_path(self.conv(x)) + self.shortcut(x))


class DownBlock(nn.Module):
    """MaxPool2d downsample + ResConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResConvBlock(in_ch, out_ch, drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """ConvTranspose2d upsample + skip concat + ResConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ResConvBlock(in_ch + skip_ch, out_ch, drop_path)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        if dy > 0 or dx > 0:
            x = F.pad(x, [max(dx // 2, 0), max(dx - dx // 2, 0),
                          max(dy // 2, 0), max(dy - dy // 2, 0)])
        if dy < 0:
            crop = (-dy) // 2
            x = x[:, :, crop:crop + skip.shape[2], :]
        if dx < 0:
            crop = (-dx) // 2
            x = x[:, :, :, crop:crop + skip.shape[3]]
        return self.conv(torch.cat([x, skip], dim=1))


def _masked_mean(feats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over directions.
    feats:      (B, N, C, H, W)
    valid_mask: (B, N, 1, 1, 1)
    """
    valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
    return (feats * valid_mask).sum(dim=1) / valid_count


# ---------------------------------------------------------------------------
# Shared encoder / decoder
# ---------------------------------------------------------------------------
class DirEncoder(nn.Module):
    """Shared encoder applied independently to each DWI direction.
    Input:  (*, 7, H, W) — signal + bval + 3x bvec + 2x position (y, x)
    Output: multi-scale feature list for skip connections
    """

    def __init__(self, f: int = 48, drop_path: float = 0.0):
        super().__init__()
        self.enc1 = ResConvBlock(7, f, drop_path)
        self.enc2 = DownBlock(f, f * 2, drop_path)
        self.enc3 = DownBlock(f * 2, f * 4, drop_path)
        self.enc4 = DownBlock(f * 4, f * 8, drop_path)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4


class DirDecoder(nn.Module):
    """Shared decoder with CBAM attention at bottleneck.
    Produces a residual image (added to input in the main model).
    """

    def __init__(self, f: int = 48, drop_path: float = 0.0):
        super().__init__()
        self.bottleneck = nn.Sequential(
            ResConvBlock(f * 8 * 2, f * 8, drop_path),
            CBAM(f * 8),
        )
        self.up3 = UpBlock(f * 8, f * 4 * 2, f * 4, drop_path)
        self.up2 = UpBlock(f * 4, f * 2 * 2, f * 2, drop_path)
        self.up1 = UpBlock(f * 2, f * 2, f, drop_path)
        self.head = nn.Conv2d(f, 1, 1)

    def forward(self, dir_feats, global_feats):
        e1_d, e2_d, e3_d, e4_d = dir_feats
        e1_g, e2_g, e3_g, e4_g = global_feats

        x = self.bottleneck(torch.cat([e4_d, e4_g], dim=1))
        x = self.up3(x, torch.cat([e3_d, e3_g], dim=1))
        x = self.up2(x, torch.cat([e2_d, e2_g], dim=1))
        x = self.up1(x, torch.cat([e1_d, e1_g], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class DenoiseUNet(nn.Module):
    """Direction-invariant Residual U-Net for DWI denoising.

    Parameters
    ----------
    base_features : int
        Feature width of the first encoder level (use MODEL_PRESETS).
    drop_path : float
        DropPath rate for stochastic depth regularization.
    """

    def __init__(self, base_features: int = 48, drop_path: float = 0.0):
        super().__init__()
        self.dir_encoder = DirEncoder(base_features, drop_path)
        self.dir_decoder = DirDecoder(base_features, drop_path)

    def forward(
        self,
        noisy_dwi: torch.Tensor,
        pad_mask: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        pos_enc: torch.Tensor,
        dir_chunk_size: int = 16,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_dwi : (B, N, H, W)
        pad_mask  : (B, N)  — 1.0 = real direction, 0.0 = padding
        bvals     : (B, N)
        bvecs     : (B, 3, N)
        pos_enc   : (B, 2, H, W) — normalized (y, x) position coordinates
        dir_chunk_size : int — max directions per encoder/decoder pass
        """
        B, N, H, W = noisy_dwi.shape
        BN = B * N

        # Per-direction input: signal + bval + 3x bvec + 2x position
        sigs = noisy_dwi.reshape(BN, 1, H, W)
        bvals_norm = (bvals / 1000.0).reshape(BN, 1, 1, 1).expand(-1, -1, H, W)
        bvecs_r = bvecs.transpose(1, 2).reshape(BN, 3, 1, 1).expand(-1, -1, H, W)
        # Repeat position encoding for each direction: (B, 2, H, W) -> (BN, 2, H, W)
        pos_r = pos_enc.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(BN, 2, H, W)
        x = torch.cat([sigs, bvals_norm, bvecs_r, pos_r], dim=1)  # (BN, 7, H, W)

        # Chunked encoding
        e1_parts, e2_parts, e3_parts, e4_parts = [], [], [], []
        for i in range(0, BN, dir_chunk_size):
            c1, c2, c3, c4 = self.dir_encoder(x[i:i + dir_chunk_size])
            e1_parts.append(c1)
            e2_parts.append(c2)
            e3_parts.append(c3)
            e4_parts.append(c4)
        e1 = torch.cat(e1_parts, dim=0)
        e2 = torch.cat(e2_parts, dim=0)
        e3 = torch.cat(e3_parts, dim=0)
        e4 = torch.cat(e4_parts, dim=0)

        f4, h4, w4 = e4.shape[1:]
        f3, h3, w3 = e3.shape[1:]
        f2, h2, w2 = e2.shape[1:]
        f1, h1, w1 = e1.shape[1:]

        # Reshape for aggregation: (B, N, C, h, w)
        e4_r = e4.reshape(B, N, f4, h4, w4)
        e3_r = e3.reshape(B, N, f3, h3, w3)
        e2_r = e2.reshape(B, N, f2, h2, w2)
        e1_r = e1.reshape(B, N, f1, h1, w1)

        # Masked mean aggregation
        valid = pad_mask.to(e4_r.dtype).reshape(B, N, 1, 1, 1)
        g4 = _masked_mean(e4_r, valid)
        g3 = _masked_mean(e3_r, valid)
        g2 = _masked_mean(e2_r, valid)
        g1 = _masked_mean(e1_r, valid)

        # Chunked decoding
        flat_to_batch = torch.arange(BN, device=noisy_dwi.device) // N
        dwi_parts = []
        for i in range(0, BN, dir_chunk_size):
            s = slice(i, i + dir_chunk_size)
            b_idx = flat_to_batch[s]
            residual = self.dir_decoder(
                (e1[s], e2[s], e3[s], e4[s]),
                (g1[b_idx], g2[b_idx], g3[b_idx], g4[b_idx]),
            )
            dwi_parts.append(residual)
        residual_all = torch.cat(dwi_parts, dim=0)

        # Residual learning: denoised = noisy + learned_residual
        denoised = sigs + residual_all
        denoised = denoised.reshape(B, N, H, W)
        denoised = denoised * pad_mask.unsqueeze(-1).unsqueeze(-1)
        return denoised


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
