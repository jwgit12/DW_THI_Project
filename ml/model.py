"""
Channel-invariant U-Net for the DTI pretext task.

Each DWI direction is processed independently by a *shared* spatial encoder.
Features are aggregated across directions (masked mean), then decoded into:
  - DTI head  → (B, 6, H, W)
  - DWI head  → per-direction reconstruction via shared decoder

This design naturally handles subjects with different numbers of gradient
directions in the same batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        if dy != 0 or dx != 0:
            x = F.pad(x, [0, dx, 0, dy])
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Shared direction encoder / decoder
# ---------------------------------------------------------------------------
class DirectionEncoder(nn.Module):
    """Small encoder applied independently to each DWI direction.

    Input : (*, 6, H, W)  — signal, mask-flag, bval, 3xbvec
    Output: multi-scale feature list for skip connections
    """

    def __init__(self, f: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(6, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)

    def forward(self, x):
        e1 = self.enc1(x)       # (*, f,   H,   W)
        e2 = self.enc2(e1)      # (*, f*2, H/2, W/2)
        e3 = self.enc3(e2)      # (*, f*4, H/4, W/4)
        return e1, e2, e3


class DirectionDecoder(nn.Module):
    """Shared decoder that reconstructs one DWI direction.

    Input : global context features (multi-scale) + per-direction features
    Output: (*, 1, H, W)
    """

    def __init__(self, f: int = 32):
        super().__init__()
        # Takes concatenation of per-dir features + global features at each scale
        self.up2 = UpBlock(f * 4 * 2, f * 2 * 2, f * 2)
        self.up1 = UpBlock(f * 2, f * 2, f)
        self.head = nn.Conv2d(f, 1, 1)

    def forward(self, dir_feats, global_feats):
        e1_d, e2_d, e3_d = dir_feats
        e1_g, e2_g, e3_g = global_feats

        x = torch.cat([e3_d, e3_g], dim=1)  # (*, f*8, h, w)
        skip2 = torch.cat([e2_d, e2_g], dim=1)
        skip1 = torch.cat([e1_d, e1_g], dim=1)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class PretextUNet(nn.Module):
    """Channel-invariant pretext U-Net.

    Parameters
    ----------
    base_features : int
        Feature width of the first encoder level.
    """

    def __init__(self, base_features: int = 32):
        super().__init__()
        f = base_features

        # Shared per-direction encoder
        self.dir_encoder = DirectionEncoder(f)

        # DTI decoder: takes aggregated global features → 6 DTI components
        self.dti_up2 = UpBlock(f * 4, f * 2, f * 2)
        self.dti_up1 = UpBlock(f * 2, f, f)
        self.dti_head = nn.Conv2d(f, 6, 1)

        # Shared per-direction DWI decoder
        self.dir_decoder = DirectionDecoder(f)

    def forward(
        self,
        directions: torch.Tensor,
        dir_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        dir_chunk_size: int = 16,
    ):
        """
        Parameters
        ----------
        directions : (B, N, H, W)  noisy DWI (masked directions zeroed)
        dir_mask   : (B, N)         1.0 = kept, 0.0 = direction-masked
        pad_mask   : (B, N)         1.0 = real direction, 0.0 = batch padding
        bvals      : (B, N)         b-values
        bvecs      : (B, 3, N)      b-vectors
        dir_chunk_size : int        max directions processed at once (memory control)

        Returns
        -------
        dwi_pred : (B, N, H, W)  reconstructed DWI per direction
        dti_pred : (B, 6, H, W)  predicted DTI 6-component tensor
        """
        B, N, H, W = directions.shape
        BN = B * N

        # Build per-direction input: (B*N, 6, H, W)
        sigs = directions.reshape(BN, 1, H, W)
        mask_flags = dir_mask.reshape(BN, 1, 1, 1).expand(-1, -1, H, W)
        bvals_norm = (bvals / 1000.0).reshape(BN, 1, 1, 1).expand(-1, -1, H, W)
        bvecs_r = bvecs.transpose(1, 2).reshape(BN, 3, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([sigs, mask_flags, bvals_norm, bvecs_r], dim=1)  # (BN, 6, H, W)

        # --- Chunked encoding (limits peak memory) ---
        e1_parts, e2_parts, e3_parts = [], [], []
        for i in range(0, BN, dir_chunk_size):
            c1, c2, c3 = self.dir_encoder(x[i : i + dir_chunk_size])
            e1_parts.append(c1)
            e2_parts.append(c2)
            e3_parts.append(c3)
        e1 = torch.cat(e1_parts, dim=0)
        e2 = torch.cat(e2_parts, dim=0)
        e3 = torch.cat(e3_parts, dim=0)

        f_ch = e3.shape[1]
        h3, w3 = e3.shape[2:]
        h2, w2 = e2.shape[2:]
        h1, w1 = e1.shape[2:]

        # Reshape for aggregation: (B, N, C, h, w)
        e3_r = e3.reshape(B, N, f_ch, h3, w3)
        e2_r = e2.reshape(B, N, f_ch // 2, h2, w2)
        e1_r = e1.reshape(B, N, f_ch // 4, h1, w1)

        # Masked mean aggregation (ignore padded + masked directions)
        valid = (pad_mask * dir_mask).reshape(B, N, 1, 1, 1)
        valid_count = valid.sum(dim=1).clamp(min=1)

        g3 = (e3_r * valid).sum(dim=1) / valid_count
        g2 = (e2_r * valid[..., :h2, :w2]).sum(dim=1) / valid_count[..., :h2, :w2]
        g1 = (e1_r * valid[..., :h1, :w1]).sum(dim=1) / valid_count[..., :h1, :w1]

        # --- DTI head: decode aggregated features ---
        dti = self.dti_up2(g3, g2)
        dti = self.dti_up1(dti, g1)
        dti_pred = self.dti_head(dti)  # (B, 6, H, W)

        # --- DWI head: chunked per-direction reconstruction ---
        g3_exp = g3.unsqueeze(1).expand_as(e3_r).reshape(BN, f_ch, h3, w3)
        g2_exp = g2.unsqueeze(1).expand_as(e2_r).reshape(BN, f_ch // 2, h2, w2)
        g1_exp = g1.unsqueeze(1).expand_as(e1_r).reshape(BN, f_ch // 4, h1, w1)

        dwi_parts = []
        for i in range(0, BN, dir_chunk_size):
            s = slice(i, i + dir_chunk_size)
            dwi_parts.append(
                self.dir_decoder(
                    (e1[s], e2[s], e3[s]),
                    (g1_exp[s], g2_exp[s], g3_exp[s]),
                )
            )
        dwi_single = torch.cat(dwi_parts, dim=0)  # (BN, 1, H, W)

        dwi_pred = dwi_single.reshape(B, N, H, W)
        return dwi_pred, dti_pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
