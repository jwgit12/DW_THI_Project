"""Loss functions for DWI denoising models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


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


class DTILoss(nn.Module):
    """MSE on 6D tensor + optional FA/MD MAE regularisation."""

    def __init__(self, lambda_scalar: float = cfg.LAMBDA_SCALAR):
        super().__init__()
        self.lambda_scalar = lambda_scalar

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mask is not None:
            mask_4d = mask.unsqueeze(1)  # (B, 1, H, W)
            n_elem = mask_4d.sum() * pred.shape[1]
            tensor_loss = ((pred - target) ** 2 * mask_4d).sum() / n_elem.clamp(min=1)
        else:
            tensor_loss = F.mse_loss(pred, target)

        if self.lambda_scalar > 0:
            pred_fa, pred_md = tensor6_to_fa_md(pred)
            tgt_fa, tgt_md = tensor6_to_fa_md(target)
            if mask is not None:
                n_spatial = mask.sum().clamp(min=1)
                fa_loss = (torch.abs(pred_fa - tgt_fa) * mask).sum() / n_spatial
                md_loss = (torch.abs(pred_md - tgt_md) * mask).sum() / n_spatial
            else:
                fa_loss = F.l1_loss(pred_fa, tgt_fa)
                md_loss = F.l1_loss(pred_md, tgt_md)
            total = tensor_loss + self.lambda_scalar * (fa_loss + md_loss)
            return total, {
                "tensor_mse": tensor_loss.item(),
                "fa_mae": fa_loss.item(),
                "md_mae": md_loss.item(),
            }

        return tensor_loss, {"tensor_mse": tensor_loss.item()}
