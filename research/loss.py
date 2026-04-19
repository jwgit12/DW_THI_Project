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


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 / Charbonnier penalty: sqrt(x^2 + eps^2).

    Behaves like MSE near zero and like MAE for large residuals, so it is
    less sensitive to outlier voxels (bad DTI fits) than plain MSE.
    """
    return torch.sqrt(x * x + eps * eps)


def _spatial_grad_mag(x: torch.Tensor) -> torch.Tensor:
    """Return |grad_x| + |grad_y| of a (B, H, W) map (replicate padding)."""
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    # Pad back to original shape so the result aligns for masked ops.
    gx = F.pad(gx.abs(), (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy.abs(), (0, 0, 0, 1), mode="replicate")
    return gx + gy


class DTILoss(nn.Module):
    """Charbonnier tensor loss + FA/MD MAE + optional FA edge loss.

    - Tensor term: Charbonnier on (pred - target), masked to brain voxels.
      Robust replacement for MSE that tolerates a few noisy target voxels
      without letting them dominate the gradient.
    - Scalar term: FA/MD MAE (unchanged from the original formulation).
    - Edge term: MAE between |grad(FA_pred)| and |grad(FA_tgt)| — pushes
      the model to preserve boundaries in the FA map. Disabled when
      ``lambda_edge == 0``.
    """

    def __init__(
        self,
        lambda_scalar: float = cfg.LAMBDA_SCALAR,
        lambda_edge: float = cfg.LAMBDA_EDGE,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.lambda_scalar = lambda_scalar
        self.lambda_edge = lambda_edge
        self.charbonnier_eps = charbonnier_eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        residual = pred - target
        charb = _charbonnier(residual, self.charbonnier_eps)

        if mask is not None:
            mask_4d = mask.unsqueeze(1)  # (B, 1, H, W)
            n_elem = mask_4d.sum() * pred.shape[1]
            tensor_loss = (charb * mask_4d).sum() / n_elem.clamp(min=1)
            with torch.no_grad():
                tensor_mse_val = ((residual ** 2) * mask_4d).sum() / n_elem.clamp(min=1)
        else:
            tensor_loss = charb.mean()
            with torch.no_grad():
                tensor_mse_val = (residual ** 2).mean()

        metrics = {
            "tensor_loss": tensor_loss.item(),
            "tensor_mse": tensor_mse_val.item(),
        }

        total = tensor_loss

        if self.lambda_scalar > 0 or self.lambda_edge > 0:
            pred_fa, pred_md = tensor6_to_fa_md(pred)
            tgt_fa, tgt_md = tensor6_to_fa_md(target)

            if mask is not None:
                n_spatial = mask.sum().clamp(min=1)
                fa_loss = (torch.abs(pred_fa - tgt_fa) * mask).sum() / n_spatial
                md_loss = (torch.abs(pred_md - tgt_md) * mask).sum() / n_spatial
            else:
                fa_loss = F.l1_loss(pred_fa, tgt_fa)
                md_loss = F.l1_loss(pred_md, tgt_md)
            metrics["fa_mae"] = fa_loss.item()
            metrics["md_mae"] = md_loss.item()

            if self.lambda_scalar > 0:
                total = total + self.lambda_scalar * (fa_loss + md_loss)

            if self.lambda_edge > 0:
                pred_edges = _spatial_grad_mag(pred_fa)
                tgt_edges = _spatial_grad_mag(tgt_fa)
                if mask is not None:
                    n_spatial = mask.sum().clamp(min=1)
                    edge_loss = (torch.abs(pred_edges - tgt_edges) * mask).sum() / n_spatial
                else:
                    edge_loss = F.l1_loss(pred_edges, tgt_edges)
                metrics["fa_edge"] = edge_loss.item()
                total = total + self.lambda_edge * edge_loss

        return total, metrics
