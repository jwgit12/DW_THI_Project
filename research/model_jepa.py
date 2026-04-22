"""JEPA-augmented QSpaceUNet for DWI -> DTI denoising.

A thin wrapper around ``QSpaceUNet`` that adds a self-distillation
(Joint-Embedding Predictive Architecture) auxiliary objective:

- Student: noisy DWI -> q_encoder -> U-Net -> 6D DTI  (task head, end-to-end)
- Teacher: clean DWI -> q_encoder_ema                 (features only, stop-grad)
- Predictor:           student q-features -> predict teacher q-features

The JEPA loss is computed in the q-encoder feature space (same H×W as
the U-Net input). At inference only the student path runs, so the
state_dict can be filtered to pure ``QSpaceUNet`` keys and loaded by
the unchanged evaluation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg
from research.model import QSpaceEncoder, UNet2D, cholesky_to_tensor6


class JEPAPredictor(nn.Module):
    """1x1 -> 3x3 -> 1x1 conv head that predicts teacher features."""

    def __init__(self, feat_dim: int, hidden: int = cfg.JEPA_PRED_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAQSpaceUNet(nn.Module):
    """QSpaceUNet + EMA teacher + feature-predictor for JEPA training.

    The student (``q_encoder`` + ``unet``) is structurally identical to
    ``QSpaceUNet`` so weights can be copied back after training for
    inference/evaluation with no code changes downstream.
    """

    def __init__(
        self,
        max_n: int,
        feat_dim: int = cfg.FEAT_DIM,
        channels: tuple[int, ...] = tuple(cfg.UNET_CHANNELS),
        cholesky: bool = False,
        dropout: float = cfg.DROPOUT,
        ema_momentum: float = cfg.JEPA_EMA,
        pred_hidden: int = cfg.JEPA_PRED_HIDDEN,
    ):
        super().__init__()
        self.max_n = max_n
        self.cholesky = cholesky
        self.ema_momentum = float(ema_momentum)

        # Student (task) path — keys match QSpaceUNet exactly.
        self.q_encoder = QSpaceEncoder(feat_dim=feat_dim)
        self.unet = UNet2D(feat_dim, out_ch=6, channels=channels, dropout=dropout)

        # Teacher (EMA) q-encoder — same structure, no gradient.
        self.target_q_encoder = QSpaceEncoder(feat_dim=feat_dim)
        for p_t, p in zip(self.target_q_encoder.parameters(), self.q_encoder.parameters()):
            p_t.data.copy_(p.data)
            p_t.requires_grad_(False)

        # Predictor: student features -> teacher features.
        self.predictor = JEPAPredictor(feat_dim=feat_dim, hidden=pred_hidden)

    # -----------------------------------------------------------------------
    # EMA update of the teacher q-encoder.
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else float(momentum)
        for p_t, p in zip(self.target_q_encoder.parameters(), self.q_encoder.parameters()):
            p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)

    # -----------------------------------------------------------------------
    # Forward passes.
    # -----------------------------------------------------------------------
    def _task_head(self, features: torch.Tensor) -> torch.Tensor:
        out = self.unet(features)
        if self.cholesky:
            out = cholesky_to_tensor6(out)
        return out

    def forward(
        self,
        signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-compatible forward: returns 6D DTI like QSpaceUNet."""
        features = self.q_encoder(signal, bvals, bvecs, vol_mask)
        return self._task_head(features)

    def forward_jepa(
        self,
        noisy_signal: torch.Tensor,
        clean_signal: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
        vol_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (dti_pred, predictor_output, teacher_features)."""
        student_feats = self.q_encoder(noisy_signal, bvals, bvecs, vol_mask)
        dti = self._task_head(student_feats)

        pred_feats = self.predictor(student_feats)
        with torch.no_grad():
            teacher_feats = self.target_q_encoder(
                clean_signal, bvals, bvecs, vol_mask,
            ).detach()
        return dti, pred_feats, teacher_feats

    # -----------------------------------------------------------------------
    # Checkpoint helpers.
    # -----------------------------------------------------------------------
    def student_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a state_dict containing only the student (QSpaceUNet) keys."""
        sd = self.state_dict()
        return {
            k: v for k, v in sd.items()
            if k.startswith("q_encoder.") or k.startswith("unet.")
        }


def jepa_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Masked Charbonnier loss between predicted and teacher q-features.

    ``mask`` is expected as (B, H, W); broadcast across the feature channels.
    """
    residual = pred - target
    charb = torch.sqrt(residual * residual + eps * eps)
    if mask is not None:
        m = mask.unsqueeze(1)  # (B, 1, H, W)
        n = (m.sum() * pred.shape[1]).clamp(min=1.0)
        return (charb * m).sum() / n
    return charb.mean()
