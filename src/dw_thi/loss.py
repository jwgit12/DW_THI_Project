"""Loss for fODF SH-coefficient prediction from noisy DWI.

Terms (all masked to brain voxels when a mask is supplied):

- ``fodf_loss``       Charbonnier on the SH coefficient residual.
- ``fodf_corr_loss``  1 - mean cosine-similarity between predicted/target
                      coefficient vectors. Captures the angular fit
                      independent of magnitude.
- ``fodf_sf_loss``    Charbonnier on sphere-sampled values, with each
                      direction weighted by the (normalized) target lobe
                      amplitude. Concentrates capacity on real fiber
                      directions instead of empty sphere.
- ``fodf_peak_loss``  Direct supervision of amplitudes at the target's
                      top-K peak directions per voxel.
- ``fodf_nonneg_loss`` Hinge penalty on negative SF amplitudes.
- ``fodf_power_loss`` Per-ℓ-band power-spectrum match (rotation-invariant
                      angular-content term — penalizes blurring/sharpening
                      of the angular distribution without caring about the
                      specific orientation of the lobes).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix

import config as cfg


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt(x^2 + eps^2) — MSE near 0, MAE on outliers."""
    return torch.sqrt(x * x + eps * eps)


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average a (B, C, H, W) tensor over brain voxels and channels."""
    if mask is None:
        return value.mean()
    mask_4d = mask.unsqueeze(1)
    n_elem = mask_4d.sum() * value.shape[1]
    return (value * mask_4d).sum() / n_elem.clamp(min=1)


def _infer_symmetric_sh_order(n_coeffs: int) -> int:
    """Infer even SH order from symmetric-basis coefficient count."""
    for sh_order in range(0, 32, 2):
        if (sh_order + 1) * (sh_order + 2) // 2 == n_coeffs:
            return sh_order
    raise ValueError(f"Cannot infer even SH order from {n_coeffs} coefficients.")


def _sh_band_slices(n_coeffs: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for each even ℓ-band in descoteaux07 order."""
    sh_order = _infer_symmetric_sh_order(n_coeffs)
    bands: list[tuple[int, int]] = []
    start = 0
    for ell in range(0, sh_order + 1, 2):
        end = start + (2 * ell + 1)
        bands.append((start, end))
        start = end
    return bands


class FodfLoss(nn.Module):
    """Multi-term loss for fODF SH coefficient prediction."""

    def __init__(
        self,
        lambda_fodf: float = cfg.LAMBDA_FODF,
        lambda_fodf_corr: float = cfg.LAMBDA_FODF_CORR,
        lambda_fodf_sf: float = cfg.LAMBDA_FODF_SF,
        lambda_fodf_peak: float = cfg.LAMBDA_FODF_PEAK,
        lambda_fodf_nonneg: float = cfg.LAMBDA_FODF_NONNEG,
        lambda_fodf_power: float = cfg.LAMBDA_FODF_POWER,
        fodf_loss_sphere: str = cfg.FODF_LOSS_SPHERE,
        fodf_sf_chunk_size: int = cfg.FODF_SF_CHUNK_SIZE,
        fodf_peak_topk: int = cfg.FODF_PEAK_TOPK,
        fodf_peak_weight: float = cfg.FODF_PEAK_WEIGHT,
        fodf_peak_gamma: float = cfg.FODF_PEAK_GAMMA,
        fodf_peak_rel_threshold: float = cfg.FODF_PEAK_REL_THRESHOLD,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.lambda_fodf = lambda_fodf
        self.lambda_fodf_corr = lambda_fodf_corr
        self.lambda_fodf_sf = lambda_fodf_sf
        self.lambda_fodf_peak = lambda_fodf_peak
        self.lambda_fodf_nonneg = lambda_fodf_nonneg
        self.lambda_fodf_power = lambda_fodf_power
        self.fodf_loss_sphere = fodf_loss_sphere
        self.fodf_sf_chunk_size = max(1, int(fodf_sf_chunk_size))
        self.fodf_peak_topk = max(1, int(fodf_peak_topk))
        self.fodf_peak_weight = fodf_peak_weight
        self.fodf_peak_gamma = fodf_peak_gamma
        self.fodf_peak_rel_threshold = fodf_peak_rel_threshold
        self.charbonnier_eps = charbonnier_eps
        self._fodf_sf_n_coeffs = 0
        self._band_slices: list[tuple[int, int]] = []
        self._band_n_coeffs = 0
        self.register_buffer("_fodf_sf_matrix", torch.empty(0), persistent=False)

    def _sf_matrix_for(
        self,
        n_coeffs: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._fodf_sf_matrix.numel() > 0 and self._fodf_sf_n_coeffs == n_coeffs:
            if self._fodf_sf_matrix.device != device or self._fodf_sf_matrix.dtype != dtype:
                self._fodf_sf_matrix = self._fodf_sf_matrix.to(device=device, dtype=dtype)
            return self._fodf_sf_matrix

        sh_order = _infer_symmetric_sh_order(n_coeffs)
        sphere = get_sphere(name=self.fodf_loss_sphere)
        matrix = sh_to_sf_matrix(
            sphere,
            sh_order_max=sh_order,
            basis_type="descoteaux07",
            legacy=True,
            return_inv=False,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape[0] != n_coeffs and matrix.shape[1] == n_coeffs:
            matrix = matrix.T
        if matrix.shape[0] != n_coeffs:
            raise ValueError(
                f"SH-to-SF matrix shape {matrix.shape} is incompatible with "
                f"{n_coeffs} fODF coefficients."
            )

        self._fodf_sf_n_coeffs = n_coeffs
        self._fodf_sf_matrix = torch.as_tensor(matrix, device=device, dtype=dtype)
        return self._fodf_sf_matrix

    def _bands_for(self, n_coeffs: int) -> list[tuple[int, int]]:
        if self._band_n_coeffs != n_coeffs or not self._band_slices:
            self._band_slices = _sh_band_slices(n_coeffs)
            self._band_n_coeffs = n_coeffs
        return self._band_slices

    def _target_topk_sf(
        self,
        target_fodf: torch.Tensor,
        sf_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(self.fodf_peak_topk, sf_matrix.shape[1])
        b, _, h, w = target_fodf.shape
        top_values = target_fodf.new_full((b, k, h, w), -float("inf"))
        top_indices = torch.zeros((b, k, h, w), device=target_fodf.device, dtype=torch.long)

        with torch.no_grad():
            for start in range(0, sf_matrix.shape[1], self.fodf_sf_chunk_size):
                end = min(start + self.fodf_sf_chunk_size, sf_matrix.shape[1])
                target_sf = torch.einsum(
                    "bchw,cv->bvhw",
                    target_fodf,
                    sf_matrix[:, start:end],
                ).clamp_min(0)
                chunk_indices = torch.arange(
                    start,
                    end,
                    device=target_fodf.device,
                    dtype=torch.long,
                ).view(1, -1, 1, 1)
                chunk_indices = chunk_indices.expand(b, -1, h, w)

                merged_values = torch.cat([top_values, target_sf], dim=1)
                merged_indices = torch.cat([top_indices, chunk_indices], dim=1)
                top_values, order = merged_values.topk(k, dim=1)
                top_indices = torch.gather(merged_indices, 1, order)

        return top_values.clamp_min(0), top_indices

    @staticmethod
    def _sample_sf_at_indices(
        sh_coeffs: torch.Tensor,
        sf_matrix: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        basis_by_direction = sf_matrix.T
        sh_hw = sh_coeffs.permute(0, 2, 3, 1)
        sampled = []
        for peak_i in range(indices.shape[1]):
            basis = basis_by_direction[indices[:, peak_i]]
            sampled.append((sh_hw * basis).sum(dim=-1))
        return torch.stack(sampled, dim=1)

    def _surface_fodf_terms(
        self,
        pred_fodf: torch.Tensor,
        target_fodf: torch.Tensor,
        mask: torch.Tensor | None,
        sf_matrix: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        top_values, top_indices = self._target_topk_sf(target_fodf, sf_matrix)
        target_peak = top_values[:, :1]

        sf_num = pred_fodf.new_zeros(())
        sf_den = pred_fodf.new_zeros(())
        sf_mae_num = pred_fodf.new_zeros(())
        sf_mae_den = pred_fodf.new_zeros(())
        nonneg_num = pred_fodf.new_zeros(())
        nonneg_den = pred_fodf.new_zeros(())
        mask_4d = mask.unsqueeze(1).to(dtype=pred_fodf.dtype) if mask is not None else None

        for start in range(0, sf_matrix.shape[1], self.fodf_sf_chunk_size):
            end = min(start + self.fodf_sf_chunk_size, sf_matrix.shape[1])
            basis_chunk = sf_matrix[:, start:end]
            pred_sf = torch.einsum("bchw,cv->bvhw", pred_fodf, basis_chunk)
            target_sf = torch.einsum("bchw,cv->bvhw", target_fodf, basis_chunk)
            target_norm = target_sf.clamp_min(0) / target_peak.clamp_min(1e-6)
            weight = 1.0 + self.fodf_peak_weight * target_norm.pow(self.fodf_peak_gamma)
            if mask_4d is not None:
                weight = weight * mask_4d
                count_weight = mask_4d.expand_as(pred_sf)
            else:
                count_weight = torch.ones_like(pred_sf)

            residual = pred_sf - target_sf
            sf_num = sf_num + (_charbonnier(residual, self.charbonnier_eps) * weight).sum()
            sf_den = sf_den + weight.sum()
            sf_mae_num = sf_mae_num + (residual.abs() * count_weight).sum()
            sf_mae_den = sf_mae_den + count_weight.sum()
            nonneg_num = nonneg_num + (F.relu(-pred_sf) * count_weight).sum()
            nonneg_den = nonneg_den + count_weight.sum()

        pred_peak_values = self._sample_sf_at_indices(pred_fodf, sf_matrix, top_indices)
        threshold = self.fodf_peak_rel_threshold * target_peak.clamp_min(1e-6)
        valid_peak = top_values > threshold
        if mask is not None:
            valid_peak = valid_peak & (mask.unsqueeze(1) > 0.5)

        if valid_peak.any():
            peak_residual = pred_peak_values - top_values
            peak_loss = _charbonnier(
                peak_residual[valid_peak],
                self.charbonnier_eps,
            ).mean()
            peak_mae = peak_residual[valid_peak].abs().mean()
            peak_ratio = (
                pred_peak_values[valid_peak].clamp_min(0)
                / top_values[valid_peak].clamp_min(1e-6)
            ).mean()
        else:
            peak_loss = pred_fodf.new_zeros(())
            peak_mae = pred_fodf.new_zeros(())
            peak_ratio = pred_fodf.new_zeros(())

        return {
            "fodf_sf_loss": sf_num / sf_den.clamp(min=1),
            "fodf_sf_mae": sf_mae_num / sf_mae_den.clamp(min=1),
            "fodf_peak_loss": peak_loss,
            "fodf_peak_mae": peak_mae,
            "fodf_peak_ratio": peak_ratio,
            "fodf_nonneg_loss": nonneg_num / nonneg_den.clamp(min=1),
        }

    def _power_spectrum_loss(
        self,
        pred_fodf: torch.Tensor,
        target_fodf: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Charbonnier on per-ℓ-band sqrt-power differences.

        Power per band ℓ is ``sum_m c_{ℓ,m}^2`` — invariant under sphere
        rotations. Matching it pushes the model to reproduce the target's
        angular sharpness/dispersion without penalizing small orientation
        errors that the coefficient-space loss already handles.
        """
        bands = self._bands_for(pred_fodf.shape[1])
        eps = self.charbonnier_eps
        eps_sq = eps * eps

        per_band = []
        for start, end in bands:
            pred_p = (pred_fodf[:, start:end] ** 2).sum(dim=1, keepdim=True)
            tgt_p = (target_fodf[:, start:end] ** 2).sum(dim=1, keepdim=True)
            pred_norm = torch.sqrt(pred_p + eps_sq)
            tgt_norm = torch.sqrt(tgt_p + eps_sq)
            per_band.append(pred_norm - tgt_norm)
        residual = torch.cat(per_band, dim=1)  # (B, n_bands, H, W)

        return _masked_channel_mean(_charbonnier(residual, eps), mask)

    def forward(
        self,
        pred_fodf: torch.Tensor,
        target_fodf: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_tensor_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        def metric_value(value: torch.Tensor) -> float | torch.Tensor:
            value = value.detach()
            return value if return_tensor_metrics else value.item()

        residual = pred_fodf - target_fodf
        fodf_charb = _charbonnier(residual, self.charbonnier_eps)
        fodf_loss = _masked_channel_mean(fodf_charb, mask)
        with torch.no_grad():
            fodf_mse_val = _masked_channel_mean(residual ** 2, mask)

        metrics = {
            "fodf_loss": metric_value(fodf_loss),
            "fodf_mse": metric_value(fodf_mse_val),
            "fodf_acc": metric_value(pred_fodf.new_zeros(())),
            "fodf_corr_loss": metric_value(pred_fodf.new_zeros(())),
            "fodf_sf_loss": metric_value(pred_fodf.new_zeros(())),
            "fodf_sf_mae": metric_value(pred_fodf.new_zeros(())),
            "fodf_peak_loss": metric_value(pred_fodf.new_zeros(())),
            "fodf_peak_mae": metric_value(pred_fodf.new_zeros(())),
            "fodf_peak_ratio": metric_value(pred_fodf.new_zeros(())),
            "fodf_nonneg_loss": metric_value(pred_fodf.new_zeros(())),
            "fodf_power_loss": metric_value(pred_fodf.new_zeros(())),
        }

        total = pred_fodf.new_zeros(())
        active_loss_term = False
        if self.lambda_fodf > 0:
            total = total + self.lambda_fodf * fodf_loss
            active_loss_term = True

        # Cosine similarity in coefficient space (angular fit, scale-free).
        pred_vox = pred_fodf.permute(0, 2, 3, 1).reshape(-1, pred_fodf.shape[1])
        tgt_vox = target_fodf.permute(0, 2, 3, 1).reshape(-1, target_fodf.shape[1])
        valid = tgt_vox.norm(dim=1) > 1e-8
        if mask is not None:
            valid = valid & (mask.reshape(-1) > 0.5)

        if valid.any():
            cosine = F.cosine_similarity(pred_vox[valid], tgt_vox[valid], dim=1, eps=1e-8)
            fodf_acc = cosine.mean()
            fodf_corr_loss = 1.0 - fodf_acc
        else:
            fodf_acc = pred_fodf.new_zeros(())
            fodf_corr_loss = pred_fodf.new_zeros(())

        metrics["fodf_acc"] = metric_value(fodf_acc)
        metrics["fodf_corr_loss"] = metric_value(fodf_corr_loss)
        if self.lambda_fodf_corr > 0:
            total = total + self.lambda_fodf_corr * fodf_corr_loss
            active_loss_term = True

        if (
            self.lambda_fodf_sf > 0
            or self.lambda_fodf_peak > 0
            or self.lambda_fodf_nonneg > 0
        ):
            sf_matrix = self._sf_matrix_for(
                pred_fodf.shape[1],
                device=pred_fodf.device,
                dtype=pred_fodf.dtype,
            )
            surface_terms = self._surface_fodf_terms(pred_fodf, target_fodf, mask, sf_matrix)
            for name, value in surface_terms.items():
                metrics[name] = metric_value(value)
            if self.lambda_fodf_sf > 0:
                total = total + self.lambda_fodf_sf * surface_terms["fodf_sf_loss"]
                active_loss_term = True
            if self.lambda_fodf_peak > 0:
                total = total + self.lambda_fodf_peak * surface_terms["fodf_peak_loss"]
                active_loss_term = True
            if self.lambda_fodf_nonneg > 0:
                total = total + self.lambda_fodf_nonneg * surface_terms["fodf_nonneg_loss"]
                active_loss_term = True

        if self.lambda_fodf_power > 0:
            power_loss = self._power_spectrum_loss(pred_fodf, target_fodf, mask)
            metrics["fodf_power_loss"] = metric_value(power_loss)
            total = total + self.lambda_fodf_power * power_loss
            active_loss_term = True

        if not active_loss_term:
            raise ValueError(
                "No active fODF loss is configured. Enable at least one of "
                "lambda_fodf, lambda_fodf_corr, lambda_fodf_sf, lambda_fodf_peak, "
                "lambda_fodf_nonneg, or lambda_fodf_power."
            )

        return total, metrics
