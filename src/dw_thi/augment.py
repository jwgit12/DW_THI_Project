"""On-the-fly DWI degradation + augmentation used by DWISliceDataset.

All helpers operate on contiguous float32 stacks shaped ``(N, H, W)`` — i.e.
one 2D slice with ``N`` diffusion-weighted volumes. This is the shape produced
after slicing a subject's 4D clean ``target_dwi`` along any of the three
spatial axes.

Performance notes
-----------------
* ``scipy.fft.rfft2`` is ~10x faster than ``numpy.fft.fft2`` for these sizes
  because the input is real-valued. The mask on the rfft output is not exactly
  equivalent to ``2*ry × 2*rx`` in the two-sided spectrum (off-by-one in x due
  to Hermitian symmetry), but for a stochastic augmentation where
  ``keep_fraction`` is drawn from a range this asymmetry is irrelevant.
* ``rng.standard_normal(..., dtype=np.float32)`` avoids a float64->float32
  cast.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.fft as sfft
import torch

# Each DataLoader worker runs its own FFT threads. Cap per-call threads so
# total thread count stays ≤ cpu_count when multiple workers run in parallel.
# With 4 workers on a 16-core CPU: 4 × 4 = 16 threads → fully utilised.
# Override with DW_FFT_WORKERS env var if needed.
_FFT_WORKERS: int = int(os.environ.get("DW_FFT_WORKERS", max(1, (os.cpu_count() or 4) // 4)))


def lowres_kspace_cutout(slice_nhw: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Zero out the high-frequency k-space of each 2D (H, W) slice.

    Parameters
    ----------
    slice_nhw : (N, H, W) float32
    keep_fraction : float in (0, 1]
        Fraction of each spatial axis kept around DC.

    Returns
    -------
    lowres : (N, H, W) float32
    """
    _, h, w = slice_nhw.shape
    k = sfft.rfft2(slice_nhw, axes=(-2, -1), workers=_FFT_WORKERS)
    ry = max(1, int(h * keep_fraction / 2))
    rx = max(1, int(w * keep_fraction / 2))
    k[:, ry:h - ry, :] = 0
    k[:, :, rx:] = 0
    lowres = sfft.irfft2(k, s=(h, w), axes=(-2, -1), workers=_FFT_WORKERS).astype(np.float32)
    return lowres


_NOISE_GAUSSIAN = "gaussian"
_NOISE_RICIAN = "rician"
_NOISE_CHI = "chi"
_VALID_NOISE = (_NOISE_GAUSSIAN, _NOISE_RICIAN, _NOISE_CHI)


def _per_volume_sigma(slice_nhw: np.ndarray, rel_noise_level: float) -> np.ndarray:
    slice_max = slice_nhw.reshape(slice_nhw.shape[0], -1).max(axis=1)
    return (rel_noise_level * slice_max).astype(np.float32).reshape(-1, 1, 1)


def add_scaled_gaussian_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise with per-slice ``sigma = rel_noise_level * max(slice)``.

    Kept for back-compat. Use :func:`add_magnitude_noise` for the publication
    pipeline (Rician / non-central chi).
    """
    sigma = _per_volume_sigma(slice_nhw, rel_noise_level)
    noise = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    return slice_nhw + noise


def add_rician_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Rician noise to a magnitude DWI stack.

    Standard magnitude-MR noise model (Gudbjartsson & Patz, 1995):

        M = sqrt((S + eta_r)^2 + eta_i^2),  eta_r, eta_i ~ N(0, sigma)

    ``sigma`` is set per volume to ``rel_noise_level * max(slice)`` so the
    interpretation matches the legacy Gaussian path: high-SNR voxels behave
    like ``S + eta_r`` (variance ``sigma^2``), while low-SNR voxels exhibit
    the well-known positive Rician bias.
    """
    sigma = _per_volume_sigma(slice_nhw, rel_noise_level)
    eta_r = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    eta_i = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    real = slice_nhw + eta_r
    return np.sqrt(real * real + eta_i * eta_i, dtype=np.float32)


def add_noncentral_chi_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
    n_coils: int,
) -> np.ndarray:
    """Add non-central chi noise — multi-coil sum-of-squares MR magnitude.

    For ``n_coils=1`` this collapses to Rician. For ``n_coils > 1`` the result
    follows a non-central chi distribution with 2*n_coils degrees of freedom
    (Constantinides 1997; Aja-Fernández 2009), the appropriate model for SoS
    reconstruction from N receive coils with uniform sensitivity:

        M = sqrt(sum_{i=1..N} (S * delta_{i,1} + eta_r_i)^2 + eta_i_i^2)

    The signal sits in a single virtual channel, which keeps the noise-free
    expectation equal to ``S`` and the asymptotic high-SNR std equal to sigma.
    """
    if n_coils <= 0:
        raise ValueError(f"n_coils must be >= 1, got {n_coils}.")
    if n_coils == 1:
        return add_rician_noise(slice_nhw, rel_noise_level, rng)
    sigma = _per_volume_sigma(slice_nhw, rel_noise_level)
    eta_r0 = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    eta_i0 = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    real0 = slice_nhw + eta_r0
    sumsq = real0 * real0 + eta_i0 * eta_i0
    extra_shape = (2 * (n_coils - 1) + 1, *slice_nhw.shape)  # 2*(N-1) + 1 == 2N-1
    extras = rng.standard_normal(extra_shape, dtype=np.float32) * sigma
    sumsq += np.einsum("k...,k...->...", extras, extras)
    return np.sqrt(sumsq, dtype=np.float32)


def add_magnitude_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
    distribution: str = _NOISE_RICIAN,
    n_coils: int = 1,
) -> np.ndarray:
    """Dispatch helper: ``"gaussian" | "rician" | "chi"``."""
    distribution = (distribution or _NOISE_RICIAN).lower()
    if distribution not in _VALID_NOISE:
        raise ValueError(
            f"Unknown noise distribution {distribution!r}; expected one of {_VALID_NOISE}."
        )
    if distribution == _NOISE_GAUSSIAN:
        return add_scaled_gaussian_noise(slice_nhw, rel_noise_level, rng)
    if distribution == _NOISE_RICIAN:
        return add_rician_noise(slice_nhw, rel_noise_level, rng)
    return add_noncentral_chi_noise(slice_nhw, rel_noise_level, rng, n_coils)


def degrade_dwi_slice(
    slice_nhw: np.ndarray,
    keep_fraction: float,
    rel_noise_level: float,
    rng: np.random.Generator | None = None,
    *,
    noise_distribution: str = _NOISE_RICIAN,
    n_coils: int = 1,
) -> np.ndarray:
    """Apply central-k-space cutout + magnitude noise to a clean DWI slice stack.

    The default noise model is **Rician** — the established magnitude-MR
    distribution for DWI denoising publications. Pass ``noise_distribution="chi"``
    with ``n_coils > 1`` to simulate multi-coil sum-of-squares reconstruction.
    """
    if rng is None:
        rng = np.random.default_rng()
    lowres = lowres_kspace_cutout(slice_nhw, keep_fraction)
    return add_magnitude_noise(
        lowres, rel_noise_level, rng,
        distribution=noise_distribution, n_coils=n_coils,
    )


def degrade_dwi_volume(
    dwi_xyzn: np.ndarray,
    keep_fraction: float,
    rel_noise_level: float,
    seed: int | None = None,
    *,
    noise_distribution: str = _NOISE_RICIAN,
    n_coils: int = 1,
) -> np.ndarray:
    """Degrade a full 4D DWI volume deterministically. Used by eval/visualizer.

    Parameters
    ----------
    dwi_xyzn : (X, Y, Z, N) float32
    keep_fraction, rel_noise_level : scalar, applied uniformly to every slice.
    seed : int or None
        If given, the noise is fully reproducible (for evaluation).
    noise_distribution : ``"rician"`` (default), ``"chi"``, or ``"gaussian"``.
    n_coils : int
        Number of coils for non-central chi (``>=1``; ``1`` ≡ Rician).

    Returns
    -------
    degraded : (X, Y, Z, N) float32
    """
    rng = np.random.default_rng(seed)
    x, y, z, n = dwi_xyzn.shape
    # Process axial slices: reshape (X, Y, Z, N) -> (Z*N, X, Y) per axial slice.
    perm = np.ascontiguousarray(dwi_xyzn.transpose(2, 3, 0, 1)).reshape(z * n, x, y)
    lowres = lowres_kspace_cutout(perm, keep_fraction)
    noisy = add_magnitude_noise(
        lowres, rel_noise_level, rng,
        distribution=noise_distribution, n_coils=n_coils,
    )
    return np.ascontiguousarray(
        noisy.reshape(z, n, x, y).transpose(2, 3, 0, 1)
    )


# ---------------------------------------------------------------------------
# Spatial-flip transforms
# ---------------------------------------------------------------------------

# Order of channels in the stored 6D tensor: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
# When the image is mirrored along world axis a, the tensor transforms as
# D' = F_a D F_a^T, which flips the sign of every off-diagonal that touches a.
DTI6D_OFFDIAG_SIGN = {
    # world_axis -> tuple of channel indices to negate
    0: (1, 3),  # x-flip: Dxy, Dxz
    1: (1, 4),  # y-flip: Dxy, Dyz
    2: (3, 4),  # z-flip: Dxz, Dyz
}


def flip_dti6d_sign(tgt_chw: np.ndarray, world_axis: int) -> np.ndarray:
    """Return a copy of ``tgt_chw`` with off-diagonals sign-flipped for axis."""
    out = tgt_chw.copy()
    for c in DTI6D_OFFDIAG_SIGN[world_axis]:
        out[c] = -out[c]
    return out


def flip_bvecs(bvecs_3n: np.ndarray, world_axis: int) -> np.ndarray:
    """Negate the component of ``bvecs`` that matches the flipped world axis."""
    out = bvecs_3n.copy()
    out[world_axis] = -out[world_axis]
    return out


# ---------------------------------------------------------------------------
# GPU degradation helpers (cuFFT — ~50× faster than CPU scipy path)
# ---------------------------------------------------------------------------

def gpu_degrade_dwi_batch(
    signal: torch.Tensor,
    keep_fraction: torch.Tensor,
    noise_level: torch.Tensor,
    *,
    noise_distribution: str = _NOISE_RICIAN,
    n_coils: int = 1,
) -> torch.Tensor:
    """K-space cutout + magnitude noise for a full batch on GPU (cuFFT).

    Semantically identical to ``degrade_dwi_slice`` but operates on
    ``(B, N, H, W)`` batches and uses ``torch.fft.rfft2``, which is
    ~50× faster than the per-sample scipy path on CUDA.

    Parameters
    ----------
    signal : (B, N, H, W) float32, on device
    keep_fraction : (B,) float32 — fraction of each spatial axis kept around DC
    noise_level : (B,) float32 — rel_noise_level per sample
    noise_distribution : ``"rician"`` (default), ``"chi"``, or ``"gaussian"``.
    n_coils : int
        Coil count for ``"chi"`` (``1`` ≡ Rician).

    Returns
    -------
    degraded : same shape as ``signal`` float32
    """
    if signal.ndim != 4:
        raise ValueError(
            f"gpu_degrade_dwi_batch expected 4D signal, got {tuple(signal.shape)}."
        )
    distribution = (noise_distribution or _NOISE_RICIAN).lower()
    if distribution not in _VALID_NOISE:
        raise ValueError(
            f"Unknown noise distribution {distribution!r}; expected one of {_VALID_NOISE}."
        )
    if distribution == _NOISE_CHI and n_coils <= 0:
        raise ValueError(f"n_coils must be >= 1, got {n_coils}.")
    B, N, H, W = signal.shape

    k = torch.fft.rfft2(signal, dim=(-2, -1))  # (B, N, H, W//2+1) complex

    ry = (H * keep_fraction / 2).long().clamp(min=1)   # (B,)
    rx = (W * keep_fraction / 2).long().clamp(min=1)   # (B,)

    freq_h = torch.arange(H, device=signal.device)           # (H,)
    freq_w = torch.arange(W // 2 + 1, device=signal.device)  # (W//2+1,)

    # Vectorized per-sample frequency masks (no Python loops).
    row_keep = (freq_h[None] < ry[:, None]) | (freq_h[None] >= H - ry[:, None])  # (B, H)
    col_keep = freq_w[None] < rx[:, None]                                          # (B, W//2+1)
    mask = (row_keep[:, :, None] & col_keep[:, None, :]).unsqueeze(1)             # (B, 1, H, W//2+1)

    lowres = torch.fft.irfft2(k * mask, s=(H, W), dim=(-2, -1))  # (B, N, H, W)

    slice_max = lowres.reshape(B, N, -1).amax(dim=-1)            # (B, N)
    sigma = (noise_level[:, None] * slice_max).view(B, N, 1, 1)

    if distribution == _NOISE_GAUSSIAN:
        noise = torch.randn_like(lowres).mul_(sigma)
        return lowres.add_(noise)

    # Rician / non-central chi share the same magnitude formula; chi adds
    # 2*(n_coils-1) extra zero-mean Gaussian channels to the sum-of-squares.
    eta_r = torch.randn_like(lowres).mul_(sigma)
    eta_i = torch.randn_like(lowres).mul_(sigma)
    real = lowres.add_(eta_r)
    sumsq = real.mul_(real).addcmul_(eta_i, eta_i)
    if distribution == _NOISE_CHI and n_coils > 1:
        for _ in range(2 * (n_coils - 1)):
            extra = torch.randn_like(sumsq).mul_(sigma)
            sumsq.addcmul_(extra, extra)
    return sumsq.sqrt_()


def gpu_b0_normalize_batch(
    signal: torch.Tensor,
    b0_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize each batch item by its mean-b0 intensity (GPU counterpart of
    ``compute_b0_norm`` + the b0-normalization step in ``DWISliceDataset``).

    Parameters
    ----------
    signal : (B, N, H, W) float32
    b0_mask : (B, N) bool — True for b0 volumes (bval < threshold)

    Returns
    -------
    normalized : same shape as ``signal`` float32
    """
    if signal.ndim != 4:
        raise ValueError(
            f"gpu_b0_normalize_batch expected 4D signal, got {tuple(signal.shape)}."
        )
    B, N, H, W = signal.shape
    b0_float = b0_mask.to(dtype=signal.dtype)                           # (B, N)
    n_b0 = b0_float.sum(dim=1).clamp(min=1.0)                          # (B,)
    mean_b0 = (signal * b0_float[:, :, None, None]).sum(dim=1) / n_b0[:, None, None]  # (B, H, W)

    max_val = mean_b0.reshape(B, -1).amax(dim=1)                       # (B,)
    brain = mean_b0 > (0.1 * max_val)[:, None, None]                   # (B, H, W)
    brain_sum = (mean_b0 * brain.float()).reshape(B, -1).sum(dim=1)    # (B,)
    brain_count = brain.reshape(B, -1).sum(dim=1).float().clamp(min=1) # (B,)
    b0_norm = (brain_sum / brain_count).clamp(min=1e-6)                # (B,)

    has_b0 = b0_mask.any(dim=1)                                        # (B,)
    b0_norm = torch.where(has_b0, b0_norm, torch.ones_like(b0_norm))

    signal.div_(b0_norm[:, None, None, None])
    return signal
