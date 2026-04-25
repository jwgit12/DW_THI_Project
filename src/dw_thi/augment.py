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


def add_scaled_gaussian_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise with per-slice ``sigma = rel_noise_level * max(slice)``.

    Matches the legacy low-resolution/noise degradation but operates on a
    (N, H, W) stack and takes a single noise level that can vary per call.
    """
    slice_max = slice_nhw.reshape(slice_nhw.shape[0], -1).max(axis=1)
    sigma = (rel_noise_level * slice_max).astype(np.float32).reshape(-1, 1, 1)
    noise = rng.standard_normal(slice_nhw.shape, dtype=np.float32) * sigma
    return slice_nhw + noise


def degrade_dwi_slice(
    slice_nhw: np.ndarray,
    keep_fraction: float,
    rel_noise_level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply central-k-space cutout + Gaussian noise to a clean DWI slice stack."""
    if rng is None:
        rng = np.random.default_rng()
    lowres = lowres_kspace_cutout(slice_nhw, keep_fraction)
    return add_scaled_gaussian_noise(lowres, rel_noise_level, rng)


def degrade_dwi_volume(
    dwi_xyzn: np.ndarray,
    keep_fraction: float,
    rel_noise_level: float,
    seed: int | None = None,
) -> np.ndarray:
    """Degrade a full 4D DWI volume deterministically. Used by eval/visualizer.

    Parameters
    ----------
    dwi_xyzn : (X, Y, Z, N) float32
    keep_fraction, rel_noise_level : scalar, applied uniformly to every slice.
    seed : int or None
        If given, the noise is fully reproducible (for evaluation).

    Returns
    -------
    degraded : (X, Y, Z, N) float32
    """
    rng = np.random.default_rng(seed)
    x, y, z, n = dwi_xyzn.shape
    # Process axial slices: reshape (X, Y, Z, N) -> (Z*N, X, Y) per axial slice.
    perm = np.ascontiguousarray(dwi_xyzn.transpose(2, 3, 0, 1)).reshape(z * n, x, y)
    lowres = lowres_kspace_cutout(perm, keep_fraction)
    noisy = add_scaled_gaussian_noise(lowres, rel_noise_level, rng)
    return np.ascontiguousarray(
        noisy.reshape(z, n, x, y).transpose(2, 3, 0, 1)
    )


# ---------------------------------------------------------------------------
# GPU degradation helpers (cuFFT — ~50× faster than CPU scipy path)
# ---------------------------------------------------------------------------

def gpu_degrade_dwi_batch(
    signal: torch.Tensor,
    keep_fraction: torch.Tensor,
    noise_level: torch.Tensor,
) -> torch.Tensor:
    """K-space cutout + Gaussian noise for a full batch on GPU (cuFFT).

    Semantically identical to ``degrade_dwi_slice`` but operates on
    ``(B, N, H, W)`` tensors and uses ``torch.fft.rfft2``, which is
    ~50× faster than the per-sample scipy path on CUDA.

    Parameters
    ----------
    signal : (B, N, H, W) float32, on device
    keep_fraction : (B,) float32 — fraction of each spatial axis kept around DC
    noise_level : (B,) float32 — rel_noise_level per sample

    Returns
    -------
    degraded : (B, N, H, W) float32
    """
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
    noise = torch.randn_like(lowres)
    noise.mul_(sigma)
    lowres.add_(noise)
    return lowres


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
    normalized : (B, N, H, W) float32
    """
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
