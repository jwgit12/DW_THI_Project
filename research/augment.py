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

import numpy as np
import scipy.fft as sfft


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
    k = sfft.rfft2(slice_nhw, axes=(-2, -1), workers=-1)
    ry = max(1, int(h * keep_fraction / 2))
    rx = max(1, int(w * keep_fraction / 2))
    k[:, ry:h - ry, :] = 0
    k[:, :, rx:] = 0
    lowres = sfft.irfft2(k, s=(h, w), axes=(-2, -1), workers=-1).astype(np.float32)
    return lowres


def add_scaled_gaussian_noise(
    slice_nhw: np.ndarray,
    rel_noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise with per-slice ``sigma = rel_noise_level * max(slice)``.

    Matches the semantics of ``functions.lowres_noise`` but operates on a
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
