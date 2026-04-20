"""GPU-accelerated DWI denoising baselines.

Currently exposes a PyTorch-based MP-PCA (Marchenko-Pastur PCA) denoiser
that is a drop-in replacement for :func:`dipy.denoise.localpca.mppca`.
"""
from .mppca_torch import mppca, mppca_torch

__all__ = ["mppca", "mppca_torch"]
