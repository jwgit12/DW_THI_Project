"""GPU-accelerated MP-PCA denoising for 4D DWI data.

Mathematically equivalent to ``dipy.denoise.localpca.mppca`` with
``pca_method='eig'``, but:

* vectorised over every patch in a z-slab and run on GPU (CUDA or MPS)
* uses a Gram-matrix formulation, so the eigendecomposition is always on
  an ``n x n`` matrix (``n`` = num samples per patch, e.g. 125 for a
  5x5x5 patch) instead of a ``d x d`` matrix (``d`` = number of DWI
  volumes, often 150+). For typical DWI data this is ``(n/d)^3`` ~2x
  cheaper per patch.
* transparently offloads only the eigendecomposition to CPU on Apple MPS
  (PyTorch 2.11 does not yet implement ``aten::_linalg_eigh`` for MPS).
  Unified memory on M-series chips makes those copies effectively free.

Reference
---------
Veraart, J., Novikov, D.S., Christiaens, D., Ades-Aron, B., Sijbers, J.,
Fieremans, E., 2016. Denoising of diffusion MRI using random matrix
theory. NeuroImage 142, 394-406.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# Cache which devices cannot run eigh natively (populated on first failure).
_EIGH_NEEDS_CPU_FALLBACK = {"mps"}


def _batched_eigh(mats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition with a CPU fallback for devices that
    don't implement it natively.

    As of PyTorch 2.11, ``torch.linalg.eigh`` is not implemented on MPS
    (see pytorch/pytorch#141287). We transparently copy the batch to CPU
    for those devices, run the eigh there and move the result back.
    Unified memory on Apple Silicon makes the transfer cost negligible
    compared with the eigh itself.

    If a future PyTorch adds native MPS support, the first successful
    call removes the fallback for subsequent batches.
    """
    dev_type = mats.device.type
    if dev_type not in _EIGH_NEEDS_CPU_FALLBACK:
        try:
            return torch.linalg.eigh(mats)
        except (NotImplementedError, RuntimeError):
            _EIGH_NEEDS_CPU_FALLBACK.add(dev_type)
    cpu_mats = mats.detach().to("cpu")
    ev, evec = torch.linalg.eigh(cpu_mats)
    return ev.to(mats.device), evec.to(mats.device)


def _mp_classifier(L: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Vectorised Marchenko-Pastur classifier.

    Parameters
    ----------
    L : (B, S) tensor of ascending eigenvalues, already truncated to the
        top ``min(d, n-1)`` values per DIPY's convention.
    n_samples : int. Number of voxels in a patch.

    Returns
    -------
    var : (B,) estimated noise variance per patch.
    """
    B, S = L.shape
    dtype = L.dtype
    dev = L.device

    m_range = torch.arange(1, S + 1, device=dev, dtype=dtype)
    cum = torch.cumsum(L, dim=-1)
    means = cum / m_range  # means[:, m-1] = mean(L[0..m-1])

    coef = 4.0 * torch.sqrt(m_range / float(n_samples))
    r = L - L[:, :1] - coef.unsqueeze(0) * means  # (B, S)

    # Replicates DIPY's loop: starting at m = S and decreasing, find the
    # first m for which r(m) <= 0.
    r_rev = torch.flip(r, dims=[-1])
    is_non_pos = r_rev <= 0
    first_k = is_non_pos.to(torch.uint8).argmax(dim=-1)
    any_non_pos = is_non_pos.any(dim=-1)
    m = torch.where(any_non_pos, S - first_k, torch.ones_like(first_k))

    gather_idx = (m - 1).to(torch.int64).unsqueeze(-1)
    var = torch.gather(means, 1, gather_idx).squeeze(-1)
    return var


def _extract_z_slab(
    arr: torch.Tensor, pr: np.ndarray, cz: int
) -> torch.Tensor:
    """Return contiguous patches for one z-center index.

    Returns (cx, cy, Px, Py, Pz, N) with cz=0 meaning center at z=pr[2].
    """
    Px, Py, Pz = int(2 * pr[0] + 1), int(2 * pr[1] + 1), int(2 * pr[2] + 1)
    block = arr[:, :, cz : cz + Pz, :]              # (X, Y, Pz, N)
    block = block.unfold(0, Px, 1).unfold(1, Py, 1)  # (cx, cy, Pz, N, Px, Py)
    block = block.permute(0, 1, 4, 5, 2, 3).contiguous()
    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mppca_torch(
    arr: Union[np.ndarray, torch.Tensor],
    *,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    patch_radius: Union[int, Tuple[int, int, int], np.ndarray] = 2,
    return_sigma: bool = False,
    out_dtype: Optional[np.dtype] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    suppress_warning: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """GPU-accelerated MP-PCA denoising.

    A drop-in replacement for :func:`dipy.denoise.localpca.mppca`.

    Parameters
    ----------
    arr : (X, Y, Z, N) ndarray or tensor
        DWI data to denoise.
    mask : (X, Y, Z) bool ndarray, optional
        Patches are only processed when their centre voxel is inside the
        mask. Voxels outside the mask are set to zero in the output.
    patch_radius : int or length-3 array, default 2
        Patch radius in voxels. A value of 2 yields 5x5x5 patches.
    return_sigma : bool
        If True, also return the estimated noise std per voxel.
    out_dtype : numpy dtype, optional
        Output dtype. Defaults to ``arr.dtype``.
    device : str, optional
        ``"cuda"``, ``"mps"`` or ``"cpu"``. Auto-detected if ``None``.
    batch_size : int, optional
        Maximum number of patches processed in a single eigh call.
        Default is one whole z-slab (``cx * cy`` patches).  Lower to
        reduce peak memory on small GPUs.
    suppress_warning : bool
        Silence the "num_samples - 1 < dim" warning.

    Returns
    -------
    denoised : (X, Y, Z, N) ndarray
    sigma : (X, Y, Z) ndarray, if ``return_sigma=True``
    """
    # --- Normalise inputs --------------------------------------------------
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy()
        in_dtype = arr_np.dtype
    else:
        arr_np = np.asarray(arr)
        in_dtype = arr_np.dtype

    if arr_np.ndim != 4:
        raise ValueError("MP-PCA denoising expects a 4D array.")

    if out_dtype is None:
        out_dtype = in_dtype
    calc_dtype = torch.float64 if in_dtype == np.float64 else torch.float32

    dev = _select_device(device)

    # Broadcast patch_radius and zero-out size-1 spatial axes, matching DIPY.
    if isinstance(patch_radius, int):
        pr = np.ones(3, dtype=int) * patch_radius
    else:
        pr = np.asarray(patch_radius, dtype=int).reshape(-1)
        if pr.size != 3:
            raise ValueError("patch_radius must be an int or length-3 array.")
    for i in range(3):
        if arr_np.shape[i] == 1:
            pr[i] = 0

    patch_size = 2 * pr + 1
    n = int(np.prod(patch_size))
    if n == 1:
        raise ValueError("Cannot have only 1 sample; increase patch_radius.")

    X, Y, Z, N_dim = arr_np.shape
    for ax, (s, ps) in enumerate(zip((X, Y, Z), patch_size)):
        if s != 1 and s < ps:
            raise ValueError(
                f"Spatial axis {ax} ({s}) smaller than patch size ({ps})."
            )

    if (n - 1) < N_dim and not suppress_warning:
        warnings.warn(
            f"Number of samples {N_dim} - 1 < Dimensionality {n}. "
            f"Consider increasing patch_radius.",
            UserWarning,
            stacklevel=2,
        )

    t = torch.as_tensor(arr_np, dtype=calc_dtype, device=dev)

    if mask is not None:
        mask_np = np.asarray(mask).astype(bool)
        if mask_np.shape != (X, Y, Z):
            raise ValueError("mask must have the same spatial shape as arr.")
        mask_t = torch.as_tensor(mask_np, device=dev)
    else:
        mask_t = None

    S_trunc = int(min(N_dim, n - 1))
    tau_factor = 1.0 + math.sqrt(N_dim / n)
    tau_factor_sq = tau_factor * tau_factor
    # Eigendecompose the smaller of the covariance and Gram matrices.
    # Covariance is (N x N), Gram is (n x n); the nonzero spectra match.
    use_cov = N_dim < n

    cx = X - 2 * int(pr[0])
    cy = Y - 2 * int(pr[1])
    cz_range = Z - 2 * int(pr[2])
    if cx <= 0 or cy <= 0 or cz_range <= 0:
        raise ValueError("Volume is too small for the chosen patch radius.")

    Px, Py, Pz = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])

    theta = torch.zeros((X, Y, Z), dtype=calc_dtype, device=dev)
    thetax = torch.zeros((X, Y, Z, N_dim), dtype=calc_dtype, device=dev)
    if return_sigma:
        var_sum = torch.zeros((X, Y, Z), dtype=calc_dtype, device=dev)

    for cz in range(cz_range):
        patches_6d = _extract_z_slab(t, pr, cz)  # (cx, cy, Px, Py, Pz, N)
        patches = patches_6d.view(cx * cy, n, N_dim)

        if mask_t is not None:
            center_mask = mask_t[
                int(pr[0]) : X - int(pr[0]),
                int(pr[1]) : Y - int(pr[1]),
                cz + int(pr[2]),
            ].reshape(-1)
            valid_idx = center_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            proc_patches = patches.index_select(0, valid_idx)
        else:
            valid_idx = None
            proc_patches = patches

        B_total = proc_patches.shape[0]
        bs = batch_size if batch_size is not None else B_total

        weighted_Xest = torch.empty_like(proc_patches)
        theta_per_patch = torch.empty(B_total, dtype=calc_dtype, device=dev)
        if return_sigma:
            var_per_patch = torch.empty(B_total, dtype=calc_dtype, device=dev)

        for b0 in range(0, B_total, bs):
            b1 = min(b0 + bs, B_total)
            batch = proc_patches[b0:b1]               # (B, n, N)
            M = batch.mean(dim=1, keepdim=True)       # (B, 1, N)
            Xc = batch - M                            # (B, n, N)

            if use_cov:
                # Covariance (N x N) is smaller than Gram (n x n).
                C = torch.matmul(Xc.transpose(-1, -2), Xc) / n
                ev, evec = _batched_eigh(C)           # (B, N), (B, N, N)
                spectrum_len = N_dim
            else:
                G = torch.matmul(Xc, Xc.transpose(-1, -2)) / n
                ev, evec = _batched_eigh(G)           # (B, n), (B, n, n)
                spectrum_len = n

            # MP classifier on the top-S eigenvalues (matches DIPY
            # truncation of L to size nvoxels - 1 when d > n - 1).
            L = ev[:, spectrum_len - S_trunc :]
            var_b = _mp_classifier(L, n)               # (B,)

            tau = tau_factor_sq * var_b                # (B,)
            sig_mask = (ev >= tau.unsqueeze(-1)).to(ev.dtype)
            num_signal = sig_mask.sum(dim=-1)          # (B,)
            this_theta = 1.0 / (1.0 + num_signal)      # (B,)

            # Reconstruction. The projection onto the signal subspace is
            # the same in both formulations; only the matmul shape changes.
            W_m = evec * sig_mask.unsqueeze(1)
            if use_cov:
                # Xest = Xc @ V_sig V_sig^T + M  (V_sig are columns of W_m)
                proj = torch.matmul(Xc, W_m)          # (B, n, N)
                Xest = torch.matmul(proj, W_m.transpose(-1, -2)) + M
            else:
                # Xest = U_sig U_sig^T Xc + M
                proj = torch.matmul(W_m.transpose(-1, -2), Xc)  # (B, n, N)
                Xest = torch.matmul(W_m, proj) + M

            weighted_Xest[b0:b1] = Xest * this_theta.view(-1, 1, 1)
            theta_per_patch[b0:b1] = this_theta
            if return_sigma:
                var_per_patch[b0:b1] = var_b

        # Scatter the per-valid-patch results back into (cx*cy, ...) space
        if valid_idx is not None:
            Xest_full = torch.zeros(cx * cy, n, N_dim, dtype=calc_dtype, device=dev)
            theta_full = torch.zeros(cx * cy, dtype=calc_dtype, device=dev)
            Xest_full.index_copy_(0, valid_idx, weighted_Xest)
            theta_full.index_copy_(0, valid_idx, theta_per_patch)
            if return_sigma:
                var_full = torch.zeros(cx * cy, dtype=calc_dtype, device=dev)
                var_full.index_copy_(0, valid_idx, var_per_patch * theta_per_patch)
        else:
            Xest_full = weighted_Xest
            theta_full = theta_per_patch
            if return_sigma:
                var_full = var_per_patch * theta_per_patch

        # Overlap-add this z-slab into the global accumulators.
        Xest_r = Xest_full.view(cx, cy, Px, Py, Pz, N_dim)
        theta_r = theta_full.view(cx, cy)
        if return_sigma:
            var_r = var_full.view(cx, cy)

        for dx in range(Px):
            for dy in range(Py):
                for dz_ in range(Pz):
                    z_idx = cz + dz_
                    thetax[dx : dx + cx, dy : dy + cy, z_idx, :].add_(
                        Xest_r[:, :, dx, dy, dz_, :]
                    )
                    theta[dx : dx + cx, dy : dy + cy, z_idx].add_(theta_r)
                    if return_sigma:
                        var_sum[dx : dx + cx, dy : dy + cy, z_idx].add_(var_r)

    # Normalise overlap-add, clip and finalise.
    theta_safe = torch.clamp(theta, min=1e-12)
    denoised = thetax / theta_safe.unsqueeze(-1)
    denoised = denoised.clamp(min=0)
    touched = theta > 0
    denoised = denoised * touched.unsqueeze(-1)

    if mask_t is not None:
        denoised = denoised * mask_t.unsqueeze(-1)

    out_np = denoised.detach().to("cpu").numpy().astype(out_dtype, copy=False)

    if return_sigma:
        var_mean = var_sum / theta_safe
        var_mean[theta == 0] = 0
        sigma_np = torch.sqrt(torch.clamp(var_mean, min=0)).detach().to("cpu").numpy()
        if mask_t is not None:
            sigma_np = sigma_np * mask_np.astype(sigma_np.dtype)
        return out_np, sigma_np.astype(out_dtype, copy=False)

    return out_np


# Public alias matching DIPY's name for drop-in usage.
mppca = mppca_torch
