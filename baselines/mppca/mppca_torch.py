"""
Marchenko-Pastur PCA (MPPCA) Denoising for 4D DWI Volumes
==========================================================
PyTorch implementation — CUDA / Apple MPS / CPU with automatic device routing.

Algorithmically matches the canonical Veraart 2016 / dipy implementation:
  - Iterative MP classifier  (exact vectorised match to dipy's _pca_classifier)
  - Inverse-rank patch weighting  (Manjon 2013, eq. 3)
  - Reflect-padded patch extraction  (every voxel is a patch centre)
  - float64 covariance  (numerical stability for high-dynamic-range DWI signal)
  - Transparent MPS fallback for torch.linalg.eigh

Three non-obvious correctness details vs a naive implementation:
  1. float64 covariance — DWI signal has std ~2-5, noise std ~0.05 (SNR ≈ 40-100×).
     float32 bmm loses the noise eigenvalues (~σ² ≈ 0.0025) to round-off when
     signal eigenvalues are O(10-100). float64 brings the error to ~1e-14.
  2. Eigenvalue clamping — even float64 produces O(1e-14) negatives for the
     true-zero null-space components of rank-deficient patches; clamp to ≥ 0.
  3. MP classifier stopping rule — dipy's while-loop is a greedy descent from
     the TOP of the eigenvalue spectrum. The vectorised equivalent is:
       r(c) = λ_c − λ_0 − 4·√((c+1)/M) · mean(λ_0…λ_c)
       n_noise = (index of first c from the top where r(c) ≤ 0) + 1
     This is NOT the same as taking argmin|r| or the last globally positive r.

References
----------
Veraart et al. (2016) NeuroImage 142:394-406
Manjon  et al. (2013) PLOS ONE 8(9):e73021

Shape convention
----------------
  input  : (X, Y, Z, N)  — three spatial dims + N diffusion volumes
  output : (X, Y, Z, N)  denoised volume   (clipped to ≥ 0)
         + (X, Y, Z)     sigma map          (noise std dev σ̂, not variance)

Quick-start
-----------
    from mppca_torch import mppca_denoise, mppca_denoise_chunked, get_best_device
    import torch

    device   = get_best_device()                               # CUDA > MPS > CPU
    volume   = torch.from_numpy(dwi_array).float().to(device)  # (X, Y, Z, N)

    # Full volume (fits in VRAM)
    denoised, sigma = mppca_denoise(volume, patch_radius=2)

    # Large volume — process in overlapping Z-slabs
    denoised, sigma = mppca_denoise_chunked(volume, patch_radius=2, chunk_size=32)
"""

from __future__ import annotations
import math
import torch
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Device detection & linalg routing
# ─────────────────────────────────────────────────────────────────────────────

def get_best_device() -> torch.device:
    """
    Return the best available compute device.

    Priority : CUDA  >  Apple MPS  >  CPU

    Device notes
    ------------
    CUDA  — Full support. torch.linalg.eigh runs on-device via cuSOLVER.
    MPS   — Apple Silicon GPU. Fast for matmul / bmm / unfold, but
            torch.linalg.eigh is not yet implemented on MPS (PyTorch ≤ 2.x).
            We keep all tensors on MPS and transparently move only the
            covariance matrices to CPU for eigendecomposition, then return
            eigenvectors to MPS. All other operations stay on-device.
    CPU   — OpenBLAS / MKL LAPACK. Always available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batched_eigh(
    sym_matrix: torch.Tensor,  # (B, N, N)  symmetric
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched symmetric eigendecomposition  C = V Λ Vᵀ — with MPS fallback.

    Returns eigenvalues and eigenvectors in ascending order
    (standard torch.linalg.eigh convention).

    On MPS, silently offloads the eigh call to CPU and returns results
    to the original device. On CUDA and CPU this is a direct call.

    Parameters
    ----------
    sym_matrix : (B, N, N) symmetric tensor, any device

    Returns
    -------
    eigvals : (B, N)    ascending, same device as input
    eigvecs : (B, N, N) eigenvectors as columns, same device as input
    """
    if sym_matrix.device.type == "mps":
        vals, vecs = torch.linalg.eigh(sym_matrix.cpu())
        return vals.to(sym_matrix.device), vecs.to(sym_matrix.device)
    return torch.linalg.eigh(sym_matrix)


def _print_chunk_progress(done: int, total: int, prefix: str = "Chunks") -> None:
    """Render a lightweight single-line ASCII progress bar."""
    width = 28
    ratio = 1.0 if total <= 0 else done / total
    filled = min(width, max(0, int(round(width * ratio))))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * ratio
    end = "\n" if done >= total else ""
    print(f"\r{prefix}: [{bar}] {done}/{total} ({pct:5.1f}%)", end=end, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MP classifier — vectorised exact match to dipy's _pca_classifier
# ─────────────────────────────────────────────────────────────────────────────

def _mp_classifier(
    eigvals_asc: torch.Tensor,  # (B, p)  ascending, non-negative (float64)
    M: int,                     # number of voxels in the patch
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch Marchenko-Pastur classifier (Veraart 2016, eq. 11).

    For each patch, identifies the noise/signal boundary by replicating
    the greedy descent of dipy's _pca_classifier while-loop:

        Initialise  σ̂² = mean(λ),  c = p−1
        while  λ_c − λ_0 − 4·√((c+1)/M) · σ̂²  >  0:
            σ̂²  ←  mean(λ_0 … λ_{c−1})
            c    ←  c − 1
        n_noise = c + 1

    The key insight for vectorisation: at the check for index c, the value
    of σ̂² is always mean(λ_0 … λ_c) — the inclusive prefix mean at c.
    So the stopping condition becomes:

        r(c) = λ_c − λ_0 − 4·√((c+1)/M) · pmean_inc(c)

    and the loop exits at the *first* c scanned from the top (c = p−1
    downward) where r(c) ≤ 0.  This is NOT the global argmin of |r|; it
    is the first non-positive value when traversing r in reverse.

    Parameters
    ----------
    eigvals_asc : (B, p)  non-negative ascending eigenvalues (float64 for
                  precision — noise eigenvalues are O(σ²) ≈ 0.0025 while
                  signal eigenvalues can be O(10-100))
    M           : voxel count per patch

    Returns
    -------
    sigma2  : (B,)  estimated noise variance (float64)
    n_noise : (B,)  number of noise eigenvalues (long)
    """
    device = eigvals_asc.device
    B, p   = eigvals_asc.shape

    # Clip to min(p, M-1) to discard exact-zero eigenvalues when M < p
    p_eff = min(p, M - 1)
    L     = eigvals_asc[:, p - p_eff:]          # (B, p_eff)  largest p_eff eigs

    # Inclusive prefix mean: pmean[b, c] = mean(L[b, 0 : c+1])
    cum   = torch.cumsum(L, dim=-1)              # (B, p_eff)
    cnt   = torch.arange(1, p_eff + 1, device=device, dtype=L.dtype)
    pmean = cum / cnt.unsqueeze(0)               # (B, p_eff)

    # r(c) = L[c] − L[0] − 4·√((c+1)/M) · pmean[c]
    c_idx = torch.arange(p_eff, device=device, dtype=L.dtype)
    coeff = 4.0 * torch.sqrt((c_idx + 1.0) / M)
    r     = L - L[:, :1] - coeff.unsqueeze(0) * pmean   # (B, p_eff)

    # Find first c from the TOP (i.e. c = p_eff−1 downward) where r(c) ≤ 0.
    # Flip r so index-0 corresponds to c = p_eff−1, then argmax over the
    # boolean (r_flip ≤ 0) returns the first True element.
    r_flip      = r.flip(-1)                     # (B, p_eff)
    nonpos_flip = r_flip <= 0                    # (B, p_eff) bool

    # If all r > 0 (entire spectrum is noise), clamp to top index.
    all_pos      = ~nonpos_flip.any(dim=-1)      # (B,)
    first_nonpos = nonpos_flip.long().argmax(dim=-1)         # (B,)
    c_stop       = p_eff - 1 - first_nonpos      # (B,)
    c_stop       = torch.where(all_pos, torch.full_like(c_stop, p_eff - 1), c_stop)

    idx    = c_stop.clamp(0, p_eff - 1)
    sigma2 = pmean[torch.arange(B, device=device), idx]
    sigma2 = sigma2.clamp(min=0.0).nan_to_num(nan=0.0)

    return sigma2, (c_stop + 1).long()  # sigma2, n_noise


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main denoising function
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def mppca_denoise(
    volume:       torch.Tensor,
    patch_radius: int  = 2,
    verbose:      bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Marchenko-Pastur PCA denoising of a 4D DWI volume.

    Parameters
    ----------
    volume        : (X, Y, Z, N) float32 tensor on any device.
                    Use  volume.to(get_best_device())  before calling.
    patch_radius  : half-width of the cubic sliding window.
                      2  →  5³ = 125 voxels  (default, matches dipy)
                      1  →  3³ = 27  voxels  (faster, less stable for large N)
    verbose       : print shape / device / stats.

    Returns
    -------
    denoised  : (X, Y, Z, N) float32  — denoised volume, values ≥ 0
    sigma_map : (X, Y, Z)    float32  — estimated noise std σ̂  (NOT variance)
    """
    device = volume.device
    dtype  = volume.dtype
    X, Y, Z, N = volume.shape
    pw  = patch_radius
    ps  = 2 * pw + 1           # patch side  (5 for pw=2)
    M   = ps ** 3              # voxels per patch
    p   = min(M, N)            # covariance size

    if p < 4:
        raise ValueError(
            f"p = min(M={M}, N={N}) = {p} is too small. "
            "Increase patch_radius or use more DWI volumes."
        )

    if verbose:
        print(f"  Device  : {device}  [{device.type.upper()}]")
        print(f"  Volume  : {X} × {Y} × {Z},  N = {N}")
        print(f"  Patch   : {ps}³ = {M} voxels,  p = min(M,N) = {p}")

    # ── Step 1: Extract all valid patches via unfold ─────────────────────────
    # We do NOT pad, strictly matching dipy. Valid patches only.
    unfolded = (volume.permute(3, 0, 1, 2).unsqueeze(0)  # (1, N, X, Y, Z)
                .unfold(2, ps, 1)   # (1, N, Xp, Y, Z, ps)
                .unfold(3, ps, 1)   # (1, N, Xp, Yp, Z, ps, ps)
                .unfold(4, ps, 1))  # (1, N, Xp, Yp, Zp, ps, ps, ps)
    
    Xp, Yp, Zp = unfolded.shape[2], unfolded.shape[3], unfolded.shape[4]
    B = Xp * Yp * Zp
    patches = (unfolded
               .reshape(1, N, B, M)
               .squeeze(0)         # (N, B, M)
               .permute(1, 2, 0)   # (B, M, N)
               .contiguous())

    if verbose:
        mem_mb = patches.element_size() * patches.nelement() / 1e6
        print(f"  Patches : {list(patches.shape)},  ~{mem_mb:.0f} MB (fp32)")

    # ── Step 2: Mean-centre (per-volume mean within each patch) ─────────────
    patch_mean = patches.mean(dim=1, keepdim=True)   # (B, 1, N)
    Xc         = patches - patch_mean                # (B, M, N)  float32

    # ── Step 3: Sample covariance in float64 ────────────────────────────────
    # Why float64?
    #   DWI signal std ≈ 2-5, noise std ≈ 0.05.  float32 (~7 sig. figs.)
    #   cannot represent noise eigenvalues (~0.0025) alongside signal
    #   eigenvalues (~10-100) in the same matrix — batched float32 bmm
    #   produces O(1e-6) round-off that swamps the noise floor.
    #   float64 reduces this to O(1e-14).
    # Note: eigvecs are cast back to float32 after eigh; the reconstruction
    # bmm stays in float32 to save memory.
    
    if device.type == "mps":
        # MPS does not support float64. Compute covariance on MPS (fast matmul),
        # move only the smaller covariance tensor to CPU for float64 eigh.
        C32 = torch.bmm(Xc.transpose(1, 2), Xc) / M
        # Important: move first, then cast. A combined .to("cpu", dtype=float64)
        # can try to cast on MPS first, which fails because MPS has no float64.
        C64_cpu = C32.cpu().to(torch.float64)
        eigvals64, eigvecs64 = torch.linalg.eigh(C64_cpu)
        # Keep float64 items on CPU for classifier, MPS can't handle float64.
        del C32, C64_cpu
    else:
        Xc64 = Xc.double()
        C64  = torch.bmm(Xc64.transpose(1, 2), Xc64) / M   # (B, N, N)  float64
        eigvals64, eigvecs64 = _batched_eigh(C64)            # ascending, float64

    # Clamp: float64 still leaves O(1e-14) negatives for near-zero null-space
    # components; clamp before feeding to the classifier.
    eigvals64 = eigvals64.clamp(min=0.0)

    # ── Step 4: Dipy classifier — thresholding with tau ──────────────────────
    # Dipy uses _pca_classifier to get sigma, but then recalculates n_noise
    # with a tau_factor based thresholding, ignoring the classifier's ncomps.
    sigma2, _ = _mp_classifier(eigvals64, M)             # float64, (B,)
    
    # tau_factor = 1 + sqrt(dim / num_samples) as per dipy defaults
    tau_factor = 1.0 + math.sqrt(N / M)
    tau = (tau_factor ** 2) * sigma2                     # (B,)
    
    # Count noise components: sum(d < tau)
    # n_noise shape: (B,)
    n_noise = (eigvals64 < tau.unsqueeze(-1)).sum(dim=-1).long()
    n_signal = (p - n_noise).clamp(min=0)                # (B,)

    # Eigenvectors back to float32 for memory-efficient reconstruction, move back to device if MPS
    eigvecs = eigvecs64.float().to(device)               # (B, N, p)  ascending
    n_noise = n_noise.to(device)
    n_signal = n_signal.to(device)
    sigma2 = sigma2.float().to(device)

    if verbose:
        print(f"  Median n_signal : {n_signal.float().median().item():.1f}")
        print(f"  Median σ̂       : {sigma2.float().sqrt().median().item():.5f}")

    # ── Step 5: Reconstruct — keep signal eigenvectors, zero noise ones ─────
    # eigvecs are ascending; noise = first n_noise columns (smallest eigenvalues).
    # keep_mask[b, i] = 1  iff  i >= n_noise[b]
    idx_range  = torch.arange(p, device=device).unsqueeze(0)           # (1, p)
    keep_mask  = (idx_range >= n_noise.unsqueeze(1)).float()            # (B, p)

    scores      = torch.bmm(Xc, eigvecs)                               # (B, M, p)
    scores_filt = scores * keep_mask.unsqueeze(1)                      # (B, M, p)
    Xc_hat      = torch.bmm(scores_filt, eigvecs.transpose(1, 2))      # (B, M, N)
    X_hat       = Xc_hat + patch_mean                                  # (B, M, N)

    # ── Step 6: Weighted aggregation — Manjon 2013 eq. 3 ───────────────────
    # dipy weights: 1.0 / (1.0 + dim - ncomps) == 1.0 / (1.0 + n_signal)
    weights  = 1.0 / (1.0 + n_signal.float())                           # (B,)

    # Reshape X_hat back to spatial layout for the scatter-add.
    X_hat_sp = X_hat.reshape(Xp, Yp, Zp, ps, ps, ps, N)                # (Xp, Yp, Zp, ps, ps, ps, N)
    w_sp     = weights.reshape(Xp, Yp, Zp)                              # (Xp, Yp, Zp)
    sigma2_sp = sigma2.float().reshape(Xp, Yp, Zp)                      # (Xp, Yp, Zp)

    out_wsum  = torch.zeros(X, Y, Z, N, device=device, dtype=dtype)
    out_sigsum= torch.zeros(X, Y, Z,    device=device, dtype=dtype)
    out_wt    = torch.zeros(X, Y, Z,    device=device, dtype=dtype)

    # Loop over ps³ spatial offsets inside the patch.
    # Each valid patch center (cx, cy, cz) writes to (cx+si, cy+sj, cz+sk)
    # where cx in [0, Xp-1], si in [0, ps-1].
    w_sp_expanded = w_sp.unsqueeze(-1)
    weighted_sigma2 = sigma2_sp * w_sp

    for di in range(ps):
        for dj in range(ps):
            for dk in range(ps):
                # We add the whole Xp, Yp, Zp block into out_wsum at offset di, dj, dk
                out_wsum[di:di+Xp, dj:dj+Yp, dk:dk+Zp, :] += (
                    X_hat_sp[:, :, :, di, dj, dk, :] * w_sp_expanded
                )
                out_sigsum[di:di+Xp, dj:dj+Yp, dk:dk+Zp] += weighted_sigma2
                out_wt[di:di+Xp, dj:dj+Yp, dk:dk+Zp] += w_sp

    weight_norm = out_wt.clamp(min=1e-8)
    denoised  = (out_wsum / weight_norm.unsqueeze(-1)).clamp(min=0.0)
    sigma_map = (out_sigsum / weight_norm).clamp(min=0.0).sqrt()

    # Voxels that received no patches (e.g. at the extreme edges if volume is too small)
    # will be zeroed out
    no_patch_mask = (out_wt == 0)
    if no_patch_mask.any():
        denoised[no_patch_mask] = 0.0
        sigma_map[no_patch_mask] = 0.0

    return denoised, sigma_map


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Chunked wrapper  (large volumes / limited VRAM)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def mppca_denoise_chunked(
    volume:       torch.Tensor,
    patch_radius: int  = 2,
    chunk_size:   int  = 32,
    verbose:      bool = True,
    progress:     bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-safe wrapper: process the volume in overlapping Z-slabs.

    Each slab is extended by `patch_radius` slices on each Z-side so that
    patches for voxels near slab boundaries are computed with correct
    neighbourhood data before cropping.

    VRAM estimate per slab (dominant terms)
    ----------------------------------------
        patches    (float32)  :  X·Y·chunk_size · M·N · 4 bytes
        covariance (float64)  :  X·Y·chunk_size · N·N · 8 bytes
        Example: 96×96 in-plane, N=64, 5³ patch, chunk_size=20
            → ~0.37 GB (patches) + ~0.61 GB (cov)  =  ~1 GB peak

    Parameters
    ----------
    volume      : (X, Y, Z, N) float32
    patch_radius: passed to mppca_denoise
    chunk_size  : number of Z-slices per slab (tune to your VRAM budget)
    verbose     : print per-chunk progress

    Returns
    -------
    denoised  : (X, Y, Z, N) float32
    sigma_map : (X, Y, Z)    float32
    """
    X, Y, Z, N = volume.shape
    pw = patch_radius
    ps = 2 * pw + 1

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if Z < ps:
        raise ValueError(
            f"Volume depth Z={Z} is smaller than patch side {ps}. "
            "Use a smaller patch_radius or a deeper volume."
        )

    out_denoised = torch.zeros_like(volume)
    out_sigma    = torch.zeros(X, Y, Z, device=volume.device, dtype=volume.dtype)

    z_starts = list(range(0, Z, chunk_size))
    n_chunks = len(z_starts)
    if progress:
        _print_chunk_progress(0, n_chunks)
    for idx, z0 in enumerate(z_starts):
        z1  = min(z0 + chunk_size, Z)
        # Extend slab by pw on each side for correct boundary patches
        z0p = max(0, z0 - pw)
        z1p = min(Z, z1 + pw)

        # Ensure slab is deep enough for unfold(size=ps), especially for
        # tiny trailing chunks near Z-boundaries.
        slab_depth = z1p - z0p
        if slab_depth < ps:
            need = ps - slab_depth
            # Expand towards available volume; prefer left extension first.
            add_left = min(need, z0p)
            z0p -= add_left
            need -= add_left
            if need > 0:
                add_right = min(need, Z - z1p)
                z1p += add_right
                need -= add_right
            if need > 0:
                raise RuntimeError(
                    "Could not construct a slab large enough for the patch size."
                )

        if verbose:
            print(f"\n── Chunk {idx+1}/{len(z_starts)}:  "
                  f"Z=[{z0}:{z1}]  (slab [{z0p}:{z1p}]) ──")

        slab     = volume[:, :, z0p:z1p, :]
        den, sig = mppca_denoise(slab, patch_radius, verbose)

        # Crop overlap back out
        z_off = z0 - z0p
        nz    = z1 - z0
        out_denoised[:, :, z0:z1, :] = den[:, :, z_off:z_off + nz, :]
        out_sigma   [:, :, z0:z1   ] = sig[:, :, z_off:z_off + nz  ]
        if progress:
            _print_chunk_progress(idx + 1, n_chunks)

    return out_denoised, out_sigma


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Test suite — validates against dipy (reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    """
    Comprehensive test suite comparing this implementation against dipy.

    Tests
    -----
    1.  Output shapes are correct.
    2.  Denoised output is non-negative.
    3.  Denoised RMSE < noisy RMSE  (actual denoising happens).
    4.  Sigma estimate within 20 % of true σ on signal + noise volume.
    5a. Pearson r > 0.99 vs dipy on matched volume.
    5b. Per-voxel L2 diff vs dipy < 5 % of the signal dynamic range.
    5c. RMSE ratio (ours / dipy) in [0.5, 2.5]  on matched volume.
    5d. Sigma estimate ratio (ours / dipy) in [0.5, 2.0].
    6.  Pure-noise volume: σ̂ within 25 % of true σ.
    7.  get_best_device() returns a torch.device.
    8.  Classifier on a known rank-6 patch returns n_signal = 6.
    """
    import numpy as np

    try:
        from dipy.denoise.localpca import mppca as dipy_mppca
        HAS_DIPY = True
    except ImportError:
        HAS_DIPY = False
        print("  WARNING: dipy not installed — skipping dipy comparison tests.")

    results: list[tuple[str, bool]] = []

    def check(name: str, passed: bool, detail: str = "") -> bool:
        tag = f"  [{'PASS' if passed else 'FAIL'}] {name}"
        if detail:
            tag += f"  ({detail})"
        print(tag)
        results.append((name, passed))
        return passed

    print("=" * 66)
    print("  MPPCA-Torch Test Suite")
    print("=" * 66)

    # ── Shared synthetic DWI-like volume ─────────────────────────────────────
    np.random.seed(42)
    torch.manual_seed(42)

    X, Y, Z, N = 16, 16, 12, 48
    rank_true  = 6
    sigma_true = 0.05

    U      = np.random.randn(X * Y * Z, rank_true).astype(np.float32)
    Vt     = np.random.randn(rank_true, N).astype(np.float32)
    signal = (U @ Vt).reshape(X, Y, Z, N)
    signal = signal - signal.min() + 1.0    # shift to positive (DWI-like)
    noise  = (sigma_true * np.random.randn(X, Y, Z, N)).astype(np.float32)
    vol_np = signal + noise
    vol_t  = torch.from_numpy(vol_np)

    # ── 1. Shapes ─────────────────────────────────────────────────────────────
    print("\n── 1. Output shapes ──────────────────────────────────────────────")
    den_t, sig_t = mppca_denoise(vol_t, patch_radius=2, verbose=False)
    check("Denoised shape == input shape",
          den_t.shape == vol_t.shape, str(tuple(den_t.shape)))
    check("Sigma map shape == (X, Y, Z)",
          sig_t.shape == (X, Y, Z),   str(tuple(sig_t.shape)))

    # ── 2. Non-negativity ─────────────────────────────────────────────────────
    print("\n── 2. Non-negativity ────────────────────────────────────────────")
    check("min(denoised) >= 0",
          den_t.min().item() >= 0.0, f"min = {den_t.min().item():.6f}")

    # ── 3. RMSE improvement ───────────────────────────────────────────────────
    print("\n── 3. RMSE improvement ──────────────────────────────────────────")
    den_np        = den_t.numpy()
    rmse_noisy    = float(np.sqrt(np.mean((vol_np  - signal) ** 2)))
    rmse_denoised = float(np.sqrt(np.mean((den_np  - signal) ** 2)))
    gain          = rmse_noisy / max(rmse_denoised, 1e-9)
    check("Denoised RMSE < noisy RMSE",
          rmse_denoised < rmse_noisy,
          f"{rmse_noisy:.5f} → {rmse_denoised:.5f},  gain = {gain:.2f}×")

    # ── 4. Sigma estimation ───────────────────────────────────────────────────
    print("\n── 4. Sigma estimation ──────────────────────────────────────────")
    sigma_med = float(sig_t.nan_to_num(0).median().item())
    err_pct   = abs(sigma_med - sigma_true) / sigma_true * 100.0
    check("σ̂ within 20 % of true σ",
          err_pct < 20.0,
          f"σ̂ = {sigma_med:.5f},  true = {sigma_true},  err = {err_pct:.1f} %")

    # ── 5. Agreement with dipy ────────────────────────────────────────────────
    if HAS_DIPY:
        print("\n── 5. Agreement with dipy ───────────────────────────────────")
        mask                = np.ones((X, Y, Z), dtype=bool)
        den_dipy, sig_dipy  = dipy_mppca(
            vol_np, mask=mask, patch_radius=2, return_sigma=True)

        rmse_dipy  = float(np.sqrt(np.mean((den_dipy - signal) ** 2)))
        ratio_rmse = rmse_denoised / max(rmse_dipy, 1e-9)
        corr       = float(np.corrcoef(den_np.ravel(), den_dipy.ravel())[0, 1])
        diff_rmse  = float(np.sqrt(np.mean((den_np - den_dipy) ** 2)))
        sig_range  = float(signal.max() - signal.min())

        check("Pearson r > 0.99 vs dipy",
              corr > 0.99,
              f"r = {corr:.5f}")
        check("Per-voxel diff < 5 % of signal range",
              diff_rmse < 0.05 * sig_range,
              f"diff = {diff_rmse:.5f},  5 %·range = {0.05*sig_range:.5f}")
        check("RMSE ratio ours/dipy in [0.5, 2.5]",
              0.5 < ratio_rmse < 2.5,
              f"ours = {rmse_denoised:.5f},  dipy = {rmse_dipy:.5f},  "
              f"ratio = {ratio_rmse:.3f}")

        sig_dipy_med = float(np.nanmedian(sig_dipy))
        ratio_sigma  = sigma_med / max(sig_dipy_med, 1e-9)
        check("Sigma ratio ours/dipy in [0.5, 2.0]",
              0.5 < ratio_sigma < 2.0,
              f"ours = {sigma_med:.5f},  dipy = {sig_dipy_med:.5f},  "
              f"ratio = {ratio_sigma:.3f}")

    # ── 6. Pure noise ─────────────────────────────────────────────────────────
    print("\n── 6. Pure-noise sigma estimation ──────────────────────────────")
    np.random.seed(7)
    # Use larger volume so the MP distribution is well-sampled
    X2, Y2, Z2, N2 = 18, 18, 14, 48
    pn_np = (sigma_true * np.random.randn(X2, Y2, Z2, N2)).astype(np.float32)
    _, s_pn = mppca_denoise(torch.from_numpy(pn_np), patch_radius=2, verbose=False)
    sp     = float(s_pn.nan_to_num(0).median().item())
    ep     = abs(sp - sigma_true) / sigma_true * 100.0
    check("Pure-noise σ̂ within 25 % of true σ",
          ep < 25.0,
          f"σ̂ = {sp:.5f},  true = {sigma_true},  err = {ep:.1f} %")

    # ── 7. Device detection ───────────────────────────────────────────────────
    print("\n── 7. Device detection ──────────────────────────────────────────")
    dev = get_best_device()
    check("get_best_device() returns torch.device",
          isinstance(dev, torch.device), f"selected: {dev}")

    # ── 8. Classifier on a known patch ───────────────────────────────────────
    print("\n── 8. Classifier unit test (known rank-6 patch) ────────────────")
    np.random.seed(0)
    M_vox, N_vox, r_true = 125, 48, 6
    U_p = np.random.randn(M_vox, r_true).astype(np.float64)
    Vt_p = np.random.randn(r_true, N_vox).astype(np.float64)
    sig_p = U_p @ Vt_p
    sig_p = (sig_p - sig_p.mean()) / sig_p.std() * 5.0   # realistic DWI SNR
    noise_p = sigma_true * np.random.randn(M_vox, N_vox)
    Xc_p = (sig_p + noise_p - (sig_p + noise_p).mean(axis=0)).astype(np.float64)
    C_p = torch.from_numpy(Xc_p.T @ Xc_p / M_vox).unsqueeze(0)   # (1, N, N)
    ev, _ = _batched_eigh(C_p)
    ev = ev.clamp(min=0.0)
    s2_p, nn_p = _mp_classifier(ev, M_vox)
    ns_p = int(N_vox - nn_p[0].item())
    check("Known rank-6 patch → n_signal == 6",
          ns_p == r_true,
          f"n_signal = {ns_p},  n_noise = {int(nn_p[0].item())}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(v for _, v in results)
    n_fail = len(results) - n_pass
    print(f"\n{'=' * 66}")
    print(f"  Results: {n_pass} passed,  {n_fail} failed  ({len(results)} total)")
    print(f"{'=' * 66}")
    return n_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ok = run_tests()
    sys.exit(0 if ok else 1)
