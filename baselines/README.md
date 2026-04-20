# `baselines` — GPU-accelerated DWI denoising baselines

A small package of high-performance, GPU-accelerated baselines for
diffusion MRI denoising. Currently ships a single function:

| Baseline | API | Backend |
|----------|-----|---------|
| **MP-PCA** (Marchenko–Pastur PCA, Veraart et al. 2016) | `baselines.mppca` | PyTorch — CUDA, Apple MPS, or CPU |

The implementation is a drop-in replacement for
[`dipy.denoise.localpca.mppca`](https://docs.dipy.org/stable/reference/dipy.denoise.html)
with `pca_method='eig'`. Results agree with DIPY to float32 precision
(typical `max|diff|` ≈ 1e-5 on inputs scaled to ~100).

## Why

DIPY's reference implementation is a pure-Python triple nested loop over
voxels that calls SciPy's `eigh` on every patch. For a `130×130×25×150`
scan with `patch_radius=2` (125 samples per 5×5×5 patch) that is
≈333 000 independent 125×125 eigendecompositions wrapped in Python
overhead — roughly 3–4 minutes on a modern CPU.

The GPU version in this package performs exactly the same math but
computes all patches in a z-slab in parallel, which brings big speedups
on any device with batched BLAS/LAPACK available.

## Installation

No extra dependencies beyond the project's `requirements.txt` (torch,
numpy, dipy). The package is imported from the repository root:

```python
from baselines import mppca
```

## Usage

```python
import numpy as np
from baselines import mppca

arr = np.load("dwi.npy")           # (X, Y, Z, N), float32 or float64
denoised = mppca(arr, patch_radius=2)

# Optional: return the estimated noise std
denoised, sigma = mppca(arr, patch_radius=2, return_sigma=True)

# Optional: brain mask (only centres inside the mask are processed;
#          voxels outside are zeroed in the output)
denoised = mppca(arr, patch_radius=2, mask=brain_mask)

# Force a specific backend (default: cuda > mps > cpu)
denoised = mppca(arr, patch_radius=2, device="mps")

# Limit peak memory by processing smaller patch batches
denoised = mppca(arr, patch_radius=2, batch_size=2048)
```

### Full signature

```python
mppca(
    arr,                      # (X, Y, Z, N) ndarray or torch.Tensor
    *,
    mask=None,                # optional (X, Y, Z) bool mask
    patch_radius=2,           # int or length-3 array
    return_sigma=False,
    out_dtype=None,           # defaults to arr.dtype
    device=None,              # "cuda" | "mps" | "cpu", auto if None
    batch_size=None,          # patches per eigh; default = one z-slab
    suppress_warning=False,
)
```

## What it does under the hood

For each voxel `(i, j, k)` not on the boundary:

1. extract the 5×5×5×N patch `X` (125 samples × N features),
2. mean-centre and compute the Gram matrix `G = X X^T / n`
   (125×125 — always smaller than the full covariance when `N ≥ 125`),
3. eigendecompose `G`, classify noise eigenvalues with the
   Marchenko–Pastur threshold,
4. reconstruct the signal subspace projection `U_sig U_sig^T X_centred + µ`,
5. overlap-add the patch back into the volume with the redundancy
   weighting `1 / (1 + #signal components)` from Manjón et al. 2013.

The math is the same as DIPY's — the speedup is purely from doing all
patches in a z-slab in parallel on a vectorised batched eigh. See the
top-of-file docstring of `mppca_torch.py` for references.

### Smaller-matrix trick

DIPY always eigendecomposes the N×N covariance. We pick the smaller of
the two algebraically equivalent matrices:

* `N ≥ n` (common DWI case, e.g. N=150, n=125 for a 5×5×5 patch): use
  the n×n Gram matrix. Its nonzero eigenvalues coincide with the
  covariance's, and the reconstruction `X @ V_sig V_sig^T` can be
  rewritten as `U_sig @ U_sig^T @ X` with the Gram eigenvectors.
* `N < n` (e.g. a b0+60-direction shell with a 5×5×5 patch): use the
  original N×N covariance.

So eigh always runs on an `min(n, N) × min(n, N)` matrix.

### MPS caveat

PyTorch 2.11 does not yet implement `torch.linalg.eigh` on Apple MPS
(see [pytorch/pytorch#141287](https://github.com/pytorch/pytorch/issues/141287)).
We transparently detour only the eigh call through CPU; unified memory
on M-series chips makes the copy negligible, and the rest of the
pipeline (patch extraction, matmul, overlap-add) still runs on the GPU.
On CUDA, eigh is native and the whole pipeline stays on-device.

## Benchmarks

Benchmarks are included in `benchmark.py`:

```bash
python -m baselines.benchmark                             # preset sizes
python -m baselines.benchmark --shape 130 130 25 150      # custom
python -m baselines.benchmark --sizes full --repeats 3
python -m baselines.benchmark --no_dipy                   # torch only
```

The script times both the torch implementation and DIPY's reference on
the same synthetic volume and reports max/mean absolute differences.

### Measured timings

Measured with `python -m baselines.benchmark`. DIPY uses its default
`pca_method='eig'`. See `benchmark.py` for the synthetic data generator.

_(Values below come from the M4-Pro 48 GB used during development;
your numbers will differ.)_

| Shape (X, Y, Z, N) | Patches | DIPY (CPU) | Torch (MPS) | Speed-up | max\|diff\| |
|---|---:|---:|---:|---:|---:|
| 60×60×15×80    |  34 496 |  11.7 s |  7.6 s | 1.5× | 4.6e-5 |
| 100×100×20×120 | 147 456 | 109.7 s | 66.9 s | 1.6× | 5.0e-5 |
| 130×130×25×150 | 333 396 | 757.5 s | 163.1 s | 4.6× | 5.0e-5 |

The speedup grows with problem size: the constant PyTorch/MPS setup
cost amortises, while DIPY's per-voxel Python overhead scales linearly
with patch count. Run
`python -m baselines.benchmark --sizes small medium full` locally to
repopulate these rows on your own hardware.

## Correctness

`mppca_torch` is verified numerically against DIPY on every benchmark
call. The expected agreement for float32 inputs is:

* `max|diff| ≲ 1e-4 * max|dipy|` (≈ machine epsilon ×  condition number
  of a symmetric eigendecomposition),
* `mean|diff| ≲ 1e-6 * max|dipy|`.

Differences come from the eigh running on different LAPACK drivers
between DIPY (SciPy's `eigh`) and PyTorch (Accelerate on macOS, MAGNA
on CUDA), not from an algorithmic mismatch.

## Limitations

* Only `pca_method='eig'`; no `'svd'` path. The two are numerically
  equivalent for full-rank symmetric PSD matrices.
* Only the MP-PCA variant (`sigma=None`, auto-estimated). DIPY's
  `genpca`/`localpca` with user-supplied `sigma` is not yet covered.
* Output dtype follows DIPY: input float32 → internal float32,
  input float64 → internal float64. Casting to half-precision is not
  supported (eigh on fp16 is ill-conditioned).

## Layout

```
baselines/
├── README.md          ← you are here
├── __init__.py        ← re-exports `mppca`
├── mppca_torch.py     ← the implementation
└── benchmark.py       ← `python -m baselines.benchmark`
```
