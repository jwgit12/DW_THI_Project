import numpy as np


# -----------------------------
# Convert 6D tensor → 3x3 matrix
# -----------------------------
def tensor6_to_matrix(t):
    """
    Input: (..., 6)
    Output: (..., 3, 3)
    Order assumed: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    """

    Dxx = t[..., 0]
    Dyy = t[..., 1]
    Dzz = t[..., 2]
    Dxy = t[..., 3]
    Dxz = t[..., 4]
    Dyz = t[..., 5]

    mat = np.stack([
        np.stack([Dxx, Dxy, Dxz], axis=-1),
        np.stack([Dxy, Dyy, Dyz], axis=-1),
        np.stack([Dxz, Dyz, Dzz], axis=-1)
    ], axis=-2)

    return mat


# -----------------------------
# Mean Diffusivity (MD)
# -----------------------------
def compute_md_from_tensor6(tensor6):
    mat = tensor6_to_matrix(tensor6)

    eigvals = np.linalg.eigvalsh(mat)
    md = np.mean(eigvals, axis=-1)

    return md


# -----------------------------
# Fractional Anisotropy (FA)
# -----------------------------
def compute_fa_from_tensor6(tensor6):
    mat = tensor6_to_matrix(tensor6)

    eigvals = np.linalg.eigvalsh(mat)

    l1 = eigvals[..., 0]
    l2 = eigvals[..., 1]
    l3 = eigvals[..., 2]

    mean = (l1 + l2 + l3) / 3.0

    numerator = ((l1 - mean)**2 + (l2 - mean)**2 + (l3 - mean)**2)
    denominator = (l1**2 + l2**2 + l3**2) + 1e-8

    fa = np.sqrt(1.5 * numerator / denominator)

    return fa
