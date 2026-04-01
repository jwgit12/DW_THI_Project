import numpy as np
import matplotlib.pyplot as plt
import torch
from dipy.denoise.localpca import mppca as dipy_mppca
from baselines.mppca.mppca_torch import mppca_denoise

def main():
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

    # Torch implementation
    print("Running Torch MPPCA...")
    den_t, sig_t = mppca_denoise(vol_t, patch_radius=2, verbose=False)
    den_np = den_t.numpy()

    # dipy implementation
    print("Running Dipy MPPCA...")
    mask = np.ones((X, Y, Z), dtype=bool)
    den_dipy, sig_dipy = dipy_mppca(vol_np, mask=mask, patch_radius=2, return_sigma=True)

    # Difference
    diff = np.abs(den_np - den_dipy)

    # Plot
    # Select a middle slice and the first volume (or a specific one)
    z_slice = Z // 2
    v_idx = 0

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.title("Noisy Signal")
    plt.imshow(vol_np[:, :, z_slice, v_idx], cmap='gray')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Torch MPPCA")
    plt.imshow(den_np[:, :, z_slice, v_idx], cmap='gray')
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Dipy MPPCA")
    plt.imshow(den_dipy[:, :, z_slice, v_idx], cmap='gray')
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Absolute Difference")
    # vmin and vmax can be set to highlight the small differences
    plt.imshow(diff[:, :, z_slice, v_idx], cmap='hot', vmin=0, vmax=diff.max())
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("mppca_comparison.png")
    print("Comparison plot saved to mppca_comparison.png")

if __name__ == "__main__":
    main()

