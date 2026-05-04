import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime

import config
from dataset import DWIDataset2D
from model import MRDDenoiser

# -------------------------
# Utilities
# -------------------------
def pct_scale(x, pmin=1, pmax=99):
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return x, 0, 1
    vmin, vmax = np.percentile(vals, (pmin, pmax))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return x, vmin, vmax

def brain_mask_from_b0(dwi, bvals):
    # dwi: (N,H,W)
    b0 = dwi[bvals < 50]
    if b0.size == 0:
        b0 = dwi[:1]
    b0m = b0.mean(0)
    thr = np.percentile(b0m, 60) * 0.3  # simple, robust
    mask = (b0m > thr).astype(np.uint8)
    return mask

def apply_mask(x, mask):
    return x * mask

# -------------------------
# DTI (same math, no deps)
# -------------------------
def build_B(bvals, bvecs):
    B = []
    for i in range(len(bvals)):
        g = bvecs[:, i]
        b = bvals[i]
        B.append([
            -b * g[0]**2,
            -b * g[1]**2,
            -b * g[2]**2,
            -2*b * g[0]*g[1],
            -2*b * g[0]*g[2],
            -2*b * g[1]*g[2],
        ])
    return np.array(B)

def fit_tensor_slice(dwi, bvals, bvecs):
    # dwi: (N,H,W)
    dwi = np.clip(dwi, 1e-6, None)
    m = bvals > 50
    dwi = dwi[m]
    bvals = bvals[m]
    bvecs = bvecs[:, m]

    B = build_B(bvals, bvecs)
    Bpinv = np.linalg.pinv(B)

    H, W = dwi.shape[1:]
    T = np.zeros((6, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            s = dwi[:, i, j]
            T[:, i, j] = Bpinv @ np.log(s)
    return T

def tensor_to_fa_md(T):
    H, W = T.shape[1:]
    fa = np.zeros((H, W), dtype=np.float32)
    md = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            Dxx, Dxy, Dyy, Dxz, Dyz, Dzz = T[:, i, j]
            M = np.array([[Dxx, Dxy, Dxz],
                          [Dxy, Dyy, Dyz],
                          [Dxz, Dyz, Dzz]], dtype=np.float32)
            w = np.linalg.eigvalsh(M)
            w = np.sort(w)[::-1]
            m = w.mean()
            md[i, j] = m
            num = np.sqrt(((w - m)**2).sum())
            den = np.sqrt((w**2).sum() + 1e-12)
            fa[i, j] = np.sqrt(1.5) * num / den
    return fa, md

# -------------------------
# Model helpers
# -------------------------
def load_model(path, device):
    m = MRDDenoiser(in_channels=130).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def denoise(model, x):
    with torch.no_grad():
        return (x - model(x))[0].cpu().numpy()

# -------------------------
# Visualization
# -------------------------
def visualize(m1_path, m2_path, idx=0):
    os.makedirs("visuals", exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    ds = DWIDataset2D(mode="test")
    x_noisy, x_clean, tensor_gt, bvals, bvecs = ds[idx]

    x_noisy_t = x_noisy.unsqueeze(0).to(device)

    m1 = load_model(m1_path, device)
    m2 = load_model(m2_path, device)

    out1 = denoise(m1, x_noisy_t)
    out2 = denoise(m2, x_noisy_t)
    noisy = x_noisy_t[0].cpu().numpy()

    # mask
    mask = brain_mask_from_b0(noisy, bvals)

    # tensors
    Tn = fit_tensor_slice(noisy, bvals, bvecs)
    T1 = fit_tensor_slice(out1, bvals, bvecs)
    T2 = fit_tensor_slice(out2, bvals, bvecs)
    Tgt = tensor_gt.numpy()

    # FA/MD
    fa_n, md_n = tensor_to_fa_md(Tn)
    fa_1, md_1 = tensor_to_fa_md(T1)
    fa_2, md_2 = tensor_to_fa_md(T2)
    fa_gt, md_gt = tensor_to_fa_md(Tgt)

    # apply mask
    fa_n = apply_mask(fa_n, mask)
    fa_1 = apply_mask(fa_1, mask)
    fa_2 = apply_mask(fa_2, mask)
    fa_gt = apply_mask(fa_gt, mask)

    md_n = apply_mask(md_n, mask)
    md_1 = apply_mask(md_1, mask)
    md_2 = apply_mask(md_2, mask)
    md_gt = apply_mask(md_gt, mask)

    # shared scales per row (critical)
    _, fa_vmin, fa_vmax = pct_scale(fa_gt, 1, 99)
    _, md_vmin, md_vmax = pct_scale(md_gt, 1, 99)

    # differences (symmetric)
    d_fa_1 = fa_1 - fa_gt
    d_fa_2 = fa_2 - fa_gt
    d_md_1 = md_1 - md_gt
    d_md_2 = md_2 - md_gt

    def sym_lim(a, b):
        m = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-8)
        return -m, m

    fa_lo, fa_hi = sym_lim(d_fa_1, d_fa_2)
    md_lo, md_hi = sym_lim(d_md_1, d_md_2)

    # plot
    fig, axs = plt.subplots(2, 6, figsize=(22, 8))

    # FA row
    axs[0,0].imshow(fa_n, cmap='viridis', vmin=fa_vmin, vmax=fa_vmax); axs[0,0].set_title("Noisy")
    axs[0,1].imshow(fa_1, cmap='viridis', vmin=fa_vmin, vmax=fa_vmax); axs[0,1].set_title("MRD")
    axs[0,2].imshow(fa_2, cmap='viridis', vmin=fa_vmin, vmax=fa_vmax); axs[0,2].set_title("MRD+DTI")
    axs[0,3].imshow(fa_gt, cmap='viridis', vmin=fa_vmin, vmax=fa_vmax); axs[0,3].set_title("GT")
    axs[0,4].imshow(d_fa_1, cmap='bwr', vmin=fa_lo, vmax=fa_hi); axs[0,4].set_title("Δ FA (MRD-GT)")
    axs[0,5].imshow(d_fa_2, cmap='bwr', vmin=fa_lo, vmax=fa_hi); axs[0,5].set_title("Δ FA (DTI-GT)")

    # MD row
    axs[1,0].imshow(md_n, cmap='magma', vmin=md_vmin, vmax=md_vmax); axs[1,0].set_title("Noisy")
    axs[1,1].imshow(md_1, cmap='magma', vmin=md_vmin, vmax=md_vmax); axs[1,1].set_title("MRD")
    axs[1,2].imshow(md_2, cmap='magma', vmin=md_vmin, vmax=md_vmax); axs[1,2].set_title("MRD+DTI")
    axs[1,3].imshow(md_gt, cmap='magma', vmin=md_vmin, vmax=md_vmax); axs[1,3].set_title("GT")
    axs[1,4].imshow(d_md_1, cmap='bwr', vmin=md_lo, vmax=md_hi); axs[1,4].set_title("Δ MD (MRD-GT)")
    axs[1,5].imshow(d_md_2, cmap='bwr', vmin=md_lo, vmax=md_hi); axs[1,5].set_title("Δ MD (DTI-GT)")

    for i in range(2):
        for j in range(6):
            axs[i,j].axis('off')

    plt.tight_layout()
    out = f"visuals/fa_md_pub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_fa_md_publication.py mrd_model.pth mrd_dti_model.pth [idx]")
        exit()
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    visualize(sys.argv[1], sys.argv[2], idx)
