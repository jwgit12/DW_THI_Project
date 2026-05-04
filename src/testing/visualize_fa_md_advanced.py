import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# FA / MD from tensor6
# -------------------------
def tensor6_to_fa_md(tensor6):

    H, W = tensor6.shape[1:]

    fa = np.zeros((H, W))
    md = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            Dxx, Dxy, Dyy, Dxz, Dyz, Dzz = tensor6[:, i, j]

            M = np.array([
                [Dxx, Dxy, Dxz],
                [Dxy, Dyy, Dyz],
                [Dxz, Dyz, Dzz]
            ])

            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.clip(eigvals, 1e-6, None)

            md_ = np.mean(eigvals)
            md[i, j] = md_

            num = np.sqrt(((eigvals - md_)**2).sum())
            den = np.sqrt((eigvals**2).sum() + 1e-12)

            fa[i, j] = np.sqrt(1.5) * num / den

    return fa, md


# -------------------------
# LOAD MODEL
# -------------------------
def load_model(path, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# -------------------------
# MAIN
# -------------------------
def visualize(model1_path, model2_path, idx=0):

    os.makedirs("visuals", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = DWIDataset2D(mode="test")

    # ✅ FIX: unpack ALL values
    x_noisy, x_clean, tensor_gt, bvals, bvecs = dataset[idx]

    x_noisy_t = x_noisy.unsqueeze(0).to(device)

    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)

    with torch.no_grad():
        out1 = (x_noisy_t - model1(x_noisy_t))[0].cpu().numpy()
        out2 = (x_noisy_t - model2(x_noisy_t))[0].cpu().numpy()

    noisy = x_noisy.numpy()
    clean = x_clean.numpy()
    den1 = out1
    den2 = out2

    # -------------------------
    # USE GT TENSOR ONLY
    # -------------------------
    tensor_gt = tensor_gt.numpy()

    fa_gt, md_gt = tensor6_to_fa_md(tensor_gt)

    # -------------------------
    # NORMALIZATION
    # -------------------------
    def norm(x):
        vmin, vmax = np.percentile(x[x > 0], (1, 99)) if np.any(x > 0) else (0, 1)
        return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    fa_gt_n = norm(fa_gt)
    md_gt_n = norm(md_gt)

    # -------------------------
    # DWI COMPARISON (channel)
    # -------------------------
    ch = 0

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: DWI
    axs[0,0].imshow(noisy[ch], cmap='gray')
    axs[0,0].set_title("Noisy")

    axs[0,1].imshow(den1[ch], cmap='gray')
    axs[0,1].set_title("MRD")

    axs[0,2].imshow(den2[ch], cmap='gray')
    axs[0,2].set_title("MRD+DTI")

    axs[0,3].imshow(clean[ch], cmap='gray')
    axs[0,3].set_title("GT")

    # Row 2: TRUE STRUCTURE
    axs[1,0].imshow(fa_gt_n, cmap='viridis')
    axs[1,0].set_title("FA (GT)")

    axs[1,1].imshow(md_gt_n, cmap='magma')
    axs[1,1].set_title("MD (GT)")

    axs[1,2].axis('off')
    axs[1,3].axis('off')

    for i in range(2):
        for j in range(4):
            axs[i,j].axis('off')

    plt.tight_layout()

    save_path = f"visuals/simple_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("✅ Saved:", save_path)


if __name__ == "__main__":
    visualize(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 0)
