import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# MODEL RUNNER
# -------------------------
def run_model(model_path, x, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        noise = model(x)
        return x - noise


# -------------------------
# NORMALIZATION
# -------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# -------------------------
# MAIN VIS
# -------------------------
def visualize(model1_path, model2_path, idx=0, channel=0):

    os.makedirs("visuals", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    dataset = DWIDataset2D(mode="test")
    print(f"Total samples: {len(dataset)}")

    # ---- Load sample safely ----
    sample = dataset[idx]
    x_noisy = sample[0]
    x_clean = sample[1]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # ---- Run models ----
    print("Running models...")
    out1 = run_model(model1_path, x_noisy, device)
    out2 = run_model(model2_path, x_noisy, device)

    # ---- Select channel ----
    noisy = x_noisy[0, channel].cpu().numpy()
    clean = x_clean[0, channel].cpu().numpy()
    den1 = out1[0, channel].cpu().numpy()
    den2 = out2[0, channel].cpu().numpy()

    # ---- Errors ----
    err1 = den1 - clean
    err2 = den2 - clean

    err1_abs = np.abs(err1)
    err2_abs = np.abs(err2)

    # ---- Normalize for display ----
    clean_norm = normalize(clean)
    err1_vis = err1_abs / (err1_abs.max() + 1e-8)
    err2_vis = err2_abs / (err2_abs.max() + 1e-8)

    # ---- Difference between models ----
    model_diff = den1 - den2
    diff_vis = model_diff / (np.max(np.abs(model_diff)) + 1e-8)

    # -------------------------
    # STATS (IMPORTANT FOR PRESENTATION)
    # -------------------------
    print("\n--- ERROR STATS ---")
    print(f"MRD mean abs error:      {err1_abs.mean():.6f}")
    print(f"MRD+Tensor mean abs error: {err2_abs.mean():.6f}")
    print(f"Model difference mean:   {np.abs(model_diff).mean():.6f}")
    print("--------------------\n")

    # -------------------------
    # PLOT
    # -------------------------
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Inputs
    axs[0, 0].imshow(noisy, cmap='gray')
    axs[0, 0].set_title("Noisy")

    axs[0, 1].imshow(clean, cmap='gray')
    axs[0, 1].set_title("Ground Truth")

    axs[0, 2].imshow(den1, cmap='gray')
    axs[0, 2].set_title("MRD")

    axs[0, 3].imshow(den2, cmap='gray')
    axs[0, 3].set_title("MRD+Tensor")

    # Row 2: Error maps
    axs[1, 0].imshow(err1_vis, cmap='hot')
    axs[1, 0].set_title("MRD Error")

    axs[1, 1].imshow(err2_vis, cmap='hot')
    axs[1, 1].set_title("MRD+Tensor Error")

    axs[1, 2].imshow(diff_vis, cmap='bwr', vmin=-1, vmax=1)
    axs[1, 2].set_title("Model Difference")

    axs[1, 3].axis('off')

    # Row 3: Overlay (KEY FOR INTERPRETATION)
    axs[2, 0].imshow(clean_norm, cmap='gray')
    axs[2, 0].imshow(err1_vis, cmap='Reds', alpha=0.5)
    axs[2, 0].set_title("MRD Overlay")

    axs[2, 1].imshow(clean_norm, cmap='gray')
    axs[2, 1].imshow(err2_vis, cmap='Reds', alpha=0.5)
    axs[2, 1].set_title("MRD+Tensor Overlay")

    axs[2, 2].imshow(diff_vis, cmap='bwr', vmin=-1, vmax=1)
    axs[2, 2].set_title("Signed Difference")

    axs[2, 3].axis('off')

    # cleanup
    for i in range(3):
        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"visuals/compare_{timestamp}_idx{idx}_ch{channel}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}\n")


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("python visualize_compare_clean.py <mrd> <tensor> [idx] [channel]\n")
        exit()

    model1 = sys.argv[1]
    model2 = sys.argv[2]

    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ch = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    visualize(model1, model2, idx, ch)
