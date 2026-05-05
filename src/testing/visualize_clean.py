# visualize_clean.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# MODEL LOADING
# -------------------------
def load_model(path, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def run_model(model, x):
    with torch.no_grad():
        noise = model(x)
        return x - noise


# -------------------------
# NORMALIZATION
# -------------------------
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


# -------------------------
# VISUALIZER
# -------------------------
def visualize(model1_path, model2_path, sample_idx=0, channel=0):

    os.makedirs("./visuals", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset...")
    dataset = DWIDataset2D(mode="test")
    print(f"Total samples: {len(dataset)}")

    # robust unpack (your dataset returns 5 values now)
    sample = dataset[sample_idx]
    x_noisy = sample[0]
    x_clean = sample[1]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # -------------------------
    # RUN MODELS
    # -------------------------
    print("Running models...")

    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)

    out1 = run_model(model1, x_noisy)
    out2 = run_model(model2, x_noisy)

    # -------------------------
    # SELECT CHANNEL
    # -------------------------
    noisy = x_noisy[0, channel].cpu().numpy()
    clean = x_clean[0, channel].cpu().numpy()
    den1 = out1[0, channel].cpu().numpy()
    den2 = out2[0, channel].cpu().numpy()

    # normalize for visualization
    noisy_n = normalize(noisy)
    clean_n = normalize(clean)
    den1_n = normalize(den1)
    den2_n = normalize(den2)

    # -------------------------
    # ERRORS
    # -------------------------
    err1 = den1 - clean
    err2 = den2 - clean
    diff_models = den2 - den1

    err1_abs = np.abs(err1)
    err2_abs = np.abs(err2)

    # normalize error maps for visibility
    err1_vis = err1_abs / (err1_abs.max() + 1e-8)
    err2_vis = err2_abs / (err2_abs.max() + 1e-8)

    signed_diff = diff_models / (np.max(np.abs(diff_models)) + 1e-8)

    # -------------------------
    # PRINT STATS
    # -------------------------
    print("\n--- ERROR STATS ---")
    print(f"MRD mean abs error:      {err1_abs.mean():.6f}")
    print(f"MRD+Tensor mean abs error: {err2_abs.mean():.6f}")
    print(f"Model difference mean:   {np.abs(diff_models).mean():.6f}")
    print("--------------------\n")

    # -------------------------
    # PLOT
    # -------------------------
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))

    # ===== ROW 1: STRUCTURE =====
    axs[0, 0].imshow(noisy_n, cmap="viridis")
    axs[0, 0].set_title("Noisy")

    axs[0, 1].imshow(den1_n, cmap="viridis")
    axs[0, 1].set_title("MRD")

    axs[0, 2].imshow(den2_n, cmap="viridis")
    axs[0, 2].set_title("MRD+Tensor")

    axs[0, 3].imshow(clean_n, cmap="viridis")
    axs[0, 3].set_title("Ground Truth")

    # ===== ROW 2: ERRORS =====
    axs[1, 0].imshow(err1_vis, cmap="magma")
    axs[1, 0].set_title("MRD Error")

    axs[1, 1].imshow(err2_vis, cmap="magma")
    axs[1, 1].set_title("MRD+Tensor Error")

    axs[1, 2].imshow(signed_diff, cmap="coolwarm", vmin=-1, vmax=1)
    axs[1, 2].set_title("Model Difference")

    # overlay (best for presentations)
    axs[1, 3].imshow(clean_n, cmap="gray")
    axs[1, 3].imshow(err2_vis, cmap="inferno", alpha=0.5)
    axs[1, 3].set_title("Error Overlay")

    # cleanup
    for i in range(2):
        for j in range(4):
            axs[i, j].axis("off")

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    save_path = f"./visuals/clean_vis_sample{sample_idx}_ch{channel}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved to: {save_path}\n")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("python visualize_clean.py mrd_model.pth mrd_tensor_model.pth [sample_idx] [channel]\n")
        exit()

    model1 = sys.argv[1]
    model2 = sys.argv[2]

    sample_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    channel = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    visualize(model1, model2, sample_idx, channel)
