import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def load_model(model_path, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_model(model, x_noisy):
    with torch.no_grad():
        noise_pred = model(x_noisy)
        x_denoised = x_noisy - noise_pred
    return x_denoised


def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def visualize(model1_path, model2_path, sample_idx=0, channel=0):

    os.makedirs("results", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # LOAD DATA
    # -------------------------
    print("\nLoading dataset...")
    dataset = DWIDataset2D(mode="test")
    print(f"Total test samples: {len(dataset)}")

    x_noisy, x_clean, _ = dataset[sample_idx]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # -------------------------
    # LOAD MODELS
    # -------------------------
    print("\nLoading models...")
    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)

    # -------------------------
    # RUN INFERENCE
    # -------------------------
    print("Running inference...")
    out1 = run_model(model1, x_noisy)
    out2 = run_model(model2, x_noisy)

    # -------------------------
    # SELECT CHANNEL
    # -------------------------
    noisy = x_noisy[0, channel].cpu().numpy()
    clean = x_clean[0, channel].cpu().numpy()
    den1 = out1[0, channel].cpu().numpy()
    den2 = out2[0, channel].cpu().numpy()

    clean_norm = normalize(clean)

    # -------------------------
    # COMPUTE ERRORS
    # -------------------------
    err1 = den1 - clean
    err2 = den2 - clean
    diff_models = den1 - den2

    err1_abs = np.abs(err1)
    err2_abs = np.abs(err2)

    # normalize for visualization
    err1_vis = err1_abs / (err1_abs.max() + 1e-8)
    err2_vis = err2_abs / (err2_abs.max() + 1e-8)

    diff_vis = np.abs(diff_models) / (np.abs(diff_models).max() + 1e-8)

    signed_err = err2 / (np.max(np.abs(err2)) + 1e-8)

    # -------------------------
    # PRINT STATS
    # -------------------------
    print("\n--- ERROR STATS ---")
    print(f"MRD mean abs error:      {err1_abs.mean():.6f}")
    print(f"MRD+DTI mean abs error:  {err2_abs.mean():.6f}")
    print(f"Model difference mean:   {np.abs(diff_models).mean():.6f}")
    print("--------------------\n")

    # -------------------------
    # PLOT
    # -------------------------
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: Base images
    axs[0, 0].imshow(noisy, cmap='gray')
    axs[0, 0].set_title("Noisy")

    axs[0, 1].imshow(clean, cmap='gray')
    axs[0, 1].set_title("Ground Truth")

    axs[0, 2].imshow(den2, cmap='gray')
    axs[0, 2].set_title("MRD+DTI Output")

    # Row 2: MRD
    axs[1, 0].imshow(den1, cmap='gray')
    axs[1, 0].set_title("MRD Output")

    axs[1, 1].imshow(clean_norm, cmap='gray')
    axs[1, 1].imshow(err1_vis, cmap='Reds', alpha=0.6)
    axs[1, 1].set_title("MRD Error Overlay")

    axs[1, 2].imshow(err1_vis, cmap='hot')
    axs[1, 2].set_title("MRD Error Map")

    # Row 3: MRD+DTI
    axs[2, 0].imshow(den2, cmap='gray')
    axs[2, 0].set_title("MRD+DTI Output")

    axs[2, 1].imshow(clean_norm, cmap='gray')
    axs[2, 1].imshow(err2_vis, cmap='Reds', alpha=0.6)
    axs[2, 1].set_title("MRD+DTI Error Overlay")

    axs[2, 2].imshow(signed_err, cmap='bwr', vmin=-1, vmax=1)
    axs[2, 2].set_title("Signed Error (Blue=Under, Red=Over)")

    # cleanup axes
    for i in range(3):
        for j in range(3):
            axs[i, j].axis('off')

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    save_path = f"results/overlay_sample{sample_idx}_ch{channel}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved visualization to: {save_path}\n")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("python visualize_overlay.py <model1> <model2> [sample_idx] [channel]\n")
        print("Example:")
        print("python visualize_overlay.py mrd_model.pth mrd_dti_model.pth 10 20\n")
        exit()

    model1 = sys.argv[1]
    model2 = sys.argv[2]

    sample_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    channel = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    visualize(model1, model2, sample_idx, channel)
