# visualize_compare.py

import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def run_model(model_path, x_noisy, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        noise_pred = model(x_noisy)
        x_denoised = x_noisy - noise_pred

    return x_denoised


def visualize_compare(model1_path, model2_path, sample_idx=0, channel=0):

    os.makedirs("results", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset...")
    dataset = DWIDataset2D(mode="test")
    print(f"Total test samples: {len(dataset)}")

    # ---- Get sample ----
    x_noisy, x_clean, _ = dataset[sample_idx]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # ---- Run both models ----
    print("\nRunning Model 1:", model1_path)
    out1 = run_model(model1_path, x_noisy, device)

    print("Running Model 2:", model2_path)
    out2 = run_model(model2_path, x_noisy, device)

    # ---- Select channel ----
    noisy = x_noisy[0, channel].cpu().numpy()
    clean = x_clean[0, channel].cpu().numpy()
    den1 = out1[0, channel].cpu().numpy()
    den2 = out2[0, channel].cpu().numpy()

    # ---- Errors ----
    err1 = np.abs(den1 - clean)
    err2 = np.abs(den2 - clean)

    # amplify error for visibility
    err1_vis = err1 * 10
    err2_vis = err2 * 10

    # difference between models
    diff_models = np.abs(den1 - den2) * 10

    # normalize for stable visualization
    vmax_img = clean.max()

    # ---- Plot ----
    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Row 1: Inputs
    axs[0, 0].imshow(noisy, cmap='gray', vmin=0, vmax=vmax_img)
    axs[0, 0].set_title("Noisy")

    axs[0, 1].imshow(clean, cmap='gray', vmin=0, vmax=vmax_img)
    axs[0, 1].set_title("Ground Truth")

    axs[0, 2].axis('off')
    axs[0, 3].axis('off')

    # Row 2: Model 1
    axs[1, 0].imshow(den1, cmap='gray', vmin=0, vmax=vmax_img)
    axs[1, 0].set_title("MRD Output")

    axs[1, 1].imshow(err1_vis, cmap='hot')
    axs[1, 1].set_title("MRD Error (×10)")

    axs[1, 2].imshow(err1, cmap='viridis')
    axs[1, 2].set_title("MRD Error (true scale)")

    axs[1, 3].axis('off')

    # Row 3: Model 2
    axs[2, 0].imshow(den2, cmap='gray', vmin=0, vmax=vmax_img)
    axs[2, 0].set_title("MRD + DTI Output")

    axs[2, 1].imshow(err2_vis, cmap='hot')
    axs[2, 1].set_title("MRD+DTI Error (×10)")

    axs[2, 2].imshow(err2, cmap='viridis')
    axs[2, 2].set_title("MRD+DTI Error (true scale)")

    axs[2, 3].imshow(diff_models, cmap='coolwarm')
    axs[2, 3].set_title("Difference Between Models (×10)")

    # cleanup
    for i in range(3):
        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout()

    # ---- Save ----
    save_path = f"results/compare_sample{sample_idx}_ch{channel}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n✅ Saved comparison to: {save_path}\n")

    # ---- Print stats ----
    print("----- ERROR STATS -----")
    print(f"MRD mean error: {err1.mean():.6f}")
    print(f"MRD+DTI mean error: {err2.mean():.6f}")
    print(f"Model difference mean: {diff_models.mean()/10:.6f}")
    print("------------------------\n")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("python visualize_compare.py <model1> <model2> [sample_idx] [channel]\n")
        print("Example:")
        print("python visualize_compare.py mrd_model.pth mrd_dti_model.pth 0 0\n")
        exit()

    model1 = sys.argv[1]
    model2 = sys.argv[2]

    sample_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    channel = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    visualize_compare(model1, model2, sample_idx, channel)
