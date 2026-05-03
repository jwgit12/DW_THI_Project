import torch
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def visualize(model_path, sample_idx=0, channel=0):

    os.makedirs("results", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print(f"\nLoading dataset...")
    dataset = DWIDataset2D(mode="test")

    print(f"Total test samples: {len(dataset)}")

    # -------------------------
    # GET SPECIFIC SAMPLE
    # -------------------------
    x_noisy, x_clean, _ = dataset[sample_idx]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # -------------------------
    # LOAD MODEL
    # -------------------------
    print(f"\nLoading model: {model_path}")

    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -------------------------
    # INFERENCE
    # -------------------------
    with torch.no_grad():
        noise_pred = model(x_noisy)
        x_denoised = x_noisy - noise_pred

    # -------------------------
    # SELECT CHANNEL
    # -------------------------
    noisy_img = x_noisy[0, channel].cpu().numpy()
    clean_img = x_clean[0, channel].cpu().numpy()
    denoised_img = x_denoised[0, channel].cpu().numpy()

    error_img = abs(denoised_img - clean_img)

    # -------------------------
    # PLOT
    # -------------------------
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(noisy_img, cmap='gray')
    axs[0].set_title("Noisy")

    axs[1].imshow(clean_img, cmap='gray')
    axs[1].set_title("Clean")

    axs[2].imshow(denoised_img, cmap='gray')
    axs[2].set_title("Denoised")

    axs[3].imshow(error_img, cmap='hot')
    axs[3].set_title("Error")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    model_name = os.path.basename(model_path).replace(".pth", "")
    save_path = f"results/{model_name}_sample{sample_idx}_ch{channel}.png"

    plt.savefig(save_path)
    plt.close()

    print(f"\n✅ Saved visualization to: {save_path}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python visualize.py <model_path> [sample_idx] [channel]\n")
        print("Example:")
        print("python visualize.py mrd_model.pth 0 0\n")
        exit()

    model_path = sys.argv[1]

    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    channel = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    visualize(model_path, sample_idx, channel)
