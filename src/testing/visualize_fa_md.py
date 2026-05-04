import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

import config
from dataset import DWIDataset2D
from model import MRDDenoiser
from functions import compute_fa_from_tensor6, compute_md_from_tensor6


def load_model(model_path, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_model(model, x):
    with torch.no_grad():
        noise = model(x)
        return x - noise


def tensor_to_fa_md(tensor):
    t = np.transpose(tensor, (1, 2, 0))  # (H, W, 6)
    fa = compute_fa_from_tensor6(t)
    md = compute_md_from_tensor6(t)
    return fa, md


def visualize(model1_path, model2_path, sample_idx=0):

    os.makedirs("visuals", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = DWIDataset2D(mode="test")
    x_noisy, x_clean, tensor_gt = dataset[sample_idx]

    x_noisy = x_noisy.unsqueeze(0).to(device)

    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)

    out1 = run_model(model1, x_noisy)[0].cpu().numpy()
    out2 = run_model(model2, x_noisy)[0].cpu().numpy()

    tensor_gt = tensor_gt.numpy()

    # ---- FA / MD ----
    fa_gt, md_gt = tensor_to_fa_md(tensor_gt)

    # baseline (same tensor for now)
    fa_1, md_1 = fa_gt, md_gt
    fa_2, md_2 = fa_gt, md_gt

    # ---- Errors ----
    diff_fa_1 = np.abs(fa_1 - fa_gt)
    diff_fa_2 = np.abs(fa_2 - fa_gt)

    diff_md_1 = np.abs(md_1 - md_gt)
    diff_md_2 = np.abs(md_2 - md_gt)

    # ---- Plot ----
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # FA
    axs[0, 0].imshow(fa_gt, cmap='gray')
    axs[0, 0].set_title("FA GT")

    axs[0, 1].imshow(fa_1, cmap='gray')
    axs[0, 1].set_title("MRD")

    axs[0, 2].imshow(diff_fa_1, cmap='hot')
    axs[0, 2].set_title("MRD Error")

    axs[0, 3].imshow(diff_fa_2, cmap='hot')
    axs[0, 3].set_title("MRD+DTI Error")

    # MD
    axs[1, 0].imshow(md_gt, cmap='gray')
    axs[1, 0].set_title("MD GT")

    axs[1, 1].imshow(md_1, cmap='gray')
    axs[1, 1].set_title("MRD")

    axs[1, 2].imshow(diff_md_1, cmap='hot')
    axs[1, 2].set_title("MRD Error")

    axs[1, 3].imshow(diff_md_2, cmap='hot')
    axs[1, 3].set_title("MRD+DTI Error")

    for i in range(2):
        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"visuals/fa_md_compare_{timestamp}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved visualization to {save_path}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage:")
        print("python visualize_fa_md.py <model1> <model2> [sample_idx]")
        exit()

    m1 = sys.argv[1]
    m2 = sys.argv[2]
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    visualize(m1, m2, idx)
