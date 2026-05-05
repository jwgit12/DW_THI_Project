import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(path, device):
    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ============================================================
# RUN MODEL
# ============================================================
def run_model(model, x):
    with torch.no_grad():
        noise = model(x)
        out = x - noise
    return out


# ============================================================
# NORMALIZATION (robust)
# ============================================================
def norm(x):
    vmin, vmax = np.percentile(x, (1, 99))
    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)


# ============================================================
# MAIN VISUALIZATION
# ============================================================
def visualize(mrd_path, tensor_path, ssm_path, sample_idx=0, channel=0):

    os.makedirs("visuals", exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset...")
    dataset = DWIDataset2D(mode="test")
    print(f"Total samples: {len(dataset)}")

    # -------------------------
    # LOAD SAMPLE
    # -------------------------
    x_noisy, x_clean, *_ = dataset[sample_idx]

    x_noisy = x_noisy.unsqueeze(0).to(device)
    x_clean = x_clean.unsqueeze(0).to(device)

    # -------------------------
    # LOAD MODELS
    # -------------------------
    print("Loading models...")
    mrd = load_model(mrd_path, device)
    tensor = load_model(tensor_path, device)
    ssm = load_model(ssm_path, device)

    # -------------------------
    # RUN
    # -------------------------
    print("Running inference...")
    out_mrd = run_model(mrd, x_noisy)
    out_tensor = run_model(tensor, x_noisy)
    out_ssm = run_model(ssm, x_noisy)

    # -------------------------
    # SELECT CHANNEL
    # -------------------------
    noisy = x_noisy[0, channel].cpu().numpy()
    clean = x_clean[0, channel].cpu().numpy()

    mrd_img = out_mrd[0, channel].cpu().numpy()
    tensor_img = out_tensor[0, channel].cpu().numpy()
    ssm_img = out_ssm[0, channel].cpu().numpy()

    # -------------------------
    # ERRORS
    # -------------------------
    err_mrd = np.abs(mrd_img - clean)
    err_tensor = np.abs(tensor_img - clean)
    err_ssm = np.abs(ssm_img - clean)

    # consistent error scale
    err_max = max(err_mrd.max(), err_tensor.max(), err_ssm.max())

    # normalize base images
    noisy_n = norm(noisy)
    clean_n = norm(clean)
    mrd_n = norm(mrd_img)
    tensor_n = norm(tensor_img)
    ssm_n = norm(ssm_img)

    # -------------------------
    # PLOT
    # -------------------------
    fig, axs = plt.subplots(2, 5, figsize=(18, 8))

    # ---- Row 1: images ----
    axs[0, 0].imshow(noisy_n, cmap='gray')
    axs[0, 0].set_title("Noisy")

    axs[0, 1].imshow(clean_n, cmap='gray')
    axs[0, 1].set_title("Ground Truth")

    axs[0, 2].imshow(mrd_n, cmap='gray')
    axs[0, 2].set_title("MRD")

    axs[0, 3].imshow(tensor_n, cmap='gray')
    axs[0, 3].set_title("MRD + Tensor")

    axs[0, 4].imshow(ssm_n, cmap='gray')
    axs[0, 4].set_title("MRD + SSM")

    # ---- Row 2: errors ----
    axs[1, 0].axis('off')

    axs[1, 1].axis('off')

    axs[1, 2].imshow(err_mrd, cmap='coolwarm', vmin=0, vmax=err_max)
    axs[1, 2].set_title("MRD Error")

    axs[1, 3].imshow(err_tensor, cmap='coolwarm', vmin=0, vmax=err_max)
    axs[1, 3].set_title("Tensor Error")

    axs[1, 4].imshow(err_ssm, cmap='coolwarm', vmin=0, vmax=err_max)
    axs[1, 4].set_title("SSM Error")

    # cleanup
    for i in range(2):
        for j in range(5):
            axs[i, j].axis('off')

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    save_path = f"visuals/dwi_compare_sample{sample_idx}_ch{channel}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n✅ Saved: {save_path}")

    # -------------------------
    # PRINT METRICS (quick)
    # -------------------------
    print("\n--- ERROR SUMMARY ---")
    print(f"MRD mean error: {err_mrd.mean():.6f}")
    print(f"Tensor mean error: {err_tensor.mean():.6f}")
    print(f"SSM mean error: {err_ssm.mean():.6f}")
    print("----------------------\n")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("\nUsage:")
        print("python visualize_all_models_dwi.py mrd.pth tensor.pth ssm.pth [sample_idx] [channel]\n")
        exit()

    mrd_path = sys.argv[1]
    tensor_path = sys.argv[2]
    ssm_path = sys.argv[3]

    sample_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    channel = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    visualize(mrd_path, tensor_path, ssm_path, sample_idx, channel)
