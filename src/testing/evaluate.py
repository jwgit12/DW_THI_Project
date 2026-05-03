import torch
import numpy as np
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def compute_dti_proxy(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True)
    return torch.cat([mean, var], dim=1)


def get_result_filepath(model_path):
    os.makedirs("results", exist_ok=True)

    # Extract model name (remove folder + extension)
    model_name = os.path.basename(model_path).replace(".pth", "")

    return os.path.join("results", f"{model_name}_metrics.txt")


def save_results(model_path, dwi_mse, dti_mse):
    filepath = get_result_filepath(model_path)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a") as f:
        f.write("\n----------------------------------------\n")
        f.write(f"Time: {now}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"DWI MSE: {dwi_mse:.6f}\n")
        f.write(f"DTI MSE: {dti_mse:.6f}\n")
        f.write("----------------------------------------\n")

    print(f"\n✅ Results saved to {filepath}\n")


def evaluate(model_path):

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print(f"Evaluating model: {model_path}")
    print("==============================\n")

    # -------------------------
    # DATASET
    # -------------------------
    print("Loading test dataset...")
    dataset = DWIDataset2D(mode="test")

    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print(f"Test samples: {len(dataset)}\n")

    # -------------------------
    # MODEL
    # -------------------------
    print("Loading model...")

    model = MRDDenoiser(in_channels=130).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Failed to load model: {model_path}")
        print(e)
        return

    model.eval()

    # -------------------------
    # METRICS
    # -------------------------
    dwi_losses = []
    dti_losses = []

    criterion = torch.nn.MSELoss()

    print("Running inference...\n")

    # -------------------------
    # INFERENCE LOOP
    # -------------------------
    with torch.no_grad():
        for batch_idx, (x_noisy, x_clean, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            noise_pred = model(x_noisy)
            x_denoised = x_noisy - noise_pred

            # ---- DWI LOSS ----
            loss_dwi = criterion(x_denoised, x_clean)
            dwi_losses.append(loss_dwi.item())

            # ---- DTI LOSS ----
            dti_pred = compute_dti_proxy(x_denoised)
            dti_gt = compute_dti_proxy(x_clean)

            loss_dti = criterion(dti_pred, dti_gt)
            dti_losses.append(loss_dti.item())

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

    # -------------------------
    # FINAL RESULTS
    # -------------------------
    dwi_mse = np.mean(dwi_losses)
    dti_mse = np.mean(dti_losses)

    print("\n===== RESULTS =====")
    print(f"Model: {model_path}")
    print(f"DWI MSE: {dwi_mse:.6f}")
    print(f"DTI MSE: {dti_mse:.6f}")
    print("===================\n")

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    save_results(model_path, dwi_mse, dti_mse)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python evaluate.py <model_path>\n")
        print("Example:")
        print("python evaluate.py mrd_model.pth\n")
        exit()

    model_path = sys.argv[1]
    evaluate(model_path)
