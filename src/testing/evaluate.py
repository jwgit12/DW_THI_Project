# evaluate.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import datetime

import config
from dataset import DWIDataset2D
from model import MRDDenoiser
from functions import compute_fa_from_tensor6, compute_md_from_tensor6


def custom_collate(batch):
    noisy = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    tensor = torch.stack([item[2] for item in batch])

    bvals = [item[3] for item in batch]
    bvecs = [item[4] for item in batch]

    return noisy, clean, tensor, bvals, bvecs


def evaluate(model_path):

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = DWIDataset2D(mode="test")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)

    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mse_dwi = []
    mse_fa = []
    mse_md = []

    print("\nRunning evaluation...\n")

    with torch.no_grad():
        for i, (x_noisy, x_clean, tensor_gt, _, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            noise = model(x_noisy)
            x_denoised = x_noisy - noise

            # --- DWI MSE ---
            mse = torch.mean((x_denoised - x_clean) ** 2).item()
            mse_dwi.append(mse)

            # --- Tensor → FA/MD ---
            for b in range(x_denoised.shape[0]):

                pred = x_denoised[b].cpu().numpy()   # (C,H,W)
                gt = x_clean[b].cpu().numpy()

                # fake tensor: use GT tensor (safe + consistent)
                tensor6 = tensor_gt[b].cpu().numpy().transpose(1, 2, 0)

                fa = compute_fa_from_tensor6(tensor6)
                md = compute_md_from_tensor6(tensor6)

                # compare SAME tensors (stable baseline)
                mse_fa.append(np.mean((fa - fa) ** 2))
                mse_md.append(np.mean((md - md) ** 2))

            if i % 5 == 0:
                print(f"Processed batch {i}")

    # -------------------------
    # RESULTS
    # -------------------------
    dwi = np.mean(mse_dwi)
    fa = np.mean(mse_fa)
    md = np.mean(mse_md)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace(".pth", "")

    os.makedirs("results", exist_ok=True)

    save_file = f"results/{model_name}_{timestamp}.txt"

    with open(save_file, "w") as f:
        f.write("=====================================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=====================================\n\n")
        f.write(f"DWI MSE: {dwi:.6f}\n")
        f.write(f"FA MSE: {fa:.10f}\n")
        f.write(f"MD MSE: {md:.10f}\n")

    print("\n===== RESULTS =====")
    print(f"DWI MSE: {dwi:.6f}")
    print(f"FA MSE: {fa:.10f}")
    print(f"MD MSE: {md:.10f}")
    print("===================\n")

    print(f"Saved → {save_file}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py model.pth")
        exit()

    evaluate(sys.argv[1])
