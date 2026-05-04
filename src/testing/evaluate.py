import torch
import numpy as np
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# DTI PROXY
# -------------------------
def compute_dti_proxy(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True)
    return torch.cat([mean, var], dim=1)


# -------------------------
# BUILD DESIGN MATRIX
# -------------------------
def build_B_matrix(bvals, bvecs):
    B = []
    for i in range(len(bvals)):
        g = bvecs[:, i]
        b = bvals[i]
        B.append([
            -b * g[0]**2,
            -b * g[1]**2,
            -b * g[2]**2,
            -2*b * g[0]*g[1],
            -2*b * g[0]*g[2],
            -2*b * g[1]*g[2],
        ])
    return np.array(B)


# -------------------------
# SLICE DTI
# -------------------------
def estimate_tensor_slice(dwi, bvals, bvecs):
    dwi = np.clip(dwi, 1e-6, None)

    mask = bvals > 50
    dwi = dwi[mask]
    bvals = bvals[mask]
    bvecs = bvecs[:, mask]

    B = build_B_matrix(bvals, bvecs)
    B_pinv = np.linalg.pinv(B)

    H, W = dwi.shape[1:]
    tensor = np.zeros((6, H, W))

    for i in range(H):
        for j in range(W):
            S = dwi[:, i, j]
            logS = np.log(S)
            D = B_pinv @ logS
            tensor[:, i, j] = D

    return tensor


# -------------------------
# FA / MD
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

            eigvals, _ = np.linalg.eigh(M)
            eigvals = np.sort(eigvals)[::-1]

            md_ = np.mean(eigvals)
            md[i, j] = md_

            numerator = np.sqrt(((eigvals - md_)**2).sum())
            denominator = np.sqrt((eigvals**2).sum() + 1e-12)

            fa[i, j] = np.sqrt(1.5) * numerator / denominator

    return fa, md


# -------------------------
# SAVE RESULTS
# -------------------------
def save_results(model_path, dwi, dti, fa, md):
    os.makedirs("results", exist_ok=True)

    model_name = os.path.basename(model_path).replace(".pth", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = f"results/{model_name}_{timestamp}.txt"

    with open(filepath, "w") as f:
        f.write("=====================================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=====================================\n\n")

        f.write(f"DWI MSE: {dwi:.6f}\n")
        f.write(f"DTI Proxy MSE: {dti:.6f}\n")
        f.write(f"FA MSE: {fa:.10f}\n")
        f.write(f"MD MSE: {md:.12f}\n")

    print(f"\n✅ Results saved to: {filepath}\n")


# -------------------------
# MAIN
# -------------------------
def evaluate(model_path):

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = DWIDataset2D(mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MRDDenoiser(in_channels=130).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dwi_losses = []
    dti_losses = []
    fa_losses = []
    md_losses = []

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (x_noisy, x_clean, tensor_gt, bvals, bvecs) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            noise = model(x_noisy)
            x_denoised = x_noisy - noise

            # ---- DWI ----
            loss_dwi = criterion(x_denoised, x_clean)
            dwi_losses.append(loss_dwi.item())

            # ---- DTI PROXY ----
            dti_pred = compute_dti_proxy(x_denoised)
            dti_gt = compute_dti_proxy(x_clean)
            loss_dti = criterion(dti_pred, dti_gt)
            dti_losses.append(loss_dti.item())

            # ---- REAL DTI ----
            dwi_np = x_denoised[0].cpu().numpy()

            tensor_pred = estimate_tensor_slice(
                dwi_np,
                bvals[0],
                bvecs[0]
            )

            fa_pred, md_pred = tensor6_to_fa_md(tensor_pred)

            tensor_gt_np = tensor_gt[0].numpy()
            fa_gt, md_gt = tensor6_to_fa_md(tensor_gt_np)

            fa_losses.append(np.mean((fa_pred - fa_gt)**2))
            md_losses.append(np.mean((md_pred - md_gt)**2))

            if batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}")

    dwi_mse = np.mean(dwi_losses)
    dti_mse = np.mean(dti_losses)
    fa_mse = np.mean(fa_losses)
    md_mse = np.mean(md_losses)

    print("\n===== RESULTS =====")
    print(f"DWI MSE: {dwi_mse:.6f}")
    print(f"DTI Proxy MSE: {dti_mse:.6f}")
    print(f"FA MSE: {fa_mse:.10f}")
    print(f"MD MSE: {md_mse:.12f}")
    print("===================\n")

    save_results(model_path, dwi_mse, dti_mse, fa_mse, md_mse)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_path>")
        exit()

    evaluate(sys.argv[1])
