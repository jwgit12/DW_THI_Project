# train_mrd_dti.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# FAST TENSOR PROXY
# -------------------------
def compute_tensor_proxy(x):
    """
    x: (B, C, H, W)

    Returns:
        tensor-like (B, 6, H, W)
    """

    # normalize per voxel
    x_norm = x / (x.mean(dim=1, keepdim=True) + 1e-6)

    mean = torch.mean(x_norm, dim=1, keepdim=True)
    var = torch.var(x_norm, dim=1, keepdim=True)

    tensor = torch.cat([
        mean,
        var,
        mean * var,
        mean ** 2,
        var ** 2,
        torch.sqrt(var + 1e-6)
    ], dim=1)

    return tensor


# -------------------------
# CHANNEL-WISE NORMALIZATION (CRITICAL)
# -------------------------
def normalize_tensor(t):
    """
    Normalize each channel independently
    t: (B, 6, H, W)
    """
    mean = t.mean(dim=(0, 2, 3), keepdim=True)
    std  = t.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return (t - mean) / std


# -------------------------
# COLLATE
# -------------------------
def custom_collate(batch):
    noisy = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    tensor = torch.stack([item[2] for item in batch])

    bvals = [item[3] for item in batch]
    bvecs = [item[4] for item in batch]

    return noisy, clean, tensor, bvals, bvecs


# -------------------------
# TRAIN
# -------------------------
def train():

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    dataset = DWIDataset2D(mode="train")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate
    )

    model = MRDDenoiser(in_channels=130).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 🔥 IMPORTANT CHANGE
    LAMBDA = 1.0

    print("\nStarting MRD + TENSOR training...\n")

    for epoch in range(3):

        total_loss = 0

        for batch_idx, (x_noisy, x_clean, tensor_gt, _, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            tensor_gt = tensor_gt.to(device)

            # -------------------------
            # FORWARD
            # -------------------------
            noise_pred = model(x_noisy)
            x_denoised = x_noisy - noise_pred

            # -------------------------
            # DWI LOSS
            # -------------------------
            loss_dwi = criterion(x_denoised, x_clean)

            # -------------------------
            # TENSOR LOSS (FIXED)
            # -------------------------
            tensor_pred = compute_tensor_proxy(x_denoised)

            tensor_gt_norm   = normalize_tensor(tensor_gt)
            tensor_pred_norm = normalize_tensor(tensor_pred)

            loss_tensor = criterion(tensor_pred_norm, tensor_gt_norm)

            # -------------------------
            # TOTAL LOSS
            # -------------------------
            loss = loss_dwi + LAMBDA * loss_tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"[MRD+TENSOR] Epoch {epoch+1} Batch {batch_idx} "
                    f"Loss: {loss.item():.6f} "
                    f"(DWI={loss_dwi.item():.6f}, TENSOR={loss_tensor.item():.6f})"
                )

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(loader):.6f}\n")

    torch.save(model.state_dict(), "mrd_tensor_model.pth")
    print("Saved: mrd_tensor_model.pth")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    train()
