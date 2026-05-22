# mrd_dti_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def compute_tensor_proxy(x):
    """
    Lightweight proxy for tensor supervision
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True)
    return torch.cat([mean, var], dim=1)


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

    LAMBDA = 0.1

    print("\nStarting MRD + TENSOR (FINAL) training...\n")

    for epoch in range(3):

        total_loss = 0

        for batch_idx, (x_noisy, x_clean, tensor_gt, _, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            tensor_gt = tensor_gt.to(device)

            noise_pred = model(x_noisy)
            x_denoised = x_noisy - noise_pred

            # -------------------------
            # DWI LOSS
            # -------------------------
            loss_dwi = criterion(x_denoised, x_clean)

            # -------------------------
            # TENSOR PROXY LOSS
            # -------------------------
            proxy_pred = compute_tensor_proxy(x_denoised)
            proxy_gt = compute_tensor_proxy(x_clean)

            loss_tensor = criterion(proxy_pred, proxy_gt)

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


def custom_collate(batch):
    noisy = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    tensor = torch.stack([item[2] for item in batch])

    bvals = [item[3] for item in batch]
    bvecs = [item[4] for item in batch]

    return noisy, clean, tensor, bvals, bvecs


if __name__ == "__main__":
    train()
