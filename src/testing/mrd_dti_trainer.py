# train_mrd_dti.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


def compute_dti_proxy(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True)
    return torch.cat([mean, var], dim=1)


def train():

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    dataset = DWIDataset2D(mode="train")

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = MRDDenoiser(in_channels=130).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    LAMBDA = 0.1

    print("\nStarting MRD + DTI training...\n")

    for epoch in range(3):

        total_loss = 0

        for batch_idx, (x_noisy, x_clean, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            noise_pred = model(x_noisy)
            x_denoised = x_noisy - noise_pred

            loss_dwi = criterion(x_denoised, x_clean)

            dti_pred = compute_dti_proxy(x_denoised)
            dti_gt = compute_dti_proxy(x_clean)

            loss_dti = criterion(dti_pred, dti_gt)

            loss = loss_dwi + LAMBDA * loss_dti

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"[MRD+DTI] Epoch {epoch+1} Batch {batch_idx} "
                    f"Loss: {loss.item():.6f} "
                    f"(DWI={loss_dwi.item():.6f}, DTI={loss_dti.item():.6f})"
                )

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(loader):.6f}\n")

    torch.save(model.state_dict(), "mrd_dti_model.pth")
    print("Saved: mrd_dti_model.pth")


if __name__ == "__main__":
    train()
