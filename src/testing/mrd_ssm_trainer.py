# train_mrd_ssm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# ============================================================
# COLLATE (matches your current dataset structure)
# ============================================================
def custom_collate(batch):
    noisy = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    tensor = torch.stack([item[2] for item in batch])

    bvals = [item[3] for item in batch] if len(batch[0]) > 3 else None
    bvecs = [item[4] for item in batch] if len(batch[0]) > 4 else None

    return noisy, clean, tensor, bvals, bvecs


# ============================================================
# CHANNEL MASK (SSM)
# ============================================================
def create_channel_mask(x, keep_fraction):
    """
    x: (B, C, H, W)

    Returns:
        mask: same shape
    """
    B, C, H, W = x.shape

    # mask per channel (broadcast spatially)
    mask = (torch.rand(B, C, 1, 1, device=x.device) < keep_fraction).float()

    return mask


# ============================================================
# TRAIN
# ============================================================
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

    print("\nStarting MRD + SSM (corrected) training...\n")

    for epoch in range(5):

        total_loss = 0

        for batch_idx, (x_noisy, x_clean, _, _, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            # --------------------------------------------------
            # CREATE MASK
            # --------------------------------------------------
            mask = create_channel_mask(x_noisy, config.KEEP_FRACTION)

            # masked input
            x_masked = x_noisy * mask

            # --------------------------------------------------
            # FORWARD
            # --------------------------------------------------
            noise_pred = model(x_masked)
            x_denoised = x_masked - noise_pred

            # --------------------------------------------------
            # SSM LOSS (only masked regions)
            # --------------------------------------------------
            loss_ssm = ((x_denoised - x_clean) * (1 - mask)).pow(2).mean()

            # --------------------------------------------------
            # STANDARD DENOISING LOSS (full image)
            # --------------------------------------------------
            loss_denoise = criterion(x_denoised, x_clean)

            # --------------------------------------------------
            # FINAL LOSS (IMPORTANT)
            # --------------------------------------------------
            loss = loss_ssm + 0.1 * loss_denoise

            # --------------------------------------------------
            # BACKPROP
            # --------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"[SSM] Epoch {epoch+1} Batch {batch_idx} "
                    f"Loss: {loss.item():.6f} "
                    f"(SSM={loss_ssm.item():.6f}, DENOISE={loss_denoise.item():.6f})"
                )

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(loader):.6f}\n")

    torch.save(model.state_dict(), "mrd_ssm_model.pth")
    print("Saved: mrd_ssm_model.pth")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    train()
