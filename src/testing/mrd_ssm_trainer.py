# train_mrd_ssm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DWIDataset2D
from model import MRDDenoiser


# -------------------------
# CUSTOM COLLATE (UNCHANGED STYLE)
# -------------------------
def custom_collate(batch):
    noisy = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    tensor = torch.stack([item[2] for item in batch])

    # optional (not used here but keeps compatibility)
    bvals = [item[3] for item in batch] if len(batch[0]) > 3 else None
    bvecs = [item[4] for item in batch] if len(batch[0]) > 4 else None

    return noisy, clean, tensor, bvals, bvecs


# -------------------------
# SSM MASKING
# -------------------------
def apply_mask(x, keep_fraction):
    """
    Random mask over full tensor (channel + spatial)
    """
    mask = (torch.rand_like(x) < keep_fraction).float()
    x_masked = x * mask
    return x_masked, mask


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
    criterion = nn.MSELoss(reduction='none')  # IMPORTANT

    print("\nStarting MRD + SSM training...\n")

    for epoch in range(3):

        total_loss = 0

        for batch_idx, (x_noisy, x_clean, _, _, _) in enumerate(loader):

            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            # -------------------------
            # APPLY MASK
            # -------------------------
            x_masked, mask = apply_mask(x_noisy, config.KEEP_FRACTION)

            # -------------------------
            # FORWARD
            # -------------------------
            noise_pred = model(x_masked)
            x_recon = x_masked - noise_pred

            # -------------------------
            # LOSS ONLY ON MASKED REGIONS
            # -------------------------
            loss_map = criterion(x_recon, x_clean)

            masked_region = (1 - mask)  # where data was removed
            loss = (loss_map * masked_region).sum() / (masked_region.sum() + 1e-8)

            # -------------------------
            # OPTIMIZE
            # -------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"[SSM] Epoch {epoch+1} Batch {batch_idx} "
                    f"Loss: {loss.item():.6f}"
                )

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(loader):.6f}\n")

    torch.save(model.state_dict(), "mrd_ssm_model.pth")
    print("Saved: mrd_ssm_model.pth")


if __name__ == "__main__":
    train()
