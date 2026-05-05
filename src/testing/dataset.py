# dataset.py

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


# -------------------------
# DTI COMPUTATION (FIXED)
# -------------------------
def compute_dti_numpy(data, bvals, bvecs):
    """
    data: (H, W, N)
    bvals: (M,)
    bvecs: (M, 3)
    """

    # -----------------------------
    # CRITICAL FIX: ALIGN SHAPES
    # -----------------------------
    n_channels = data.shape[-1]

    if len(bvals) != n_channels:
        print(f"[WARNING] Mismatch detected: data={n_channels}, bvals={len(bvals)}")
        min_n = min(len(bvals), n_channels)

        bvals = bvals[:min_n]
        bvecs = bvecs[:min_n]
        data = data[..., :min_n]

    # -----------------------------
    # Remove b0 images (bvals ~ 0)
    # -----------------------------
    mask = bvals > 50   # safer than >0
    if mask.sum() < 6:
        # not enough directions → return zeros
        H, W, _ = data.shape
        return np.zeros((H, W, 6), dtype=np.float32)

    data = data[..., mask]
    # Ensure correct shape
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T   # (N,3)

    bvecs = bvecs[mask]
    bvals = bvals[mask]

    # -----------------------------
    # Build design matrix
    # -----------------------------
    gx, gy, gz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]

    X = np.stack([
        gx * gx,
        gy * gy,
        gz * gz,
        2 * gx * gy,
        2 * gx * gz,
        2 * gy * gz
    ], axis=1)  # (N,6)

    # -----------------------------
    # Log signal
    # -----------------------------
    eps = 1e-6
    S = np.log(data + eps)

    H, W, N = S.shape
    S = S.reshape(-1, N)

    # -----------------------------
    # Solve least squares
    # -----------------------------
    try:
        D, *_ = np.linalg.lstsq(X, S.T, rcond=None)
        D = D.T.reshape(H, W, 6)
    except:
        D = np.zeros((H, W, 6), dtype=np.float32)

    return D.astype(np.float32)


# -------------------------
# NOISE
# -------------------------
def lowres_noise(data, noise_max=0.1):
    noise = np.random.normal(0, noise_max, data.shape)
    return data + noise


# -------------------------
# DATASET
# -------------------------
class DWIDataset2D(Dataset):
    def __init__(self, mode="train"):

        self.samples = []

        TARGET_N = 130
        TARGET_HW = (128, 128)

        print(f"Loading dataset ({mode}) from:", config.DATA_DIR)

        dwi_files = glob.glob(
            os.path.join(config.DATA_DIR, "sub-*", "ses-*", "dwi", "*_dwi.nii.gz")
        )

        print(f"Found {len(dwi_files)} DWI files\n")

        for dwi_path in dwi_files:

            subject_id = dwi_path.split("/")[-4]

            # -------------------------
            # SPLIT
            # -------------------------
            if mode == "train" and subject_id not in config.TRAIN_SUBJECTS:
                continue
            if mode == "test" and subject_id not in config.TEST_SUBJECTS:
                continue

            bval_path = dwi_path.replace(".nii.gz", ".bval")
            bvec_path = dwi_path.replace(".nii.gz", ".bvec")

            img = nib.load(dwi_path)
            data = img.get_fdata()

            bvals = np.loadtxt(bval_path)
            bvecs = np.loadtxt(bvec_path)

            # ensure correct shape for bvecs (3, N) → (N, 3)
            if bvecs.shape[0] == 3:
                bvecs = bvecs.T

            num_volumes = data.shape[-1]

            if len(bvals) != num_volumes:
                print(f"[WARNING] Mismatch detected: data={num_volumes}, bvals={len(bvals)}")

                min_len = min(len(bvals), num_volumes)

                # trim everything consistently
                data = data[..., :min_len]
                bvals = bvals[:min_len]
                bvecs = bvecs[:min_len]

                print(f"[FIXED] Trimmed to {min_len} volumes")


            degraded = lowres_noise(data, noise_max=config.NOISE_MAX)

            X, Y, Z, N = data.shape

            for z in range(Z):

                clean = data[:, :, z, :]
                noisy = degraded[:, :, z, :]

                # -------------------------
                # FIX CHANNELS FIRST (CRITICAL)
                # -------------------------
                def fix_channels(x):
                    if x.shape[-1] > TARGET_N:
                        return x[..., :TARGET_N]
                    elif x.shape[-1] < TARGET_N:
                        pad = TARGET_N - x.shape[-1]
                        return np.concatenate(
                            [x, np.zeros((*x.shape[:2], pad))], axis=-1
                        )
                    return x

                clean = fix_channels(clean)
                noisy = fix_channels(noisy)

                # -------------------------
                # TRANSPOSE (C, H, W)
                # -------------------------
                clean = np.transpose(clean, (2, 0, 1))
                noisy = np.transpose(noisy, (2, 0, 1))

                # -------------------------
                # RESIZE FIRST (CRITICAL)
                # -------------------------
                def resize(x):
                    x = torch.tensor(x, dtype=torch.float32)
                    x = F.interpolate(
                        x.unsqueeze(0),
                        size=TARGET_HW,
                        mode='bilinear',
                        align_corners=False
                    )
                    return x.squeeze(0)

                clean = resize(clean)
                noisy = resize(noisy)

                # -------------------------
                # NOW COMPUTE TENSOR (FIXED)
                # -------------------------
                # Convert back to (H, W, C)
                clean_np = clean.permute(1, 2, 0).cpu().numpy()

                tensor_slice = compute_dti_numpy(clean_np, bvals, bvecs)

                # transpose tensor -> (6, H, W)
                tensor_slice = np.transpose(tensor_slice, (2, 0, 1))
                tensor_slice = torch.tensor(tensor_slice, dtype=torch.float32)

                self.samples.append((
                    noisy,
                    clean,
                    tensor_slice,
                    bvals.astype(np.float32),
                    bvecs.astype(np.float32)
                ))

        print(f"Total slices ({mode}): {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
