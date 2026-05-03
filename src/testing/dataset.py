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
# SIMPLE DTI (unchanged)
# -------------------------
def compute_dti_numpy(data, bvals, bvecs):
    X, Y, Z, N = data.shape

    data = np.clip(data, 1e-6, None)

    mask = bvals > 50
    bvals = bvals[mask]
    bvecs = bvecs[:, mask]
    data = data[..., mask]

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
    B = np.array(B)
    B_pinv = np.linalg.pinv(B)

    tensor = np.zeros((X, Y, Z, 6))

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                S = data[x, y, z, :]
                logS = np.log(S)
                D = B_pinv @ logS
                tensor[x, y, z, :] = D

    return tensor


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

            subject_id = dwi_path.split("/")[-4]  # sub-XX

            # -------------------------
            # SPLIT FILTER
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

            tensor = compute_dti_numpy(data, bvals, bvecs)
            degraded = lowres_noise(data, noise_max=config.NOISE_MAX)

            X, Y, Z, N = data.shape

            for z in range(Z):

                clean = data[:, :, z, :]
                noisy = degraded[:, :, z, :]
                tensor_slice = tensor[:, :, z, :]

                # -------------------------
                # FIX CHANNELS
                # -------------------------
                def fix_channels(x):
                    if x.shape[-1] > TARGET_N:
                        x = x[..., :TARGET_N]
                    elif x.shape[-1] < TARGET_N:
                        pad = TARGET_N - x.shape[-1]
                        x = np.concatenate(
                            [x, np.zeros((*x.shape[:2], pad))], axis=-1
                        )
                    return x

                clean = fix_channels(clean)
                noisy = fix_channels(noisy)

                # -------------------------
                # TRANSPOSE
                # -------------------------
                clean = np.transpose(clean, (2, 0, 1))
                noisy = np.transpose(noisy, (2, 0, 1))
                tensor_slice = np.transpose(tensor_slice, (2, 0, 1))

                # -------------------------
                # RESIZE
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
                tensor_slice = resize(tensor_slice)

                self.samples.append((noisy, clean, tensor_slice))

        print(f"Total slices ({mode}): {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
