# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from functions import load_all_dwi, lowres_noise


class DWIDataset2D(Dataset):

    def __init__(self, mode="train"):

        self.mode = mode
        self.root = config.DATA_PATH

        print(f"\nLoading dataset ({mode}) from: {self.root}")

        self.data_entries = load_all_dwi(self.root)

        print(f"\nFound {len(self.data_entries)} DWI files\n")

        self.samples = []

        for entry in self.data_entries:

            data = entry["data"]  # (X, Y, Z, N)
            bvals = entry["bvals"]
            bvecs = entry["bvecs"]
            path = entry["path"]

            # load precomputed tensor
            tensor_path = path.replace(".nii.gz", "_tensor6.npy")

            if not os.path.exists(tensor_path):
                print(f"[WARN] Missing tensor: {tensor_path}")
                continue

            tensor6 = np.load(tensor_path)  # (X, Y, Z, 6)

            # degrade DWI
            noisy = lowres_noise(data)

            _, _, Z, _ = data.shape

            for z in range(Z):

                clean_slice = data[:, :, z, :]       # (X, Y, N)
                noisy_slice = noisy[:, :, z, :]
                tensor_slice = tensor6[:, :, z, :]   # (X, Y, 6)

                # transpose to channel-first
                clean_slice = np.transpose(clean_slice, (2, 0, 1))
                noisy_slice = np.transpose(noisy_slice, (2, 0, 1))
                tensor_slice = np.transpose(tensor_slice, (2, 0, 1))

                self.samples.append((
                    noisy_slice.astype(np.float32),
                    clean_slice.astype(np.float32),
                    tensor_slice.astype(np.float32),
                    bvals,
                    bvecs
                ))

        print(f"\nTotal slices ({mode}): {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_noisy, x_clean, tensor, bvals, bvecs = self.samples[idx]

        return (
            torch.from_numpy(x_noisy),
            torch.from_numpy(x_clean),
            torch.from_numpy(tensor),
            bvals,
            bvecs
        )
