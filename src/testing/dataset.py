# dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset

import config
from functions import load_all_dwi, lowres_noise, compute_dti, tensor_to_6d


class DWIDataset2D(Dataset):

    def __init__(self, mode="train"):

        self.mode = mode
        self.data_root = config.DATA_DIR

        print(f"\nLoading dataset ({mode}) from: {self.data_root}")

        self.datasets = load_all_dwi(self.data_root)

        print(f"Found {len(self.datasets)} DWI files\n")

        self.samples = []

        for ds in self.datasets:

            data = ds["data"]  # (X, Y, Z, N)
            gtab = ds["gtab"]

            # degrade
            noisy = lowres_noise(data)

            # compute REAL tensor
            tensor = compute_dti(data, gtab)           # (X,Y,Z,3,3)
            tensor6 = tensor_to_6d(tensor)             # (X,Y,Z,6)

            X, Y, Z, N = data.shape

            for z in range(Z):

                clean_slice = data[:, :, z, :]
                noisy_slice = noisy[:, :, z, :]
                tensor_slice = tensor6[:, :, z, :]

                # reshape to (C,H,W)
                clean_slice = np.transpose(clean_slice, (2, 0, 1))
                noisy_slice = np.transpose(noisy_slice, (2, 0, 1))
                tensor_slice = np.transpose(tensor_slice, (2, 0, 1))

                self.samples.append((
                    noisy_slice.astype(np.float32),
                    clean_slice.astype(np.float32),
                    tensor_slice.astype(np.float32),
                    ds["bvals"],
                    ds["bvecs"]
                ))

        # split
        split = int(0.75 * len(self.samples))

        if mode == "train":
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]

        print(f"\nTotal slices ({mode}): {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy, clean, tensor, bvals, bvecs = self.samples[idx]

        return (
            torch.from_numpy(noisy),
            torch.from_numpy(clean),
            torch.from_numpy(tensor),
            bvals,
            bvecs
        )
