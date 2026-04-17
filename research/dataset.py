"""DWI slice dataset for training DTI prediction models."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

import config as cfg
from functions import compute_b0_norm, compute_brain_mask_from_dwi


class DWISliceDataset(Dataset):
    """Loads 2D axial slices from a zarr DWI dataset.

    Each sample returns:
        input:      (max_n, H, W)   padded DWI signal, normalised by mean b0
        target:     (6, H, W)       6D DTI tensor, scaled by dti_scale
        bvals:      (max_n,)        normalised b-values (/ max_bval), padded
        bvecs:      (3, max_n)      b-vectors, padded
        vol_mask:   (max_n,)        1 for real volumes, 0 for padding
        brain_mask: (H, W)          binary brain mask (1 = brain, 0 = background)
    """

    def __init__(
        self,
        zarr_path: str,
        subject_keys: list[str],
        augment: bool = False,
        b0_threshold: float = cfg.B0_THRESHOLD,
    ):
        self.zarr_path = zarr_path
        self.subject_keys = list(subject_keys)
        self.augment = augment
        self.b0_threshold = b0_threshold

        store = zarr.open_group(zarr_path, mode="r")

        self.max_n = 0
        self.samples: list[tuple[str, int]] = []

        global_max_bval = 0.0
        all_dti_abs = []

        for key in self.subject_keys:
            grp = store[key]
            n_volumes = grp["bvals"].shape[0]
            n_slices = grp["target_dti_6d"].shape[2]
            self.max_n = max(self.max_n, n_volumes)
            for z in range(n_slices):
                self.samples.append((key, z))

            bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
            global_max_bval = max(global_max_bval, float(bvals.max()))

            dti = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
            nonzero = np.abs(dti[dti != 0])
            if nonzero.size > 0:
                all_dti_abs.append(nonzero)

        # Adaptive normalisation constants derived from the data
        self.max_bval = global_max_bval if global_max_bval > 0 else 1.0
        if all_dti_abs:
            pooled = np.concatenate(all_dti_abs)
            self.dti_scale = float(1.0 / np.percentile(pooled, 99))
        else:
            self.dti_scale = 1.0

        # Pre-compute 3D brain masks per subject from clean target DWI
        self._brain_masks: dict[str, np.ndarray] = {}
        for key in self.subject_keys:
            grp = store[key]
            dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
            bvals_raw = np.asarray(grp["bvals"][:], dtype=np.float32)
            self._brain_masks[key] = compute_brain_mask_from_dwi(
                dwi, bvals_raw, self.b0_threshold,
            )

        self._store = None

    @property
    def store(self):
        # Lazy open so pickling (DataLoader workers) works
        if self._store is None:
            self._store = zarr.open_group(self.zarr_path, mode="r")
        return self._store

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key, z = self.samples[idx]
        grp = self.store[key]

        input_slice = np.asarray(grp["input_dwi"][:, :, z, :], dtype=np.float32)  # (H, W, N)
        target_slice = np.asarray(grp["target_dti_6d"][:, :, z, :], dtype=np.float32)  # (H, W, 6)
        bvals = np.asarray(grp["bvals"][:], dtype=np.float32)  # (N,)
        bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)  # (3, N)

        N = bvals.shape[0]

        # Brain mask for this slice
        bmask = self._brain_masks[key][:, :, z].astype(np.float32)  # (H, W)

        # Channels-first: (N, H, W) and (6, H, W)
        input_slice = input_slice.transpose(2, 0, 1)
        target_slice = target_slice.transpose(2, 0, 1)

        # Scale DTI tensor to ~O(1) range for balanced training
        # Clamp to remove extreme outliers from bad DTI fits at brain edges
        target_slice = np.clip(target_slice * self.dti_scale, -3.0, 3.0)

        # Normalise input by mean b0 signal
        b0_idx = bvals < self.b0_threshold
        if b0_idx.any():
            mean_b0 = input_slice[b0_idx].mean(axis=0)  # (H, W)
            b0_norm = compute_b0_norm(mean_b0)
            if b0_norm > 0:
                input_slice = input_slice / b0_norm

        # Normalise bvals to [0, 1] using the max b-value in the dataset
        bvals_norm = bvals / self.max_bval

        # Pad to max_n
        if N < self.max_n:
            pad = self.max_n - N
            input_slice = np.pad(input_slice, ((0, pad), (0, 0), (0, 0)))
            bvals_norm = np.pad(bvals_norm, (0, pad))
            bvecs = np.pad(bvecs, ((0, 0), (0, pad)))

        vol_mask = np.zeros(self.max_n, dtype=np.float32)
        vol_mask[:N] = 1.0

        # Augmentation: random flips
        if self.augment:
            if random.random() > 0.5:
                input_slice = np.flip(input_slice, axis=1).copy()
                target_slice = np.flip(target_slice, axis=1).copy()
                bmask = np.flip(bmask, axis=0).copy()
            if random.random() > 0.5:
                input_slice = np.flip(input_slice, axis=2).copy()
                target_slice = np.flip(target_slice, axis=2).copy()
                bmask = np.flip(bmask, axis=1).copy()

        return {
            "input": torch.from_numpy(input_slice),
            "target": torch.from_numpy(target_slice),
            "bvals": torch.from_numpy(bvals_norm),
            "bvecs": torch.from_numpy(bvecs),
            "vol_mask": torch.from_numpy(vol_mask),
            "brain_mask": torch.from_numpy(bmask),
        }
