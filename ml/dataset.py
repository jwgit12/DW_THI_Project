"""
PyTorch Dataset for the DTI masked-signal-prediction pretext task.

Loads 2-D axial slices from the Zarr pretext store and applies random
gradient-direction masking.  Returns per-direction tensors so that a
channel-invariant model can handle subjects with different numbers of
gradient directions in the same batch (via the custom ``collate_fn``).

Returns per sample (before collation)
-------------------------------------
directions   : (N, H, W)   float32  – noisy DWI per direction (masked ones zeroed)
dir_mask     : (N,)         float32  – 1.0 = kept, 0.0 = masked
target_dwi   : (N, H, W)   float32  – clean DWI (all directions)
target_dti   : (6, H, W)   float32  – clean 6-component DTI
bvals        : (N,)         float32
bvecs        : (3, N)       float32
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class PretextDWIDataset(Dataset):
    """2-D axial-slice dataset with random DWI direction masking."""

    def __init__(
        self,
        zarr_path: str,
        subject_indices: list[int] | None = None,
        mask_fraction: float = 0.4,
        normalize: bool = True,
    ):
        super().__init__()
        self.store = zarr.open(zarr_path, mode="r")
        all_subjects = sorted(self.store.group_keys())

        if subject_indices is not None:
            self.subjects = [all_subjects[i] for i in subject_indices]
        else:
            self.subjects = all_subjects

        self.mask_fraction = mask_fraction
        self.normalize = normalize

        # Pre-compute index → (subject_idx_in_list, z_slice)
        self._index_map: list[tuple[int, int]] = []
        for s_idx, name in enumerate(self.subjects):
            nz = self.store[name]["input_dwi"].shape[2]
            for z in range(nz):
                self._index_map.append((s_idx, z))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        s_idx, z = self._index_map[idx]
        name = self.subjects[s_idx]
        grp = self.store[name]

        # (H, W, N) and (H, W, 6)
        noisy_slice = grp["input_dwi"][:, :, z, :].astype(np.float32)
        clean_slice = grp["target_dwi"][:, :, z, :].astype(np.float32)
        dti_slice = grp["target_dti_6d"][:, :, z, :].astype(np.float32)
        bvals = grp["bvals"][:].astype(np.float32)
        bvecs = grp["bvecs"][:].astype(np.float32)

        n_dirs = noisy_slice.shape[-1]

        # Robust per-slice normalization
        if self.normalize:
            for arr in (noisy_slice, clean_slice):
                vmax = np.percentile(arr, 99) + 1e-8
                arr /= vmax
            dti_max = np.abs(dti_slice).max() + 1e-8
            dti_slice /= dti_max

        # Random gradient-direction masking
        n_mask = max(1, int(n_dirs * self.mask_fraction))
        mask_indices = np.random.choice(n_dirs, size=n_mask, replace=False)
        dir_mask = np.ones(n_dirs, dtype=np.float32)
        dir_mask[mask_indices] = 0.0

        masked_dirs = noisy_slice * dir_mask[None, None, :]  # (H, W, N)

        # Transpose to (N, H, W) / (6, H, W)
        return {
            "directions": torch.from_numpy(masked_dirs.transpose(2, 0, 1)),
            "dir_mask": torch.from_numpy(dir_mask),
            "target_dwi": torch.from_numpy(clean_slice.transpose(2, 0, 1)),
            "target_dti": torch.from_numpy(dti_slice.transpose(2, 0, 1)),
            "bvals": torch.from_numpy(bvals),
            "bvecs": torch.from_numpy(bvecs),
        }


def pretext_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-N direction tensors to the max N in the batch.

    Padding directions are zeroed and marked with ``dir_mask=0`` and
    ``pad_mask=0`` so models can ignore them during aggregation.
    """
    max_n = max(s["directions"].shape[0] for s in batch)
    B = len(batch)
    H, W = batch[0]["directions"].shape[1:]

    directions = torch.zeros(B, max_n, H, W)
    dir_mask = torch.zeros(B, max_n)
    pad_mask = torch.zeros(B, max_n)          # 1 = real direction, 0 = padding
    target_dwi = torch.zeros(B, max_n, H, W)
    bvals = torch.zeros(B, max_n)
    bvecs = torch.zeros(B, 3, max_n)
    target_dti = torch.stack([s["target_dti"] for s in batch])  # (B, 6, H, W)

    for i, s in enumerate(batch):
        n = s["directions"].shape[0]
        directions[i, :n] = s["directions"]
        dir_mask[i, :n] = s["dir_mask"]
        pad_mask[i, :n] = 1.0
        target_dwi[i, :n] = s["target_dwi"]
        bvals[i, :n] = s["bvals"]
        bvecs[i, :, :n] = s["bvecs"]

    return {
        "directions": directions,
        "dir_mask": dir_mask,
        "pad_mask": pad_mask,
        "target_dwi": target_dwi,
        "target_dti": target_dti,
        "bvals": bvals,
        "bvecs": bvecs,
        "n_dirs": torch.tensor([s["directions"].shape[0] for s in batch]),
    }
