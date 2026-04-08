"""
PyTorch Dataset for supervised DWI denoising.

Loads 2-D axial slices from the Zarr pretext store.  Each sample contains
all diffusion directions as channels.

Supports optional patch-wise training: random crops of configurable size
are extracted from each slice, with normalized (x, y) position encoding
so the model retains spatial context within the full image.

Augmentation (when enabled): random horizontal/vertical flips and 90-degree
rotations applied jointly to noisy and clean slices.  These are safe for DWI
since bvals/bvecs are per-direction metadata independent of spatial orientation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class DWIDenoiseDataset(Dataset):
    """2-D axial-slice dataset for supervised DWI denoising."""

    def __init__(
        self,
        zarr_path: str,
        subject_indices: list[int] | None = None,
        normalize: bool = True,
        augment: bool = False,
        patch_size: int | None = None,
    ):
        super().__init__()
        self.store = zarr.open(zarr_path, mode="r")
        all_subjects = sorted(self.store.group_keys())

        if subject_indices is not None:
            self.subjects = [all_subjects[i] for i in subject_indices]
        else:
            self.subjects = all_subjects

        self.normalize = normalize
        self.augment = augment
        self.patch_size = patch_size

        # Pre-compute index -> (subject_idx, z_slice)
        self._index_map: list[tuple[int, int]] = []
        for s_idx, name in enumerate(self.subjects):
            nz = self.store[name]["input_dwi"].shape[2]
            for z in range(nz):
                self._index_map.append((s_idx, z))

    def __len__(self) -> int:
        return len(self._index_map)

    def _apply_augmentation(
        self, noisy: np.ndarray, clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random flips and 180-degree rotation on (N, H, W) arrays.

        90/270-degree rotations are skipped because they swap H and W,
        which breaks collation when images are non-square.
        """
        if np.random.random() > 0.5:
            noisy = noisy[:, :, ::-1].copy()
            clean = clean[:, :, ::-1].copy()
        if np.random.random() > 0.5:
            noisy = noisy[:, ::-1, :].copy()
            clean = clean[:, ::-1, :].copy()
        # 180-degree rotation preserves H and W
        if np.random.random() > 0.5:
            noisy = np.rot90(noisy, 2, axes=(1, 2)).copy()
            clean = np.rot90(clean, 2, axes=(1, 2)).copy()
        return noisy, clean

    def __getitem__(self, idx: int):
        s_idx, z = self._index_map[idx]
        name = self.subjects[s_idx]
        grp = self.store[name]

        # (H, W, N) slices
        noisy_slice = grp["input_dwi"][:, :, z, :].astype(np.float32)
        clean_slice = grp["target_dwi"][:, :, z, :].astype(np.float32)
        bvals = grp["bvals"][:].astype(np.float32)
        bvecs = grp["bvecs"][:].astype(np.float32)  # (3, N)

        # Joint normalization — same scale so residual is meaningful
        scale = np.float32(1.0)
        if self.normalize:
            combined_max = max(
                np.percentile(noisy_slice, 99),
                np.percentile(clean_slice, 99),
            )
            scale = np.float32(combined_max + 1e-8)
            noisy_slice = noisy_slice / scale
            clean_slice = clean_slice / scale

        # Transpose to (N, H, W)
        noisy_slice = noisy_slice.transpose(2, 0, 1)
        clean_slice = clean_slice.transpose(2, 0, 1)

        if self.augment:
            noisy_slice, clean_slice = self._apply_augmentation(
                noisy_slice, clean_slice,
            )

        N, H, W = noisy_slice.shape

        # Patch cropping (if enabled)
        if self.patch_size is not None and self.patch_size < min(H, W):
            ps = self.patch_size
            y0 = np.random.randint(0, H - ps + 1)
            x0 = np.random.randint(0, W - ps + 1)
            noisy_slice = noisy_slice[:, y0:y0 + ps, x0:x0 + ps]
            clean_slice = clean_slice[:, y0:y0 + ps, x0:x0 + ps]
        else:
            y0, x0 = 0, 0
            ps = None

        pH, pW = noisy_slice.shape[1], noisy_slice.shape[2]

        # Position encoding: normalized (y, x) coordinates within the full slice
        if ps is not None:
            y_coords = np.linspace(y0 / H, (y0 + ps - 1) / H, pH, dtype=np.float32)
            x_coords = np.linspace(x0 / W, (x0 + ps - 1) / W, pW, dtype=np.float32)
        else:
            y_coords = np.linspace(0, 1, pH, dtype=np.float32)
            x_coords = np.linspace(0, 1, pW, dtype=np.float32)
        # (pH, pW) grids
        pos_y, pos_x = np.meshgrid(y_coords, x_coords, indexing="ij")
        # Stack to (2, pH, pW)
        pos_enc = np.stack([pos_y, pos_x], axis=0)

        return {
            "noisy_dwi": torch.from_numpy(noisy_slice),
            "clean_dwi": torch.from_numpy(clean_slice),
            "bvals": torch.from_numpy(bvals),
            "bvecs": torch.from_numpy(bvecs),
            "pos_enc": torch.from_numpy(pos_enc),
            "scale": torch.tensor(scale),
            "subject": name,
            "slice_idx": z,
        }


def denoise_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-N direction tensors to the max N in the batch."""
    max_n = max(s["noisy_dwi"].shape[0] for s in batch)
    B = len(batch)
    H, W = batch[0]["noisy_dwi"].shape[1:]

    noisy_dwi = torch.zeros(B, max_n, H, W)
    clean_dwi = torch.zeros(B, max_n, H, W)
    pad_mask = torch.zeros(B, max_n)
    bvals = torch.zeros(B, max_n)
    bvecs = torch.zeros(B, 3, max_n)
    pos_enc = torch.zeros(B, 2, H, W)
    scales = torch.zeros(B)
    subjects = []
    slice_indices = []

    for i, s in enumerate(batch):
        n = s["noisy_dwi"].shape[0]
        noisy_dwi[i, :n] = s["noisy_dwi"]
        clean_dwi[i, :n] = s["clean_dwi"]
        pad_mask[i, :n] = 1.0
        bvals[i, :n] = s["bvals"]
        bvecs[i, :, :n] = s["bvecs"]
        pos_enc[i] = s["pos_enc"]
        scales[i] = s["scale"]
        subjects.append(s["subject"])
        slice_indices.append(s["slice_idx"])

    return {
        "noisy_dwi": noisy_dwi,
        "clean_dwi": clean_dwi,
        "pad_mask": pad_mask,
        "bvals": bvals,
        "bvecs": bvecs,
        "pos_enc": pos_enc,
        "scales": scales,
        "subjects": subjects,
        "slice_indices": torch.tensor(slice_indices),
        "n_dirs": torch.tensor([s["noisy_dwi"].shape[0] for s in batch]),
    }
