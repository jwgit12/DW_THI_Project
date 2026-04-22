"""DWI slice dataset with on-the-fly degradation and multi-axis slicing.

The Zarr store holds only clean data (`target_dwi`, `target_dti_6d`,
`bvals`, `bvecs`). The noisy model input is synthesized on the fly in
`__getitem__`, so every epoch sees a different noise realisation and a
different k-space cutout for the same underlying slice.

Slices can be drawn from any of the three spatial axes; padding to a
canonical (H, W) is applied at the end of `__getitem__` so batches stack
cleanly with the default collate.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

import config as cfg
from functions import compute_b0_norm, compute_brain_mask_from_dwi
from research.augment import degrade_dwi_slice


# Order of channels in the stored 6D tensor: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
# When the image is mirrored along world axis a, the tensor transforms as
# D' = F_a D F_a^T, which flips the sign of every off-diagonal that touches a.
_DTI6D_OFFDIAG_SIGN = {
    # world_axis -> list of (channel_index_to_negate,)
    0: (1, 3),  # x-flip: Dxy, Dxz
    1: (1, 4),  # y-flip: Dxy, Dyz
    2: (3, 4),  # z-flip: Dxz, Dyz
}


def _flip_dti6d_sign(tgt_chw: np.ndarray, world_axis: int) -> np.ndarray:
    """Return a copy of ``tgt_chw`` with off-diagonals sign-flipped for axis."""
    out = tgt_chw.copy()
    for c in _DTI6D_OFFDIAG_SIGN[world_axis]:
        out[c] = -out[c]
    return out


def _flip_bvecs(bvecs_3n: np.ndarray, world_axis: int) -> np.ndarray:
    """Negate the component of ``bvecs`` that matches the flipped world axis."""
    out = bvecs_3n.copy()
    out[world_axis] = -out[world_axis]
    return out


class DWISliceDataset(Dataset):
    """Loads 2D DWI slices and degrades them on the fly.

    Each sample returns:
        input:      (max_n, H, W)   padded noisy DWI (generated on the fly)
        target:     (6, H, W)       6D DTI tensor, scaled by dti_scale
        bvals:      (max_n,)        normalised b-values
        bvecs:      (3, max_n)      b-vectors
        vol_mask:   (max_n,)        1 for real volumes, 0 for padding
        brain_mask: (H, W)          binary brain mask (1 inside brain)
    """

    def __init__(
        self,
        zarr_path: str,
        subject_keys: list[str],
        *,
        augment: bool = False,
        b0_threshold: float = cfg.B0_THRESHOLD,
        use_brain_mask: bool = True,
        on_the_fly_degradation: bool = True,
        keep_fraction_range: tuple[float, float] = (
            cfg.KEEP_FRACTION_MIN, cfg.KEEP_FRACTION_MAX,
        ),
        noise_range: tuple[float, float] = (cfg.NOISE_MIN, cfg.NOISE_MAX),
        random_axis: bool = cfg.RANDOM_SLICE_AXIS,
        slice_axes: tuple[int, ...] = cfg.SLICE_AXES,
        aug_flip: bool = cfg.AUG_FLIP,
        aug_intensity: float = cfg.AUG_INTENSITY,
        aug_volume_dropout: float = cfg.AUG_VOLUME_DROPOUT,
        canonical_hw: tuple[int, int] | None = None,
        eval_mode: bool = False,
        eval_keep_fraction: float = cfg.EVAL_KEEP_FRACTION,
        eval_noise_level: float = cfg.EVAL_NOISE_LEVEL,
        eval_seed: int = cfg.EVAL_DEGRADE_SEED,
        return_clean_input: bool = False,
    ):
        self.zarr_path = zarr_path
        self.subject_keys = list(subject_keys)
        self.augment = augment
        self.b0_threshold = b0_threshold
        self.use_brain_mask = use_brain_mask
        self.on_the_fly_degradation = on_the_fly_degradation
        self.keep_fraction_range = tuple(keep_fraction_range)
        self.noise_range = tuple(noise_range)
        self.random_axis = random_axis
        self.slice_axes = tuple(slice_axes) if random_axis else (2,)
        self.aug_flip = aug_flip
        self.aug_intensity = float(aug_intensity)
        self.aug_volume_dropout = float(aug_volume_dropout)
        self.eval_mode = eval_mode
        self.eval_keep_fraction = float(eval_keep_fraction)
        self.eval_noise_level = float(eval_noise_level)
        self.eval_seed = int(eval_seed)
        self.return_clean_input = bool(return_clean_input)

        store = zarr.open_group(zarr_path, mode="r")

        # ── Pre-load all subject arrays into RAM (dominant memory cost) ─────
        self._data: dict[str, dict[str, np.ndarray]] = {}
        self._brain_masks: dict[str, np.ndarray | None] = {}
        self.samples: list[tuple[str, int, int]] = []  # (key, axis, slice_index)

        self.max_n = 0
        global_max_bval = 0.0
        all_dti_abs: list[np.ndarray] = []
        max_h, max_w = 0, 0

        for key in self.subject_keys:
            grp = store[key]
            target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
            target_dti = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
            bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
            bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)

            self._data[key] = {
                "target_dwi": target_dwi,
                "target_dti_6d": target_dti,
                "bvals": bvals,
                "bvecs": bvecs,
            }

            if self.use_brain_mask:
                self._brain_masks[key] = compute_brain_mask_from_dwi(
                    target_dwi, bvals, self.b0_threshold,
                )
            else:
                self._brain_masks[key] = None

            N = bvals.shape[0]
            self.max_n = max(self.max_n, N)
            global_max_bval = max(global_max_bval, float(bvals.max()))

            nonzero = np.abs(target_dti[target_dti != 0])
            if nonzero.size > 0:
                all_dti_abs.append(nonzero)

            X, Y, Z, _ = target_dwi.shape
            for axis in self.slice_axes:
                n_slices = target_dwi.shape[axis]
                for s in range(n_slices):
                    self.samples.append((key, axis, s))
                other = [d for i, d in enumerate((X, Y, Z)) if i != axis]
                max_h = max(max_h, other[0])
                max_w = max(max_w, other[1])

        self.canonical_hw = (
            tuple(canonical_hw) if canonical_hw is not None else (max_h, max_w)
        )

        self.max_bval = global_max_bval if global_max_bval > 0 else 1.0
        if all_dti_abs:
            pooled = np.concatenate(all_dti_abs)
            self.dti_scale = float(1.0 / np.percentile(pooled, 99))
        else:
            self.dti_scale = 1.0

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key, axis, s = self.samples[idx]
        d = self._data[key]
        bmask_3d = self._brain_masks[key]

        # ── Extract clean 2D slice along the chosen axis ────────────────────
        target_dwi = d["target_dwi"]      # (X, Y, Z, N)
        target_dti = d["target_dti_6d"]   # (X, Y, Z, 6)
        bvals = d["bvals"]                # (N,)
        bvecs = d["bvecs"]                # (3, N)

        if axis == 0:
            clean = target_dwi[s]                 # (Y, Z, N)
            tgt = target_dti[s]                   # (Y, Z, 6)
            bmask = bmask_3d[s] if bmask_3d is not None else None   # (Y, Z)
        elif axis == 1:
            clean = target_dwi[:, s]              # (X, Z, N)
            tgt = target_dti[:, s]                # (X, Z, 6)
            bmask = bmask_3d[:, s] if bmask_3d is not None else None  # (X, Z)
        else:  # axis == 2
            clean = target_dwi[:, :, s]           # (X, Y, N)
            tgt = target_dti[:, :, s]             # (X, Y, 6)
            bmask = bmask_3d[:, :, s] if bmask_3d is not None else None  # (X, Y)

        # (H, W, C) -> (C, H, W) channel-first; contiguous for later in-place ops.
        clean_nhw = np.ascontiguousarray(clean.transpose(2, 0, 1))     # (N, H, W)
        tgt_chw = np.ascontiguousarray(tgt.transpose(2, 0, 1))         # (6, H, W)
        if bmask is not None:
            bmask_hw = np.ascontiguousarray(bmask.astype(np.float32))
        else:
            bmask_hw = np.ones(clean_nhw.shape[1:], dtype=np.float32)

        N = bvals.shape[0]
        bvals_n = bvals.copy()
        bvecs_n = bvecs.copy()

        # ── On-the-fly degradation (fresh noise and cutout each call) ───────
        if self.on_the_fly_degradation:
            if self.eval_mode:
                kf = self.eval_keep_fraction
                nl = self.eval_noise_level
                rng = np.random.default_rng(self.eval_seed + idx)
            else:
                kf = float(np.random.uniform(*self.keep_fraction_range))
                nl = float(np.random.uniform(*self.noise_range))
                rng = np.random.default_rng()
            input_nhw = degrade_dwi_slice(clean_nhw, kf, nl, rng)
        else:
            input_nhw = clean_nhw.copy()

        # ── Scale DTI to O(1) range for balanced training ───────────────────
        tgt_chw = np.clip(tgt_chw * self.dti_scale, -3.0, 3.0).astype(np.float32)

        # ── Normalise input by the mean-b0 of the (noisy) input ────────────
        b0_idx = bvals_n < self.b0_threshold
        if b0_idx.any():
            mean_b0 = input_nhw[b0_idx].mean(axis=0)
            b0_norm = compute_b0_norm(mean_b0)
            if b0_norm > 0:
                input_nhw = input_nhw / b0_norm

        # ── Optional clean-DWI view for JEPA teacher (normalised by its own
        # clean b0 so the teacher sees natural-scale clean signals).
        clean_nhw_out: np.ndarray | None = None
        if self.return_clean_input:
            clean_nhw_out = clean_nhw.copy()
            if b0_idx.any():
                mean_b0_clean = clean_nhw_out[b0_idx].mean(axis=0)
                cb0_norm = compute_b0_norm(mean_b0_clean)
                if cb0_norm > 0:
                    clean_nhw_out = clean_nhw_out / cb0_norm

        bvals_norm = bvals_n / self.max_bval

        # ── Spatial + intensity augmentations (training only) ────────────────
        # Flipping an anatomical axis in the image also inverts the corresponding
        # components of D (off-diagonals that touch the flipped axis) and bvec
        # components. Without this, the (signal, bvecs) -> tensor mapping becomes
        # contradictory for ~50% of the training batch.
        if self.augment:
            if self.aug_flip:
                # Map (slice_axis, flipped_hw_axis) -> world axis (0=x, 1=y, 2=z)
                world_axes_by_slice = {
                    0: (1, 2),  # sagittal slice: H=Y, W=Z
                    1: (0, 2),  # coronal slice:  H=X, W=Z
                    2: (0, 1),  # axial slice:    H=X, W=Y
                }
                h_world, w_world = world_axes_by_slice[axis]
                if random.random() > 0.5:
                    input_nhw = input_nhw[:, ::-1, :]
                    tgt_chw = tgt_chw[:, ::-1, :]
                    bmask_hw = bmask_hw[::-1, :]
                    if clean_nhw_out is not None:
                        clean_nhw_out = clean_nhw_out[:, ::-1, :]
                    tgt_chw = _flip_dti6d_sign(tgt_chw, world_axis=h_world)
                    bvecs_n = _flip_bvecs(bvecs_n, world_axis=h_world)
                if random.random() > 0.5:
                    input_nhw = input_nhw[:, :, ::-1]
                    tgt_chw = tgt_chw[:, :, ::-1]
                    bmask_hw = bmask_hw[:, ::-1]
                    if clean_nhw_out is not None:
                        clean_nhw_out = clean_nhw_out[:, :, ::-1]
                    tgt_chw = _flip_dti6d_sign(tgt_chw, world_axis=w_world)
                    bvecs_n = _flip_bvecs(bvecs_n, world_axis=w_world)
                input_nhw = np.ascontiguousarray(input_nhw)
                tgt_chw = np.ascontiguousarray(tgt_chw)
                bmask_hw = np.ascontiguousarray(bmask_hw)
                if clean_nhw_out is not None:
                    clean_nhw_out = np.ascontiguousarray(clean_nhw_out)
            if self.aug_intensity > 0.0:
                scale = 1.0 + np.random.uniform(
                    -self.aug_intensity, self.aug_intensity,
                )
                input_nhw = input_nhw * np.float32(scale)
            if self.aug_volume_dropout > 0.0:
                drop = np.random.random(N) < self.aug_volume_dropout
                if drop.any():
                    input_nhw[drop] = 0.0
                    # vol_mask is built later from N; record drop mask
                    dropped_volumes = drop
                else:
                    dropped_volumes = None
            else:
                dropped_volumes = None
        else:
            dropped_volumes = None

        # ── Pad diffusion dimension to max_n (variable across subjects) ─────
        if N < self.max_n:
            pad = self.max_n - N
            input_nhw = np.pad(input_nhw, ((0, pad), (0, 0), (0, 0)))
            bvals_norm = np.pad(bvals_norm, (0, pad))
            bvecs_n = np.pad(bvecs_n, ((0, 0), (0, pad)))
            if clean_nhw_out is not None:
                clean_nhw_out = np.pad(clean_nhw_out, ((0, pad), (0, 0), (0, 0)))

        vol_mask = np.zeros(self.max_n, dtype=np.float32)
        vol_mask[:N] = 1.0
        if dropped_volumes is not None:
            vol_mask[:N][dropped_volumes] = 0.0

        # ── Pad spatial dims to canonical (H, W) so batches stack ───────────
        ch, cw = self.canonical_hw
        h, w = input_nhw.shape[1:]
        if (h, w) != (ch, cw):
            ph = ch - h
            pw = cw - w
            input_nhw = np.pad(input_nhw, ((0, 0), (0, ph), (0, pw)))
            tgt_chw = np.pad(tgt_chw, ((0, 0), (0, ph), (0, pw)))
            bmask_hw = np.pad(bmask_hw, ((0, ph), (0, pw)))
            if clean_nhw_out is not None:
                clean_nhw_out = np.pad(clean_nhw_out, ((0, 0), (0, ph), (0, pw)))

        sample = {
            "input": torch.from_numpy(np.ascontiguousarray(input_nhw)),
            "target": torch.from_numpy(np.ascontiguousarray(tgt_chw)),
            "bvals": torch.from_numpy(bvals_norm.astype(np.float32)),
            "bvecs": torch.from_numpy(bvecs_n.astype(np.float32)),
            "vol_mask": torch.from_numpy(vol_mask),
            "brain_mask": torch.from_numpy(np.ascontiguousarray(bmask_hw)),
        }
        if clean_nhw_out is not None:
            sample["clean_input"] = torch.from_numpy(
                np.ascontiguousarray(clean_nhw_out)
            )
        return sample
