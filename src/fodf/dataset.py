"""DWI slice dataset with on-the-fly degradation and multi-axis slicing.

The fODF Zarr store holds clean inputs and SH targets only — it deliberately
omits ``target_dti_6d`` (the standard pipeline owns that). The noisy model
input is synthesised in ``__getitem__`` so every epoch sees a fresh noise
realisation and k-space cutout for the same underlying slice.

Slices can be drawn from any of the three spatial axes; padding to a canonical
``(H, W)`` is applied at the end of ``__getitem__`` so batches stack cleanly
with the default collate.
"""

from __future__ import annotations

import logging
from os import PathLike
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from dw_thi.augment import degrade_dwi_slice, flip_bvecs
from dw_thi.preprocessing import compute_b0_norm, compute_brain_mask_from_dwi
from dw_thi.runtime import path_str

from . import defaults as cfg

log = logging.getLogger(__name__)

# Process-local zarr store cache — one open store per (process, path).
# Each DataLoader worker (spawned process on Windows) gets its own entry on
# first __getitem__ call, so we never pickle the large zarr store object.
_process_zarr_stores: dict[str, zarr.Group] = {}


def _get_zarr_group(zarr_path: str) -> zarr.Group:
    """Return a cached, read-only zarr root group for this process."""
    if zarr_path not in _process_zarr_stores:
        _process_zarr_stores[zarr_path] = zarr.open_group(zarr_path, mode="r")
    return _process_zarr_stores[zarr_path]


def _spatial_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a 2D spatial slice from a 3D/4D volume."""
    if axis == 0:
        return array[index]
    if axis == 1:
        return array[:, index]
    return array[:, :, index]


def _context_indices(center: int, n_slices: int, depth: int) -> list[int]:
    radius = depth // 2
    return [
        min(max(center + offset, 0), n_slices - 1)
        for offset in range(-radius, radius + 1)
    ]


def _slice_to_chw(arr, axis: int, index: int) -> np.ndarray:
    """Slice ``(X, Y, Z, C)`` along ``axis`` and return a contiguous ``(C, H, W)``.

    Folds slice + transpose + contiguous-copy into one allocation. Accepts
    zarr Arrays or numpy arrays since ``_spatial_slice`` on a zarr Array
    returns a freshly materialised numpy view either way.
    """
    sl = _spatial_slice(arr, axis, index)
    return np.ascontiguousarray(np.asarray(sl).transpose(2, 0, 1), dtype=np.float32)


def _extract_dwi_context_chw(
    dwi_xyzn,
    axis: int,
    center: int,
    depth: int,
) -> np.ndarray:
    """Return a contiguous DWI slice ``(N, H, W)`` or context stack ``(N, D, H, W)``.

    Edge slices are clamped to within the volume so the depth dimension is
    always exactly ``depth``.
    """
    if depth == 1:
        return _slice_to_chw(dwi_xyzn, axis, center)
    indices = _context_indices(center, dwi_xyzn.shape[axis], depth)
    # Stack into (D, H, W, N) first so the final transpose yields a single copy.
    stacked = np.stack(
        [np.asarray(_spatial_slice(dwi_xyzn, axis, idx)) for idx in indices],
        axis=0,
    )
    return np.ascontiguousarray(stacked.transpose(3, 0, 1, 2), dtype=np.float32)


def _degrade_model_signal(
    signal: np.ndarray,
    keep_fraction: float,
    noise_level: float,
    rng: np.random.Generator,
    *,
    noise_distribution: str = "rician",
    n_coils: int = 1,
) -> np.ndarray:
    kwargs = {"noise_distribution": noise_distribution, "n_coils": n_coils}
    if signal.ndim == 3:
        return degrade_dwi_slice(signal, keep_fraction, noise_level, rng, **kwargs)
    if signal.ndim == 4:
        n, d, h, w = signal.shape
        degraded = degrade_dwi_slice(
            signal.reshape(n * d, h, w), keep_fraction, noise_level, rng, **kwargs,
        )
        return degraded.reshape(n, d, h, w)
    raise ValueError(f"Unexpected model signal shape: {signal.shape}")


def _b0_normalize_model_signal(signal: np.ndarray, b0_idx: np.ndarray) -> np.ndarray:
    if not b0_idx.any():
        return signal
    if signal.ndim == 3:
        mean_b0 = signal[b0_idx].mean(axis=0)
        b0_norm = compute_b0_norm(mean_b0)
        if b0_norm > 0:
            signal *= np.float32(1.0 / b0_norm)
        return signal
    if signal.ndim == 4:
        for depth_idx in range(signal.shape[1]):
            mean_b0 = signal[b0_idx, depth_idx].mean(axis=0)
            b0_norm = compute_b0_norm(mean_b0)
            if b0_norm > 0:
                signal[:, depth_idx] *= np.float32(1.0 / b0_norm)
        return signal
    raise ValueError(f"Unexpected model signal shape: {signal.shape}")


def _pad_signal(signal: np.ndarray, n_pad: int, ph: int, pw: int) -> np.ndarray:
    """Pad ``(N, H, W)`` or ``(N, D, H, W)`` along N and trailing two spatial axes."""
    if n_pad == 0 and ph == 0 and pw == 0:
        return signal
    if signal.ndim == 3:
        return np.pad(signal, ((0, n_pad), (0, ph), (0, pw)))
    if signal.ndim == 4:
        return np.pad(signal, ((0, n_pad), (0, 0), (0, ph), (0, pw)))
    raise ValueError(f"Unexpected model signal shape: {signal.shape}")


def _sh_n_coeffs(order: int) -> int:
    if order < 0 or order % 2 != 0:
        raise ValueError(f"fODF SH order must be a non-negative even integer, got {order}.")
    return (order + 1) * (order + 2) // 2


def _infer_sh_order(n_coeffs: int) -> int:
    for order in range(0, 32, 2):
        if _sh_n_coeffs(order) == int(n_coeffs):
            return order
    raise ValueError(f"Unsupported fODF SH coefficient count: {n_coeffs}")


class DWISliceDataset(Dataset):
    """Loads 2D DWI slices and degrades them on the fly (fODF target only).

    Each sample returns:
        input:        (max_n, H, W) or (max_n, D, H, W)
                                      padded noisy DWI (generated on the fly)
        target_fodf:  (C, H, W)       SH coefficients for the slice
        bvals:        (max_n,)        normalised b-values
        bvecs:        (3, max_n)      b-vectors
        vol_mask:     (max_n,)        1 for real volumes, 0 for padding
        brain_mask:   (H, W)          binary brain mask (1 inside brain)
    """

    def __init__(
        self,
        zarr_path: str | PathLike[str],
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
        noise_distribution: str = cfg.NOISE_DISTRIBUTION,
        n_coils: int = cfg.NOISE_COILS,
        random_axis: bool = cfg.RANDOM_SLICE_AXIS,
        slice_axes: tuple[int, ...] = cfg.SLICE_AXES,
        aug_flip: bool = cfg.AUG_FLIP,
        aug_intensity: float = cfg.AUG_INTENSITY,
        aug_volume_dropout: float = cfg.AUG_VOLUME_DROPOUT,
        aug_bvec_mask_prob: float = cfg.AUG_BVEC_MASK_PROB,
        aug_bvec_mask_min_keep: float = cfg.AUG_BVEC_MASK_MIN_KEEP,
        aug_bvec_mask_max_keep: float = cfg.AUG_BVEC_MASK_MAX_KEEP,
        eval_bvec_mask_keep: float = cfg.EVAL_BVEC_MASK_KEEP,
        canonical_hw: tuple[int, int] | None = None,
        eval_mode: bool = False,
        eval_keep_fraction: float = cfg.EVAL_KEEP_FRACTION,
        eval_noise_level: float = cfg.EVAL_NOISE_LEVEL,
        eval_seed: int = cfg.EVAL_DEGRADE_SEED,
        gpu_degrade: bool = False,
        context_slices: int = 1,
        target_fodf_sh_order: int | None = None,
        preload: bool = False,
    ):
        self.zarr_path = path_str(zarr_path)
        self.subject_keys = list(subject_keys)
        self.augment = augment
        self.b0_threshold = b0_threshold
        self.use_brain_mask = use_brain_mask
        self.on_the_fly_degradation = on_the_fly_degradation
        self.keep_fraction_range = tuple(keep_fraction_range)
        self.noise_range = tuple(noise_range)
        self.noise_distribution = str(noise_distribution).lower()
        self.n_coils = int(n_coils)
        self.random_axis = random_axis
        self.slice_axes = tuple(slice_axes) if random_axis else (2,)
        self.aug_flip = bool(aug_flip)
        self.aug_intensity = float(aug_intensity)
        self.aug_volume_dropout = float(aug_volume_dropout)
        self.aug_bvec_mask_prob = float(aug_bvec_mask_prob)
        self.aug_bvec_mask_min_keep = float(aug_bvec_mask_min_keep)
        self.aug_bvec_mask_max_keep = float(aug_bvec_mask_max_keep)
        self.eval_bvec_mask_keep = float(eval_bvec_mask_keep)
        self.eval_mode = eval_mode
        self.eval_keep_fraction = float(eval_keep_fraction)
        self.eval_noise_level = float(eval_noise_level)
        self.eval_seed = int(eval_seed)
        self.context_slices = int(context_slices)
        if self.context_slices < 1 or self.context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd integer.")
        # GPU degrade: skip CPU FFT in __getitem__; batch dict carries kf/nl/b0_mask
        # for run_epoch to apply cuFFT on the full batch (>50x faster on CUDA).
        # Disabled in eval_mode to keep deterministic per-sample degradation.
        self.gpu_degrade = bool(gpu_degrade) and not eval_mode and on_the_fly_degradation

        # Open the store for metadata scanning; also seed the process-local cache
        # so the main process reuses this handle in __getitem__ (no double-open).
        store = zarr.open_group(self.zarr_path, mode="r")
        _process_zarr_stores[self.zarr_path] = store

        # Only bvals, bvecs, and brain_masks are kept in RAM (a few MB total).
        # The large DWI / SH arrays are loaded slice-by-slice from zarr in
        # __getitem__, which lets DataLoader workers run without pickling GBs.
        self._data: dict[str, dict[str, np.ndarray]] = {}
        self._brain_masks: dict[str, np.ndarray | None] = {}
        self.samples: list[tuple[str, int, int]] = []  # (key, axis, slice_index)
        self.has_fodf = False
        self.fodf_n_coeffs = 0
        self.stored_fodf_n_coeffs = 0
        self.target_fodf_sh_order = target_fodf_sh_order

        self.max_n = 0
        global_max_bval = 0.0
        max_h, max_w = 0, 0
        expected_fodf_n_coeffs: int | None = None

        for key in self.subject_keys:
            grp = store[key]
            target_dwi_arr = grp["target_dwi"]
            X, Y, Z, _ = target_dwi_arr.shape
            bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
            bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)

            self._data[key] = {"bvals": bvals, "bvecs": bvecs}

            target_dwi_for_mask = None
            if self.use_brain_mask and "brain_mask" in set(grp.array_keys()):
                self._brain_masks[key] = np.asarray(grp["brain_mask"][:], dtype=bool)
            elif self.use_brain_mask:
                log.warning(
                    "%s has no stored brain_mask; recomputing with DIPY median_otsu. "
                    "Rebuild with build_fodf_dataset.py for production runs.",
                    key,
                )
                target_dwi_for_mask = np.asarray(target_dwi_arr[:], dtype=np.float32)
                self._brain_masks[key] = compute_brain_mask_from_dwi(
                    target_dwi_for_mask, bvals, self.b0_threshold,
                )
            else:
                self._brain_masks[key] = None

            N = bvals.shape[0]
            self.max_n = max(self.max_n, N)
            global_max_bval = max(global_max_bval, float(bvals.max()))

            has_fodf = "target_fodf_sh" in set(grp.array_keys())
            if has_fodf:
                n_coeffs = int(grp["target_fodf_sh"].shape[-1])
                if expected_fodf_n_coeffs is None:
                    expected_fodf_n_coeffs = n_coeffs
                elif n_coeffs != expected_fodf_n_coeffs:
                    raise ValueError(
                        f"Inconsistent target_fodf_sh coefficient count: "
                        f"{key} has {n_coeffs}, expected {expected_fodf_n_coeffs}."
                    )
            elif expected_fodf_n_coeffs is not None:
                raise ValueError(
                    f"{key} is missing target_fodf_sh but other subjects provide it."
                )

            for axis in self.slice_axes:
                n_slices = target_dwi_arr.shape[axis]
                for s in range(n_slices):
                    self.samples.append((key, axis, s))
                other = [d for i, d in enumerate((X, Y, Z)) if i != axis]
                max_h = max(max_h, other[0])
                max_w = max(max_w, other[1])

            if target_dwi_for_mask is not None:
                del target_dwi_for_mask

        self.canonical_hw = (
            tuple(canonical_hw) if canonical_hw is not None else (max_h, max_w)
        )
        self.has_fodf = expected_fodf_n_coeffs is not None
        self.stored_fodf_n_coeffs = expected_fodf_n_coeffs or 0
        self.fodf_n_coeffs = self.stored_fodf_n_coeffs
        if self.has_fodf:
            stored_order = _infer_sh_order(self.stored_fodf_n_coeffs)
            if self.target_fodf_sh_order is None:
                self.target_fodf_sh_order = stored_order
            else:
                requested_coeffs = _sh_n_coeffs(int(self.target_fodf_sh_order))
                if requested_coeffs > self.stored_fodf_n_coeffs:
                    raise ValueError(
                        f"Requested fODF training order l={self.target_fodf_sh_order} "
                        f"requires {requested_coeffs} coefficients, but the dataset only "
                        f"stores {self.stored_fodf_n_coeffs} (l={stored_order})."
                    )
                self.fodf_n_coeffs = requested_coeffs
                if self.fodf_n_coeffs < self.stored_fodf_n_coeffs:
                    log.info(
                        "Training on truncated fODF SH order l=%d (%d/%d coeffs).",
                        self.target_fodf_sh_order,
                        self.fodf_n_coeffs,
                        self.stored_fodf_n_coeffs,
                    )
        if self.has_fodf and self.augment and self.aug_flip:
            log.warning(
                "Disabling aug_flip for %s because mirrored fODF SH targets would "
                "need an SH-basis reflection transform that is not applied here.",
                self.zarr_path,
            )
            self.aug_flip = False

        self.max_bval = global_max_bval if global_max_bval > 0 else 1.0
        # No DTI head trains here; keep the attribute so checkpoints / shared
        # train.py code referencing dti_scale don't break.
        self.dti_scale = 1.0

        if preload:
            preload_dataset_in_worker(self, preload_fodf=True)

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key, axis, s = self.samples[idx]
        d = self._data[key]
        bmask_3d = self._brain_masks[key]

        bvals = d["bvals"]   # (N,)
        bvecs = d["bvecs"]   # (3, N)

        # Workers set self._preloaded via worker_init_fn (see
        # preload_dataset_in_worker below). RAM access is ~200× faster than
        # zarr chunk reads; fall back to lazy zarr in the main process.
        preloaded = getattr(self, "_preloaded", None)

        if preloaded is not None:
            arr_dwi = preloaded[key]["target_dwi"]
            clean_signal = _extract_dwi_context_chw(
                arr_dwi, axis, s, self.context_slices,
            )
            arr_fodf = preloaded[key].get("target_fodf_sh")
            if arr_fodf is not None:
                tgt_fodf_chw = _slice_to_chw(arr_fodf, axis, s)
            elif self.has_fodf:
                # PRELOAD_FODF=False intentionally avoids caching SH targets in RAM.
                # Fall back to lazy zarr reads so sample dicts still include the
                # supervision tensor expected by the fODF training loop.
                tgt_fodf_chw = _slice_to_chw(
                    _get_zarr_group(self.zarr_path)[key]["target_fodf_sh"],
                    axis,
                    s,
                )
            else:
                tgt_fodf_chw = None
        else:
            grp = _get_zarr_group(self.zarr_path)[key]
            clean_signal = _extract_dwi_context_chw(
                grp["target_dwi"], axis, s, self.context_slices,
            )
            tgt_fodf_chw = (
                _slice_to_chw(grp["target_fodf_sh"], axis, s)
                if self.has_fodf
                else None
            )

        if tgt_fodf_chw is not None and self.fodf_n_coeffs > 0:
            tgt_fodf_chw = tgt_fodf_chw[: self.fodf_n_coeffs]

        bmask = (
            _spatial_slice(bmask_3d, axis, s) if bmask_3d is not None else None
        )
        if bmask is not None:
            bmask_hw = np.ascontiguousarray(bmask, dtype=np.float32)
        else:
            bmask_hw = np.ones(clean_signal.shape[-2:], dtype=np.float32)

        N = bvals.shape[0]
        # Cached bvals/bvecs are shared across workers — copy before any
        # modification so we never alias them in returned tensors.
        bvals_n = bvals.copy()
        bvecs_n = bvecs.copy()

        # ── On-the-fly degradation (fresh noise and cutout each call) ───────
        degrade_kf = degrade_nl = None
        b0_idx = bvals_n < self.b0_threshold
        if self.on_the_fly_degradation:
            if self.eval_mode:
                kf = self.eval_keep_fraction
                nl = self.eval_noise_level
                rng = np.random.default_rng(self.eval_seed + idx)
                input_signal = _degrade_model_signal(
                    clean_signal, kf, nl, rng,
                    noise_distribution=self.noise_distribution,
                    n_coils=self.n_coils,
                )
            elif self.gpu_degrade:
                # Degradation deferred to GPU in run_epoch (cuFFT, ~50x faster).
                # Return clean signal here; batch dict carries the params needed.
                degrade_kf = float(np.random.uniform(*self.keep_fraction_range))
                degrade_nl = float(np.random.uniform(*self.noise_range))
                input_signal = clean_signal
            else:
                kf = float(np.random.uniform(*self.keep_fraction_range))
                nl = float(np.random.uniform(*self.noise_range))
                rng = np.random.default_rng()
                input_signal = _degrade_model_signal(
                    clean_signal, kf, nl, rng,
                    noise_distribution=self.noise_distribution,
                    n_coils=self.n_coils,
                )
        else:
            input_signal = clean_signal

        # ── Normalise input by the mean-b0 of the (noisy) input ────────────
        # Skipped in gpu_degrade mode: run_epoch applies gpu_b0_normalize_batch
        # after cuFFT degradation so normalization is over the degraded signal.
        if not self.gpu_degrade:
            if input_signal is clean_signal:
                input_signal = clean_signal.copy()
            input_signal = _b0_normalize_model_signal(input_signal, b0_idx)

        bvals_norm = bvals_n / self.max_bval

        # ── Spatial + intensity augmentations (training only) ────────────────
        if self.augment:
            if self.aug_flip:
                if random.random() > 0.5:
                    if input_signal.ndim == 3:
                        input_signal = input_signal[:, ::-1, :]
                    else:
                        input_signal = input_signal[:, :, ::-1, :]
                    if tgt_fodf_chw is not None:
                        tgt_fodf_chw = tgt_fodf_chw[:, ::-1, :]
                    bmask_hw = bmask_hw[::-1, :]
                    # h-flip flips one world axis; mirror the matching bvec component.
                    h_world = (1, 0, 0)[axis]
                    bvecs_n = flip_bvecs(bvecs_n, world_axis=h_world)
                if random.random() > 0.5:
                    if input_signal.ndim == 3:
                        input_signal = input_signal[:, :, ::-1]
                    else:
                        input_signal = input_signal[:, :, :, ::-1]
                    if tgt_fodf_chw is not None:
                        tgt_fodf_chw = tgt_fodf_chw[:, :, ::-1]
                    bmask_hw = bmask_hw[:, ::-1]
                    w_world = (2, 2, 1)[axis]
                    bvecs_n = flip_bvecs(bvecs_n, world_axis=w_world)
                input_signal = np.ascontiguousarray(input_signal)
                if tgt_fodf_chw is not None:
                    tgt_fodf_chw = np.ascontiguousarray(tgt_fodf_chw)
                bmask_hw = np.ascontiguousarray(bmask_hw)
            if self.aug_intensity > 0.0:
                scale = np.float32(
                    1.0 + np.random.uniform(-self.aug_intensity, self.aug_intensity)
                )
                if input_signal is clean_signal:
                    input_signal = clean_signal.copy()
                input_signal *= scale
            if self.aug_volume_dropout > 0.0:
                drop = np.random.random(N) < self.aug_volume_dropout
                if drop.any():
                    if input_signal is clean_signal:
                        input_signal = clean_signal.copy()
                    input_signal[drop] = 0.0
                    dropped_volumes = drop
                else:
                    dropped_volumes = None
            else:
                dropped_volumes = None
            if self.aug_bvec_mask_prob > 0.0 and random.random() < self.aug_bvec_mask_prob:
                dw_idx = np.where(~b0_idx)[0]
                if len(dw_idx) > 0:
                    keep_frac = np.random.uniform(
                        self.aug_bvec_mask_min_keep, self.aug_bvec_mask_max_keep
                    )
                    n_keep = max(1, int(round(len(dw_idx) * keep_frac)))
                    keep_set = np.random.choice(dw_idx, size=n_keep, replace=False)
                    bvec_drop = np.zeros(N, dtype=bool)
                    bvec_drop[dw_idx] = True
                    bvec_drop[keep_set] = False
                    if bvec_drop.any():
                        if input_signal is clean_signal:
                            input_signal = clean_signal.copy()
                        input_signal[bvec_drop] = 0.0
                        dropped_volumes = (
                            bvec_drop if dropped_volumes is None
                            else dropped_volumes | bvec_drop
                        )
        else:
            dropped_volumes = None

        # ── Deterministic degradation at eval time (volume dropout + bvec mask) ─
        if self.eval_mode:
            if self.aug_volume_dropout > 0.0:
                rng_drop = np.random.default_rng(self.eval_seed + idx + 200_000)
                drop = rng_drop.random(N) < self.aug_volume_dropout
                if drop.any():
                    if input_signal is clean_signal:
                        input_signal = clean_signal.copy()
                    input_signal[drop] = 0.0
                    dropped_volumes = drop
            dw_idx = np.where(~b0_idx)[0]
            if len(dw_idx) > 0:
                rng_mask = np.random.default_rng(self.eval_seed + idx + 100_000)
                n_keep = max(1, int(round(len(dw_idx) * self.eval_bvec_mask_keep)))
                keep_set = rng_mask.choice(dw_idx, size=n_keep, replace=False)
                bvec_drop = np.zeros(N, dtype=bool)
                bvec_drop[dw_idx] = True
                bvec_drop[keep_set] = False
                if bvec_drop.any():
                    if input_signal is clean_signal:
                        input_signal = clean_signal.copy()
                    input_signal[bvec_drop] = 0.0
                    dropped_volumes = (
                        bvec_drop if dropped_volumes is None
                        else dropped_volumes | bvec_drop
                    )

        # ── Pad diffusion + spatial dims in one allocation per array ────────
        n_pad = self.max_n - N
        ch, cw = self.canonical_hw
        h, w = input_signal.shape[-2:]
        ph = ch - h
        pw = cw - w

        input_signal = _pad_signal(input_signal, n_pad, ph, pw)
        if n_pad > 0:
            bvals_norm = np.pad(bvals_norm, (0, n_pad))
            bvecs_n = np.pad(bvecs_n, ((0, 0), (0, n_pad)))
        if tgt_fodf_chw is not None and (ph > 0 or pw > 0):
            tgt_fodf_chw = np.pad(tgt_fodf_chw, ((0, 0), (0, ph), (0, pw)))
        if ph > 0 or pw > 0:
            bmask_hw = np.pad(bmask_hw, ((0, ph), (0, pw)))

        vol_mask = np.zeros(self.max_n, dtype=np.float32)
        vol_mask[:N] = 1.0
        if dropped_volumes is not None:
            vol_mask[:N][dropped_volumes] = 0.0

        result = {
            "input": torch.from_numpy(np.ascontiguousarray(input_signal)),
            "bvals": torch.from_numpy(np.ascontiguousarray(bvals_norm, dtype=np.float32)),
            "bvecs": torch.from_numpy(np.ascontiguousarray(bvecs_n, dtype=np.float32)),
            "vol_mask": torch.from_numpy(vol_mask),
            "brain_mask": torch.from_numpy(np.ascontiguousarray(bmask_hw)),
        }
        if tgt_fodf_chw is not None:
            result["target_fodf"] = torch.from_numpy(np.ascontiguousarray(tgt_fodf_chw))
        if degrade_kf is not None:
            # b0_mask padded to max_n; run_epoch uses it for GPU b0 normalization.
            b0_mask_full = np.zeros(self.max_n, dtype=bool)
            b0_mask_full[:N] = b0_idx
            result["degrade_kf"] = torch.tensor(degrade_kf, dtype=torch.float32)
            result["degrade_nl"] = torch.tensor(degrade_nl, dtype=torch.float32)
            result["b0_mask"] = torch.from_numpy(b0_mask_full)
        return result


def dwi_worker_init(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: pre-load zarr data into this worker's RAM.

    Workers spawn fresh processes (Windows) or fork (Linux/macOS). In either
    case the dataset object carried into the worker contains only metadata
    (~10 MB). This function then loads the full DWI / SH arrays from zarr into
    that worker's own RAM so subsequent ``__getitem__`` calls run at RAM speed
    rather than hitting zarr every sample.

    Pass directly to DataLoader::

        DataLoader(ds, ..., worker_init_fn=dwi_worker_init)
    """
    import torch.utils.data as _tud
    info = _tud.get_worker_info()
    if info is not None:
        preload_dataset_in_worker(info.dataset, preload_fodf=cfg.PRELOAD_FODF)


def preload_dataset_in_worker(
    dataset: DWISliceDataset,
    preload_fodf: bool = True,
) -> None:
    """Load all subject arrays into this process's RAM.

    Call from DataLoader's ``worker_init_fn`` so each worker has in-memory
    access to the data instead of making zarr chunk reads per sample. Also
    usable from the main process (``preload=True`` on the dataset) for
    ``num_workers=0`` runs — most notably MPS, where the GPU degrade path is
    disabled and CPU FFT throughput dominates loader time.

    Set ``preload_fodf=False`` to skip caching ``target_fodf_sh`` — useful
    when RAM is tight, since the OS filesystem cache keeps zarr reads fast
    after the first epoch anyway.
    """
    store = zarr.open_group(dataset.zarr_path, mode="r")
    preloaded: dict[str, dict[str, np.ndarray]] = {}
    for key in dataset.subject_keys:
        grp = store[key]
        entry: dict[str, np.ndarray] = {
            "target_dwi": np.asarray(grp["target_dwi"][:], dtype=np.float32),
        }
        if preload_fodf and "target_fodf_sh" in set(grp.array_keys()):
            entry["target_fodf_sh"] = np.asarray(grp["target_fodf_sh"][:], dtype=np.float32)
        preloaded[key] = entry
    dataset._preloaded = preloaded
