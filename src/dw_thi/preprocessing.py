"""Preprocessing and dataset-building utilities.

The production Zarr dataset stores clean DWI, fitted clean DTI targets,
gradients, and a precomputed 3D brain mask. The brain mask is generated from
the mean b0 image with DIPY's median_otsu in build_pretext_dataset.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import zarr
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
from tqdm import tqdm

import config as cfg
from .augment import degrade_dwi_volume


def parse_dwi_entities(dwi_path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse BIDS-style subject/session/run entities from a DWI path."""
    basename = os.path.basename(os.fspath(dwi_path))
    stem = basename.removesuffix(".nii.gz")

    bids_stem = stem.split("__")[-1]
    if bids_stem.endswith("_dwi"):
        bids_stem = bids_stem[:-4]

    entities: dict[str, str] = {}
    key_parts: list[str] = []
    for part in bids_stem.split("_"):
        if "-" not in part:
            continue
        name, _ = part.split("-", 1)
        if name in {"sub", "ses", "run", "acq", "dir"}:
            entities[name] = part
            key_parts.append(part)

    if "sub" not in entities:
        raise ValueError(f"Could not parse subject from DWI filename: {basename}")

    key = "_".join(key_parts) if key_parts else entities["sub"]
    return {
        "subject": entities["sub"],
        "session": entities.get("ses", ""),
        "run": entities.get("run", ""),
        "key": key,
    }


def find_dwi_datasets(root_dir: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Find DWI NIfTI files with matching bval/bvec sidecars."""
    dwi_files = glob.glob(os.path.join(os.fspath(root_dir), "*_dwi.nii.gz"))
    datasets: list[dict[str, str]] = []

    for dwi_path in dwi_files:
        base = dwi_path.replace(".nii.gz", "")
        bval_path = base + ".bval"
        bvec_path = base + ".bvec"

        if os.path.exists(bval_path) and os.path.exists(bvec_path):
            datasets.append(
                {
                    "dwi": dwi_path,
                    "bval": bval_path,
                    "bvec": bvec_path,
                    **parse_dwi_entities(dwi_path),
                }
            )
        else:
            print(f"Missing gradients for {dwi_path}")

    return datasets


def load_dwi_dataset(entry: dict[str, str]) -> dict[str, object]:
    """Load a DWI NIfTI plus gradient files."""
    img = nib.load(entry["dwi"])
    data = img.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(entry["bval"], entry["bvec"])
    bvals = np.asarray(bvals, dtype=np.float32)
    bvecs = np.asarray(bvecs, dtype=np.float32)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    gtab = gradient_table(bvals, bvecs=bvecs.T, b0_threshold=cfg.B0_THRESHOLD)
    return {
        "data": data,
        "affine": img.affine,
        "bvals": bvals,
        "bvecs": bvecs,
        "gtab": gtab,
        "path": entry["dwi"],
    }


def compute_dti(data: np.ndarray, gtab, mask: np.ndarray | None = None) -> np.ndarray:
    """Fit a DTI tensor and return the full 3x3 quadratic form."""
    tenmodel = TensorModel(gtab, fit_method=cfg.DTI_FIT_METHOD)
    tenfit = tenmodel.fit(data, mask=mask)
    return tenfit.quadratic_form.astype(np.float32)


def tensor_to_6d(tensor: np.ndarray) -> np.ndarray:
    """Convert a symmetric 3x3 tensor field to Dxx,Dxy,Dyy,Dxz,Dyz,Dzz order."""
    return np.stack(
        [
            tensor[..., 0, 0],
            tensor[..., 0, 1],
            tensor[..., 1, 1],
            tensor[..., 0, 2],
            tensor[..., 1, 2],
            tensor[..., 2, 2],
        ],
        axis=-1,
    ).astype(np.float32)


def compute_b0_norm(mean_b0_slice: np.ndarray) -> float:
    """Return a robust scalar normalization factor from one mean-b0 slice."""
    threshold = 0.1 * float(np.max(mean_b0_slice))
    brain_voxels = mean_b0_slice[mean_b0_slice > threshold]
    return float(brain_voxels.mean()) if brain_voxels.size > 0 else 1.0


def mean_b0_volume(
    dwi_4d: np.ndarray,
    bvals: np.ndarray,
    b0_threshold: float = cfg.B0_THRESHOLD,
) -> np.ndarray:
    """Compute a 3D reference volume from b0 images, falling back to all volumes."""
    b0_idx = np.asarray(bvals) < b0_threshold
    if b0_idx.any():
        return dwi_4d[..., b0_idx].mean(axis=-1).astype(np.float32)
    return dwi_4d.mean(axis=-1).astype(np.float32)


def compute_brain_mask_from_dwi(
    dwi_4d: np.ndarray,
    bvals: np.ndarray,
    b0_threshold: float = cfg.B0_THRESHOLD,
    median_radius: int = cfg.BRAIN_MASK_MEDIAN_RADIUS,
    numpass: int = cfg.BRAIN_MASK_NUMPASS,
    dilate: int | None = cfg.BRAIN_MASK_DILATE,
    finalize_mask: bool = cfg.BRAIN_MASK_FINALIZE,
) -> np.ndarray:
    """Compute a 3D brain mask from mean-b0 signal using DIPY median_otsu."""
    ref = mean_b0_volume(dwi_4d, bvals, b0_threshold=b0_threshold)
    if not np.isfinite(ref).all():
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)

    if float(ref.max() - ref.min()) <= 1e-8:
        return np.ones(ref.shape, dtype=bool)

    _, mask = median_otsu(
        ref,
        median_radius=int(median_radius),
        numpass=int(numpass),
        autocrop=False,
        dilate=dilate,
        finalize_mask=bool(finalize_mask),
    )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != ref.shape or not mask.any():
        return ref > (0.1 * float(ref.max()))
    return mask


def show_kspace(img: np.ndarray) -> np.ndarray:
    k = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(k)).astype(np.float32)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    return (x - vmin) / (vmax - vmin + 1e-8)


def _normalize_pair_01(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([a, b], axis=0).astype(np.float32)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    scale = vmax - vmin + 1e-8
    return (a - vmin) / scale, (b - vmin) / scale


def save_qc_plot(
    subject_id: str,
    clean_dwi: np.ndarray,
    degraded_dwi: np.ndarray,
    brain_mask: np.ndarray,
    bvals: np.ndarray,
    output_dir: Path,
    keep_fraction: float,
    noise_level: float,
) -> None:
    """Save a compact build-time QC plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    z = clean_dwi.shape[2] // 2
    non_b0 = np.where(bvals >= cfg.B0_THRESHOLD)[0]
    b = int(non_b0[0]) if non_b0.size > 0 else 0
    b0_idx = np.where(bvals < cfg.B0_THRESHOLD)[0]
    b0_vol = int(b0_idx[0]) if b0_idx.size > 0 else b

    clean_slice, degraded_slice = _normalize_pair_01(
        clean_dwi[:, :, z, b],
        degraded_dwi[:, :, z, b],
    )
    clean_kspace = np.rot90(show_kspace(clean_slice), 1)
    degraded_kspace = np.rot90(show_kspace(degraded_slice), 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    axes[0, 0].imshow(np.rot90(_normalize_01(clean_dwi[:, :, z, b0_vol]), 1), cmap="gray")
    axes[0, 0].set_title("b0 image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.rot90(brain_mask[:, :, z].astype(np.float32), 1), cmap="gray")
    axes[0, 1].set_title("median_otsu brain mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.rot90(clean_slice, 1), cmap="gray")
    axes[0, 2].set_title("Clean DWI")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(np.rot90(degraded_slice, 1), cmap="gray")
    axes[1, 0].set_title(f"Preview degraded (kf={keep_fraction}, n={noise_level})")
    axes[1, 0].axis("off")

    diff = np.rot90(clean_slice - degraded_slice, 1)
    im = axes[1, 1].imshow(diff, cmap="bwr")
    axes[1, 1].set_title("DWI difference")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(clean_kspace, cmap="gray")
    axes[1, 2].set_title("Clean k-space")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(degraded_kspace, cmap="gray")
    axes[2, 0].set_title("Degraded k-space")
    axes[2, 0].axis("off")

    axes[2, 1].plot(bvals, ".")
    axes[2, 1].set_title("b-values")
    axes[2, 1].set_xlabel("Volume index")
    axes[2, 1].set_ylabel("b-value")

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        0.95,
        (
            f"subject: {subject_id}\n"
            f"shape: {clean_dwi.shape}\n"
            f"slice z={z}, volume={b}\n"
            f"mask voxels: {int(brain_mask.sum())}\n"
            f"median_radius={cfg.BRAIN_MASK_MEDIAN_RADIUS}, numpass={cfg.BRAIN_MASK_NUMPASS}\n"
            f"training degradation is sampled on the fly"
        ),
        va="top",
        fontsize=10,
    )

    fig.suptitle(f"QC sample: {subject_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject_id}_qc.png", dpi=140)
    plt.close(fig)


def validate_store(store: zarr.Group) -> None:
    """Validate the production Zarr contract."""
    required_keys = {"target_dwi", "target_dti_6d", "bvals", "bvecs", "brain_mask"}
    for subject_id in sorted(store.group_keys()):
        group = store[subject_id]
        missing = required_keys.difference(set(group.array_keys()))
        if missing:
            raise ValueError(f"{subject_id} missing arrays: {sorted(missing)}")

        target_shape = group["target_dwi"].shape
        tensor_shape = group["target_dti_6d"].shape
        mask_shape = group["brain_mask"].shape

        if len(tensor_shape) != 4 or tensor_shape[:3] != target_shape[:3] or tensor_shape[-1:] != (6,):
            raise ValueError(f"{subject_id} invalid target_dti_6d shape: {tensor_shape}")
        if mask_shape != target_shape[:3]:
            raise ValueError(f"{subject_id} invalid brain_mask shape: {mask_shape}")

        for key in required_keys:
            arr = group[key][:]
            if not np.isfinite(arr).all():
                raise ValueError(f"{subject_id} contains non-finite values in {key}")
        if not np.asarray(group["brain_mask"][:], dtype=bool).any():
            raise ValueError(f"{subject_id} brain_mask is empty")


def validate_unique_subject_keys(entries: list[dict[str, str]]) -> None:
    paths_by_key: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        paths_by_key[entry["key"]].append(entry["dwi"])

    duplicates = {key: paths for key, paths in paths_by_key.items() if len(paths) > 1}
    if not duplicates:
        return

    lines = ["Duplicate Zarr group keys found. Conflicting DWI files:"]
    for key, paths in sorted(duplicates.items()):
        lines.append(f"  {key}:")
        lines.extend(f"    - {path}" for path in sorted(paths))
    raise ValueError("\n".join(lines))


def build_pretext_dataset(args: argparse.Namespace) -> dict[str, object]:
    """Build the clean production Zarr dataset from raw DWI files."""
    entries = sorted(find_dwi_datasets(args.data_dir), key=lambda d: d["dwi"])
    if args.max_subjects is not None:
        entries = entries[: args.max_subjects]

    if not entries:
        raise FileNotFoundError(f"No DWI datasets found in: {args.data_dir}")
    validate_unique_subject_keys(entries)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(output_path), mode="w")
    store.attrs["format_version"] = 3
    store.attrs["source_data_dir"] = str(Path(args.data_dir).resolve())
    store.attrs["degradation"] = "on_the_fly"
    store.attrs["brain_mask"] = {
        "source": "mean_b0",
        "method": "dipy.segment.mask.median_otsu",
        "median_radius": cfg.BRAIN_MASK_MEDIAN_RADIUS,
        "numpass": cfg.BRAIN_MASK_NUMPASS,
        "dilate": cfg.BRAIN_MASK_DILATE,
        "finalize_mask": cfg.BRAIN_MASK_FINALIZE,
    }
    store.attrs["degradation_ranges"] = {
        "keep_fraction": [cfg.KEEP_FRACTION_MIN, cfg.KEEP_FRACTION_MAX],
        "noise_level": [cfg.NOISE_MIN, cfg.NOISE_MAX],
    }

    print(f"Found {len(entries)} subject entries")
    for idx, entry in enumerate(tqdm(entries, desc="Building Zarr dataset")):
        sample = load_dwi_dataset(entry)
        clean_dwi = np.asarray(sample["data"], dtype=np.float32)
        bvals = np.asarray(sample["bvals"], dtype=np.float32)
        bvecs = np.asarray(sample["bvecs"], dtype=np.float32)

        brain_mask = compute_brain_mask_from_dwi(clean_dwi, bvals)
        tensor_clean_6d = tensor_to_6d(compute_dti(clean_dwi, sample["gtab"], mask=brain_mask))

        subject_id = entry["key"]
        group = store.create_group(subject_id)
        group.attrs["source_dwi"] = entry["dwi"]
        group.attrs["original_subject"] = entry["subject"]
        group.attrs["original_session"] = entry["session"]
        group.attrs["original_run"] = entry["run"]

        group.create_array("target_dwi", data=clean_dwi)
        group.create_array("target_dti_6d", data=tensor_clean_6d)
        group.create_array("brain_mask", data=brain_mask.astype(np.uint8))
        group.create_array("bvals", data=bvals)
        group.create_array("bvecs", data=bvecs)

        if idx < args.plot_subjects:
            degraded_preview = degrade_dwi_volume(
                clean_dwi,
                keep_fraction=args.plot_keep_fraction,
                rel_noise_level=args.plot_noise_level,
                seed=cfg.EVAL_DEGRADE_SEED,
            )
            save_qc_plot(
                subject_id=subject_id,
                clean_dwi=clean_dwi,
                degraded_dwi=degraded_preview,
                brain_mask=brain_mask,
                bvals=bvals,
                output_dir=Path(args.plot_dir),
                keep_fraction=args.plot_keep_fraction,
                noise_level=args.plot_noise_level,
            )

    validate_store(store)

    summary = {
        "output": str(output_path.resolve()),
        "subjects": len(list(store.group_keys())),
        "qc_plot_dir": str(Path(args.plot_dir).resolve()) if args.plot_subjects > 0 else None,
    }
    print("Build complete")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the clean production Zarr dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing DWI NIfTI + bval/bvec files")
    parser.add_argument("--output", type=str, default=cfg.DATASET_ZARR_PATH, help="Output Zarr path")
    parser.add_argument("--plot_subjects", type=int, default=3, help="Number of first subjects to export QC plots for")
    parser.add_argument("--plot_dir", type=str, default=cfg.DATASET_QC_DIR, help="Directory to store QC plot PNGs")
    parser.add_argument("--plot_keep_fraction", type=float, default=cfg.EVAL_KEEP_FRACTION)
    parser.add_argument("--plot_noise_level", type=float, default=cfg.EVAL_NOISE_LEVEL)
    parser.add_argument("--max_subjects", type=int, default=None, help="Optional cap for quick test runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    build_pretext_dataset(args)


if __name__ == "__main__":
    main()
