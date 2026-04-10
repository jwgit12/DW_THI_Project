#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from tqdm import tqdm

from functions import compute_dti, find_dwi_datasets, load_dwi_dataset, lowres_noise, show_kspace, tensor_to_6d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Zarr pretext dataset from DWI NIfTI files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing *_dwi.nii.gz + .bval/.bvec files")
    parser.add_argument("--output", type=str, required=True, help="Output Zarr path (e.g. dataset/pretext_dataset.zarr)")
    parser.add_argument("--keep_fraction", type=float, default=0.6, help="Central k-space keep fraction for low-res degradation")
    parser.add_argument("--noise_min", type=float, default=0.01, help="Minimum relative Gaussian noise level")
    parser.add_argument("--noise_max", type=float, default=0.10, help="Maximum relative Gaussian noise level")
    parser.add_argument("--plot_subjects", type=int, default=3, help="Number of first subjects to export QC plots for")
    parser.add_argument("--plot_dir", type=str, default="dataset/pretext_dataset_qc", help="Directory to store QC plot PNGs")
    parser.add_argument("--max_subjects", type=int, default=None, help="Optional cap for quick test runs")
    return parser.parse_args()


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
    bvals: np.ndarray,
    output_dir: Path,
    keep_fraction: float,
    noise_min: float,
    noise_max: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    z = clean_dwi.shape[2] // 2
    non_b0 = np.where(bvals >= 50)[0]
    b = int(non_b0[0]) if non_b0.size > 0 else 0

    clean_slice, degraded_slice = _normalize_pair_01(
        clean_dwi[:, :, z, b],
        degraded_dwi[:, :, z, b],
    )
    clean_kspace = np.rot90(show_kspace(clean_slice), 1)
    degraded_kspace = np.rot90(show_kspace(degraded_slice), 1)
    diff_kspace = clean_kspace - degraded_kspace

    b0_idx = np.where(bvals < 50)[0]
    b0_vol = int(b0_idx[0]) if b0_idx.size > 0 else b

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    axes[0, 0].imshow(np.rot90(_normalize_01(clean_dwi[:, :, z, b0_vol]), 1), cmap="gray")
    axes[0, 0].set_title("b0 image example slice")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.rot90(clean_slice, 1), cmap="gray")
    axes[0, 1].set_title("Ground Truth DWI")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.rot90(degraded_slice, 1), cmap="gray")
    axes[0, 2].set_title("Low-res + Noisy")
    axes[0, 2].axis("off")

    diff = np.rot90(clean_slice - degraded_slice, 1)
    im = axes[1, 0].imshow(diff, cmap="bwr")
    axes[1, 0].set_title("Difference map")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(clean_kspace, cmap="gray")
    axes[1, 1].set_title("Ground Truth k-space")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(degraded_kspace, cmap="gray")
    axes[1, 2].set_title("Low-res + Noisy k-space")
    axes[1, 2].axis("off")

    k_im = axes[2, 0].imshow(diff_kspace, cmap="gray")
    axes[2, 0].set_title("Difference k-space")
    axes[2, 0].axis("off")
    fig.colorbar(k_im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    axes[2, 1].plot(bvals, ".")
    axes[2, 1].set_title("b-values per volume")
    axes[2, 1].set_xlabel("Volume index")
    axes[2, 1].set_ylabel("b-value")

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        0.95,
        (
            f"subject: {subject_id}\n"
            f"shape: {clean_dwi.shape}\n"
            f"slice z={z}, volume b={b}\n"
            f"keep_fraction={keep_fraction}\n"
            f"noise=[{noise_min}, {noise_max}]\n"
        ),
        va="top",
        fontsize=10,
    )

    fig.suptitle(f"QC sample: {subject_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject_id}_qc.png", dpi=140)
    plt.close(fig)


def validate_store(store: zarr.Group) -> None:
    required_keys = {"input_dwi", "target_dwi", "target_dti_6d", "bvals", "bvecs"}
    for subject_id in sorted(store.group_keys()):
        group = store[subject_id]
        missing = required_keys.difference(set(group.array_keys()))
        if missing:
            raise ValueError(f"{subject_id} missing arrays: {sorted(missing)}")

        input_shape = group["input_dwi"].shape
        target_shape = group["target_dwi"].shape
        tensor_arr = group["target_dti_6d"]
        tensor_shape = tensor_arr.shape

        if input_shape != target_shape:
            raise ValueError(f"{subject_id} input/target shape mismatch: {input_shape} vs {target_shape}")
        if tensor_arr.ndim != 4:
            raise ValueError(f"{subject_id} invalid target_dti_6d shape: {tensor_shape}")
        if tensor_shape[:3] != input_shape[:3] or tensor_shape[-1:] != (6,):
            raise ValueError(f"{subject_id} invalid target_dti_6d shape: {tensor_shape}")

        for key in required_keys:
            arr = group[key][:]
            if not np.isfinite(arr).all():
                raise ValueError(f"{subject_id} contains non-finite values in {key}")


def main() -> None:
    args = parse_args()

    entries = sorted(find_dwi_datasets(args.data_dir), key=lambda d: d["dwi"])
    if args.max_subjects is not None:
        entries = entries[: args.max_subjects]

    if not entries:
        raise FileNotFoundError(f"No DWI datasets found in: {args.data_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(output_path), mode="w")
    store.attrs["format_version"] = 1
    store.attrs["source_data_dir"] = str(Path(args.data_dir).resolve())
    store.attrs["degradation"] = {
        "keep_fraction": args.keep_fraction,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }

    print(f"Found {len(entries)} subject entries")
    for idx, entry in enumerate(tqdm(entries, desc="Building Zarr dataset")):
        sample = load_dwi_dataset(entry)
        clean_dwi = sample["data"].astype(np.float32)

        degraded_dwi = lowres_noise(
            clean_dwi,
            keep_fraction=args.keep_fraction,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
        ).astype(np.float32)

        tensor_clean_6d = tensor_to_6d(compute_dti(clean_dwi, sample["gtab"])).astype(np.float32)

        subject_id = f"subject_{idx:03d}"
        group = store.create_group(subject_id)
        group.attrs["source_dwi"] = entry["dwi"]

        group.create_array("input_dwi", data=degraded_dwi)
        group.create_array("target_dwi", data=clean_dwi)
        group.create_array("target_dti_6d", data=tensor_clean_6d)
        group.create_array("bvals", data=sample["bvals"].astype(np.float32))
        group.create_array("bvecs", data=sample["bvecs"].astype(np.float32))

        if idx < args.plot_subjects:
            save_qc_plot(
                subject_id=subject_id,
                clean_dwi=clean_dwi,
                degraded_dwi=degraded_dwi,
                bvals=sample["bvals"],
                output_dir=Path(args.plot_dir),
                keep_fraction=args.keep_fraction,
                noise_min=args.noise_min,
                noise_max=args.noise_max,
            )

    validate_store(store)

    summary = {
        "output": str(output_path.resolve()),
        "subjects": len(list(store.group_keys())),
        "qc_plot_dir": str(Path(args.plot_dir).resolve()) if args.plot_subjects > 0 else None,
    }
    print("Build complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
