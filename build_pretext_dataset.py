"""
Build a pretext-task dataset from DWI MRT files.

For each subject in the source directory this script:
  1. Loads the DWI NIfTI volume together with b-values / b-vectors.
  2. Creates a *noisy* version of the DWI volume (k-space under-sampling + Gaussian noise)
     using ``lowres_noise`` from functions.py.
  3. Computes the clean 6-component DTI tensor image (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)
     using dipy's TensorModel and ``tensor_to_6d``.
  4. Stores everything in a Zarr store for fast random access during training.

Zarr layout
-----------
pretext_dataset.zarr/
├── subject_000/
│   ├── input_dwi        (X, Y, Z, N)  float32 – noisy DWI
│   ├── bvals             (N,)          float32
│   ├── bvecs             (3, N)        float32
│   ├── target_dti_6d     (X, Y, Z, 6) float32 – clean DTI 6-component
│   └── target_dwi        (X, Y, Z, N) float32 – clean DWI
├── subject_001/
│   └── …
└── attrs: {num_subjects, keep_fraction, noise_min, noise_max}
"""

import argparse
import sys
import time

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

# Re-use the helpers already defined in the project
from functions import (
    find_dwi_datasets,
    load_dwi_dataset,
    lowres_noise,
    compute_dti,
    tensor_to_6d,
)


def _optimal_chunks(shape: tuple, last_dim: int | None = None) -> tuple:
    """Return reasonable chunk sizes for a 4-D (or 3-D + channels) array.

    Strategy: keep full X×Y slices together and chunk along Z so that each
    chunk is roughly 4–16 MB (good default for Zarr / training data loaders).
    """
    x, y, z = shape[:3]
    c = shape[3] if len(shape) == 4 else 1

    # target ~ 8 MB per chunk  (float32 → 4 bytes)
    target_bytes = 8 * 1024 * 1024
    slice_bytes = x * y * c * 4  # one Z-slice
    z_chunk = max(1, min(z, target_bytes // slice_bytes))

    if len(shape) == 4:
        return (x, y, z_chunk, c)
    return (x, y, z_chunk)


def build_dataset(
    data_dir: str,
    output_path: str,
    keep_fraction: float = 0.5,
    noise_min: float = 0.01,
    noise_max: float = 0.05,
) -> None:
    """Build the pretext-task Zarr dataset."""

    entries = find_dwi_datasets(data_dir)
    if not entries:
        print(f"[ERROR] No DWI datasets found in '{data_dir}'.")
        sys.exit(1)

    print(f"Found {len(entries)} DWI dataset(s) in '{data_dir}'.")

    store = zarr.open(output_path, mode="w")
    store.attrs["num_subjects"] = len(entries)
    store.attrs["keep_fraction"] = keep_fraction
    store.attrs["noise_min"] = noise_min
    store.attrs["noise_max"] = noise_max

    t0 = time.time()

    for idx, entry in enumerate(tqdm(entries, desc="Processing subjects")):
        subject = load_dwi_dataset(entry)
        data: np.ndarray = subject["data"]      # (X, Y, Z, N)
        bvals: np.ndarray = subject["bvals"]    # (N,)
        bvecs: np.ndarray = subject["bvecs"]    # (3, N) or (N, 3) → normalised to (3, N)
        gtab = subject["gtab"]

        # --- 1. Create noisy input ------------------------------------------------
        noisy = lowres_noise(
            data,
            keep_fraction=keep_fraction,
            noise_min=noise_min,
            noise_max=noise_max,
        )

        # --- 2. Compute clean 6-D DTI target --------------------------------------
        tensor = compute_dti(data, gtab)       # (X, Y, Z, 3, 3)
        dti_6d = tensor_to_6d(tensor)          # (X, Y, Z, 6)

        # --- 3. Persist to Zarr ---------------------------------------------------
        grp_name = f"subject_{idx:03d}"
        grp = store.create_group(grp_name)

        _compressors = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]

        grp.create_array(
            "input_dwi",
            data=noisy.astype(np.float32),
            chunks=_optimal_chunks(noisy.shape),
            compressors=_compressors,
        )
        grp.create_array("bvals", data=bvals.astype(np.float32))
        grp.create_array("bvecs", data=bvecs.astype(np.float32))
        grp.create_array(
            "target_dti_6d",
            data=dti_6d.astype(np.float32),
            chunks=_optimal_chunks(dti_6d.shape),
            compressors=_compressors,
        )
        grp.create_array(
            "target_dwi",
            data=data.astype(np.float32),
            chunks=_optimal_chunks(data.shape),
            compressors=_compressors,
        )

        # Store the original path for traceability
        grp.attrs["source_path"] = entry["dwi"]

    elapsed = time.time() - t0

    # ------ Quick sanity check ------------------------------------------------
    print(f"\n✅  Done in {elapsed:.1f}s – wrote {output_path}")
    print(f"    Subjects: {store.attrs['num_subjects']}")
    for name in sorted(store.group_keys()):
        g = store[name]
        print(
            f"    {name}: input_dwi {g['input_dwi'].shape}  "
            f"target_dwi {g['target_dwi'].shape}  "
            f"target_dti_6d {g['target_dti_6d'].shape}  "
            f"bvals {g['bvals'].shape}  bvecs {g['bvecs'].shape}"
        )
        # Stop after showing first three to keep output concise
        if int(name.split("_")[1]) >= 2:
            remaining = store.attrs["num_subjects"] - 3
            if remaining > 0:
                print(f"    … and {remaining} more subject(s).")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Build a pretext-task dataset (Zarr) from DWI MRT files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/dataset_v1",
        help="Directory containing *_dwi.nii.gz / .bval / .bvec files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/pretext_dataset.zarr",
        help="Output path for the Zarr store.",
    )
    parser.add_argument(
        "--keep_fraction",
        type=float,
        default=0.5,
        help="Fraction of k-space to keep during degradation (0–1).",
    )
    parser.add_argument(
        "--noise_min",
        type=float,
        default=0.01,
        help="Minimum noise level (relative to slice max).",
    )
    parser.add_argument(
        "--noise_max",
        type=float,
        default=0.05,
        help="Maximum noise level (relative to slice max).",
    )
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        keep_fraction=args.keep_fraction,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
    )


if __name__ == "__main__":
    main()
