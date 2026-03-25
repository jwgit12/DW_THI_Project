"""
DTI ML Dataset Builder
Loads DWI data, applies degradation, computes DTI targets, and saves an ML-ready
Zarr dataset with one group per subject and one sub-group per Z slice.
Processes subjects sequentially to avoid memory overflows.
Generates QC visualizations in output_dir/qc/.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import zarr
from functions import (
    find_dwi_datasets, load_dwi_dataset, lowres_noise,
    compute_dti, tensor_to_6d, compute_fa_from_tensor6,
    compute_md_from_tensor6, compute_color_fa_from_tensor6,
    brain_mask, norm, split_b0_dwi, show_kspace, radial_profile, add_noise,
    apply_kspace_mask,
)


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "dataset/dataset_v2"
DEFAULT_OUTPUT = "dti_ml_dataset_v2.zarr"
KEEP_FRACTION = 0.6
NOISE_MIN = 0.01
NOISE_MAX = 0.10
VIS_SLICE_FRAC = 0.45          # which axial slice to visualize (fraction of Z)
VIS_VOL = 10                   # which DWI volume to visualize


# ── QC Visualizations ────────────────────────────────────────────────────────

def plot_overview(clean, degraded, tensor6_clean, tensor6_deg, mask, z, vol,
                  subject_id, qc_dir):
    """Single multi-panel QC figure per subject."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(f"Subject {subject_id}", fontsize=16, y=0.98)

    def _show(ax, img, title, cmap="gray"):
        ax.imshow(np.rot90(img, 1), cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 0: DWI slices
    gt_s = clean[:, :, z, vol]
    dg_s = degraded[:, :, z, vol]
    gt_n = (gt_s - gt_s.min()) / (gt_s.max() - gt_s.min() + 1e-8)
    dg_n = (dg_s - dg_s.min()) / (dg_s.max() - dg_s.min() + 1e-8)

    _show(axes[0, 0], gt_n, "Clean DWI")
    _show(axes[0, 1], dg_n, "Degraded DWI")
    im = axes[0, 2].imshow(np.rot90(gt_n - dg_n, 1), cmap="bwr",
                            vmin=-0.3, vmax=0.3)
    axes[0, 2].set_title("Difference", fontsize=10); axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    _show(axes[0, 3], mask[:, :, z].astype(float), "Brain mask")

    # Row 1: DTI metrics (clean)
    fa = compute_fa_from_tensor6(tensor6_clean) * mask
    md = compute_md_from_tensor6(tensor6_clean) * mask
    cfa = compute_color_fa_from_tensor6(tensor6_clean) * mask[..., None]
    _show(axes[1, 0], norm(fa[:, :, z]), "FA (clean)")
    _show(axes[1, 1], norm(md[:, :, z]), "MD (clean)")
    _show(axes[1, 2], np.clip(norm(cfa[:, :, z]), 0, 1), "Color FA (clean)")

    # Row 1 col 3: b-value distribution
    axes[1, 3].axis("on")
    b0, dwi = split_b0_dwi(clean, np.zeros(clean.shape[-1]))  # just for count
    axes[1, 3].text(0.5, 0.5, f"Volumes: {clean.shape[-1]}\nShape: {clean.shape[:3]}",
                    ha="center", va="center", fontsize=12,
                    transform=axes[1, 3].transAxes)
    axes[1, 3].set_title("Info", fontsize=10)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    # Row 2: DTI metrics (degraded)
    fa_d = compute_fa_from_tensor6(tensor6_deg) * mask
    md_d = compute_md_from_tensor6(tensor6_deg) * mask
    cfa_d = compute_color_fa_from_tensor6(tensor6_deg) * mask[..., None]
    _show(axes[2, 0], norm(fa_d[:, :, z]), "FA (degraded)")
    _show(axes[2, 1], norm(md_d[:, :, z]), "MD (degraded)")
    _show(axes[2, 2], np.clip(norm(cfa_d[:, :, z]), 0, 1), "Color FA (degraded)")

    # Row 2 col 3: frequency comparison
    axes[2, 3].axis("on")
    gt_prof = radial_profile(gt_n)
    dg_prof = radial_profile(dg_n)
    axes[2, 3].plot(gt_prof, label="Clean", linewidth=1)
    axes[2, 3].plot(dg_prof, label="Degraded", linewidth=1)
    axes[2, 3].set_yscale("log")
    axes[2, 3].set_xlabel("Spatial freq"); axes[2, 3].set_ylabel("Magnitude")
    axes[2, 3].set_title("Radial spectrum", fontsize=10)
    axes[2, 3].legend(fontsize=8); axes[2, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(qc_dir, f"qc_subject_{subject_id:03d}.png"), dpi=120)
    plt.close(fig)


def plot_dataset_summary(stats, qc_dir):
    """Summary figure across all subjects."""
    fa_means = [s["fa_mean"] for s in stats]
    md_means = [s["md_mean"] for s in stats]
    snr_vals = [s["snr"] for s in stats]
    n_vols   = [s["n_vols"] for s in stats]
    ids = list(range(len(stats)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dataset Summary", fontsize=14)

    axes[0, 0].bar(ids, fa_means, color="steelblue")
    axes[0, 0].set_ylabel("Mean FA"); axes[0, 0].set_title("FA across subjects")

    axes[0, 1].bar(ids, md_means, color="coral")
    axes[0, 1].set_ylabel("Mean MD"); axes[0, 1].set_title("MD across subjects")

    axes[1, 0].bar(ids, snr_vals, color="seagreen")
    axes[1, 0].set_ylabel("SNR (b0)"); axes[1, 0].set_title("SNR across subjects")
    axes[1, 0].set_xlabel("Subject")

    axes[1, 1].bar(ids, n_vols, color="mediumpurple")
    axes[1, 1].set_ylabel("# volumes (after harmonize)"); axes[1, 1].set_title("Volume count")
    axes[1, 1].set_xlabel("Subject")

    plt.tight_layout()
    fig.savefig(os.path.join(qc_dir, "dataset_summary.png"), dpi=120)
    plt.close(fig)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def estimate_snr(data, bvals):
    """Simple SNR estimate: mean(b0 brain) / std(b0 background corner)."""
    b0 = data[..., bvals < 50].mean(axis=-1)
    mid_z = b0.shape[2] // 2
    sl = b0[:, :, mid_z]
    corner = sl[:10, :10]
    signal = np.mean(sl[sl > np.percentile(sl, 50)])
    noise_std = np.std(corner) + 1e-12
    return signal / noise_std


def process_subject(entry, i, root, qc_dir, skip_qc):
    """Process a single subject sequentially and write slices to Zarr."""
    sample = load_dwi_dataset(entry)
    data, bvals, bvecs, gtab = (
        sample["data"], sample["bvals"], sample["bvecs"], sample["gtab"]
    )

    # Brain mask from mean DWI (compute first so DTI can use it)
    _, dwi_vols = split_b0_dwi(data, bvals)
    mean_dwi = np.mean(dwi_vols, axis=-1)
    mask, _ = brain_mask(mean_dwi)

    # Degrade
    degraded = lowres_noise(data, keep_fraction=KEEP_FRACTION,
                            noise_min=NOISE_MIN, noise_max=NOISE_MAX)

    # DTI fit (only within brain mask)
    tensor6_clean = tensor_to_6d(compute_dti(data, gtab, mask=mask))
    tensor6_deg = tensor_to_6d(compute_dti(degraded, gtab, mask=mask))

    # Stats
    fa = compute_fa_from_tensor6(tensor6_clean) * mask
    md = compute_md_from_tensor6(tensor6_clean) * mask
    snr = estimate_snr(data, bvals)
    brain_voxels = mask > 0
    stat = {
        "fa_mean": float(np.mean(fa[brain_voxels])) if brain_voxels.any() else 0,
        "md_mean": float(np.mean(md[brain_voxels])) if brain_voxels.any() else 0,
        "snr": float(snr),
        "n_vols": int(data.shape[-1]),
    }

    # QC visualization
    if not skip_qc:
        z = int(data.shape[2] * VIS_SLICE_FRAC)
        vol = min(VIS_VOL, data.shape[-1] - 1)
        plot_overview(data, degraded, tensor6_clean, tensor6_deg, mask,
                      z, vol, i, qc_dir)

    # ── Write to Zarr per Z slice ────────────────────────────────────────
    subj_grp = root.create_group(f"subject_{i:03d}", overwrite=True)
    subj_grp.create_array("bvals", data=bvals.astype(np.float32))
    subj_grp.create_array("bvecs", data=bvecs.astype(np.float32))

    n_z = data.shape[2]
    for z_idx in range(n_z):
        sl_grp = subj_grp.create_group(f"slice_{z_idx:03d}")
        sl_grp.create_array("input", data=degraded[:, :, z_idx, :].astype(np.float32))
        sl_grp.create_array("target", data=tensor6_clean[:, :, z_idx, :].astype(np.float32))
        sl_grp.create_array("mask", data=mask[:, :, z_idx].astype(np.float32))

    print(f"  [{i:3d}] shape={data.shape} slices={n_z} "
          f"FA={stat['fa_mean']:.3f} MD={stat['md_mean']:.6f} SNR={stat['snr']:.1f}")

    # Free memory before next subject
    del data, degraded, tensor6_clean, tensor6_deg, fa, md, mask, sample
    return stat


def build_dataset(data_dir, output_path, qc_dir, skip_qc=False):
    entries = find_dwi_datasets(data_dir)
    if not entries:
        print(f"No DWI datasets found in {data_dir}")
        return

    print(f"Found {len(entries)} subjects in {data_dir}")
    os.makedirs(qc_dir, exist_ok=True)

    # Open Zarr store (directory store on disk)
    root = zarr.open_group(output_path, mode="w")

    stats = []
    for i, entry in enumerate(entries):
        stat = process_subject(entry, i, root, qc_dir, skip_qc)
        stats.append(stat)

    # Summary plot
    if not skip_qc:
        plot_dataset_summary(stats, qc_dir)

    print(f"\nSaved {len(stats)} subjects ({output_path})")
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DTI ML dataset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory with *_dwi.nii.gz files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output .zarr directory path")
    parser.add_argument("--qc-dir", default="qc",
                        help="Directory for QC visualizations")
    parser.add_argument("--skip-qc", action="store_true",
                        help="Skip QC visualization generation")
    args = parser.parse_args()

    build_dataset(args.data_dir, args.output, args.qc_dir, args.skip_qc)