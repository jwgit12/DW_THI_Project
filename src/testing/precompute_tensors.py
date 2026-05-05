# precompute_tensors.py

import os
import time
import numpy as np

from functions import (
    find_dwi_datasets,
    load_dwi_dataset,
    compute_dti,
    tensor_to_6d
)


SAVE_SUFFIX = "_tensor6.npy"


def get_save_path(dwi_path):
    base = dwi_path.replace(".nii.gz", "")
    return base + SAVE_SUFFIX


def already_exists(save_path):
    return os.path.exists(save_path)


def main(root_dir):

    print("\n" + "="*50)
    print("🚀 PRECOMPUTE DTI TENSORS STARTED")
    print("="*50)

    print(f"\n📂 Root directory: {root_dir}")

    # -------------------------
    # FIND DATASETS
    # -------------------------
    print("\n🔍 Scanning for DWI datasets...")
    datasets = find_dwi_datasets(root_dir)

    total = len(datasets)
    print(f"\n📊 Total datasets found: {total}")

    if total == 0:
        print("\n❌ No datasets found. Check paths.")
        return

    # -------------------------
    # PROCESS LOOP
    # -------------------------
    start_all = time.time()

    for idx, entry in enumerate(datasets):

        dwi_path = entry["dwi"]
        save_path = get_save_path(dwi_path)

        print("\n" + "-"*50)
        print(f"📁 [{idx+1}/{total}] Processing:")
        print(dwi_path)

        if already_exists(save_path):
            print("✅ Tensor already exists → skipping")
            continue

        start = time.time()

        try:
            # -------------------------
            # LOAD DATA
            # -------------------------
            print("📥 Loading DWI + gradients...")
            ds = load_dwi_dataset(entry)

            data = ds["data"]
            gtab = ds["gtab"]

            print(f"   ✔ Data shape: {data.shape}")

            # -------------------------
            # COMPUTE TENSOR
            # -------------------------
            print("🧠 Computing tensor (DIPY)... this is slow ⏳")

            tensor = compute_dti(data, gtab)

            print("   ✔ Tensor computed")

            # -------------------------
            # CONVERT TO 6D
            # -------------------------
            print("🔄 Converting tensor → 6D format...")

            tensor6 = tensor_to_6d(tensor)

            print(f"   ✔ Tensor6 shape: {tensor6.shape}")

            # -------------------------
            # SAVE
            # -------------------------
            print("💾 Saving tensor...")

            tensor6 = tensor6.astype(np.float32)
            np.save(save_path, tensor6)

            print(f"   ✔ Saved → {save_path}")

        except Exception as e:
            print("❌ ERROR during processing!")
            print(f"   File: {dwi_path}")
            print(f"   Error: {e}")
            continue

        end = time.time()
        print(f"⏱️  Time for this dataset: {end - start:.2f} sec")

    end_all = time.time()

    print("\n" + "="*50)
    print("🎉 ALL DONE")
    print(f"⏱️ Total time: {end_all - start_all:.2f} sec")
    print("="*50 + "\n")


if __name__ == "__main__":

    ROOT = "/home/nerkar/data/ds003047"
    main(ROOT)
