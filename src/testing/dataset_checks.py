import os
import glob
import numpy as np

data_dir = "/home/nerkar/data/ds003047"

dwi_files = glob.glob(
    os.path.join(data_dir, "sub-*", "ses-*", "dwi", "*_dwi.nii.gz")
)

print(f"Found {len(dwi_files)} DWI files\n")

for path in sorted(dwi_files):

    bval_path = path.replace(".nii.gz", ".bval")

    if not os.path.exists(bval_path):
        print(f"Missing bval: {path}")
        continue

    bvals = np.loadtxt(bval_path)

    print(f"{path}")
    print(f" → directions: {len(bvals)}\n")
