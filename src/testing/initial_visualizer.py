import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------
# 1. SET DATA PATH
# ----------------------------
data_dir = "/home/nerkar/data/ds003047"

# ----------------------------
# 2. FIND ALL DWI FILES (WITH SESSIONS)
# ----------------------------
dwi_files = glob.glob(
    os.path.join(data_dir, "sub-*", "ses-*", "dwi", "*_dwi.nii.gz")
)

print(f"Found {len(dwi_files)} DWI files")

assert len(dwi_files) > 0, "No DWI files found!"

# ----------------------------
# 3. LOAD ONE SAMPLE
# ----------------------------
dwi_path = dwi_files[0]

bval_path = dwi_path.replace(".nii.gz", ".bval")
bvec_path = dwi_path.replace(".nii.gz", ".bvec")

print("\n--- FILES ---")
print(dwi_path)
print(bval_path)
print(bvec_path)

# Load image
img = nib.load(dwi_path)
data = img.get_fdata()   # (X, Y, Z, N)

# Load gradients
bvals = np.loadtxt(bval_path)
bvecs = np.loadtxt(bvec_path)

print("\n--- DATA INFO ---")
print(f"Shape: {data.shape}")
print(f"bvals: {bvals.shape}")
print(f"bvecs: {bvecs.shape}")

# ----------------------------
# 4. VISUALIZE ONE SLICE
# ----------------------------
X, Y, Z, N = data.shape

z = Z // 2
d = 0

slice_2d = data[:, :, z, d]
slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())

plt.imshow(np.rot90(slice_2d), cmap="gray")
plt.title("Clean DWI slice")
plt.axis("off")

plt.savefig("test_slice.png", bbox_inches="tight")
plt.close()

# ----------------------------
# 5. PREPARE MODEL INPUT
# ----------------------------
slice_full = data[:, :, z, :]  # (H, W, N)
slice_full = np.transpose(slice_full, (2, 0, 1))  # (N, H, W)

print("\n--- MODEL INPUT ---")
print(f"Slice shape (N,H,W): {slice_full.shape}")
