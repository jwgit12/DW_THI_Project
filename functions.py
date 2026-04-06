import os
import glob
import nibabel as nib
import numpy as np
import cv2
from joblib import Parallel, delayed
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel


#%% DWI dataset handling
def find_dwi_datasets(root_dir):
    dwi_files = glob.glob(os.path.join(root_dir, "*_dwi.nii.gz"))
    
    datasets = []
    
    for dwi_path in dwi_files:
        base = dwi_path.replace(".nii.gz", "")
        
        bval_path = base + ".bval"
        bvec_path = base + ".bvec"
        
        if os.path.exists(bval_path) and os.path.exists(bvec_path):
            datasets.append({
                "dwi": dwi_path,
                "bval": bval_path,
                "bvec": bvec_path
            })
        else:
            print(f"Missing gradients for {dwi_path}")
    
    return datasets

def load_dwi_dataset(entry):
    # Load image
    img = nib.load(entry["dwi"])
    data = img.get_fdata()  # shape: (X, Y, Z, N)
    
    # Load gradients
    bvals, bvecs = read_bvals_bvecs(entry["bval"], entry["bvec"])
    
    # Ensure correct shape
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    
    # Build gradient table
    gtab = gradient_table(bvals, bvecs)
    
    return {
        "data": data,
        "affine": img.affine,
        "bvals": bvals,
        "bvecs": bvecs,
        "gtab": gtab,
        "path": entry["dwi"]
    }

def load_all_dwi(root_dir):
    entries = find_dwi_datasets(root_dir)
    
    dataset = []
    for e in entries:
        dataset.append(load_dwi_dataset(e))
    
    return dataset


#%% Image degradation
def apply_kspace_mask(slice_2d, keep_fraction=0.5):
    kspace = np.fft.fftshift(np.fft.fft2(slice_2d))

    nx, ny = kspace.shape
    cx, cy = nx // 2, ny // 2

    keep_x = int(nx * keep_fraction / 2)
    keep_y = int(ny * keep_fraction / 2)

    mask = np.zeros_like(kspace)
    mask[cx - keep_x:cx + keep_x, cy - keep_y:cy + keep_y] = 1

    kspace_filtered = kspace * mask

    img_lowres = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_filtered)))

    return img_lowres

def add_noise(slice_2d, noise_level):
    sigma = noise_level * np.max(slice_2d)
    noise = np.random.normal(0, sigma, slice_2d.shape)
    return slice_2d + noise

def lowres_noise(data, keep_fraction=0.5, noise_min=0.01, noise_max=0.05):
    h, w = data.shape[:2]

    # Vectorized k-space filtering over all slices/volumes at once
    kspace = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1)), axes=(0, 1))

    # Build 2D mask once, broadcast over z and t
    cy, cx = h // 2, w // 2
    ry, rx = int(h * keep_fraction / 2), int(w * keep_fraction / 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[cy - ry:cy + ry, cx - rx:cx + rx] = True
    kspace[~mask] = 0

    lowres = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1)))

    # Vectorized noise: one random level per (slice, volume)
    noise_levels = np.random.uniform(noise_min, noise_max, size=data.shape[2:])
    slice_max = lowres.max(axis=(0, 1), keepdims=True)
    sigma = noise_levels * slice_max
    noise = np.random.randn(*data.shape) * sigma

    return lowres + noise

#%% DTI computations
def compute_dti(data, gtab, mask=None):
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    # Tensor element order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    tensor = tenfit.quadratic_form

    return tensor  # shape: (X, Y, Z, 3, 3)

def tensor_to_6d(tensor):
    Dxx = tensor[..., 0, 0]
    Dxy = tensor[..., 0, 1]
    Dyy = tensor[..., 1, 1]
    Dxz = tensor[..., 0, 2]
    Dyz = tensor[..., 1, 2]
    Dzz = tensor[..., 2, 2]
    
    return np.stack([Dxx, Dxy, Dyy, Dxz, Dyz, Dzz], axis=-1)

def tensor6_to_full(tensor6):
    Dxx = tensor6[..., 0]
    Dxy = tensor6[..., 1]
    Dyy = tensor6[..., 2]
    Dxz = tensor6[..., 3]
    Dyz = tensor6[..., 4]
    Dzz = tensor6[..., 5]
    
    tensor = np.zeros(tensor6.shape[:-1] + (3, 3))
    
    tensor[..., 0, 0] = Dxx
    tensor[..., 0, 1] = Dxy
    tensor[..., 1, 0] = Dxy
    
    tensor[..., 1, 1] = Dyy
    tensor[..., 0, 2] = Dxz
    tensor[..., 2, 0] = Dxz
    
    tensor[..., 1, 2] = Dyz
    tensor[..., 2, 1] = Dyz
    
    tensor[..., 2, 2] = Dzz
    
    return tensor

def tensor_to_eig(tensor):
    evals, evecs = np.linalg.eigh(tensor)
    
    # sort descending (largest eigenvalue first)
    idx = np.argsort(evals, axis=-1)[..., ::-1]
    
    evals = np.take_along_axis(evals, idx, axis=-1)
    evecs = np.take_along_axis(evecs, idx[..., None, :], axis=-1)
    
    return evals, evecs


#%% Other utilities

def compute_md_from_tensor6(tensor6):
    tensor = tensor6_to_full(tensor6)
    evals, _ = tensor_to_eig(tensor)
    
    md = np.mean(evals, axis=-1)
    return md

def compute_fa_from_tensor6(tensor6):
    tensor = tensor6_to_full(tensor6)
    evals, _ = tensor_to_eig(tensor)
    
    md = np.mean(evals, axis=-1, keepdims=True)
    
    numerator = np.sqrt(((evals - md) ** 2).sum(axis=-1))
    denominator = np.sqrt((evals ** 2).sum(axis=-1) + 1e-12)
    
    fa = np.sqrt(1.5) * numerator / denominator
    
    return fa

def compute_color_fa_from_tensor6(tensor6):
    tensor = tensor6_to_full(tensor6)
    evals, evecs = tensor_to_eig(tensor)
    
    fa = compute_fa_from_tensor6(tensor6)
    
    # principal eigenvector
    principal_dir = evecs[..., :, 0]
    
    color = np.abs(principal_dir)
    color_fa = color * fa[..., None]
    
    return color_fa

def norm(x, pmin=1, pmax=99):
    vmin, vmax = np.percentile(x[x > 0], (pmin, pmax))
    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

def split_b0_dwi(data, bvals, threshold=50):
    b0_idx = bvals < threshold
    dwi_idx = bvals >= threshold
    
    return (
        data[..., b0_idx],
        data[..., dwi_idx]
    )

def show_kspace(img):
    k = np.fft.fftshift(np.fft.fft2(img))
    k_mag = np.abs(k)
    
    # Log scaling
    k_log = np.log1p(k_mag)
    
    return k_log

def radial_profile(image):
    k = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(k)

    nx, ny = mag.shape
    cx, cy = nx // 2, ny // 2

    # Create radius map
    y, x = np.indices((nx, ny))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)

    # Radial mean
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())

    radial_mean = tbin / (nr + 1e-8)

    return radial_mean

def brain_mask(image,
               edge_percentile=92,
               sobel_ksize=5,
               close_kernel=31,
               open_kernel=11,
               min_contour_area_frac=0.02):

    def process_slice(slice2d):
        im = np.array(slice2d, dtype=np.float64)

        # Sobel gradients and magnitude
        k = sobel_ksize if sobel_ksize % 2 == 1 else sobel_ksize + 1
        sx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=k)
        sobel_mag = np.hypot(sx, sy)

        # Edge threshold
        thr = np.percentile(sobel_mag, edge_percentile)
        edges = (sobel_mag > thr).astype(np.uint8) * 255

        # Morphological closing/opening
        ck = close_kernel if close_kernel % 2 == 1 else close_kernel + 1
        ok = open_kernel if open_kernel % 2 == 1 else open_kernel + 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))

        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        edges_closed = cv2.dilate(edges_closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

        # Find largest contour
        contours_info = cv2.findContours(edges_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[-2]
        h, w = im.shape
        img_area = h * w

        mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            if areas[max_idx] >= (min_contour_area_frac * img_area):
                cv2.drawContours(mask, [contours[max_idx]], -1, 255, thickness=-1)
            else:
                # fallback to Otsu threshold if contour too small
                im_u8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, mask = cv2.threshold(im_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # fallback if no contour
            im_u8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, mask = cv2.threshold(im_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = (mask > 0).astype(np.uint8)
        return mask

    # Handle 2D, 3D, and 4D cases
    if image.ndim == 2:
        mask = process_slice(image)
        masked = image * mask
    elif image.ndim == 3:
        slices = Parallel(n_jobs=-1)(delayed(process_slice)(image[..., i]) for i in range(image.shape[-1]))
        mask = np.stack(slices, axis=-1)
        masked = image * mask
    elif image.ndim == 4:
        all_slices = Parallel(n_jobs=-1)(
            delayed(process_slice)(image[..., i, t])
            for t in range(image.shape[-1])
            for i in range(image.shape[-2])
        )
        # Reshape: we iterated t-outer, i-inner
        n_z, n_t = image.shape[-2], image.shape[-1]
        mask = np.stack(all_slices, axis=-1).reshape(image.shape[:2] + (n_z, n_t))
        masked = image * mask
    else:
        raise ValueError("Input image must be 2D, 3D, or 4D.")

    return mask, masked