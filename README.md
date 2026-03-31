# DW_THI_Project
This repository contains the research and implementation of Deep Learning architectures designed to enhance Diffusion-Weighted MRI (DW-MRI).
## What is Diffusion MRI?

Structural diffusion MRI (DW-MRI) is a non-invasive neuroimaging technique that maps the brain's white matter microstructure by measuring how water molecules diffuse within tissue. Because axon membranes and myelin sheaths constrain water movement along fiber bundles, diffusion becomes directionally dependent — revealing the orientation and integrity of white matter pathways.

**Diffusion Tensor Imaging (DTI)** models this diffusion per voxel using a mathematical tensor, from which key metrics are derived:

| Metric | Description |
|---|---|
| **Fractional Anisotropy (FA)** | Degree of directional diffusion; proxy for white matter integrity |
| **Mean Diffusivity (MD / ADC)** | Average diffusion magnitude; sensitive to tissue changes |
| **Colored FA Maps** | RGB-encoded FA showing dominant fiber orientations |

---

## The Core Problem

Diffusion MRI suffers from an inherently **low signal-to-noise ratio (SNR)** and **low spatial resolution**, limiting the accuracy of downstream analyses like tractography and microstructural modeling. This project addresses these limitations by framing them as two computational tasks:

- **Denoising** — recovering clean signal from noisy acquisitions
- **Super-Resolution** — recovering fine spatial detail from low-resolution data

The goal is to develop and benchmark deep learning architectures that tackle one or both of these tasks using real clinical diffusion MRI data.

---

## Project Structure

```
DW_THI_Project/
├── functions.py                 # Core utilities (DWI loading, noise, DTI computation)
├── build_pretext_dataset.py     # CLI: build Zarr pretext dataset from MRT files
├── visualizer.py                # Qt6 desktop app to inspect the Zarr dataset
├── ml/
│   ├── dataset.py               # PyTorch Dataset with gradient-direction masking
│   ├── model.py                 # Channel-invariant U-Net (dual-head: DTI + DWI)
│   ├── train.py                 # Training loop with Charbonnier+SSIM loss, TensorBoard
│   ├── checkpoints/             # Saved model weights
│   └── runs/                    # TensorBoard logs
├── dataset/
│   ├── dataset_v1/              # Raw DWI NIfTI files
│   └── pretext_dataset.zarr/    # Generated Zarr dataset
├── qc/                          # Quality-control images
└── dti_prep.ipynb               # Exploratory notebook
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the pretext dataset

```bash
python build_pretext_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/pretext_dataset.zarr
```

This loads each DWI subject, creates noisy inputs via k-space masking + Gaussian noise, computes clean 6D DTI tensors, and writes everything to a Zarr store.

### 3. Inspect the dataset

```bash
python visualizer.py --zarr_path dataset/pretext_dataset.zarr
```

### 4. Train the pretext model

```bash
python ml/train.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --epochs 100 \
  --batch_size 2 \
  --accum_steps 4
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ml/runs/
```

---

## Apple Silicon (MPS) Optimisation

The training pipeline is optimised for Apple Silicon Macs (M1/M2/M3/M4).
When an MPS device is detected, the following are enabled automatically:

| Optimisation | What it does |
|---|---|
| **bfloat16 autocast** | Mixed-precision training using `torch.autocast("mps", dtype=torch.bfloat16)`. bfloat16 has the same exponent range as float32 so no GradScaler is needed, avoiding the scaling overhead entirely. |
| **channels_last memory format** | Converts all Conv2d weights to NHWC layout, which maps directly to the Metal Performance Shaders kernel format. |
| **Chunked direction processing** | Encoder/decoder process at most `--dir_chunk_size` directions at once (default 16), capping peak GPU memory instead of sending all B*N images through at once. |
| **Gradient accumulation** | `--accum_steps 4` with `--batch_size 2` gives an effective batch of 8 while only keeping 2 samples in memory at a time. |
| **MPS cache clearing** | `torch.mps.empty_cache()` is called between epochs to release unused Metal allocations. |
| **Non-blocking transfers** | All `.to(device, non_blocking=True)` calls overlap CPU→GPU copies with computation. |

### Recommended settings for M4 Pro (18 GB unified memory)

```bash
python ml/train.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --epochs 100 \
  --batch_size 2 \
  --accum_steps 4 \
  --dir_chunk_size 16 \
  --base_features 32 \
  --num_workers 4
```

Increase `--batch_size` or `--base_features` if memory permits; decrease `--dir_chunk_size` if you run into out-of-memory errors.

---

## Pretext Task: Masked Signal Prediction

The model learns to predict masked DWI gradient directions and the underlying DTI tensor from partially observed, noisy diffusion data:

- **Input**: noisy DWI volume with randomly masked gradient directions + binary mask
- **Target 1**: clean DWI volume (all directions)
- **Target 2**: clean 6-component DTI tensor (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)

This pretext task forces the network to learn the physics of diffusion: how signal along one gradient direction constrains possible signals along other directions.

---

## Loss Functions

The training uses a composite loss designed for DWI denoising:

| Component | Formula | Purpose |
|---|---|---|
| **Charbonnier (DWI)** | `mean(sqrt((pred - target)^2 + eps^2))` | Smooth L1 — robust to outliers, differentiable at zero. Standard in MRI denoising. |
| **SSIM (DWI)** | `1 - SSIM(pred, target)` | Gaussian-weighted structural similarity — preserves anatomical structure and local contrast. |
| **Charbonnier (DTI)** | Same formula, applied to 6-component tensor | Penalises tensor reconstruction error directly. |

**Combined**: `loss = lambda_dwi * [(1-alpha) * Charb_DWI + alpha * (1-SSIM)] + lambda_dti * Charb_DTI`

Defaults: `alpha_ssim=0.2`, `lambda_dwi=1.0`, `lambda_dti=1.0`

---

## Evaluation Metrics

All metrics are computed during validation and logged to TensorBoard.

### Image-domain (DWI)

| Metric | What it measures | Good target |
|---|---|---|
| **PSNR (dB)** | Pixel-level reconstruction fidelity | > 30 dB |
| **SSIM** | Structural similarity (luminance, contrast, structure) | > 0.90 |

### Diffusion-derived (DTI)

These are computed by eigendecomposing the predicted 6-component DTI tensor into FA and MD maps and comparing to ground truth. These are the metrics neuroscientists actually use to assess DWI quality.

| Metric | What it measures | Good target |
|---|---|---|
| **FA MAE** | Error in fractional anisotropy (white matter integrity) | < 0.05 |
| **MD MAE** | Error in mean diffusivity (tissue microstructure) | < 5e-5 mm^2/s |

---

## TensorBoard Visualisations

Three image panels are logged every ~10 epochs:

| Panel | Columns |
|---|---|
| `val/DWI_input_pred_target_error` | Noisy masked input &#124; Prediction &#124; Clean target &#124; Absolute error |
| `val/FA_pred_target_error` | Predicted FA map &#124; Target FA map &#124; FA error |
| `val/MD_pred_target_error` | Predicted MD map &#124; Target MD map &#124; MD error |

The DWI panel shows a **masked direction** — one that was hidden from the model during inference — so the comparison demonstrates actual denoising / imputation quality.

---

## CLI Reference

```
python ml/train.py [OPTIONS]

--zarr_path         Path to Zarr dataset              (default: dataset/pretext_dataset.zarr)
--epochs            Number of training epochs          (default: 100)
--batch_size        Samples per mini-batch             (default: 2)
--accum_steps       Gradient accumulation steps        (default: 4)
--dir_chunk_size    Directions per encoder/decoder pass (default: 16)
--lr                Initial learning rate               (default: 1e-3)
--mask_fraction     Fraction of directions to mask     (default: 0.4)
--alpha_ssim        SSIM weight in DWI loss            (default: 0.2)
--lambda_dwi        DWI loss weight                    (default: 1.0)
--lambda_dti        DTI loss weight                    (default: 1.0)
--base_features     U-Net base feature width           (default: 32)
--num_workers       DataLoader worker processes        (default: 0)
--log_dir           TensorBoard log directory          (default: ml/runs/pretext)
--ckpt_dir          Checkpoint directory               (default: ml/checkpoints)
```

---

## Used Datasets
*DTI data from 'Fiber architecture in the ventromedial striatum and its relation with the bed nucleus of the stria terminalis'* : https://openneuro.org/datasets/ds003047/versions/1.0.0
## Potential Datasets
- DWI Traveling Human Phantom Study: https://openneuro.org/datasets/ds000206/versions/00002
- SUDMEX_CONN: The Mexican dataset of cocaine use disorder patients: https://openneuro.org/datasets/ds003346/versions/1.1.2
- SCA2 Diffusion Tensor Imaging: https://openneuro.org/datasets/ds001378/versions/00003
