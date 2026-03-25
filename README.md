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

## Used Datasets
*DTI data from 'Fiber architecture in the ventromedial striatum and its relation with the bed nucleus of the stria terminalis'* : https://openneuro.org/datasets/ds003047/versions/1.0.0
## Potential Datasets
- DWI Traveling Human Phantom Study: https://openneuro.org/datasets/ds000206/versions/00002
- SUDMEX_CONN: The Mexican dataset of cocaine use disorder patients: https://openneuro.org/datasets/ds003346/versions/1.1.2

## How to Check out the UI

To visualize the DW-MRI dataset with the modern Qt6 desktop dashboard:

1. **Activate the Environment:**
   Ensure your Python virtual environment is active and dependencies are installed.
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the Application:**
   Run the Qt6 desktop application:
   ```bash
   python main_qt.py
   ```
   *The application will compute DW-MRI metrics (FA/MD/cFA) and K-space representations directly within the desktop window.*
