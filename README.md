# DW_THI_Project

Supervised DWI denoising and DTI prediction for a variable-`N` diffusion dataset. A deep learning model (QSpaceUNet) is trained to predict clean 6D DTI tensors directly from degraded DWI volumes, compared against Patch2Self and MP-PCA baselines.

## Current focus

- Supervised prediction: `input_dwi (X,Y,Z,N)` → `target_dti_6d (X,Y,Z,6)`
- 2D slice-based training with q-space encoding for variable `N`
- Tensor-aware training via combined MSE + differentiable FA/MD loss
- Subject-level train/val/test split with held-out evaluation
- Lightweight PyQt dataset inspection for raw DWI and derived DTI maps
- Lightweight PyQt comparison of Patch2Self, MP-PCA, and research metrics

## Dataset contract

The supervised Zarr dataset lives at `dataset/pretext_dataset_new.zarr/` and contains per-subject groups:

| Array | Shape | Description |
|---|---|---|
| `input_dwi` | `(X, Y, Z, N)` | noisy / degraded DWI |
| `target_dwi` | `(X, Y, Z, N)` | clean DWI target |
| `target_dti_6d` | `(X, Y, Z, 6)` | tensor target `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]` |
| `bvals` | `(N,)` | diffusion b-values |
| `bvecs` | `(3, N)` | diffusion gradient directions |

The dataset contains 11 biological subjects (18 sessions total, 7 subjects have 2 sessions).
Zarr groups are named by subject and session (e.g., `sub-01_ses-1`).
Spatial dims are `(130, 132, 25)`. `N` varies across subjects (130 or 258 volumes).
The train/test/val split operates on biological subject IDs to prevent data leakage.

## Repository structure

```text
DW_THI_Project/
├── functions.py                     # Core DWI loading, degradation, and DTI helpers
├── build_pretext_dataset.py         # NIfTI -> Zarr dataset builder
├── visualizer.py                    # Single-file PyQt6 viewer for the Zarr dataset
├── comparison_viewer.py             # PyQt6 viewer for comparing evaluation CSVs
├── baselines/
│   ├── utils.py                     # Shared DWI/DTI metrics and plotting helpers
│   ├── patch2self/patch2self.py     # Patch2Self baseline evaluation
│   └── mppca/mppca.py               # MP-PCA baseline evaluation
├── research/
│   ├── ARCHITECTURE.md             # Concise model/data-flow overview
│   ├── dataset.py                   # Zarr slice-based PyTorch dataset
│   ├── model.py                     # QSpaceUNet (q-space encoder + 2D U-Net)
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation (produces CSV like baselines)
├── requirements.txt                 # Python dependencies
└── dti_prep.ipynb
```

## Visualization tool

`visualizer.py` is a lightweight desktop viewer for `dataset/pretext_dataset_new.zarr`.

- Subject selector for browsing the 18 stored subject groups
- Plane switcher for axial, coronal, and sagittal views
- Slice and diffusion-volume sliders that work with variable-`N` acquisitions
- Side-by-side panels for `input_dwi`, `target_dwi`, absolute difference, FA, MD, and color-FA
- Metadata sidebar with source file, shell counts, current `bval` / `bvec`, and a b-value scatter plot

For quick checks without opening the GUI, `--summary-only` prints the dataset shape and subject count in the terminal.

## Comparison viewer

`comparison_viewer.py` is a small PyQt desktop tool for comparing the three evaluation CSVs:

- `baselines/patch2self/results/metrics_per_subject.csv`
- `baselines/mppca/mppca_eval/metrics_per_subject.csv`
- `research/results/metrics_per_subject.csv`

It automatically:

- finds the subject intersection across all three files
- finds the quality metrics they all share
- recomputes mean values on that shared subject set instead of trusting the `MEAN` footer rows
- shows an all-metrics table, a selected-metric bar chart, and per-subject detail

This is especially useful when the research CSV only covers held-out test subjects while the baselines cover all subjects.

## QSpaceUNet

`research/model.py` implements the model:

See [research/ARCHITECTURE.md](research/ARCHITECTURE.md) for a diagram and short walkthrough of the current data flow.

- **Q-space encoder**: 1×1 conv compresses `N` DWI volumes into `C` feature channels, with FiLM conditioning from a gradient table MLP (bvals/bvecs → scale + shift)
- **2D U-Net backbone**: 4-level encoder-decoder with skip connections, GroupNorm, auto-padding for arbitrary spatial dims
- **Output**: 6-channel DTI tensor `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]`

Variable `N` is handled by zero-padding to `max_n` with a volume mask. The gradient FiLM conditioning adapts features to the acquisition protocol.

## Training

`research/train.py` trains on 2D axial slices with:

- **Loss**: MSE on 6D tensor + λ × (FA MAE + MD MAE), computed within a brain mask
- **FA/MD computation**: Frobenius norm formulation (no eigendecomposition — MPS-safe)
- **Optimizer**: AdamW with cosine annealing, gradient clipping
- **Split**: 7 train / 2 val / 2 test (biological subject-level, all sessions grouped)
- **Augmentation**: random horizontal/vertical flips
- **Early stopping**: patience=25 on validation loss

## TensorBoard monitoring

`research/train.py` writes TensorBoard event files to `{out_dir}/tb/`.

**Scalars logged every epoch:**

| Tag | Description |
|---|---|
| `loss/train`, `loss/val` | Total combined loss |
| `tensor_mse/train`, `tensor_mse/val` | 6D tensor MSE component |
| `fa_mae/train`, `fa_mae/val` | FA MAE auxiliary loss component |
| `md_mae/train`, `md_mae/val` | MD MAE auxiliary loss component |
| `lr` | Current learning rate |
| `baselines/fa_rmse/patch2self` | Patch2Self mean FA RMSE (flat reference line) |
| `baselines/fa_rmse/mppca` | MP-PCA mean FA RMSE (flat reference line) |
| *(all other `fa_*`, `adc_*` baseline metrics)* | Flat reference lines from baseline CSVs |

**Images logged every `--vis_every` epochs (default: 1):**

`val_prediction` — a 2×4 panel showing for a fixed validation slice: target FA, predicted FA, FA error map, FA scatter plot (with RMSE and R²), and the same four panels for ADC.

## Baselines

Patch2Self and MP-PCA baselines in `baselines/` evaluate in both DWI-space and DTI-space. The DL evaluation in `research/evaluate.py` uses the same metric functions from `baselines/utils.py` for direct comparison.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

Build the supervised Zarr dataset:

```bash
python3 build_pretext_dataset.py \
  --data_dir dataset/dataset_v1 \
  --output dataset/pretext_dataset_new.zarr
```

Inspect the dataset:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr
```

Open a specific subject first:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr --subject sub-06_ses-1
```

Print a dataset summary without launching Qt:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr --summary-only
```

Open the comparison viewer:

```bash
python3 comparison_viewer.py
```

Print the shared subjects/metrics and recomputed overlap-only means:

```bash
python3 comparison_viewer.py --summary-only
```

Run Patch2Self baseline:

```bash
python3 baselines/patch2self/patch2self.py --zarr_path dataset/pretext_dataset_new.zarr
```

Run MP-PCA baseline:

```bash
python3 baselines/mppca/mppca.py --zarr_path dataset/pretext_dataset_new.zarr
```

Train the DL model:

```bash
python3 -m research.train \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/runs/run_01 \
  --epochs 150 \
  --batch_size 8
```

Monitor training with TensorBoard (logs written to `{out_dir}/tb/`):

```bash
tensorboard --logdir research/runs/run_01/tb
```

Reduce visualisation overhead by generating figures less frequently:

```bash
python3 -m research.train \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/runs/run_01 \
  --vis_every 5
```

Evaluate the trained model on test subjects:

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_02/best_model.pt \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --out_dir research/results
```

Evaluate on all subjects (for full comparison with baselines):

```bash
python3 -m research.evaluate \
  --checkpoint research/runs/run_01/best_model.pt \
  --zarr_path dataset/pretext_dataset_new.zarr \
  --eval_all
```
