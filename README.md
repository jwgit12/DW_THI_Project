# DW_THI_Project

Supervised DWI restoration research code for a variable-`N` diffusion dataset. The current implementation centers on a new `research/` stack that trains and evaluates a set-conditioned 3D model against clean DWI targets and DTI tensor targets, while keeping Patch2Self and MPPCA as comparison baselines.

## Current focus

- fully supervised restoration from `input_dwi -> target_dwi`
- variable-length diffusion sets per subject (`N` differs across subjects)
- tensor-aware training via direct tensor supervision and differentiable tensor fitting
- held-out subject evaluation with fold manifests, image metrics, tensor metrics, edge metrics, and paired statistical tests

## Dataset contract

The supervised Zarr dataset lives at `dataset/pretext_dataset.zarr/` and contains per-subject groups:

| Array | Shape | Description |
|---|---|---|
| `input_dwi` | `(X, Y, Z, N)` | noisy / degraded DWI |
| `target_dwi` | `(X, Y, Z, N)` | clean DWI target |
| `target_dti_6d` | `(X, Y, Z, 6)` | tensor target `[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]` |
| `bvals` | `(N,)` | diffusion b-values |
| `bvecs` | `(3, N)` | diffusion gradient directions |

The local dataset currently contains `18` subjects. In this dataset `N` varies, with both `130` and `258` direction stacks present.

## Research stack

The new implementation lives under `research/`:

```text
research/
├── data/
│   ├── zarr_dataset.py         # Lazy subject access, metadata, train-only clipping stats
│   ├── patch_sampler.py        # Foreground / boundary / WM patch sampling
│   ├── collate.py              # Variable-N padding and batch assembly
│   └── gradients.py            # Gradient validation, shell ids, N_ctx selection
├── models/
│   ├── aqd_net.py              # AQD-Net and fixed-channel ablation model
│   ├── volume_encoder.py       # Shared 3D encoder/decoder blocks
│   ├── set_attention.py        # Masked cross-volume attention
│   └── tensor_head.py          # Auxiliary tensor decoder head
├── losses/
│   ├── charbonnier.py          # Weighted DWI reconstruction + SSIM
│   ├── edge_loss.py            # Edge-aware boundary preservation loss
│   ├── tensor_fit.py           # Differentiable WLS tensor fit from predicted DWI
│   └── dti_metrics.py          # Torch tensor scalars and tensor losses
├── eval/
│   ├── metrics_image.py        # PSNR / SSIM / NRMSE / edge metrics
│   ├── metrics_tensor.py       # Tensor RMSE, FA/MD/AD/RD errors, FDR helpers
│   └── evaluate_fold.py        # Fold evaluation against noisy, Patch2Self, MPPCA
├── baselines/
│   ├── run_patch2self.py       # Wrapper on common preprocessing
│   └── run_mppca.py            # Wrapper on common preprocessing
├── splits/
│   └── folds.json              # 6 subject-level folds: 12 train / 3 val / 3 test
├── infer.py                    # Sliding-window + multi-context inference
├── train.py                    # Training loop with EMA, warmup+cosine, early stop
└── tests/
    └── test_research_stack.py  # Smoke tests for data/model/inference contracts
```

## AQD-Net summary

`research/models/aqd_net.py` implements the proposed model:

- shared 3D per-volume encoder
- gradient conditioning from `[is_b0, log_bval, bvec_x, bvec_y, bvec_z]`
- masked set attention across diffusion volumes
- per-volume residual decoding (`pred_dwi = x_in - residual`)
- auxiliary tensor head for `target_dti_6d`

To handle subjects with `N=258`, training and inference use the plan’s memory fallback by default:

- `context_cap = 48`
- all b0 volumes are retained
- remaining DWI volumes are selected shell-stratified with angular coverage
- inference reconstructs the full subject by running multiple contexts and stitching them back together

The default training preset is now a smaller AQD-Net for faster iteration:

- `model_preset = "small"`
- `base_channels = 16`
- `attention_depth = 1`
- `num_heads = 4`

Inference and evaluation also skip the auxiliary tensor decoder head, because only `pred_dwi` is needed on that path.

## Training objective

`research/train.py` uses the composite supervised loss from the implementation plan:

```text
L_total =
  1.00 * L_dwi
  + 0.25 * L_ssim
  + 0.20 * L_edge
  + 0.35 * L_tensor_fit
  + 0.15 * L_tensor_aux
  + 0.10 * L_fa_md
```

Implemented details include:

- per-subject normalization from the median input mean-b0 signal
- train-fold-only clipping threshold estimation
- subject-level split reuse via `research/splits/folds.json`
- foreground / boundary / white-matter patch sampling
- diffusion-geometry-safe augmentation with matching `bvec` updates
- AdamW, warmup + cosine decay, gradient clipping, and EMA
- TensorBoard scalars for epoch-level train/validation metrics plus a fixed validation denoising slice
- early stopping on the validation composite score:

```text
score =
  + PSNR_target_dwi
  + SSIM_target_dwi
  - NRMSE_target_dwi
  - tensor_RMSE
  - FA_abs_error
  - MD_abs_error
```

## Baselines

The original baseline scripts remain in `baselines/`, and the new fold evaluator uses wrappers in `research/baselines/` so baselines and model evaluation share:

- the same subject normalization
- the same masking policy
- the same tensor fitting code path
- the same image/tensor/edge metrics

Patch2Self defaults in the wrapper follow the plan:

- OLS first
- `patch_radius=0`

MPPCA defaults are fixed and documented:

- `patch_radius=2`
- `pca_method="eig"`

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
  --output dataset/pretext_dataset.zarr \
  --keep_fraction 0.5 \
  --noise_min 0.01 \
  --noise_max 0.05
```

Inspect the dataset:

```bash
python3 visualizer.py --zarr_path dataset/pretext_dataset.zarr
```

Train AQD-Net on one fold:

```bash
python3 -m research.train \
  --zarr_path dataset/pretext_dataset.zarr \
  --fold_id 0 \
  --epochs 40 \
  --batch_size 2 \
  --model_preset small \
  --patch_size 32 32 32 \
  --context_cap 48 \
  --augment
```

Run inference on one subject:

```bash
python3 -m research.infer \
  --checkpoint research/runs/fold_0/best.pt \
  --zarr_path dataset/pretext_dataset.zarr \
  --subject_id subject_000 \
  --patch_size 32 32 32
```

Evaluate one fold against noisy input, Patch2Self, and MPPCA:

```bash
python3 -m research.eval.evaluate_fold \
  --checkpoint research/runs/fold_0/best.pt \
  --zarr_path dataset/pretext_dataset.zarr \
  --fold_id 0 \
  --output_dir research/eval/results
```

Run smoke tests:

```bash
python3 -m unittest research.tests.test_research_stack
```

Launch TensorBoard for a fold run:

```bash
tensorboard --logdir research/runs
```

## Training CLI highlights

`research/train.py` supports the planned ablations directly:

- `--disable_gradient_conditioning`
- `--disable_attention`
- `--disable_aux_tensor_head`
- `--disable_tensor_fit_loss`
- `--disable_edge_loss`
- `--architecture fixed_channel`

Useful core options:

- `--model_preset small`: smaller default AQD-Net (`16 / 1 / 4`)
- `--model_preset base`: previous larger AQD-Net (`32 / 3 / 8`)
- `--context_cap 48`: enable large-`N` fallback
- `--patch_size X Y Z`: patch-based 3D training and inference
- `--use_b0_guide`: concatenate the mean-b0 patch as an extra guide channel
- `--augment`: enable geometry-safe augmentation
- `--val_overlap`: overlap used during sliding-window validation

Each fold run writes TensorBoard event files under `research/runs/fold_<id>/tensorboard/`.
The validation example image uses the first validation subject, a fixed DWI volume, and a fixed axial slice, with panels ordered as `input | prediction | target | abs_error`.

Note: the current dataset has `Z=25`, so the code clamps requested patch sizes to the subject dimensions at runtime when necessary.

## Legacy files still in repo

The repository still includes:

- `functions.py` for raw DWI / tensor helper utilities
- `build_pretext_dataset.py` for dataset creation
- `visualizer.py` for manual inspection
- `baselines/` for the original standalone baseline scripts

The active research training and evaluation path is now the `research/` package, not the older `ml/` prototype.
