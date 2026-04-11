# Research Architecture

The current research model predicts a 6-channel DTI tensor from DWI slices. It is a slice-wise pipeline: dataset preprocessing prepares one axial slice at a time, a q-space encoder compresses the diffusion volumes while conditioning on `bvals` and `bvecs`, and a 2D U-Net produces the final tensor map.

## Overview

![Architecture Overview](research/assets/Overview.png)

## Components

### Dataset

`research/dataset.py` loads each subject from zarr and returns one axial slice per sample.

-   `input`: padded DWI slice with shape `(max_n, H, W)`
-   `target`: DTI target with shape `(6, H, W)`
-   `bvals`, `bvecs`: diffusion metadata
-   `vol_mask`: marks real versus padded diffusion volumes
-   `brain_mask`: used for masked loss

Preprocessing normalizes the DWI signal by mean `b0`, scales the DTI target by `dti_scale`, normalizes `bvals`, and pads every sample to the shared `max_n`.

### QSpaceEncoder

`research/model.py` starts with a q-space encoder:

-   A `1x1` convolution compresses the diffusion dimension into `feat_dim` spatial feature maps.
-   A small MLP embeds per-volume gradient info `[bval, bx, by, bz]`.
-   `vol_mask` removes padded volumes before aggregation.
-   The aggregated embedding is split into `gamma` and `beta` and applied with FiLM:

`features = signal_features * (1 + gamma) + beta`

This lets the model adapt to variable acquisition protocols without changing the backbone.

### UNet2D

The spatial model is a 2D U-Net with:

-   channel pyramid `64 -> 128 -> 256 -> 512`
-   skip connections
-   GroupNorm and LeakyReLU blocks
-   a final `1x1` head producing 6 tensor channels

The network pads inputs automatically so arbitrary slice sizes can pass through the U-Net cleanly.

### Output and loss

The output channel order is:

`Dxx, Dxy, Dyy, Dxz, Dyz, Dzz`

`DTILoss` combines:

-   tensor MSE on the 6 output channels
-   optional FA MAE and MD MAE regularization

FA and MD are computed with `tensor6_to_fa_md`, which avoids eigendecomposition and stays differentiable.

## Training and evaluation

`research/train.py` trains the model on 2D slices with `AdamW`, cosine annealing, gradient clipping, TensorBoard logging, and early stopping. Checkpoints store both model weights and preprocessing constants such as `max_n`, `max_bval`, and `dti_scale`.

`research/evaluate.py` restores those constants, predicts one slice at a time for each subject, rebuilds the full `(X, Y, Z, 6)` DTI volume, and reports tensor RMSE plus FA and ADC metrics.

## Summary

The current architecture is:

`slice-wise DWI preprocessing -> q-space conditioning -> 2D U-Net -> 6D DTI regression`

It is protocol-aware and simple to train, but it is still a 2D model, so it does not use explicit 3D context across slices.
