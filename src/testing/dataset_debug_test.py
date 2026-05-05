# dataset_debug_test.py

import torch
import numpy as np
from collections import Counter
import traceback

from dataset import DWIDataset2D


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_shape(x):
    try:
        return tuple(x.shape)
    except:
        return "N/A"


def run_dataset_diagnostics(mode="train", max_samples=50):

    print_header(f"DATASET DIAGNOSTICS ({mode.upper()})")

    try:
        dataset = DWIDataset2D(mode=mode)
    except Exception as e:
        print("❌ FAILED TO LOAD DATASET")
        traceback.print_exc()
        return

    print(f"\nTotal samples: {len(dataset)}")

    # -----------------------------------
    # Track statistics
    # -----------------------------------
    channel_counts = []
    shape_issues = []
    nan_issues = []
    zero_issues = []
    tensor_issues = []

    print("\nChecking samples...\n")

    for i in range(min(len(dataset), max_samples)):

        try:
            sample = dataset[i]
        except Exception as e:
            print(f"\n❌ ERROR at sample {i}")
            traceback.print_exc()
            continue

        # -----------------------------------
        # Flexible unpack (IMPORTANT)
        # -----------------------------------
        if len(sample) == 5:
            x_noisy, x_clean, tensor, bvals, bvecs = sample
        elif len(sample) == 3:
            x_noisy, x_clean, tensor = sample
            bvals, bvecs = None, None
        else:
            print(f"⚠️ Unexpected sample format at index {i}: len={len(sample)}")
            continue

        # -----------------------------------
        # Shape checks
        # -----------------------------------
        noisy_shape = safe_shape(x_noisy)
        clean_shape = safe_shape(x_clean)
        tensor_shape = safe_shape(tensor)

        if noisy_shape != clean_shape:
            shape_issues.append((i, noisy_shape, clean_shape))

        # Channel count
        if len(noisy_shape) == 3:
            channel_counts.append(noisy_shape[0])

        # -----------------------------------
        # NaN / Inf checks
        # -----------------------------------
        if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
            nan_issues.append(i)

        if torch.isnan(x_clean).any() or torch.isinf(x_clean).any():
            nan_issues.append(i)

        # -----------------------------------
        # Zero / constant checks
        # -----------------------------------
        if torch.std(x_noisy) < 1e-6:
            zero_issues.append((i, "noisy"))

        if torch.std(x_clean) < 1e-6:
            zero_issues.append((i, "clean"))

        # -----------------------------------
        # Tensor sanity
        # -----------------------------------
        if isinstance(tensor, torch.Tensor):
            if tensor.numel() == 0 or torch.std(tensor) < 1e-6:
                tensor_issues.append(i)

        # -----------------------------------
        # Print first few samples
        # -----------------------------------
        if i < 5:
            print(f"Sample {i}")
            print(f"  noisy shape : {noisy_shape}")
            print(f"  clean shape : {clean_shape}")
            print(f"  tensor shape: {tensor_shape}")

            if bvals is not None:
                print(f"  bvals len   : {len(bvals)}")
                print(f"  bvecs shape : {np.array(bvecs).shape}")

            print("-" * 40)

    # -----------------------------------
    # SUMMARY
    # -----------------------------------
    print_header("SUMMARY")

    # Channel distribution
    if len(channel_counts) > 0:
        counter = Counter(channel_counts)
        print("\nChannel distribution:")
        for k, v in sorted(counter.items()):
            print(f"  {k} channels → {v} samples")

    # Shape issues
    if shape_issues:
        print(f"\n❌ Shape mismatches: {len(shape_issues)}")
        for s in shape_issues[:5]:
            print("  ", s)
    else:
        print("\n✅ All shapes consistent")

    # NaN issues
    if nan_issues:
        print(f"\n❌ NaN/Inf issues in {len(nan_issues)} samples")
    else:
        print("\n✅ No NaN/Inf issues")

    # Zero variance
    if zero_issues:
        print(f"\n⚠️ Zero/constant images: {len(zero_issues)}")
    else:
        print("\n✅ No constant images")

    # Tensor issues
    if tensor_issues:
        print(f"\n⚠️ Tensor issues: {len(tensor_issues)} samples")
    else:
        print("\n✅ Tensor looks valid")

    print("\nDiagnostics complete.\n")


if __name__ == "__main__":

    run_dataset_diagnostics(mode="train", max_samples=100)
    run_dataset_diagnostics(mode="test", max_samples=100)
