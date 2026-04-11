"""Evaluate a trained QSpaceUNet on test subjects.

Produces a CSV with DTI-level metrics (tensor RMSE, FA, ADC) that is
directly comparable to the baseline CSVs in baselines/*/results/.

Usage:
    python -m research.evaluate --checkpoint research/runs/run_01/best_model.pt
    python -m research.evaluate --checkpoint research/runs/run_01/best_model.pt --subjects subject_015 subject_016 subject_017
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.utils import dti6d_to_scalar_maps, scalar_map_metrics, save_prediction_slice_plot
from research.model import QSpaceUNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_subject(
    model: QSpaceUNet,
    zarr_path: str,
    subject_key: str,
    device: torch.device,
    b0_threshold: float = 50.0,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
) -> np.ndarray:
    """Run inference on a full 3D subject, slice by slice.

    Returns predicted DTI tensor (X, Y, Z, 6) as float32 numpy array.
    """
    store = zarr.open_group(zarr_path, mode="r")
    grp = store[subject_key]

    input_dwi = np.asarray(grp["input_dwi"][:], dtype=np.float32)  # (X, Y, Z, N)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)  # (3, N)

    X, Y, Z, N = input_dwi.shape
    max_n = model.max_n

    # Normalise bvals using the same max_bval as training
    bvals_norm = bvals / max_bval

    # Pad to max_n
    if N < max_n:
        pad = max_n - N
        bvals_norm = np.pad(bvals_norm, (0, pad))
        bvecs = np.pad(bvecs, ((0, 0), (0, pad)))
        input_dwi = np.pad(input_dwi, ((0, 0), (0, 0), (0, 0), (0, pad)))

    vol_mask = np.zeros(max_n, dtype=np.float32)
    vol_mask[:N] = 1.0

    # Prepare gradient tensors (shared across slices)
    bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(device)
    bvecs_t = torch.from_numpy(bvecs).unsqueeze(0).to(device)
    vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(device)

    # Compute normalisation factor from mean b0
    b0_idx = bvals[:N] < b0_threshold
    if b0_idx.any():
        mean_b0_vol = input_dwi[:, :, :, :N][..., b0_idx].mean(axis=-1)  # (X, Y, Z)
    else:
        mean_b0_vol = input_dwi[:, :, :, :N].mean(axis=-1)

    pred_dti = np.zeros((X, Y, Z, 6), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for z in range(Z):
            # Extract and normalise slice
            signal = input_dwi[:, :, z, :].transpose(2, 0, 1).astype(np.float32)  # (max_n, H, W)

            b0_slice = mean_b0_vol[:, :, z]
            b0_norm = float(b0_slice[b0_slice > 0.1 * b0_slice.max()].mean()) if (b0_slice > 0).any() else 1.0
            if b0_norm > 0:
                signal = signal / b0_norm

            signal_t = torch.from_numpy(signal).unsqueeze(0).to(device)

            pred = model(signal_t, bvals_t, bvecs_t, vol_mask_t)  # (1, 6, H, W)
            pred_dti[:, :, z, :] = pred[0].permute(1, 2, 0).cpu().numpy()

    # Unscale from training range back to physical units
    pred_dti = pred_dti / dti_scale

    return pred_dti


def evaluate_subject(
    model: QSpaceUNet,
    zarr_path: str,
    subject_key: str,
    device: torch.device,
    brain_mask_frac: float = 0.1,
    b0_threshold: float = 50.0,
    dti_scale: float = 1.0,
    max_bval: float = 1000.0,
) -> tuple[dict, dict]:
    """Full evaluation pipeline for one subject.

    Returns
    -------
    metrics : dict
        Scalar metrics row (goes into the CSV).
    arrays : dict
        Raw arrays needed for visualization (input_dwi, target_dwi,
        pred_dti6d, target_dti6d, bvals, bvecs).
    """
    t0 = time.time()

    store = zarr.open_group(zarr_path, mode="r")
    grp = store[subject_key]
    target_dti6d = np.asarray(grp["target_dti_6d"][:], dtype=np.float32)
    target_dwi = np.asarray(grp["target_dwi"][:], dtype=np.float32)
    input_dwi = np.asarray(grp["input_dwi"][:], dtype=np.float32)
    bvals = np.asarray(grp["bvals"][:], dtype=np.float32)
    bvecs = np.asarray(grp["bvecs"][:], dtype=np.float32)

    # Predict
    pred_dti6d = predict_subject(model, zarr_path, subject_key, device, b0_threshold, dti_scale, max_bval)

    # Brain mask
    b0_idx = bvals < b0_threshold
    if b0_idx.sum() > 0 and brain_mask_frac > 0:
        mean_b0 = target_dwi[..., b0_idx].mean(axis=-1)
        brain_mask = mean_b0 > brain_mask_frac * mean_b0.max()
    else:
        brain_mask = np.ones(target_dwi.shape[:3], dtype=bool)

    # Tensor RMSE within brain mask
    diff = pred_dti6d - target_dti6d
    tensor_rmse = float(np.sqrt(np.mean(diff[brain_mask] ** 2)))

    # FA and ADC metrics
    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_dti6d)
    tgt_fa, tgt_adc = dti6d_to_scalar_maps(target_dti6d)

    fa_m = scalar_map_metrics(tgt_fa, pred_fa, mask=brain_mask)
    adc_m = scalar_map_metrics(tgt_adc, pred_adc, mask=brain_mask)

    elapsed = time.time() - t0

    metrics = {
        "subject": subject_key,
        "elapsed_s": round(elapsed, 2),
        "tensor_rmse": round(tensor_rmse, 6),
        "fa_rmse": round(fa_m["rmse"], 6),
        "fa_mae": round(fa_m["mae"], 6),
        "fa_nrmse": round(fa_m["nrmse"], 6),
        "fa_r2": round(fa_m["r2"], 4),
        "adc_rmse": round(adc_m["rmse"], 8),
        "adc_mae": round(adc_m["mae"], 8),
        "adc_nrmse": round(adc_m["nrmse"], 6),
        "adc_r2": round(adc_m["r2"], 4),
    }
    arrays = {
        "input_dwi": input_dwi,
        "target_dwi": target_dwi,
        "pred_dti6d": pred_dti6d,
        "target_dti6d": target_dti6d,
        "bvals": bvals,
        "bvecs": bvecs,
    }
    return metrics, arrays


def main(args):
    device = get_device()
    log.info("Device: %s", device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    max_n = ckpt["max_n"]
    feat_dim = ckpt.get("feat_dim", 64)
    channels = tuple(ckpt.get("channels", [64, 128, 256, 512]))
    dti_scale = ckpt.get("dti_scale", 1.0)
    max_bval = ckpt.get("max_bval", 1000.0)
    log.info("DTI scale factor: %.4f, max_bval: %.1f", dti_scale, max_bval)

    model = QSpaceUNet(max_n=max_n, feat_dim=feat_dim, channels=channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint from epoch %d (val_loss=%.6f)", ckpt["epoch"], ckpt["val_loss"])

    # Determine subjects to evaluate
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = ckpt.get("test_subjects", [])
        if not subjects:
            store = zarr.open_group(args.zarr_path, mode="r")
            subjects = sorted(store.keys())

    if args.eval_all:
        store = zarr.open_group(args.zarr_path, mode="r")
        subjects = sorted(store.keys())

    log.info("Evaluating %d subjects", len(subjects))

    # Run evaluation
    rows = []
    plot_arrays = {}   # subject_key -> arrays dict (kept only for the plot subject)
    for subj in subjects:
        try:
            result, arrays = evaluate_subject(
                model, args.zarr_path, subj, device,
                brain_mask_frac=args.brain_mask_frac,
                b0_threshold=args.b0_threshold,
                dti_scale=dti_scale,
                max_bval=max_bval,
            )
            log.info(
                "%-14s  tensor_rmse=%.5f  FA[rmse=%.4f r2=%.3f]  ADC[rmse=%.2e r2=%.3f]  (%.1fs)",
                result["subject"],
                result["tensor_rmse"],
                result["fa_rmse"], result["fa_r2"],
                result["adc_rmse"], result["adc_r2"],
                result["elapsed_s"],
            )
            rows.append(result)
            if not plot_arrays:
                # Keep arrays for the first subject as the default plot candidate
                plot_arrays[subj] = arrays
            elif subj == args.plot_subject:
                plot_arrays[subj] = arrays
        except Exception as exc:
            log.warning("FAIL  %s  —  %s", subj, exc)

    if not rows:
        log.error("No subjects evaluated successfully.")
        return

    # Save CSV
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    metric_cols = [
        "tensor_rmse",
        "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
        "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2",
    ]
    summary = df[metric_cols].agg(["mean", "std"]).round(6).reset_index()
    summary.columns = ["subject"] + metric_cols
    summary["subject"] = summary["subject"].str.upper()
    df_out = pd.concat([df, summary], ignore_index=True)

    out_path = out_dir / "metrics_per_subject.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    # ── Visualization ─────────────────────────────────────────────────────────
    if rows and not args.skip_plot and plot_arrays:
        plot_subject = args.plot_subject if (args.plot_subject and args.plot_subject in plot_arrays) else next(iter(plot_arrays))
        if args.plot_subject and args.plot_subject not in plot_arrays:
            log.warning("Plot subject %s not found. Falling back to %s.", args.plot_subject, plot_subject)

        arrs = plot_arrays[plot_subject]
        plot_path = out_dir / f"prediction_example_{plot_subject}.png"
        try:
            plot_meta = save_prediction_slice_plot(
                input_dwi=arrs["input_dwi"],
                pred_dti6d=arrs["pred_dti6d"],
                target_dti6d=arrs["target_dti6d"],
                bvals=arrs["bvals"],
                out_path=plot_path,
                subject_key=plot_subject,
                b0_threshold=args.b0_threshold,
                target_dwi=arrs["target_dwi"],
                bvecs=arrs["bvecs"],
                brain_mask_frac=args.brain_mask_frac,
                slice_idx=args.plot_slice_idx,
                volume_idx=args.plot_volume_idx,
            )
            log.info("Saved prediction plot → %s  (z=%d, volume=%d)",
                     plot_meta["out_path"], plot_meta["slice_idx"], plot_meta["volume_idx"])
        except Exception as exc:
            log.warning("Could not save prediction plot for %s: %s", plot_subject, exc)

    # Print summary
    print(f"\n── DTI metrics (QSpaceUNet vs target_dti_6d) {'─' * 20}")
    print(df[["subject", "tensor_rmse",
              "fa_rmse", "fa_mae", "fa_nrmse", "fa_r2",
              "adc_rmse", "adc_mae", "adc_nrmse", "adc_r2"]].to_string(index=False))
    print(f"\n  MEAN  " + "  ".join(f"{c}={df[c].mean():.4f}" for c in metric_cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QSpaceUNet on test subjects")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--zarr_path", default="dataset/pretext_dataset_new.zarr")
    parser.add_argument("--out_dir", default="research/results")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subjects to evaluate (default: test subjects from checkpoint)")
    parser.add_argument("--eval_all", action="store_true",
                        help="Evaluate all subjects in the zarr store")
    parser.add_argument("--brain_mask_frac", type=float, default=0.1)
    parser.add_argument("--b0_threshold", type=float, default=50.0)
    parser.add_argument("--skip_plot", action="store_true",
                        help="Disable saving the denoising slice plot")
    parser.add_argument("--plot_subject", default=None,
                        help="Subject key to visualize (default: first evaluated subject)")
    parser.add_argument("--plot_slice_idx", type=int, default=None,
                        help="Axial slice index for visualization (default: auto)")
    parser.add_argument("--plot_volume_idx", type=int, default=None,
                        help="DWI volume index for visualization (default: auto)")

    main(parser.parse_args())
