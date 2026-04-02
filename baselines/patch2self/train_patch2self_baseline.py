"""
Train/evaluate Patch2Self as a second baseline on the project's Zarr dataset.

Example
-------
python baselines/patch2self/train_patch2self_baseline.py \
  --zarr_path dataset/pretext_dataset.zarr \
  --subject subject_000 \
  --save_models \
  --save_denoised
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.patch2self.patch2self_trainable import (  # noqa: E402
    Patch2SelfConfig,
    denoise_with_model,
    fit_patch2self,
)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))

    data_range = float(target.max() - target.min())
    if data_range <= 1e-12:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 1e-12:
        data_range = 1.0

    psnr = 10.0 * math.log10((data_range**2) / max(mse, 1e-12))
    return mse, mae, psnr


def _resolve_subjects(root: zarr.Group, subject: str | None, max_subjects: int) -> list[str]:
    subjects = sorted(root.group_keys())
    if subject is not None:
        if subject not in subjects:
            raise KeyError(f"Subject '{subject}' not found in dataset.")
        subjects = [subject]

    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    return subjects


def _bound(start: int, end: int, size: int) -> tuple[int, int]:
    s = max(0, start)
    e = size if end <= 0 else min(end, size)
    if e <= s:
        raise ValueError(f"Invalid crop bounds: start={start}, end={end}, size={size}")
    return s, e


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Patch2Self baseline on Zarr data.")
    parser.add_argument("--zarr_path", type=str, default="dataset/pretext_dataset.zarr")
    parser.add_argument("--subject", type=str, default=None, help="Single subject ID, e.g. subject_000")
    parser.add_argument("--max_subjects", type=int, default=0, help="0 = all available subjects")

    parser.add_argument("--model", choices=["ols", "ridge", "lasso"], default="ols")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--b0_threshold", type=float, default=50.0)
    parser.add_argument("--no_b0_denoising", action="store_true")
    parser.add_argument("--sketch_fraction", type=float, default=0.30)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--sketch_chunk_size", type=int, default=200000)
    parser.add_argument("--predict_chunk_size", type=int, default=200000)

    parser.add_argument("--clip_negative_vals", action="store_true")
    parser.add_argument("--no_shift_intensity", action="store_true")

    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--x1", type=int, default=0)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--y1", type=int, default=0)
    parser.add_argument("--z0", type=int, default=0)
    parser.add_argument("--z1", type=int, default=0)
    parser.add_argument("--max_dirs", type=int, default=0, help="0 = all directions")

    parser.add_argument("--save_dir", type=str, default="baselines/patch2self/runs")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--save_denoised", action="store_true")
    parser.add_argument("--metrics_csv", type=str, default="metrics.csv")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="r")
    subjects = _resolve_subjects(root, args.subject, args.max_subjects)
    if not subjects:
        raise RuntimeError("No subjects found in dataset.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_store = None
    if args.save_denoised:
        out_store = zarr.open(str(save_dir / "denoised_patch2self.zarr"), mode="a")

    rows: list[dict[str, float | int | str]] = []
    total_t0 = time.perf_counter()

    for i, subject in enumerate(subjects, start=1):
        grp = root[subject]
        shape = grp["input_dwi"].shape
        x0, x1 = _bound(args.x0, args.x1, shape[0])
        y0, y1 = _bound(args.y0, args.y1, shape[1])
        z0, z1 = _bound(args.z0, args.z1, shape[2])

        n_dirs_total = shape[3]
        n_dirs = n_dirs_total if args.max_dirs <= 0 else min(args.max_dirs, n_dirs_total)

        print(f"[{i}/{len(subjects)}] {subject} | block=({x0}:{x1}, {y0}:{y1}, {z0}:{z1}) | dirs={n_dirs}")

        input_dwi = np.asarray(grp["input_dwi"][x0:x1, y0:y1, z0:z1, :n_dirs], dtype=np.float32)
        bvals = np.asarray(grp["bvals"][:n_dirs], dtype=np.float32)

        target_dwi = None
        if "target_dwi" in grp:
            target_dwi = np.asarray(grp["target_dwi"][x0:x1, y0:y1, z0:z1, :n_dirs], dtype=np.float32)

        cfg = Patch2SelfConfig(
            model=args.model,
            alpha=args.alpha,
            b0_threshold=args.b0_threshold,
            b0_denoising=not args.no_b0_denoising,
            sketch_fraction=args.sketch_fraction,
            random_state=args.random_state,
            sketch_chunk_size=args.sketch_chunk_size,
            predict_chunk_size=args.predict_chunk_size,
            dtype=np.float32,
        )

        t0 = time.perf_counter()
        fitted = fit_patch2self(input_dwi, bvals, cfg=cfg, verbose=args.verbose)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        denoised = denoise_with_model(
            input_dwi,
            fitted,
            clip_negative_vals=args.clip_negative_vals,
            shift_intensity=not args.no_shift_intensity,
        )
        predict_seconds = time.perf_counter() - t1

        model_path = ""
        if args.save_models:
            model_path = str(save_dir / f"{subject}_patch2self_model.npz")
            fitted.save(model_path)

        if args.save_denoised and out_store is not None:
            if subject in out_store:
                del out_store[subject]
            subgrp = out_store.create_group(subject)
            subgrp.create_array("denoised_dwi", data=denoised.astype(np.float32))
            subgrp.create_array("bvals", data=bvals.astype(np.float32))
            subgrp.attrs["source_zarr"] = args.zarr_path
            subgrp.attrs["crop"] = [x0, x1, y0, y1, z0, z1]

        row: dict[str, float | int | str] = {
            "subject": subject,
            "x": x1 - x0,
            "y": y1 - y0,
            "z": z1 - z0,
            "n_dirs": n_dirs,
            "fit_seconds": round(fit_seconds, 4),
            "predict_seconds": round(predict_seconds, 4),
            "model_path": model_path,
        }

        if target_dwi is not None:
            mse, mae, psnr = _compute_metrics(denoised, target_dwi)
            row["mse"] = mse
            row["mae"] = mae
            row["psnr_db"] = psnr
            print(
                f"    fit={fit_seconds:.2f}s predict={predict_seconds:.2f}s "
                f"MSE={mse:.6f} MAE={mae:.6f} PSNR={psnr:.2f} dB"
            )
        else:
            print(f"    fit={fit_seconds:.2f}s predict={predict_seconds:.2f}s")

        rows.append(row)

    total_seconds = time.perf_counter() - total_t0

    csv_path = save_dir / args.metrics_csv
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinished Patch2Self baseline run")
    print(f"Subjects processed: {len(rows)}")
    print(f"Total time: {total_seconds:.2f}s")
    print(f"Metrics CSV: {csv_path}")
    if args.save_denoised:
        print(f"Denoised output store: {save_dir / 'denoised_patch2self.zarr'}")


if __name__ == "__main__":
    main()
