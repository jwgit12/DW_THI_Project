"""Benchmark :func:`baselines.mppca_torch` against :func:`dipy.denoise.localpca.mppca`.

Reports wall-clock time and max/mean absolute difference versus DIPY on a
range of synthetic volumes. Run as a module:

    python -m baselines.benchmark
    python -m baselines.benchmark --shape 130 130 25 150
    python -m baselines.benchmark --sizes small medium full --repeats 2

When run without ``--sizes`` it benchmarks a progression that culminates in
the typical scan size used in this project (130, 130, 25, 150).
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from baselines import mppca_torch


SHAPE_PRESETS = {
    "tiny":   (30, 30, 10, 60),
    "small":  (60, 60, 15, 80),
    "medium": (100, 100, 20, 120),
    "full":   (130, 130, 25, 150),
}

DEFAULT_CHECK_RTOL = 1e-4
DEFAULT_CHECK_ATOL = 1e-4


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _time_call(fn, *args, **kwargs) -> Tuple[float, object]:
    t0 = time.time()
    out = fn(*args, **kwargs)
    return time.time() - t0, out


def _make_volume(shape: Tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Rough DWI-like magnitudes: smooth baseline + Rician-ish noise.
    base = rng.uniform(20, 120, size=shape).astype(np.float32)
    noise = rng.normal(0, 5, size=shape).astype(np.float32)
    return np.clip(base + noise, 0, None).astype(np.float32)


def bench_once(
    shape: Tuple[int, ...],
    patch_radius: int = 2,
    device: Optional[str] = None,
    run_dipy: bool = True,
    check_result: bool = True,
    check_rtol: float = DEFAULT_CHECK_RTOL,
    check_atol: float = DEFAULT_CHECK_ATOL,
    repeats: int = 1,
    seed: int = 0,
) -> dict:
    device = device or _auto_device()
    arr = _make_volume(shape, seed)
    row = {"shape": shape, "device": device, "patch_radius": patch_radius}

    # Torch: warmup + timed runs
    _ = mppca_torch(arr[:20, :20, :6], patch_radius=patch_radius, device=device)
    torch_times = []
    for _ in range(repeats):
        t, d_torch = _time_call(
            mppca_torch, arr, patch_radius=patch_radius, device=device
        )
        torch_times.append(t)
    row["torch_time"] = float(np.mean(torch_times))
    row["torch_time_best"] = float(np.min(torch_times))

    if run_dipy:
        from dipy.denoise.localpca import mppca as dipy_mppca
        dipy_times = []
        for _ in range(repeats):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, d_dipy = _time_call(
                    dipy_mppca, arr, patch_radius=patch_radius, pca_method="eig"
                )
            dipy_times.append(t)
        row["dipy_time"] = float(np.mean(dipy_times))
        row["dipy_time_best"] = float(np.min(dipy_times))
        row["speedup"] = row["dipy_time"] / row["torch_time"]
        row["torch_shape"] = d_torch.shape
        row["dipy_shape"] = d_dipy.shape
        row["shape_matches"] = d_torch.shape == d_dipy.shape
        if row["shape_matches"]:
            diff = np.abs(d_torch - d_dipy)
            max_abs_diff = float(diff.max())
            row["max_abs_diff"] = max_abs_diff
            row["mean_abs_diff"] = float(diff.mean())
            row["rel_max_diff"] = float(
                max_abs_diff / max(1e-9, np.abs(d_dipy).max())
            )
        else:
            row["max_abs_diff"] = float("inf")
            row["mean_abs_diff"] = float("inf")
            row["rel_max_diff"] = float("inf")
        row["result_checked"] = check_result
        row["check_rtol"] = check_rtol
        row["check_atol"] = check_atol
        row["result_matches"] = bool(
            (not check_result)
            or (
                row["shape_matches"]
                and np.allclose(
                    d_torch,
                    d_dipy,
                    rtol=check_rtol,
                    atol=check_atol,
                    equal_nan=False,
                )
            )
        )

    return row


def _format_row(r: dict) -> str:
    shape_str = "x".join(str(s) for s in r["shape"])
    line = (
        f"  {shape_str:<20} device={r['device']:<5} "
        f"torch={r['torch_time']:7.2f}s"
    )
    if "dipy_time" in r:
        check_status = "SKIP"
        if r.get("result_checked", False):
            check_status = "PASS" if r["result_matches"] else "FAIL"
        line += (
            f"  dipy={r['dipy_time']:7.2f}s  speedup={r['speedup']:5.1f}x  "
            f"max|diff|={r['max_abs_diff']:.2e}  "
            f"mean|diff|={r['mean_abs_diff']:.2e}  "
            f"check={check_status}"
        )
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+",
        default=["small", "medium", "full"],
        choices=list(SHAPE_PRESETS.keys()),
        help="Preset volume sizes to benchmark.",
    )
    parser.add_argument(
        "--shape", nargs=4, type=int, default=None,
        help="Override: custom shape (X Y Z N).",
    )
    parser.add_argument(
        "--patch_radius", type=int, default=2,
        help="Patch radius in voxels. 2 -> 5x5x5 (125 samples).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
        help="Torch device for the GPU baseline. Default: auto.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Timing repeats per configuration; reports mean and best.",
    )
    parser.add_argument(
        "--no_dipy", action="store_true",
        help="Skip DIPY reference; only time the torch implementation.",
    )
    parser.add_argument(
        "--no_check", action="store_true",
        help="Do not fail when the torch and DIPY outputs differ.",
    )
    parser.add_argument(
        "--check_rtol", type=float, default=DEFAULT_CHECK_RTOL,
        help="Relative tolerance for comparing torch and DIPY outputs.",
    )
    parser.add_argument(
        "--check_atol", type=float, default=DEFAULT_CHECK_ATOL,
        help="Absolute tolerance for comparing torch and DIPY outputs.",
    )
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    shapes = [tuple(args.shape)] if args.shape else [
        SHAPE_PRESETS[s] for s in args.sizes
    ]

    print(f"Device selected : {device or _auto_device()}")
    print(f"Patch radius    : {args.patch_radius}  "
          f"(patch size {2 * args.patch_radius + 1}^3 "
          f"= {(2 * args.patch_radius + 1) ** 3} samples)")
    print(f"Repeats         : {args.repeats}")
    if args.no_dipy:
        print("DIPY reference  : skipped")
    elif args.no_check:
        print("Result check    : skipped")
    else:
        print(f"Result check    : rtol={args.check_rtol:g}, atol={args.check_atol:g}")
    print()

    print("Benchmarks:")
    failed_rows = []
    for shape in shapes:
        row = bench_once(
            shape=shape,
            patch_radius=args.patch_radius,
            device=device,
            run_dipy=not args.no_dipy,
            check_result=not args.no_check,
            check_rtol=args.check_rtol,
            check_atol=args.check_atol,
            repeats=args.repeats,
        )
        print(_format_row(row), flush=True)
        if (
            "result_matches" in row
            and row.get("result_checked", False)
            and not row["result_matches"]
        ):
            failed_rows.append(row)

    if failed_rows:
        print()
        print("Result mismatch detected:")
        for row in failed_rows:
            shape_str = "x".join(str(s) for s in row["shape"])
            print(
                f"  {shape_str}: max|diff|={row['max_abs_diff']:.6e}, "
                f"mean|diff|={row['mean_abs_diff']:.6e}, "
                f"rtol={row['check_rtol']:g}, atol={row['check_atol']:g}"
            )
            if not row["shape_matches"]:
                print(
                    f"    torch shape={row['torch_shape']}, "
                    f"DIPY shape={row['dipy_shape']}"
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
