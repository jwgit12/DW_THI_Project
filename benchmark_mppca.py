import time
import argparse
import os
import zarr
import numpy as np
import torch
from dipy.denoise.localpca import mppca as dipy_mppca
from baselines.mppca.mppca_torch import mppca_denoise_chunked, get_best_device


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_best_device()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device option: {device_arg}")


def _configure_cpu_threads(cpu_threads: int, interop_threads: int) -> tuple[int, int]:
    total_cores = os.cpu_count() or 1
    target_threads = total_cores if cpu_threads <= 0 else max(1, cpu_threads)
    torch.set_num_threads(target_threads)

    # set_num_interop_threads can only be set before parallel work starts.
    if interop_threads > 0:
        torch.set_num_interop_threads(max(1, interop_threads))

    return torch.get_num_threads(), torch.get_num_interop_threads()


def _bench_torch(
    vol_np: np.ndarray,
    device: torch.device,
    patch_radius: int,
    chunk_size: int,
    repeats: int,
    progress: bool,
) -> float:
    vol_t = torch.from_numpy(vol_np).to(device)

    # Warmup must be synchronized on async backends (CUDA/MPS)
    _ = mppca_denoise_chunked(
        vol_t,
        patch_radius=patch_radius,
        chunk_size=chunk_size,
        verbose=False,
        progress=progress,
    )
    _sync_device(device)

    times = []
    for _ in range(repeats):
        start_t = time.perf_counter()
        _ = mppca_denoise_chunked(
            vol_t,
            patch_radius=patch_radius,
            chunk_size=chunk_size,
            verbose=False,
            progress=progress,
        )
        _sync_device(device)
        times.append(time.perf_counter() - start_t)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Torch MPPCA (MPS/CPU) against Dipy.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--cpu_threads", type=int, default=0,
                        help="Torch CPU compute threads. 0 means use all logical cores.")
    parser.add_argument("--interop_threads", type=int, default=0,
                        help="Torch CPU interop threads. 0 keeps current/default value.")
    parser.add_argument("--patch_radius", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--subject", type=str, default="subject_000")
    parser.add_argument("--x0", type=int, default=30)
    parser.add_argument("--x1", type=int, default=94)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--y1", type=int, default=94)
    parser.add_argument("--z0", type=int, default=0)
    parser.add_argument("--z1", type=int, default=25)
    parser.add_argument("--max_dirs", type=int, default=64)
    parser.add_argument("--skip_dipy", action="store_true", help="Skip Dipy benchmark (useful for very large blocks).")
    parser.add_argument("--progress", action="store_true", help="Show per-chunk progress during Torch denoising.")
    parser.add_argument("--skip_cpu_baseline", action="store_true",
                        help="Skip additional CPU baseline when primary device is not CPU.")
    args = parser.parse_args()

    cpu_threads, interop_threads = _configure_cpu_threads(args.cpu_threads, args.interop_threads)
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"Torch CPU threads: {cpu_threads}")
    print(f"Torch interop threads: {interop_threads}")

    print("Loading data from Zarr dataset...")
    root = zarr.open('dataset/pretext_dataset.zarr', mode='r')
    if args.subject not in root:
        raise KeyError(f"Subject '{args.subject}' not found in dataset.")

    data = root[args.subject]['input_dwi'][
        args.x0:args.x1,
        args.y0:args.y1,
        args.z0:args.z1,
        :args.max_dirs,
    ]

    # Pre-process numpy array
    vol_np = np.array(data, dtype=np.float32)
    print(f"Data shape for benchmark: {vol_np.shape}")

    # ── Torch benchmark ──────────────────────
    device = _resolve_device(args.device)
    print(f"Running Torch MPPCA on device: {device}")
    torch_time = _bench_torch(
        vol_np,
        device=device,
        patch_radius=args.patch_radius,
        chunk_size=args.chunk_size,
        repeats=args.repeats,
        progress=args.progress,
    )
    print(f"Torch MPPCA ({device}) median time over {args.repeats} runs: {torch_time:.2f} seconds")

    # ── Torch CPU benchmark ──────────────────
    if args.skip_cpu_baseline and device.type != "cpu":
        torch_cpu_time = torch_time
        print("Skipping extra CPU baseline (--skip_cpu_baseline).")
    elif device.type == "cpu":
        torch_cpu_time = torch_time
        print("Best device is CPU; skipping duplicate CPU benchmark.")
    else:
        print("Running Torch MPPCA on device: cpu")
        torch_cpu_time = _bench_torch(
            vol_np,
            device=torch.device("cpu"),
            patch_radius=args.patch_radius,
            chunk_size=args.chunk_size,
            repeats=args.repeats,
            progress=args.progress,
        )
        print(f"Torch MPPCA (CPU) median time over {args.repeats} runs: {torch_cpu_time:.2f} seconds")

    # ── Dipy benchmark ───────────────────────
    dipy_time = None
    if not args.skip_dipy:
        print("Running Dipy MPPCA...")
        mask = np.ones(vol_np.shape[:3], dtype=bool)
        start_t = time.perf_counter()
        _, _ = dipy_mppca(vol_np, mask=mask, patch_radius=args.patch_radius, return_sigma=True)
        dipy_time = time.perf_counter() - start_t
        print(f"Dipy MPPCA completed in {dipy_time:.2f} seconds")
    else:
        print("Skipping Dipy benchmark (--skip_dipy).")

    # ── Comparison ──────────────────────────
    speedup_mps_vs_cpu = torch_cpu_time / max(torch_time, 1e-9)
    print(f"Speedup: Torch ({device}) is {speedup_mps_vs_cpu:.2f}x faster than Torch (CPU).")
    if dipy_time is not None:
        speedup_mps = dipy_time / max(torch_time, 1e-9)
        speedup_cpu = dipy_time / max(torch_cpu_time, 1e-9)
        print(f"Speedup: Torch ({device}) is {speedup_mps:.2f}x faster than Dipy.")
        print(f"Speedup: Torch (CPU) is {speedup_cpu:.2f}x faster than Dipy.")

if __name__ == "__main__":
    main()
