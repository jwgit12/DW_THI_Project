# fODF Pipeline

The fODF code is isolated from the standard FA/MD pipeline so both training
modes can coexist behind the root `train.py` entry point.

```bash
python train.py --training f-odf
python f-odf/train.py
python f-odf/build_dataset.py --data_dir dataset/dataset_v1 --output dataset/default_odf.zarr
```

The fODF-specific model, dataset, loss, train loop, evaluation, and defaults
live in `src/dw_thi/f_odf/`. Shared preprocessing, augmentation, runtime, and
metric helpers stay in `src/dw_thi/` and are called by both modes.

Mac M-series profiling:

```bash
python train.py --training f-odf \
  --out_dir runs/profile_fodf_m4 \
  --epochs 1 \
  --profile --profile_exit_after_capture
```

On MPS the PyTorch profiler records CPU-side scheduling/copy activity rather
than CUDA-style kernel times. The current defaults use bf16 autocast, one
DataLoader worker, no pinned memory, no gradient clipping, and a 362-direction
surface-loss sphere to avoid avoidable MPS synchronisation overhead.
