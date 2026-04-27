"""fODF defaults mapped from the project-level ``config.py``.

The standard FA/MD path reads the unprefixed settings in ``config.py``. The
fODF package reads the same central file through this adapter so it can keep
optimized fODF defaults without taking over the standard training defaults.
"""

from __future__ import annotations

import config as project_cfg


def _setting(name: str, fallback: object | None = None):
    fodf_name = f"FODF_{name}"
    if hasattr(project_cfg, fodf_name):
        return getattr(project_cfg, fodf_name)
    if hasattr(project_cfg, name):
        return getattr(project_cfg, name)
    if fallback is not None:
        return fallback
    raise AttributeError(f"Missing config setting {fodf_name} / {name}")


DATASET_ZARR_PATH = _setting("DATASET_ZARR_PATH")
DATASET_QC_DIR = _setting("DATASET_QC_DIR")
TRAIN_OUT_DIR = _setting("TRAIN_OUT_DIR")
EVAL_OUT_DIR = _setting("EVAL_OUT_DIR")
SEED = _setting("SEED")

NOISE_MIN = _setting("NOISE_MIN")
NOISE_MAX = _setting("NOISE_MAX")
KEEP_FRACTION_MIN = _setting("KEEP_FRACTION_MIN")
KEEP_FRACTION_MAX = _setting("KEEP_FRACTION_MAX")
KEEP_FRACTION = _setting("KEEP_FRACTION")
EVAL_KEEP_FRACTION = _setting("EVAL_KEEP_FRACTION")
EVAL_NOISE_LEVEL = _setting("EVAL_NOISE_LEVEL")
EVAL_REPEATS = _setting("EVAL_REPEATS")
EVAL_KEEP_FRACTION_MIN = _setting("EVAL_KEEP_FRACTION_MIN")
EVAL_KEEP_FRACTION_MAX = _setting("EVAL_KEEP_FRACTION_MAX")
EVAL_NOISE_MIN = _setting("EVAL_NOISE_MIN")
EVAL_NOISE_MAX = _setting("EVAL_NOISE_MAX")
EVAL_DEGRADE_SEED = _setting("EVAL_DEGRADE_SEED")
EVAL_DEFAULT_CHECKPOINT = _setting("EVAL_DEFAULT_CHECKPOINT")
EVAL_INFER_BATCH_SIZE = _setting("EVAL_INFER_BATCH_SIZE")

RANDOM_SLICE_AXIS = _setting("RANDOM_SLICE_AXIS")
SLICE_AXES = _setting("SLICE_AXES")
AUG_FLIP = _setting("AUG_FLIP")
AUG_INTENSITY = _setting("AUG_INTENSITY")
AUG_VOLUME_DROPOUT = _setting("AUG_VOLUME_DROPOUT")

B0_THRESHOLD = _setting("B0_THRESHOLD")
DTI_FIT_METHOD = _setting("DTI_FIT_METHOD")
MAX_DIFFUSIVITY = _setting("MAX_DIFFUSIVITY")
BRAIN_MASK_MEDIAN_RADIUS = _setting("BRAIN_MASK_MEDIAN_RADIUS")
BRAIN_MASK_NUMPASS = _setting("BRAIN_MASK_NUMPASS")
BRAIN_MASK_DILATE = _setting("BRAIN_MASK_DILATE")
BRAIN_MASK_FINALIZE = _setting("BRAIN_MASK_FINALIZE")

FODF_SH_ORDER = _setting("SH_ORDER")
TRAIN_FODF_SH_ORDER = _setting("TRAIN_SH_ORDER")
FODF_RESPONSE_ROI_RADII = _setting("RESPONSE_ROI_RADII")
FODF_RESPONSE_FA_THR = _setting("RESPONSE_FA_THR")
FODF_SINGLE_SHELL_TOL = _setting("SINGLE_SHELL_TOL")

TEST_SUBJECTS = _setting("TEST_SUBJECTS")
VAL_SUBJECTS = _setting("VAL_SUBJECTS")

FEAT_DIM = _setting("FEAT_DIM")
UNET_CHANNELS = _setting("UNET_CHANNELS")
CONTEXT_SLICES = _setting("CONTEXT_SLICES")
CONTEXT_FUSION_LAYERS = _setting("CONTEXT_FUSION_LAYERS")
DROPOUT = _setting("DROPOUT")
LAMBDA_FODF = _setting("LAMBDA")
LAMBDA_FODF_BAND = _setting("LAMBDA_BAND")
LAMBDA_FODF_CORR = _setting("LAMBDA_CORR")
LAMBDA_FODF_ANISO_CORR = _setting("LAMBDA_ANISO_CORR")
LAMBDA_FODF_SF = _setting("LAMBDA_SF")
LAMBDA_FODF_PEAK = _setting("LAMBDA_PEAK")
LAMBDA_FODF_NONNEG = _setting("LAMBDA_NONNEG")
LAMBDA_FODF_POWER = _setting("LAMBDA_POWER")
FODF_LOSS_SPHERE = _setting("LOSS_SPHERE")
FODF_SF_CHUNK_SIZE = _setting("SF_CHUNK_SIZE")
FODF_PEAK_TOPK = _setting("PEAK_TOPK")
FODF_PEAK_WEIGHT = _setting("PEAK_WEIGHT")
FODF_PEAK_GAMMA = _setting("PEAK_GAMMA")
FODF_PEAK_REL_THRESHOLD = _setting("PEAK_REL_THRESHOLD")
FODF_BAND_WEIGHT_GAMMA = _setting("BAND_WEIGHT_GAMMA")
FODF_POWER_WEIGHT_GAMMA = _setting("POWER_WEIGHT_GAMMA")
FODF_BAND_SCALE_FLOOR = _setting("BAND_SCALE_FLOOR")
FODF_POWER_SCALE_FLOOR = _setting("POWER_SCALE_FLOOR")
FODF_ANISO_MIN_L = _setting("ANISO_MIN_L")

EPOCHS = _setting("EPOCHS")
BATCH_SIZE = _setting("BATCH_SIZE")
LEARNING_RATE = _setting("LEARNING_RATE")
WEIGHT_DECAY = _setting("WEIGHT_DECAY")
PATIENCE = _setting("PATIENCE")
GRAD_CLIP = _setting("GRAD_CLIP")
WARMUP_EPOCHS = _setting("WARMUP_EPOCHS")
VIS_EVERY = _setting("VIS_EVERY")
NUM_WORKERS = _setting("NUM_WORKERS")
PREFETCH_FACTOR = _setting("PREFETCH_FACTOR")
PIN_MEMORY = _setting("PIN_MEMORY")
PRELOAD_FODF = _setting("PRELOAD")
AMP = _setting("AMP")
AMP_DTYPE = _setting("AMP_DTYPE")
CHANNELS_LAST = _setting("CHANNELS_LAST")
COMPILE = _setting("COMPILE")
COMPILE_MODE = _setting("COMPILE_MODE")
FUSED_ADAMW = _setting("FUSED_ADAMW")
DETERMINISTIC = _setting("DETERMINISTIC")
REQUIRE_CUDA = _setting("REQUIRE_CUDA")

PROFILE = _setting("PROFILE")
PROFILE_WAIT = _setting("PROFILE_WAIT")
PROFILE_WARMUP = _setting("PROFILE_WARMUP")
PROFILE_ACTIVE = _setting("PROFILE_ACTIVE")
PROFILE_REPEAT = _setting("PROFILE_REPEAT")
PROFILE_RECORD_SHAPES = _setting("PROFILE_RECORD_SHAPES")
PROFILE_MEMORY = _setting("PROFILE_MEMORY")
PROFILE_WITH_STACK = _setting("PROFILE_WITH_STACK")
PROFILE_WITH_FLOPS = _setting("PROFILE_WITH_FLOPS")
PROFILE_ROW_LIMIT = _setting("PROFILE_ROW_LIMIT")
PROFILE_EXIT_AFTER_CAPTURE = _setting("PROFILE_EXIT_AFTER_CAPTURE")

P2S_MODEL = _setting("P2S_MODEL")
P2S_ALPHA = _setting("P2S_ALPHA")
P2S_SHIFT_INTENSITY = _setting("P2S_SHIFT_INTENSITY")
P2S_CLIP_NEGATIVE = _setting("P2S_CLIP_NEGATIVE")
P2S_B0_DENOISING = _setting("P2S_B0_DENOISING")
MPPCA_PATCH_RADIUS = _setting("MPPCA_PATCH_RADIUS")
MPPCA_PCA_METHOD = _setting("MPPCA_PCA_METHOD")
