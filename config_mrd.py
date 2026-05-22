"""Config overrides for the MRD-CNN residual denoising experiment."""

from config import *  # noqa: F403

TRAIN_OUT_DIR = "runs/production_mrd_6d_tiny"
EVAL_OUT_DIR = "runs/evaluation_mrd"
EVAL_DEFAULT_CHECKPOINT = "runs/production_mrd_6d_tiny/best_model.pt"

# Same dataset, task, losses, metrics, and corruption defaults as the standard
# FA/MD path. These knobs only affect the MRD-CNN architecture.
MRD_DENOISE_CHANNELS = 16
MRD_DENOISE_DEPTH = 4
MRD_TENSOR_CHANNELS = FEAT_DIM  # noqa: F405
MRD_TENSOR_DEPTH = 8
MRD_RESIDUAL_SCALE = 0.5
MRD_GRAD_HIDDEN = 64
