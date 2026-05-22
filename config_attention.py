"""Config overrides for the QSpaceAttentionNet experiment."""

from config import *  # noqa: F403

TRAIN_OUT_DIR = "runs/production_attention_6d_tiny"
EVAL_OUT_DIR = "runs/evaluation_attention"
EVAL_DEFAULT_CHECKPOINT = "runs/production_attention_6d_tiny/best_model.pt"

# Keep the same task/loss/data defaults as QSpaceUNet. These only control the
# new q-space attention encoder.
ATTENTION_HEADS = 4
ATTENTION_DEPTH = 2
ATTENTION_DROPOUT = DROPOUT  # noqa: F405
ATTENTION_GRAD_HIDDEN = 128
ATTENTION_USE_SIGNAL_STATS = True
