"""Centralised hyperparameters shared across research and data prep.

Import from here instead of hardcoding values so that every script uses
the same constants.  CLI argument defaults should reference these too.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data preparation (build_pretext_dataset.py, functions.py)
# ─────────────────────────────────────────────────────────────────────────────
KEEP_FRACTION = 0.6          # central k-space fraction kept during low-res degradation
NOISE_MIN = 0.01             # minimum relative Gaussian noise level
NOISE_MAX = 0.10             # maximum relative Gaussian noise level

# ─────────────────────────────────────────────────────────────────────────────
# DWI / DTI shared constants
# ─────────────────────────────────────────────────────────────────────────────
B0_THRESHOLD = 50.0          # b-value threshold separating b0 from DWI volumes
DTI_FIT_METHOD = "WLS"       # DTI fitting algorithm: 'WLS' | 'OLS' | 'NLLS'

# ─────────────────────────────────────────────────────────────────────────────
# Subject split (biological subject IDs — all sessions stay together)
# ─────────────────────────────────────────────────────────────────────────────
TEST_SUBJECTS = ["sub-03", "sub-04"]
VAL_SUBJECTS = ["sub-05", "sub-11"]

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DIM = 64                # q-space encoder feature dimension
UNET_CHANNELS = [64, 128, 256, 512]
DROPOUT = 0.1                # spatial dropout rate in U-Net conv blocks
LAMBDA_SCALAR = 0.3          # weight for FA/MD auxiliary loss

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25                # early stopping patience (epochs)
GRAD_CLIP = 1.0              # gradient norm clipping value

# ─────────────────────────────────────────────────────────────────────────────
# Patch2Self baseline
# ─────────────────────────────────────────────────────────────────────────────
P2S_MODEL = "ols"            # 'ols' | 'ridge' | 'lasso'
P2S_ALPHA = 1.0              # regularisation for ridge/lasso
P2S_SHIFT_INTENSITY = True
P2S_CLIP_NEGATIVE = False
P2S_B0_DENOISING = True

# ─────────────────────────────────────────────────────────────────────────────
# MP-PCA baseline
# ─────────────────────────────────────────────────────────────────────────────
MPPCA_PATCH_RADIUS = 2       # local patch radius in voxels (2 → 5×5×5)
MPPCA_PCA_METHOD = "eig"     # 'eig' (faster) | 'svd' (occasionally more accurate)
