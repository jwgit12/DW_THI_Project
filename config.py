"""Centralised hyperparameters for preprocessing, training, and evaluation.

Import from here instead of hardcoding values so that every script uses
the same constants.  CLI argument defaults should reference these too.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ZARR_PATH = "dataset/default_clean.zarr"
DATASET_QC_DIR = "dataset/default_clean_qc"
TRAIN_OUT_DIR = "runs/production"
EVAL_OUT_DIR = "runs/evaluation"
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Degradation — applied on-the-fly in the training DataLoader so every epoch
# sees different noise and k-space cutouts for the same clean slice. The
# dataset build stores only the clean DWI; there is no pre-degraded array.
# ─────────────────────────────────────────────────────────────────────────────
NOISE_MIN = 0.01              # minimum relative Gaussian noise level
NOISE_MAX = 0.10              # maximum relative Gaussian noise level
KEEP_FRACTION_MIN = 0.5       # min central k-space fraction kept
KEEP_FRACTION_MAX = 0.7       # max central k-space fraction kept

# Deterministic single-value defaults used at eval time (and as a back-compat
# scalar when callers expect the legacy ``KEEP_FRACTION`` / Zarr attrs).
KEEP_FRACTION = 0.6           # midpoint of training range
EVAL_KEEP_FRACTION = 0.6
EVAL_NOISE_LEVEL = 0.055      # midpoint of training range
EVAL_REPEATS = 3              # number of independent corruptions per subject
EVAL_KEEP_FRACTION_MIN = KEEP_FRACTION_MIN
EVAL_KEEP_FRACTION_MAX = KEEP_FRACTION_MAX
EVAL_NOISE_MIN = NOISE_MIN
EVAL_NOISE_MAX = NOISE_MAX
EVAL_DEGRADE_SEED = 1234      # fixed seed so evaluation is reproducible
EVAL_DEFAULT_CHECKPOINT = "runs/production/best_model.pt"
EVAL_INFER_BATCH_SIZE = 16

# ─────────────────────────────────────────────────────────────────────────────
# Training-time augmentation
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SLICE_AXIS = True      # randomly slice along X (sagittal), Y (coronal), or Z (axial)
SLICE_AXES = (0, 1, 2)        # axes to sample from when RANDOM_SLICE_AXIS is True
AUG_FLIP = True               # random physical-mirror flips: signal + tensor + bvecs transform together
AUG_INTENSITY = 0.1           # uniform multiplicative jitter on the input (0 disables)
AUG_VOLUME_DROPOUT = 0.1      # per-volume dropout probability on the input (0 disables)

# ─────────────────────────────────────────────────────────────────────────────
# DWI / DTI shared constants
# ─────────────────────────────────────────────────────────────────────────────
B0_THRESHOLD = 50.0          # b-value threshold separating b0 from DWI volumes
DTI_FIT_METHOD = "WLS"       # DTI fitting algorithm: 'WLS' | 'OLS' | 'NLLS'
MAX_DIFFUSIVITY = 0.01       # mm²/s, eigenvalue cap for physically plausible DTI

# Brain masks are computed once in build_pretext_dataset.py and stored in Zarr.
# Training/evaluation prefer the stored mask and only fall back to recomputing
# it for old datasets that do not yet have a brain_mask array.
BRAIN_MASK_MEDIAN_RADIUS = 4
BRAIN_MASK_NUMPASS = 4
BRAIN_MASK_DILATE = 1
BRAIN_MASK_FINALIZE = True

# ─────────────────────────────────────────────────────────────────────────────
# Subject split (biological subject IDs — all sessions stay together)
# ─────────────────────────────────────────────────────────────────────────────
TEST_SUBJECTS = ["sub-03", "sub-04"]
VAL_SUBJECTS = ["sub-05", "sub-11"]

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DIM = 128                # q-space encoder feature dimension (matches channels[0])
UNET_CHANNELS = [128, 256, 512]  # 3 encoder levels; factor=8 fits (132, 130) easily
DROPOUT = 0.1                # spatial dropout rate in U-Net conv blocks
LAMBDA_SCALAR = 0.3          # weight for FA/MD auxiliary loss
LAMBDA_EDGE = 0.1            # weight for FA spatial-gradient (edge) loss

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25                # early stopping patience (epochs)
GRAD_CLIP = 1.0              # gradient norm clipping value
WARMUP_EPOCHS = 5            # linear LR warmup before cosine annealing
VIS_EVERY = 1                # TensorBoard validation figure cadence
NUM_WORKERS = 4            # -1 = OS-aware auto
PREFETCH_FACTOR = 2          # DataLoader prefetch per worker
AMP = True
AMP_DTYPE = "auto"           # 'auto' | 'bf16' | 'fp16'
CHANNELS_LAST = True         # CUDA-only channels-last conv layout
COMPILE = "auto"             # 'off' | 'auto' | 'on'
COMPILE_MODE = "max-autotune"
FUSED_ADAMW = True           # CUDA-only fused optimizer when available
DETERMINISTIC = False
REQUIRE_CUDA = False

# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────
PROFILE = False
PROFILE_WAIT = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 4
PROFILE_REPEAT = 1
PROFILE_RECORD_SHAPES = True
PROFILE_MEMORY = True
PROFILE_WITH_STACK = False
PROFILE_WITH_FLOPS = False
PROFILE_ROW_LIMIT = 20
PROFILE_EXIT_AFTER_CAPTURE = False

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
