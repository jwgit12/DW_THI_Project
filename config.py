"""Centralised hyperparameters for preprocessing, training, and evaluation.

Import from here instead of hardcoding values so that every script uses
the same constants.  CLI argument defaults should reference these too.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ZARR_PATH = "dataset/default_clean.zarr"
DATASET_QC_DIR = "dataset/default_clean_qc"
TRAIN_OUT_DIR = "runs/production_6d_tiny"
EVAL_OUT_DIR = "runs/evaluation"
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Degradation — applied on-the-fly in the training DataLoader so every epoch
# sees different noise and k-space cutouts for the same clean slice. The
# dataset build stores only the clean DWI; there is no pre-degraded array.
# ─────────────────────────────────────────────────────────────────────────────
NOISE_MIN = 0.05              # minimum relative Gaussian noise level
NOISE_MAX = 0.25              # maximum relative Gaussian noise level
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

# Brain masks are computed once in build_dataset.py and stored in Zarr.
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
FEAT_DIM = 16                # q-space encoder feature dimension (matches channels[0])
UNET_CHANNELS = [16, 32, 64]  # 3 encoder levels; factor=8 fits (132, 130) easily
DROPOUT = 0.1                # spatial dropout rate in U-Net conv blocks
LAMBDA_SCALAR = 0.3          # weight for FA/MD auxiliary loss
LAMBDA_EDGE = 0.1            # weight for FA spatial-gradient (edge) loss

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25                # early stopping patience (epochs)
GRAD_CLIP = 1.0              # gradient norm clipping value
WARMUP_EPOCHS = 5            # linear LR warmup before cosine annealing
VIS_EVERY = 1                # TensorBoard validation figure cadence
NUM_WORKERS = 0            # -1 = OS-aware auto
PREFETCH_FACTOR = 2          # DataLoader prefetch per worker
AMP = False
AMP_DTYPE = "auto"           # 'auto' | 'bf16' | 'fp16'
CHANNELS_LAST = False         # CUDA-only channels-last conv layout
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
# fODF training mode
# ─────────────────────────────────────────────────────────────────────────────
# The standard FA/MD path uses the unprefixed values above. The isolated
# ``fodf`` package reads these FODF_* values through ``fodf.defaults`` so both
# modes still share this single config file.
FODF_DATASET_ZARR_PATH = "dataset/default_odf_l4.zarr"
FODF_DATASET_QC_DIR = "dataset/default_odf_qc"
FODF_TRAIN_OUT_DIR = "runs/production_fodf_l4_tiny"
FODF_EVAL_OUT_DIR = "runs/evaluation_fodf"

FODF_NOISE_MIN = 0.05
FODF_NOISE_MAX = 0.25
FODF_KEEP_FRACTION_MIN = 0.5
FODF_KEEP_FRACTION_MAX = 0.75
FODF_KEEP_FRACTION = 0.6
FODF_EVAL_KEEP_FRACTION = 0.6
FODF_EVAL_NOISE_LEVEL = 0.055
FODF_EVAL_REPEATS = 3
FODF_EVAL_KEEP_FRACTION_MIN = FODF_KEEP_FRACTION_MIN
FODF_EVAL_KEEP_FRACTION_MAX = FODF_KEEP_FRACTION_MAX
FODF_EVAL_NOISE_MIN = FODF_NOISE_MIN
FODF_EVAL_NOISE_MAX = FODF_NOISE_MAX
FODF_EVAL_DEGRADE_SEED = EVAL_DEGRADE_SEED
FODF_EVAL_DEFAULT_CHECKPOINT = "runs/production_fodf_context_l4/best_model.pt"
FODF_EVAL_INFER_BATCH_SIZE = EVAL_INFER_BATCH_SIZE

FODF_RANDOM_SLICE_AXIS = False
FODF_SLICE_AXES = (2,)
FODF_AUG_FLIP = True
FODF_AUG_INTENSITY = 0.1
FODF_AUG_VOLUME_DROPOUT = 0.02

FODF_SH_ORDER = 4
FODF_TRAIN_SH_ORDER = 4
FODF_RESPONSE_ROI_RADII = 10
FODF_RESPONSE_FA_THR = 0.7
FODF_SINGLE_SHELL_TOL = 100.0

FODF_FEAT_DIM = 32
FODF_UNET_CHANNELS = [32, 64, 128]
FODF_CONTEXT_SLICES = 5
FODF_CONTEXT_FUSION_LAYERS = 2
FODF_DROPOUT = 0.1
FODF_LAMBDA = 0.25
FODF_LAMBDA_BAND = 0.75
FODF_LAMBDA_CORR = 0.25
FODF_LAMBDA_ANISO_CORR = 0.5
FODF_LAMBDA_SF = 2.0
FODF_LAMBDA_PEAK = 1.5
FODF_LAMBDA_NONNEG = 0.1
FODF_LAMBDA_POWER = 0.75
FODF_LOSS_SPHERE = "symmetric362"  # faster MPS surface loss; use repulsion724 for max angular density
FODF_SF_CHUNK_SIZE = 181           # two chunks for symmetric362, limiting MPS copy overhead
FODF_PEAK_TOPK = 5
FODF_PEAK_WEIGHT = 12.0
FODF_PEAK_GAMMA = 2.0
FODF_PEAK_REL_THRESHOLD = 0.15
FODF_BAND_WEIGHT_GAMMA = 0.5
FODF_POWER_WEIGHT_GAMMA = 0.5
FODF_BAND_SCALE_FLOOR = 0.02
FODF_POWER_SCALE_FLOOR = 0.02
FODF_ANISO_MIN_L = 4

FODF_EPOCHS = 220
FODF_BATCH_SIZE = 8
FODF_LEARNING_RATE = 1e-3
FODF_WEIGHT_DECAY = 5e-5
FODF_PATIENCE = 40
FODF_GRAD_CLIP = 0.0             # disabled by default on MPS; pass --grad_clip 1.0 if needed
FODF_WARMUP_EPOCHS = 8
FODF_VIS_EVERY = 1
FODF_NUM_WORKERS = 0
FODF_PREFETCH_FACTOR = 2
FODF_PIN_MEMORY = False
FODF_PRELOAD = True
FODF_AMP = False
FODF_AMP_DTYPE = "auto"
FODF_CHANNELS_LAST = False
FODF_COMPILE = "auto"
FODF_COMPILE_MODE = "reduce-overhead"
FODF_FUSED_ADAMW = True
FODF_DETERMINISTIC = False
FODF_REQUIRE_CUDA = False

# ─────────────────────────────────────────────────────────────────────────────
# Patch2Self baseline
# ─────────────────────────────────────────────────────────────────────────────
P2S_MODEL = "ridge"            # 'ols' | 'ridge' | 'lasso'
P2S_ALPHA = 0.001              # regularisation for ridge/lasso
P2S_SHIFT_INTENSITY = True
P2S_CLIP_NEGATIVE = True
P2S_B0_DENOISING = False
P2S_B0_THRESHOLD = 50  

# ─────────────────────────────────────────────────────────────────────────────
# MP-PCA baseline
# ─────────────────────────────────────────────────────────────────────────────
MPPCA_PATCH_RADIUS = 1       # local patch radius in voxels (2 → 5×5×5)
MPPCA_PCA_METHOD = "svd"     # 'eig' (faster) | 'svd' (occasionally more accurate)
MPPCA_USE_MASK = True        # restrict denoising to brain mask voxels

# ─────────────────────────────────────────────────────────────────────────────
# BM4D baseline
# bm4d 4.x only supports 'np' and 'refilter' as string profiles.
# Custom profiles (patch size, search window, n_max) use BM4DProfile objects.
# ─────────────────────────────────────────────────────────────────────────────
BM4D_SIGMA = None            # None = auto-estimate per volume via MAD
BM4D_PROFILE = 'np'         # 'np' (default) | 'refilter' (two-pass Wiener)
BM4D_PATCH_SIZE = 4       
BM4D_SEARCH_WINDOW = 9    
BM4D_N_MAX = 8            
