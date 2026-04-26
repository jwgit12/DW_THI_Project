"""Centralised hyperparameters for preprocessing, training, and evaluation.

Import from here instead of hardcoding values so that every script uses
the same constants.  CLI argument defaults should reference these too.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ZARR_PATH = "dataset/default_odf.zarr"
DATASET_QC_DIR = "dataset/default_clean_qc"
TRAIN_OUT_DIR = "runs/production_fodf_context_l6"
EVAL_OUT_DIR = "runs/evaluation_multitask"
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Degradation — applied on-the-fly in the training DataLoader so every epoch
# sees different noise and k-space cutouts for the same clean slice. The
# dataset build stores only the clean DWI; there is no pre-degraded array.
# ─────────────────────────────────────────────────────────────────────────────
NOISE_MIN = 0.02              # minimum relative Gaussian noise level
NOISE_MAX = 0.12              # maximum relative Gaussian noise level
KEEP_FRACTION_MIN = 0.5       # min central k-space fraction kept
KEEP_FRACTION_MAX = 0.75      # max central k-space fraction kept

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
EVAL_DEFAULT_CHECKPOINT = "runs/production_multitask/best_model.pt"
EVAL_INFER_BATCH_SIZE = 16

# ─────────────────────────────────────────────────────────────────────────────
# Training-time augmentation
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SLICE_AXIS = False     # quality-first fODF training: match axial validation/inference view
SLICE_AXES = (2,)             # axes to sample from when RANDOM_SLICE_AXIS is True
AUG_FLIP = True               # random physical-mirror flips: signal + tensor + bvecs transform together
AUG_INTENSITY = 0.1           # uniform multiplicative jitter on the input (0 disables)
AUG_VOLUME_DROPOUT = 0.02     # per-volume dropout probability on the input (0 disables)

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
# fODF (single-shell CSD) — fitted at build-time on clean DWI and stored in Zarr
# as spherical-harmonic coefficients (n_coeffs depends on FODF_SH_ORDER:
# 6→28, 8→45, 10→66).
# ─────────────────────────────────────────────────────────────────────────────
FODF_SH_ORDER = 8
TRAIN_FODF_SH_ORDER = 6             # cap training target/order from stored SH coeffs (use 8 for full dataset)
FODF_RESPONSE_ROI_RADII = 10        # ROI radius for auto_response_ssst (voxels)
FODF_RESPONSE_FA_THR = 0.7          # FA threshold for response-function ROI
FODF_SINGLE_SHELL_TOL = 100.0       # b-value tolerance for "single shell" check

# ─────────────────────────────────────────────────────────────────────────────
# Subject split (biological subject IDs — all sessions stay together)
# ─────────────────────────────────────────────────────────────────────────────
TEST_SUBJECTS = ["sub-03", "sub-04"]
VAL_SUBJECTS = ["sub-05", "sub-11"]

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DIM = 128                # q-space encoder feature dimension (matches channels[0])
UNET_CHANNELS = [128, 256, 512]  # 3 encoder levels; factor=8 fits (132, 130) via padding
CONTEXT_SLICES = 5            # odd axial slice stack for fODF shape context (1 disables 2.5D)
CONTEXT_FUSION_LAYERS = 2     # residual 3D blocks before collapsing back to the central slice
DROPOUT = 0.05               # spatial dropout rate in U-Net conv blocks
LAMBDA_FODF = 0.25           # raw SH coefficient anchor; kept low so l=0/l=2 do not dominate
LAMBDA_FODF_BAND = 0.75      # band-balanced SH coefficient reconstruction
LAMBDA_FODF_CORR = 0.25      # full-coefficient cosine surrogate
LAMBDA_FODF_ANISO_CORR = 0.5 # high-order cosine term for angular shape, ignoring isotropic baseline
LAMBDA_FODF_SF = 2.0         # sphere-sampled fODF reconstruction
LAMBDA_FODF_PEAK = 1.5       # extra supervision at the target's strongest fODF lobes
LAMBDA_FODF_NONNEG = 0.1     # discourage physically implausible negative fODF values
LAMBDA_FODF_POWER = 0.75     # relative per-ℓ-band SH power-spectrum match
FODF_LOSS_SPHERE = "repulsion724"  # denser sphere improves peak supervision on RTX 4090
FODF_SF_CHUNK_SIZE = 100     # directions per loss chunk; lower if GPU memory is tight
FODF_PEAK_TOPK = 5           # number of target peak directions supervised per voxel
FODF_PEAK_WEIGHT = 12.0      # surface-loss multiplier near high target fODF values
FODF_PEAK_GAMMA = 2.0        # sharper weighting toward large fODF peaks
FODF_PEAK_REL_THRESHOLD = 0.15  # ignore tiny top-k directions below this relative height
FODF_BAND_WEIGHT_GAMMA = 0.5 # >0 softly upweights high-ℓ detail bands
FODF_POWER_WEIGHT_GAMMA = 0.5
FODF_BAND_SCALE_FLOOR = 0.02 # denominator floor for per-band relative coefficient losses
FODF_POWER_SCALE_FLOOR = 0.02
FODF_ANISO_MIN_L = 4         # high-order cosine uses ℓ >= this value

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 220
BATCH_SIZE = 8
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 40                # early stopping patience (epochs)
GRAD_CLIP = 1.0              # gradient norm clipping value
WARMUP_EPOCHS = 8            # linear LR warmup before cosine annealing
VIS_EVERY = 1                # TensorBoard validation figure cadence
NUM_WORKERS = 4              # fast worker-side preload for context stacks; lower if system RAM is tight
PREFETCH_FACTOR = 4          # DataLoader prefetch per worker
PIN_MEMORY = True
PRELOAD_FODF = True          # cache target_fodf_sh in worker RAM; set False if RAM is tight
AMP = True
AMP_DTYPE = "auto"           # 'auto' | 'bf16' | 'fp16'
CHANNELS_LAST = True         # CUDA-only channels-last conv layout
COMPILE = "auto"             # 'off' | 'auto' | 'on'
COMPILE_MODE = "reduce-overhead"
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
