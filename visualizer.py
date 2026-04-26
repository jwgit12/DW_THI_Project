#!/usr/bin/env python3
"""Lightweight desktop viewer for DW_THI Zarr datasets.

Usage:
    python visualizer.py --zarr_path dataset/default_clean.zarr --dti_checkpoint runs/production_medium_more_noise/best_model.pt --fodf_checkpoint runs/production_fodf/best_model.pt
    python visualizer.py --zarr_path dataset/default_clean.zarr --checkpoint runs/production_multitask/best_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import torch
import zarr
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from matplotlib import colormaps
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as cfg
from dw_thi.augment import degrade_dwi_slice, degrade_dwi_volume
from dw_thi.model import DTI_CHANNELS, QSpaceUNet
from dw_thi.preprocessing import compute_b0_norm, compute_brain_mask_from_dwi
from dw_thi.runtime import get_device
from dw_thi.utils import (
    compute_fa_from_tensor6,
    fit_dti_to_6d,
    sanitize_dti6d,
    tensor6_to_full,
    tensor_to_eig,
)


DEGRADE_SLIDER_STEPS = 1000


def slider_to_float(value: int, low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(low + (high - low) * (value / DEGRADE_SLIDER_STEPS))


def float_to_slider(value: float, low: float, high: float) -> int:
    if high <= low:
        return 0
    clipped = min(max(float(value), low), high)
    return int(round(DEGRADE_SLIDER_STEPS * (clipped - low) / (high - low)))


def stable_degrade_seed(
    subject: str,
    plane: str,
    slice_idx: int,
    keep_slider_value: int,
    noise_slider_value: int,
) -> int:
    payload = (
        f"{cfg.EVAL_DEGRADE_SEED}|{subject}|{plane}|{slice_idx}|"
        f"{keep_slider_value}|{noise_slider_value}"
    )
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _group_shape(group) -> tuple[int, ...]:
    """Return the `(X, Y, Z, N)` shape of a subject group without requiring
    ``input_dwi`` to exist."""
    key = "input_dwi" if "input_dwi" in group.array_keys() else "target_dwi"
    return tuple(group[key].shape)

matplotlib.rcParams["font.family"] = "DejaVu Sans"
FODF_SPHERE = get_sphere(name="repulsion724")

# Pre-compute direction-encoded colour direction per face (|x|→R, |y|→G, |z|→B).
# Face centroid direction on the unit sphere stays constant across all voxels.
_fodf_face_verts = FODF_SPHERE.vertices[FODF_SPHERE.faces]          # (F, 3, 3)
_fodf_face_centroids = _fodf_face_verts.mean(axis=1)                 # (F, 3)
_fodf_face_norms = np.linalg.norm(_fodf_face_centroids, axis=1, keepdims=True)
FODF_FACE_DIRS = _fodf_face_centroids / np.maximum(_fodf_face_norms, 1e-8)  # (F, 3)


PLANE_TO_AXIS = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}

ROW_GROUPS: dict[str, tuple[str, ...]] = {
    "noisy": ("Noisy FA", "Noisy MD", "Noisy Color FA"),
    "dti": ("FA Map", "MD Map", "Color FA"),
    "pred": ("Predicted FA", "Predicted MD", "Predicted Color FA"),
    "mask": ("Brain Mask",),
    "tracts": ("Tracts (Clean)", "Tracts (Predicted)"),
}
ROW_LABELS: dict[str, str] = {
    "noisy": "Noisy Input DTI Fit",
    "dti": "DTI Maps (ground truth)",
    "pred": "NN Predictions",
    "mask": "Precomputed Brain Mask",
    "tracts": "Fiber Tracts",
}

# Tractography parameters (deterministic Euler integration on principal eigvec).
TRACT_FA_THRESHOLD = 0.15
TRACT_STEP_SIZE = 0.5
TRACT_MAX_ANGLE_COS = 0.5      # cos(60°)
TRACT_MAX_STEPS = 200
TRACT_MIN_LENGTH = 12
TRACT_SEED_DENSITY = 0.30      # fraction of valid voxels used as seeds
TRACT_SLICE_TOLERANCE = 0.75   # voxels above/below the slice that still display
TRACT_RENDER_UPSCALE = 6       # super-sample factor for the matplotlib renderer
TRACT_DISPLAY_WIDTH = 560      # pixmap width (px) for tract panels — larger than 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the DW_THI pretext Zarr dataset.")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=cfg.DATASET_ZARR_PATH,
        help="Path to the Zarr store.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject key to open first (for example: subject_010).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Legacy single-checkpoint mode. When --dti_checkpoint and "
            "--fodf_checkpoint are omitted, this checkpoint is reused for both."
        ),
    )
    parser.add_argument(
        "--dti_checkpoint",
        type=str,
        default=None,
        help="Checkpoint used for predicted FA/MD/ColorFA panels and tractography.",
    )
    parser.add_argument(
        "--fodf_checkpoint",
        type=str,
        default=None,
        help="Checkpoint used for predicted fODF visualization.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print dataset summary and exit without opening the GUI.",
    )
    return parser.parse_args()


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[QSpaceUNet, dict[str, object]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    max_n = int(ckpt["max_n"])
    feat_dim = int(ckpt.get("feat_dim", 64))
    channels = tuple(ckpt.get("channels", [64, 128, 256, 512]))
    cholesky = bool(ckpt.get("cholesky", False))
    fodf_channels = int(ckpt.get("fodf_channels", 0))
    # Older DTI/multitask checkpoints predate the dti_channels metadata; default
    # to 6 so they keep loading. New fODF-only checkpoints store dti_channels=0.
    dti_channels = int(ckpt.get("dti_channels", DTI_CHANNELS))
    dti_scale = float(ckpt.get("dti_scale", 1.0))
    max_bval = float(ckpt.get("max_bval", 1000.0))

    model = QSpaceUNet(
        max_n=max_n,
        feat_dim=feat_dim,
        channels=channels,
        cholesky=cholesky,
        fodf_channels=fodf_channels,
        dti_channels=dti_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, {
        "epoch": int(ckpt.get("epoch", -1)),
        "max_n": max_n,
        "fodf_channels": fodf_channels,
        "dti_channels": dti_channels,
        "dti_scale": dti_scale,
        "max_bval": max_bval,
    }


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def normalize_image(arr: np.ndarray, pmin: float = 1.0, pmax: float = 99.0, symmetric: bool = False) -> np.ndarray:
    """Map an array to [0, 1] for display while handling empty/flat inputs."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)

    data = arr[finite]
    if symmetric:
        vmax = _safe_percentile(np.abs(data), pmax)
        if vmax <= 1e-8:
            return np.zeros(arr.shape, dtype=np.float32)
        scaled = 0.5 + 0.5 * (arr / vmax)
        return np.clip(scaled, 0.0, 1.0)

    lo = _safe_percentile(data, pmin)
    hi = _safe_percentile(data, pmax)
    if hi - lo <= 1e-8:
        lo = float(np.min(data))
        hi = float(np.max(data))
    if hi - lo <= 1e-8:
        return np.zeros(arr.shape, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def format_float(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def summarize_shells(bvals: np.ndarray) -> str:
    rounded = np.rint(np.asarray(bvals, dtype=np.float32)).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    return ", ".join(f"{value} ({count})" for value, count in zip(values, counts))


def rotate_for_display(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 1)


def infer_fodf_sh_order(n_coeffs: int) -> int:
    for order in range(0, 20, 2):
        if (order + 1) * (order + 2) // 2 == int(n_coeffs):
            return order
    raise ValueError(f"Unsupported fODF SH coefficient count: {n_coeffs}")


def reconstruct_fodf_sf(sh_coeffs: np.ndarray) -> np.ndarray:
    sh_coeffs = np.asarray(sh_coeffs, dtype=np.float32)
    sf = sh_to_sf(
        sh_coeffs,
        FODF_SPHERE,
        sh_order_max=infer_fodf_sh_order(sh_coeffs.shape[-1]),
        basis_type="descoteaux07",
        legacy=True,
    )
    sf = np.asarray(sf, dtype=np.float32)
    sf = np.nan_to_num(sf, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(sf, 0.0, None)


def make_pixmap(arr: np.ndarray, cmap: str = "gray", symmetric: bool = False, width: int = 240) -> QPixmap:
    """Convert a 2D or HxWx3 float array to a display-ready QPixmap."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        normalized = normalize_image(rotate_for_display(arr), symmetric=symmetric)
        if cmap == "gray":
            frame = np.ascontiguousarray((normalized * 255).astype(np.uint8))
            image = QImage(
                frame.tobytes(),
                frame.shape[1],
                frame.shape[0],
                frame.shape[1],
                QImage.Format.Format_Grayscale8,
            ).copy()
        else:
            rgb = np.ascontiguousarray((colormaps[cmap](normalized)[..., :3] * 255).astype(np.uint8))
            image = QImage(
                rgb.tobytes(),
                rgb.shape[1],
                rgb.shape[0],
                3 * rgb.shape[1],
                QImage.Format.Format_RGB888,
            ).copy()
    else:
        rgb = np.ascontiguousarray(rotate_for_display(np.clip(arr, 0.0, 1.0)))
        rgb_u8 = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        image = QImage(
            rgb_u8.tobytes(),
            rgb_u8.shape[1],
            rgb_u8.shape[0],
            3 * rgb_u8.shape[1],
            QImage.Format.Format_RGB888,
        ).copy()

    return QPixmap.fromImage(image).scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


def extract_dwi_slice_nhw(array: zarr.Array, plane: str, slice_idx: int) -> np.ndarray:
    if plane == "Axial":
        clean_hwn = np.asarray(array[:, :, slice_idx, :], dtype=np.float32)
    elif plane == "Coronal":
        clean_hwn = np.asarray(array[:, slice_idx, :, :], dtype=np.float32)
    else:
        clean_hwn = np.asarray(array[slice_idx, :, :, :], dtype=np.float32)
    return np.ascontiguousarray(clean_hwn.transpose(2, 0, 1))


def extract_tensor_slice(array: zarr.Array, plane: str, slice_idx: int) -> np.ndarray:
    if plane == "Axial":
        return np.asarray(array[:, :, slice_idx, :], dtype=np.float32)
    if plane == "Coronal":
        return np.asarray(array[:, slice_idx, :, :], dtype=np.float32)
    return np.asarray(array[slice_idx, :, :, :], dtype=np.float32)


def extract_mask_slice(mask: np.ndarray, plane: str, slice_idx: int) -> np.ndarray:
    if plane == "Axial":
        return np.asarray(mask[:, :, slice_idx], dtype=bool)
    if plane == "Coronal":
        return np.asarray(mask[:, slice_idx, :], dtype=bool)
    return np.asarray(mask[slice_idx, :, :], dtype=bool)


def apply_display_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if mask is None:
        return arr
    mask_f = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        mask_f = mask_f[..., None]
    return arr * mask_f


def masked_stats(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    values = arr[finite]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


# ---------------------------------------------------------------------------
# Tractography (deterministic Euler integration of principal eigvec field)
# ---------------------------------------------------------------------------

def _fa_from_evals(evals: np.ndarray) -> np.ndarray:
    md = np.mean(evals, axis=-1, keepdims=True)
    num = np.sqrt(((evals - md) ** 2).sum(axis=-1))
    den = np.sqrt((evals ** 2).sum(axis=-1) + 1e-12)
    return (np.sqrt(1.5) * num / den).astype(np.float32)


def compute_principal_evec_field(tensor6_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pev (X, Y, Z, 3), fa (X, Y, Z))`` from a 6D tensor volume."""
    full = tensor6_to_full(tensor6_volume)
    evals, evecs = tensor_to_eig(full)
    pev = np.ascontiguousarray(evecs[..., :, 0], dtype=np.float32)
    fa = _fa_from_evals(evals)
    return pev, fa


def compute_dti_metrics(tensor6: np.ndarray) -> dict:
    """Single eigendecomposition → ``{fa, md, color_fa}``. Replaces three
    separate ``compute_*_from_tensor6`` calls (each of which does its own
    eigendecomp) on a hot path."""
    full = tensor6_to_full(tensor6)
    evals, evecs = tensor_to_eig(full)
    md_vec = np.mean(evals, axis=-1, keepdims=True)
    md = np.squeeze(md_vec, axis=-1).astype(np.float32)
    num = np.sqrt(((evals - md_vec) ** 2).sum(axis=-1))
    den = np.sqrt((evals ** 2).sum(axis=-1) + 1e-12)
    fa = (np.sqrt(1.5) * num / den).astype(np.float32)
    color_fa = (np.abs(evecs[..., :, 0]) * fa[..., None]).astype(np.float32)
    return {"fa": fa, "md": md, "color_fa": color_fa}


def _trace_streamline(
    seed: np.ndarray,
    pev: np.ndarray,
    valid: np.ndarray,
    sign: int,
    step_size: float,
    max_angle_cos: float,
    max_steps: int,
) -> np.ndarray:
    sx, sy, sz = valid.shape
    pts = [(float(seed[0]), float(seed[1]), float(seed[2]))]
    px, py, pz = pts[0]
    pdx = pdy = pdz = 0.0
    have_prev = False
    for _ in range(max_steps):
        ix = int(px); iy = int(py); iz = int(pz)
        if ix < 0 or iy < 0 or iz < 0 or ix >= sx or iy >= sy or iz >= sz:
            break
        if not valid[ix, iy, iz]:
            break
        vx = float(pev[ix, iy, iz, 0])
        vy = float(pev[ix, iy, iz, 1])
        vz = float(pev[ix, iy, iz, 2])
        if not have_prev:
            if sign < 0:
                vx = -vx; vy = -vy; vz = -vz
        else:
            dot = vx * pdx + vy * pdy + vz * pdz
            if dot < 0.0:
                vx = -vx; vy = -vy; vz = -vz
                dot = -dot
            if dot < max_angle_cos:
                break
        px += step_size * vx
        py += step_size * vy
        pz += step_size * vz
        pts.append((px, py, pz))
        pdx, pdy, pdz = vx, vy, vz
        have_prev = True
    return np.asarray(pts, dtype=np.float32)


def deterministic_track(
    pev: np.ndarray,
    fa: np.ndarray,
    mask: np.ndarray,
    fa_thresh: float = TRACT_FA_THRESHOLD,
    step_size: float = TRACT_STEP_SIZE,
    max_angle_cos: float = TRACT_MAX_ANGLE_COS,
    max_steps: int = TRACT_MAX_STEPS,
    min_streamline_length: int = TRACT_MIN_LENGTH,
    seed_density: float = TRACT_SEED_DENSITY,
    rng_seed: int = 0,
) -> list[np.ndarray]:
    """Generate streamlines (lists of (M, 3) float arrays in voxel coordinates)."""
    pev = np.ascontiguousarray(pev, dtype=np.float32)
    fa = np.ascontiguousarray(fa, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=bool)

    valid = np.ascontiguousarray((fa > fa_thresh) & mask)
    seed_idx = np.argwhere(valid)
    if seed_idx.size == 0:
        return []

    if seed_density < 1.0:
        rng = np.random.default_rng(rng_seed)
        n = max(1, int(round(seed_density * len(seed_idx))))
        sel = rng.choice(len(seed_idx), n, replace=False)
        seed_idx = seed_idx[sel]

    streamlines: list[np.ndarray] = []
    for ijk in seed_idx:
        seed = ijk.astype(np.float32) + 0.5
        forward = _trace_streamline(seed, pev, valid, +1,
                                    step_size, max_angle_cos, max_steps)
        backward = _trace_streamline(seed, pev, valid, -1,
                                     step_size, max_angle_cos, max_steps)
        if len(backward) > 1:
            full = np.concatenate([backward[:0:-1], forward], axis=0)
        else:
            full = forward
        if len(full) >= min_streamline_length:
            streamlines.append(full.astype(np.float32))
    return streamlines


def render_tract_overlay(
    streamlines: list[np.ndarray],
    plane: str,
    slice_idx: int,
    underlay_fa_slice: np.ndarray,
    slice_tolerance: float = TRACT_SLICE_TOLERANCE,
    line_width: float = 0.45,
    upscale: int = TRACT_RENDER_UPSCALE,
) -> np.ndarray:
    """Render tract segments crossing the slice as antialiased lines on a dim FA underlay.

    Returns an ``(H, W, 3)`` RGB image in the same ``(axis_a, axis_b)`` order
    as ``underlay_fa_slice`` — :func:`make_pixmap` will apply the display rotation.
    """
    axis = PLANE_TO_AXIS[plane]
    other_axes = [a for a in (0, 1, 2) if a != axis]

    underlay = normalize_image(np.asarray(underlay_fa_slice, dtype=np.float32))
    h, w = underlay.shape  # (axis_a, axis_b)

    # Render at upscale × resolution for crisp antialiased lines, then return
    # the raw RGB pixels — make_pixmap handles rotation + final downscale.
    dpi = 100
    fig = Figure(figsize=(w * upscale / dpi, h * upscale / dpi), dpi=dpi, frameon=False)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(
        underlay * 0.35, cmap="gray", vmin=0.0, vmax=1.0,
        origin="upper", extent=(0.0, float(w), float(h), 0.0),
        interpolation="nearest", aspect="auto",
    )

    segments: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for sl in streamlines:
        if len(sl) < 2:
            continue
        z = sl[:, axis]
        within = np.abs(z - (slice_idx + 0.5)) <= slice_tolerance
        idx = np.where(within)[0]
        if len(idx) < 2:
            continue
        breaks = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, breaks)
        for run in runs:
            if len(run) < 2:
                continue
            pts = sl[run]
            # imshow with extent (0,w,h,0) plots column-axis as x and row-axis as y;
            # streamline coords are (axis_a, axis_b) → x=axis_b, y=axis_a.
            xs = pts[:, other_axes[1]]
            ys = pts[:, other_axes[0]]
            line = np.column_stack([xs, ys])
            seg_pairs = np.stack([line[:-1], line[1:]], axis=1)  # (K-1, 2, 2)
            # Color each segment by the *local* fiber direction (RGB = |dx,dy,dz|).
            tangent3 = pts[1:] - pts[:-1]
            t3_mags = np.linalg.norm(tangent3, axis=1, keepdims=True)
            t3_mags = np.where(t3_mags < 1e-6, 1.0, t3_mags)
            seg_colors = np.clip(np.abs(tangent3 / t3_mags), 0.0, 1.0)
            segments.append(seg_pairs)
            colors.append(seg_colors)

    if segments:
        all_segments = np.concatenate(segments, axis=0)
        all_colors = np.concatenate(colors, axis=0)
        lc = LineCollection(
            all_segments, colors=all_colors, linewidths=line_width,
            antialiased=True, capstyle="round",
        )
        ax.add_collection(lc)

    ax.set_xlim(0.0, float(w))
    ax.set_ylim(float(h), 0.0)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[..., :3].astype(np.float32) / 255.0


class ClickableImageLabel(QLabel):
    clicked = pyqtSignal(object)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)


class ImagePanel(QGroupBox):
    image_clicked = pyqtSignal(object)

    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.display_shape: tuple[int, int] | None = None
        self.image_label = ClickableImageLabel("No data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(220, 145)
        self.image_label.setStyleSheet(
            "background: #111; color: #ddd; border: 1px solid #333; font-size: 12px;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.clicked.connect(self.image_clicked.emit)
        layout.addWidget(self.image_label, stretch=1)

        self.caption_label = QLabel("")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setStyleSheet("color: #555; font-size: 11px;")
        self.caption_label.setWordWrap(True)
        self.caption_label.setMaximumHeight(34)
        layout.addWidget(self.caption_label)

    def set_pixmap(
        self,
        pixmap: QPixmap,
        caption: str,
        data_shape: tuple[int, int] | None = None,
    ) -> None:
        self.display_shape = data_shape
        self.image_label.setText("")
        self.image_label.setPixmap(pixmap)
        self.caption_label.setText(caption)

    def set_clickable(self, enabled: bool, tooltip: str = "") -> None:
        self.image_label.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.image_label.setToolTip(tooltip if enabled else "")


class FodfComparisonWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("fODF Comparison")
        self.resize(1160, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.info_label = QLabel("Click an FA panel to inspect the local clean/predicted fODF.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(11, 5.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)

    def show_message(self, subject: str, voxel: tuple[int, int, int], message: str) -> None:
        self.setWindowTitle(
            f"fODF Comparison - {subject} voxel ({voxel[0]}, {voxel[1]}, {voxel[2]})"
        )
        self.info_label.setText(message)
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.canvas.draw_idle()

    def update_plot(
        self,
        subject: str,
        voxel: tuple[int, int, int],
        clean_sh: np.ndarray,
        pred_sh: np.ndarray | None,
    ) -> None:
        clean_sf = reconstruct_fodf_sf(clean_sh)
        pred_sf = reconstruct_fodf_sf(pred_sh) if pred_sh is not None else None
        shared_max = max(
            float(np.max(clean_sf)) if clean_sf.size else 0.0,
            float(np.max(pred_sf)) if pred_sf is not None and pred_sf.size else 0.0,
            1e-6,
        )
        self.setWindowTitle(
            f"fODF Comparison - {subject} voxel ({voxel[0]}, {voxel[1]}, {voxel[2]})"
        )

        self.figure.clear()
        axes = [
            self.figure.add_subplot(1, 2, 1, projection="3d"),
            self.figure.add_subplot(1, 2, 2, projection="3d"),
        ]
        surfaces = (
            ("Clean 3D fODF", clean_sf),
            ("Predicted 3D fODF", pred_sf),
        )
        for ax, (title, sf) in zip(axes, surfaces):
            if sf is None:
                ax.text2D(0.5, 0.5, "Predicted fODF unavailable", transform=ax.transAxes,
                          ha="center", va="center")
                ax.set_title(title)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_box_aspect((1.0, 1.0, 1.0))
                ax.view_init(elev=24, azim=36)
                continue
            if float(np.max(sf)) <= 1e-8:
                ax.text2D(0.5, 0.5, "No fODF signal at this voxel", transform=ax.transAxes,
                          ha="center", va="center")
                ax.set_title(title)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
                ax.set_zlim(-1.05, 1.05)
                ax.set_box_aspect((1.0, 1.0, 1.0))
                ax.view_init(elev=24, azim=36)
                ax.grid(True, alpha=0.25)
                continue

            radius = sf / shared_max
            coords = FODF_SPHERE.vertices * radius[:, None]

            # Direction-encoded colours: |direction| * amplitude → DEC convention.
            face_radii = radius[FODF_SPHERE.faces].mean(axis=1)       # (F,)
            face_rgb = np.clip(np.abs(FODF_FACE_DIRS) * face_radii[:, None], 0.0, 1.0)
            face_rgba = np.column_stack(
                [face_rgb, np.ones(len(face_rgb), dtype=np.float32)]
            )

            mesh = ax.plot_trisurf(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                triangles=FODF_SPHERE.faces,
                linewidth=0.0,
                antialiased=True,
                shade=False,
            )
            mesh.set_facecolors(face_rgba)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_zlim(-1.05, 1.05)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=24, azim=36)
            ax.grid(True, alpha=0.25)

        self.figure.tight_layout()
        pred_state = "available" if pred_sf is not None else "unavailable"
        self.info_label.setText(
            f"Subject: {subject} | voxel: ({voxel[0]}, {voxel[1]}, {voxel[2]}) | "
            f"predicted fODF: {pred_state}"
        )
        self.canvas.draw_idle()


def dataset_summary(zarr_path: str) -> str:
    store = zarr.open_group(zarr_path, mode="r")
    subjects = sorted(store.group_keys())
    if not subjects:
        return f"{zarr_path}: no subjects found"

    spatial_shapes = []
    volume_counts = []
    mask_count = 0
    for subject in subjects:
        group = store[subject]
        shape = _group_shape(group)
        spatial_shapes.append(shape[:3])
        volume_counts.append(shape[3])
        if "brain_mask" in set(group.array_keys()):
            mask_count += 1

    unique_shapes = sorted(set(spatial_shapes))
    unique_counts = sorted(set(volume_counts))
    example = store[subjects[0]]
    source = example.attrs.get("source_dwi", "unknown")

    lines = [
        f"Dataset: {Path(zarr_path).resolve()}",
        f"Subjects: {len(subjects)}",
        f"Spatial shapes: {unique_shapes}",
        f"Volumes per subject: {unique_counts} (min={min(volume_counts)}, max={max(volume_counts)})",
        f"Stored brain masks: {mask_count}/{len(subjects)}",
        f"Example subject: {subjects[0]} -> source={source}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Background worker infrastructure
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    metrics_ready = pyqtSignal(str, str, int, object)   # subject, plane, slice_idx, result
    noisy_metrics_ready = pyqtSignal(object, object)    # noisy_key, result
    prediction_ready = pyqtSignal(object, object)        # pred_key, result
    fodf_prediction_ready = pyqtSignal(object, object)   # pred_key, fodf_sh_volume
    clean_tracts_ready = pyqtSignal(str, object)         # subject, streamlines
    pred_tracts_ready = pyqtSignal(object, object)       # pred_tract_key, streamlines


class MetricsWorker(QRunnable):
    """Loads a DTI tensor slice from zarr and computes FA/MD/ColorFA off the main thread."""

    def __init__(self, subject: str, plane: str, slice_idx: int,
                 zarr_group, signals: WorkerSignals):
        super().__init__()
        self.subject = subject
        self.plane = plane
        self.slice_idx = slice_idx
        self.zarr_group = zarr_group
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            tensor6 = extract_tensor_slice(self.zarr_group["target_dti_6d"], self.plane, self.slice_idx)
            result = compute_dti_metrics(tensor6)
            self.signals.metrics_ready.emit(self.subject, self.plane, self.slice_idx, result)
        except Exception as exc:
            print(f"[MetricsWorker] {exc}", file=sys.stderr)


class NoisyMetricsWorker(QRunnable):
    """Fits DTI from one degraded DWI slice and computes noisy FA/MD/ColorFA."""

    def __init__(self, noisy_key: tuple, input_slice_nhw: np.ndarray,
                 bvals: np.ndarray, bvecs: np.ndarray,
                 b0_threshold: float, signals: WorkerSignals):
        super().__init__()
        self.noisy_key = noisy_key
        self.input_slice_nhw = np.ascontiguousarray(input_slice_nhw, dtype=np.float32)
        self.bvals = np.asarray(bvals, dtype=np.float32)
        self.bvecs = np.asarray(bvecs, dtype=np.float32)
        self.b0_threshold = b0_threshold
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            dwi_hwn = np.maximum(self.input_slice_nhw.transpose(1, 2, 0), 0.0)
            bvecs_n3 = self.bvecs.T if self.bvecs.shape[0] == 3 else self.bvecs
            tensor6 = fit_dti_to_6d(
                dwi_hwn,
                self.bvals,
                bvecs_n3=bvecs_n3,
                fit_method=cfg.DTI_FIT_METHOD,
                b0_threshold=self.b0_threshold,
            )
            tensor6 = sanitize_dti6d(tensor6, max_eigenvalue=cfg.MAX_DIFFUSIVITY)
            result = compute_dti_metrics(tensor6)
            self.signals.noisy_metrics_ready.emit(self.noisy_key, result)
        except Exception as exc:
            print(f"[NoisyMetricsWorker] {exc}", file=sys.stderr)


class PredictionWorker(QRunnable):
    """Runs DTI-model inference for one axial slice off the main thread."""

    def __init__(self, pred_key: tuple, subject: str, slice_idx: int,
                 input_slice_nhw: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                 model: QSpaceUNet, device: torch.device,
                 dti_scale: float, max_bval: float, b0_threshold: float,
                 model_lock: threading.Lock, signals: WorkerSignals):
        super().__init__()
        self.pred_key = pred_key
        self.subject = subject
        self.slice_idx = slice_idx
        self.input_slice_nhw = np.ascontiguousarray(input_slice_nhw, dtype=np.float32)
        self.bvals = bvals
        self.bvecs = bvecs
        self.model = model
        self.device = device
        self.dti_scale = dti_scale
        self.max_bval = max_bval
        self.b0_threshold = b0_threshold
        self.model_lock = model_lock
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            bvals = self.bvals.copy()
            bvecs = self.bvecs.copy()
            N = bvals.shape[0]
            max_n = self.model.max_n

            bvals_norm = bvals / self.max_bval
            if N < max_n:
                pad = max_n - N
                bvals_norm = np.pad(bvals_norm, (0, pad))
                bvecs = np.pad(bvecs, ((0, 0), (0, pad)))

            vol_mask = np.zeros(max_n, dtype=np.float32)
            vol_mask[:N] = 1.0

            bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(self.device)
            bvecs_t = torch.from_numpy(bvecs.astype(np.float32)).unsqueeze(0).to(self.device)
            vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(self.device)

            signal = self.input_slice_nhw
            if N < max_n:
                signal = np.pad(signal, ((0, max_n - N), (0, 0), (0, 0)))

            b0_idx = self.bvals < self.b0_threshold
            if b0_idx.any():
                b0_slice = self.input_slice_nhw[:N][b0_idx].mean(axis=0)
            else:
                b0_slice = self.input_slice_nhw[:N].mean(axis=0)
            b0_norm = compute_b0_norm(b0_slice)
            if b0_norm > 0:
                signal = signal / b0_norm

            signal_t = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0).to(self.device)

            with self.model_lock:
                with torch.no_grad():
                    pred = self.model(signal_t, bvals_t, bvecs_t, vol_mask_t)

            pred_dti, _ = self.model.split_outputs(pred)
            pred_tensor6 = pred_dti[0].permute(1, 2, 0).cpu().numpy() / self.dti_scale
            result = compute_dti_metrics(pred_tensor6)
            result["tensor6"] = np.asarray(pred_tensor6, dtype=np.float32)
            self.signals.prediction_ready.emit(self.pred_key, result)
        except Exception as exc:
            print(f"[PredictionWorker] {exc}", file=sys.stderr)


class FodfPredictionWorker(QRunnable):
    """Runs fODF-model inference for one axial slice off the main thread."""

    def __init__(
        self,
        pred_key: tuple,
        subject: str,
        slice_idx: int,
        input_slice_nhw: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        model: QSpaceUNet,
        device: torch.device,
        max_bval: float,
        b0_threshold: float,
        model_lock: threading.Lock,
        signals: WorkerSignals,
    ):
        super().__init__()
        self.pred_key = pred_key
        self.subject = subject
        self.slice_idx = slice_idx
        self.input_slice_nhw = np.ascontiguousarray(input_slice_nhw, dtype=np.float32)
        self.bvals = bvals
        self.bvecs = bvecs
        self.model = model
        self.device = device
        self.max_bval = max_bval
        self.b0_threshold = b0_threshold
        self.model_lock = model_lock
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            bvals = self.bvals.copy()
            bvecs = self.bvecs.copy()
            n = bvals.shape[0]
            max_n = self.model.max_n

            bvals_norm = bvals / self.max_bval
            if n < max_n:
                pad = max_n - n
                bvals_norm = np.pad(bvals_norm, (0, pad))
                bvecs = np.pad(bvecs, ((0, 0), (0, pad)))

            vol_mask = np.zeros(max_n, dtype=np.float32)
            vol_mask[:n] = 1.0

            bvals_t = torch.from_numpy(bvals_norm).unsqueeze(0).to(self.device)
            bvecs_t = torch.from_numpy(bvecs.astype(np.float32)).unsqueeze(0).to(self.device)
            vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(self.device)

            signal = self.input_slice_nhw
            if n < max_n:
                signal = np.pad(signal, ((0, max_n - n), (0, 0), (0, 0)))

            b0_idx = self.bvals < self.b0_threshold
            if b0_idx.any():
                b0_slice = self.input_slice_nhw[:n][b0_idx].mean(axis=0)
            else:
                b0_slice = self.input_slice_nhw[:n].mean(axis=0)
            b0_norm = compute_b0_norm(b0_slice)
            if b0_norm > 0:
                signal = signal / b0_norm

            signal_t = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0).to(self.device)

            with self.model_lock:
                with torch.no_grad():
                    pred = self.model(signal_t, bvals_t, bvecs_t, vol_mask_t)

            _, pred_fodf = self.model.split_outputs(pred)
            if pred_fodf is None:
                raise RuntimeError("The selected fODF checkpoint does not provide an fODF head.")
            fodf_sh = pred_fodf[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            self.signals.fodf_prediction_ready.emit(self.pred_key, fodf_sh)
        except Exception as exc:
            print(f"[FodfPredictionWorker] {exc}", file=sys.stderr)


class CleanTractWorker(QRunnable):
    """Computes deterministic streamlines from the ground-truth DTI volume."""

    def __init__(self, subject: str, tensor6_volume: np.ndarray,
                 brain_mask: np.ndarray, signals: WorkerSignals):
        super().__init__()
        self.subject = subject
        self.tensor6_volume = np.ascontiguousarray(tensor6_volume, dtype=np.float32)
        self.brain_mask = np.ascontiguousarray(brain_mask, dtype=bool)
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            tensor6 = sanitize_dti6d(self.tensor6_volume, max_eigenvalue=cfg.MAX_DIFFUSIVITY)
            pev, fa = compute_principal_evec_field(tensor6)
            streamlines = deterministic_track(pev, fa, self.brain_mask, rng_seed=0)
            self.signals.clean_tracts_ready.emit(self.subject, streamlines)
        except Exception as exc:
            print(f"[CleanTractWorker] {exc}", file=sys.stderr)


class PredTractWorker(QRunnable):
    """Predicts the full DTI tensor volume axial-slice-by-slice and tracks it."""

    def __init__(self, pred_tract_key: tuple, target_dwi_xyzn: np.ndarray,
                 brain_mask: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                 keep_fraction: float, noise_level: float, seed: int,
                 model: QSpaceUNet, device: torch.device,
                 dti_scale: float, max_bval: float, b0_threshold: float,
                 model_lock: threading.Lock, signals: WorkerSignals):
        super().__init__()
        self.pred_tract_key = pred_tract_key
        self.target_dwi_xyzn = np.ascontiguousarray(target_dwi_xyzn, dtype=np.float32)
        self.brain_mask = np.ascontiguousarray(brain_mask, dtype=bool)
        self.bvals = np.asarray(bvals, dtype=np.float32)
        self.bvecs = np.asarray(bvecs, dtype=np.float32)
        self.keep_fraction = float(keep_fraction)
        self.noise_level = float(noise_level)
        self.seed = int(seed)
        self.model = model
        self.device = device
        self.dti_scale = float(dti_scale)
        self.max_bval = float(max_bval)
        self.b0_threshold = float(b0_threshold)
        self.model_lock = model_lock
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            degraded = degrade_dwi_volume(
                self.target_dwi_xyzn,
                self.keep_fraction,
                self.noise_level,
                seed=self.seed,
            )

            x, y, z, n = degraded.shape
            max_n = self.model.max_n
            bvals_norm = self.bvals / self.max_bval
            bvecs = self.bvecs.astype(np.float32)
            if n < max_n:
                bvals_norm = np.pad(bvals_norm, (0, max_n - n))
                bvecs = np.pad(bvecs, ((0, 0), (0, max_n - n)))
            vol_mask = np.zeros(max_n, dtype=np.float32)
            vol_mask[:n] = 1.0

            bvals_t = torch.from_numpy(bvals_norm.astype(np.float32)).unsqueeze(0).to(self.device)
            bvecs_t = torch.from_numpy(bvecs).unsqueeze(0).to(self.device)
            vol_mask_t = torch.from_numpy(vol_mask).unsqueeze(0).to(self.device)

            b0_idx = self.bvals < self.b0_threshold
            tensor6_volume = np.zeros((x, y, z, 6), dtype=np.float32)
            with self.model_lock, torch.no_grad():
                for k in range(z):
                    slice_nhw = np.ascontiguousarray(
                        degraded[:, :, k, :].transpose(2, 0, 1)
                    )
                    if b0_idx.any():
                        b0_slice = slice_nhw[b0_idx].mean(axis=0)
                    else:
                        b0_slice = slice_nhw.mean(axis=0)
                    b0_norm = compute_b0_norm(b0_slice)
                    if n < max_n:
                        signal = np.pad(slice_nhw, ((0, max_n - n), (0, 0), (0, 0)))
                    else:
                        signal = slice_nhw
                    if b0_norm > 0:
                        signal = signal / b0_norm
                    signal_t = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0).to(self.device)
                    pred = self.model(signal_t, bvals_t, bvecs_t, vol_mask_t)
                    pred_dti, _ = self.model.split_outputs(pred)
                    tensor6_volume[:, :, k, :] = (
                        pred_dti[0].permute(1, 2, 0).cpu().numpy() / self.dti_scale
                    )

            tensor6_volume = sanitize_dti6d(tensor6_volume, max_eigenvalue=cfg.MAX_DIFFUSIVITY)
            pev, fa = compute_principal_evec_field(tensor6_volume)
            streamlines = deterministic_track(pev, fa, self.brain_mask, rng_seed=0)
            self.signals.pred_tracts_ready.emit(self.pred_tract_key, streamlines)
        except Exception as exc:
            print(f"[PredTractWorker] {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------

class DatasetViewer(QMainWindow):
    def __init__(
        self,
        zarr_path: str,
        initial_subject: str | None = None,
        dti_checkpoint_path: str | None = None,
        fodf_checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.store = zarr.open_group(zarr_path, mode="r")
        all_subjects = sorted(self.store.group_keys())
        if not all_subjects:
            raise RuntimeError(f"No subjects found in {zarr_path}")

        has_any_checkpoint = dti_checkpoint_path is not None or fodf_checkpoint_path is not None
        if has_any_checkpoint:
            allowed_ids = set(cfg.TEST_SUBJECTS + cfg.VAL_SUBJECTS)
            self.subjects = [s for s in all_subjects if any(s.startswith(sid) for sid in allowed_ids)]
            if not self.subjects:
                raise RuntimeError("No test/val subjects found in dataset")
        else:
            self.subjects = all_subjects

        self.current_group = None
        self.current_subject = ""
        self.current_shape = (0, 0, 0, 0)
        self.current_bvals = np.array([], dtype=np.float32)
        self.current_bvecs = np.empty((3, 0), dtype=np.float32)
        self.current_brain_mask: np.ndarray | None = None
        self.current_brain_mask_source = ""
        self.brain_mask_cache: dict[str, np.ndarray] = {}
        self.brain_mask_source_cache: dict[str, str] = {}
        self.degraded_slice_cache: dict[tuple, np.ndarray] = {}
        self.noisy_metric_cache: dict[tuple, dict] = {}
        self.slice_metric_cache: dict[tuple, dict] = {}
        self.pred_cache: dict[tuple, dict] = {}
        self.fodf_pred_cache: dict[tuple, np.ndarray] = {}
        self.clean_tract_cache: dict[str, list[np.ndarray]] = {}
        self.pred_tract_cache: dict[tuple, list[np.ndarray]] = {}
        self.pending_fodf_request: dict[str, object] | None = None
        self.fodf_window: FodfComparisonWindow | None = None
        self._pending_noisy_metrics: set[tuple] = set()
        self._pending_metrics: set[tuple] = set()
        self._pending_predictions: set[tuple] = set()
        self._pending_fodf_predictions: set[tuple] = set()
        self._pending_clean_tracts: set[str] = set()
        self._pending_pred_tracts: set[tuple] = set()
        self.tracts_enabled = False

        self.model: QSpaceUNet | None = None
        self.fodf_model: QSpaceUNet | None = None
        self.fodf_channels = 0
        self.dti_scale = 1.0
        self.max_bval = 1000.0
        self.fodf_max_bval = 1000.0
        self.device = torch.device("cpu")
        self.model_lock = threading.Lock()

        if has_any_checkpoint:
            self.device = get_device()
        dti_meta: dict[str, object] | None = None
        resolved_dti = Path(dti_checkpoint_path).resolve() if dti_checkpoint_path else None
        resolved_fodf = Path(fodf_checkpoint_path).resolve() if fodf_checkpoint_path else None

        if dti_checkpoint_path is not None:
            self.model, dti_meta = load_checkpoint_model(dti_checkpoint_path, self.device)
            if int(dti_meta.get("dti_channels", DTI_CHANNELS)) <= 0:
                raise RuntimeError(
                    f"{dti_checkpoint_path} has no DTI head (dti_channels=0). "
                    "Pass it as --fodf_checkpoint instead."
                )
            self.dti_scale = float(dti_meta["dti_scale"])
            self.max_bval = float(dti_meta["max_bval"])
            print(
                f"DTI model loaded from {dti_checkpoint_path} "
                f"(epoch {dti_meta['epoch']}, device={self.device})"
            )

        if fodf_checkpoint_path is not None:
            if resolved_dti is not None and resolved_fodf == resolved_dti and self.model is not None:
                self.fodf_model = self.model
                self.fodf_channels = int(dti_meta["fodf_channels"]) if dti_meta is not None else 0
                self.fodf_max_bval = float(dti_meta["max_bval"]) if dti_meta is not None else self.max_bval
                if self.fodf_channels > 0:
                    print(f"Reusing {fodf_checkpoint_path} for predicted fODFs.")
            else:
                self.fodf_model, fodf_meta = load_checkpoint_model(fodf_checkpoint_path, self.device)
                self.fodf_channels = int(fodf_meta["fodf_channels"])
                self.fodf_max_bval = float(fodf_meta["max_bval"])
                if self.fodf_channels > 0:
                    print(
                        f"fODF model loaded from {fodf_checkpoint_path} "
                        f"(epoch {fodf_meta['epoch']}, device={self.device})"
                    )
                else:
                    print(
                        f"[visualizer] {fodf_checkpoint_path} has no fODF head; "
                        "predicted fODF visualization is disabled.",
                        file=sys.stderr,
                    )
                    self.fodf_model = None

        self.worker_signals = WorkerSignals()
        self.worker_signals.metrics_ready.connect(self._on_metrics_ready)
        self.worker_signals.noisy_metrics_ready.connect(self._on_noisy_metrics_ready)
        self.worker_signals.prediction_ready.connect(self._on_prediction_ready)
        self.worker_signals.fodf_prediction_ready.connect(self._on_fodf_prediction_ready)
        self.worker_signals.clean_tracts_ready.connect(self._on_clean_tracts_ready)
        self.worker_signals.pred_tracts_ready.connect(self._on_pred_tracts_ready)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        self.setWindowTitle("DW_THI Dataset Viewer")
        self._set_compact_window_size()
        self._build_ui()
        self._load_subject_by_name(initial_subject or self.subjects[0])

    def _set_compact_window_size(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.setMinimumSize(960, 680 if self.model is not None else 580)
            self.resize(1180, 820 if self.model is not None else 720)
            return

        available = screen.availableGeometry()
        min_width = min(960, max(760, int(available.width() * 0.70)))
        min_height_target = 680 if self.model is not None else 580
        min_height = min(min_height_target, max(520, int(available.height() * 0.72)))
        self.setMinimumSize(min_width, min_height)

        target_width = min(1220, int(available.width() * 0.94))
        target_height = min(820 if self.model is not None else 700, int(available.height() * 0.90))
        self.resize(max(min_width, target_width), max(min_height, target_height))

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        page = QHBoxLayout(root)
        page.setContentsMargins(6, 6, 6, 6)
        page.setSpacing(6)
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.setSpacing(6)
        right.setSpacing(6)
        page.addLayout(left, stretch=5)
        page.addLayout(right, stretch=1)

        # ── Controls bar ──────────────────────────────────────────────────
        controls = QHBoxLayout()
        controls.setSpacing(6)
        left.addLayout(controls)

        controls.addWidget(QLabel("Subject"))
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(self.subjects)
        self.subject_combo.currentTextChanged.connect(self._load_subject_by_name)
        controls.addWidget(self.subject_combo, stretch=1)

        controls.addWidget(QLabel("Plane"))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(list(PLANE_TO_AXIS))
        self.plane_combo.currentTextChanged.connect(self._handle_plane_change)
        controls.addWidget(self.plane_combo)

        controls.addWidget(QLabel("Slice"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self._update_view)
        controls.addWidget(self.slice_slider, stretch=2)
        self.slice_label = QLabel("0 / 0")
        controls.addWidget(self.slice_label)

        # ── Image rows (QVBoxLayout so hidden rows collapse) ──────────────
        rows_container = QVBoxLayout()
        rows_container.setSpacing(4)
        left.addLayout(rows_container, stretch=1)

        self.panels: dict[str, ImagePanel] = {}
        self.row_widgets: dict[str, QWidget] = {}

        for row_key, panel_names in ROW_GROUPS.items():
            if row_key == "pred" and self.model is None:
                continue
            visible_names = panel_names
            if row_key == "tracts" and self.model is None:
                visible_names = ("Tracts (Clean)",)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            for name in visible_names:
                panel = ImagePanel(name)
                row_layout.addWidget(panel)
                self.panels[name] = panel
            self.row_widgets[row_key] = row_widget
            rows_container.addWidget(row_widget, stretch=1)
            if row_key == "tracts":
                row_widget.setVisible(False)  # hidden until user enables tract panels

        if self.fodf_model is not None and self.fodf_channels > 0:
            fa_click_tooltip = "Click to open the clean vs predicted 3D fODF for the selected voxel."
        else:
            fa_click_tooltip = "Click to open the clean 3D fODF for the selected voxel."
        if "FA Map" in self.panels:
            self.panels["FA Map"].set_clickable(True, fa_click_tooltip)
            self.panels["FA Map"].image_clicked.connect(
                lambda pos, panel_name="FA Map": self._handle_fa_panel_click(panel_name, pos)
            )
        if "Predicted FA" in self.panels:
            self.panels["Predicted FA"].set_clickable(True, fa_click_tooltip)
            self.panels["Predicted FA"].image_clicked.connect(
                lambda pos, panel_name="Predicted FA": self._handle_fa_panel_click(panel_name, pos)
            )

        # ── Right panel ───────────────────────────────────────────────────
        info_box = QGroupBox("Subject Summary")
        info_layout = QVBoxLayout(info_box)
        self.subject_info_label = QLabel("")
        self.subject_info_label.setWordWrap(True)
        self.subject_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(self.subject_info_label)
        right.addWidget(info_box)

        selection_box = QGroupBox("Current Selection")
        selection_layout = QVBoxLayout(selection_box)
        self.selection_info_label = QLabel("")
        self.selection_info_label.setWordWrap(True)
        self.selection_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        selection_layout.addWidget(self.selection_info_label)
        right.addWidget(selection_box)

        degradation_box = QGroupBox("Input Degradation")
        degradation_layout = QVBoxLayout(degradation_box)

        keep_row = QHBoxLayout()
        keep_row.addWidget(QLabel("K-space keep"))
        self.keep_slider = QSlider(Qt.Orientation.Horizontal)
        self.keep_slider.setRange(0, DEGRADE_SLIDER_STEPS)
        self.keep_slider.setSingleStep(1)
        self.keep_slider.setPageStep(50)
        self.keep_slider.setToolTip(
            f"Central k-space fraction kept during degradation "
            f"({cfg.KEEP_FRACTION_MIN:.2f} to {cfg.KEEP_FRACTION_MAX:.2f})."
        )
        self.keep_label = QLabel("")
        self.keep_label.setMinimumWidth(48)
        keep_row.addWidget(self.keep_slider, stretch=1)
        keep_row.addWidget(self.keep_label)
        degradation_layout.addLayout(keep_row)

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise"))
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, DEGRADE_SLIDER_STEPS)
        self.noise_slider.setSingleStep(1)
        self.noise_slider.setPageStep(50)
        self.noise_slider.setToolTip(
            f"Relative Gaussian noise level from training "
            f"({cfg.NOISE_MIN:.2f} to {cfg.NOISE_MAX:.2f})."
        )
        self.noise_label = QLabel("")
        self.noise_label.setMinimumWidth(58)
        noise_row.addWidget(self.noise_slider, stretch=1)
        noise_row.addWidget(self.noise_label)
        degradation_layout.addLayout(noise_row)

        self.keep_slider.setValue(
            float_to_slider(cfg.EVAL_KEEP_FRACTION, cfg.KEEP_FRACTION_MIN, cfg.KEEP_FRACTION_MAX)
        )
        self.noise_slider.setValue(
            float_to_slider(cfg.EVAL_NOISE_LEVEL, cfg.NOISE_MIN, cfg.NOISE_MAX)
        )
        self.keep_slider.valueChanged.connect(self._handle_degradation_change)
        self.noise_slider.valueChanged.connect(self._handle_degradation_change)
        self.keep_slider.sliderReleased.connect(self._update_view)
        self.noise_slider.sliderReleased.connect(self._update_view)
        self._update_degradation_labels()
        right.addWidget(degradation_box)

        # ── Visibility checkboxes ─────────────────────────────────────────
        vis_box = QGroupBox("Visible Panels")
        vis_layout = QVBoxLayout(vis_box)
        vis_layout.setSpacing(2)
        self.panel_checks: dict[str, QCheckBox] = {}

        for row_key, panel_names in ROW_GROUPS.items():
            if row_key == "pred" and self.model is None:
                continue
            visible_names = panel_names
            if row_key == "tracts" and self.model is None:
                visible_names = ("Tracts (Clean)",)
            row_header = QLabel(f"<b>{ROW_LABELS[row_key]}</b>")
            vis_layout.addWidget(row_header)
            for name in visible_names:
                cb = QCheckBox(name)
                cb.setChecked(row_key != "tracts")  # tracts off by default — expensive
                cb.toggled.connect(lambda checked, n=name: self._toggle_panel(n, checked))
                vis_layout.addWidget(cb)
                self.panel_checks[name] = cb

        right.addWidget(vis_box)

        help_box = QGroupBox("Viewer Notes")
        help_layout = QVBoxLayout(help_box)
        help_text = (
            "Row 1: FA/MD/ColorFA from the slider-degraded input slice. "
            "Row 2: DTI-derived maps computed from ground-truth tensors. "
        )
        if self.model is not None:
            help_text += "Row 3: live NN predictions (axial plane only). "
        help_text += (
            "Toggle Tracts (Clean/Predicted) under 'Visible Panels' to enable "
            "deterministic tractography over the full volume; tracts crossing "
            "the current slice are projected onto a dim FA underlay (RGB = "
            "absolute fiber direction). Computation runs once per subject "
            "(clean) or per (subject, keep, noise) (predicted). "
        )
        if self.fodf_model is not None and self.fodf_channels > 0:
            help_text += "Click an FA panel to inspect the clean and predicted 3D fODF at that voxel. "
        else:
            help_text += "Click an FA panel to inspect the clean 3D fODF at that voxel. "
        help_text += "The Brain Mask panel shows the stored Zarr mask slice when present; otherwise the viewer falls back to DIPY median_otsu. "
        help_text += "Noisy, DTI, and NN panels are masked to the target-side brain mask. "
        help_text += "Noisy DTI, ground-truth DTI, and NN panels load in the background — the GUI stays responsive."
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        right.addWidget(help_box)

    def _toggle_panel(self, panel_name: str, visible: bool) -> None:
        if panel_name in self.panels:
            self.panels[panel_name].setVisible(visible)
        # Collapse a row when *all* its panel checkboxes are unchecked.
        # Use checkbox state (not isVisible) — a freshly-shown child of a still-
        # hidden row reports isVisible()==False until the next event loop tick.
        triggered_tracts = False
        for row_key, panel_names in ROW_GROUPS.items():
            if panel_name in panel_names and row_key in self.row_widgets:
                row_visible = any(
                    self.panel_checks[n].isChecked()
                    for n in panel_names if n in self.panel_checks
                )
                self.row_widgets[row_key].setVisible(row_visible)
                if row_key == "tracts" and visible:
                    triggered_tracts = True
        if triggered_tracts:
            self._update_view()

    def _update_degradation_labels(self) -> None:
        self.keep_label.setText(f"{self.keep_fraction:.3f}")
        self.noise_label.setText(f"{100.0 * self.noise_level:.1f}%")

    def _handle_degradation_change(self) -> None:
        self._update_degradation_labels()
        self.degraded_slice_cache.clear()
        self.noisy_metric_cache.clear()
        self.pred_cache.clear()
        self.fodf_pred_cache.clear()
        self.pred_tract_cache.clear()
        self.pending_fodf_request = None
        self._pending_noisy_metrics.clear()
        self._pending_predictions.clear()
        self._pending_fodf_predictions.clear()
        self._pending_pred_tracts.clear()
        self._update_view()

    def _degradation_slider_active(self) -> bool:
        return self.keep_slider.isSliderDown() or self.noise_slider.isSliderDown()

    @property
    def plane(self) -> str:
        return self.plane_combo.currentText()

    def _prediction_result(self, axial_slice_idx: int) -> dict | None:
        return self.pred_cache.get(self._prediction_key(axial_slice_idx))

    def _fodf_prediction_result(self, axial_slice_idx: int) -> np.ndarray | None:
        return self.fodf_pred_cache.get(self._prediction_key(axial_slice_idx))

    def _start_prediction_worker(self, axial_slice_idx: int, update_panels: bool = False) -> None:
        if self.model is None:
            return
        pred_key = self._prediction_key(axial_slice_idx)
        if pred_key in self._pending_predictions or self._degradation_slider_active():
            return

        input_nhw = self._get_degraded_slice_nhw("Axial", axial_slice_idx)
        self._pending_predictions.add(pred_key)
        if update_panels:
            for p in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                self.panels[p].image_label.setPixmap(QPixmap())
                self.panels[p].image_label.setText("Computing…")
                self.panels[p].caption_label.setText("")
        self.thread_pool.start(
            PredictionWorker(
                pred_key,
                self.current_subject, axial_slice_idx,
                input_nhw,
                self.current_bvals.copy(), self.current_bvecs.copy(),
                self.model, self.device,
                self.dti_scale, self.max_bval, cfg.B0_THRESHOLD,
                self.model_lock, self.worker_signals,
            )
        )

    def _start_fodf_prediction_worker(self, axial_slice_idx: int) -> None:
        if self.fodf_model is None or self.fodf_channels <= 0:
            return
        pred_key = self._prediction_key(axial_slice_idx)
        if pred_key in self._pending_fodf_predictions or self._degradation_slider_active():
            return

        input_nhw = self._get_degraded_slice_nhw("Axial", axial_slice_idx)
        self._pending_fodf_predictions.add(pred_key)
        self.thread_pool.start(
            FodfPredictionWorker(
                pred_key,
                self.current_subject,
                axial_slice_idx,
                input_nhw,
                self.current_bvals.copy(),
                self.current_bvecs.copy(),
                self.fodf_model,
                self.device,
                self.fodf_max_bval,
                cfg.B0_THRESHOLD,
                self.model_lock,
                self.worker_signals,
            )
        )

    def _ensure_fodf_window(self) -> FodfComparisonWindow:
        if self.fodf_window is None:
            self.fodf_window = FodfComparisonWindow()
        self.fodf_window.show()
        self.fodf_window.raise_()
        self.fodf_window.activateWindow()
        return self.fodf_window

    def _panel_voxel_from_click(self, panel_name: str, pos) -> tuple[int, int, int] | None:
        panel = self.panels.get(panel_name)
        if panel is None or panel.display_shape is None:
            return None
        pixmap = panel.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        rect = panel.image_label.contentsRect()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        x0 = rect.x() + max(0.0, (rect.width() - pixmap_w) / 2.0)
        y0 = rect.y() + max(0.0, (rect.height() - pixmap_h) / 2.0)
        rel_x = float(pos.x()) - x0
        rel_y = float(pos.y()) - y0
        if rel_x < 0.0 or rel_y < 0.0 or rel_x >= pixmap_w or rel_y >= pixmap_h:
            return None

        axis_a, axis_b = panel.display_shape
        disp_col = min(int(rel_x * axis_a / max(pixmap_w, 1)), axis_a - 1)
        disp_row = min(int(rel_y * axis_b / max(pixmap_h, 1)), axis_b - 1)
        slice_row = int(np.clip(disp_col, 0, axis_a - 1))
        slice_col = int(np.clip((axis_b - 1) - disp_row, 0, axis_b - 1))
        slice_idx = self.slice_slider.value()

        if self.plane == "Axial":
            return slice_row, slice_col, slice_idx
        if self.plane == "Coronal":
            return slice_row, slice_idx, slice_col
        return slice_idx, slice_row, slice_col

    def _get_clean_fodf_sh(self, subject: str, voxel: tuple[int, int, int]) -> np.ndarray | None:
        group = self.store[subject]
        if "target_fodf_sh" not in set(group.array_keys()):
            return None
        x, y, z = voxel
        return np.asarray(group["target_fodf_sh"][x, y, z, :], dtype=np.float32)

    def _render_fodf_request(self, request: dict[str, object]) -> None:
        subject = str(request["subject"])
        voxel = tuple(int(v) for v in request["voxel"])
        clean_sh = self._get_clean_fodf_sh(subject, voxel)
        window = self._ensure_fodf_window()
        if clean_sh is None:
            window.show_message(subject, voxel, "Clean fODF SH coefficients are unavailable in this dataset.")
            return

        pred_sh = self.fodf_pred_cache.get(tuple(request["fodf_key"]))
        if pred_sh is not None:
            x, y, z = voxel
            pred_sh = np.asarray(pred_sh[x, y, :], dtype=np.float32)
        window.update_plot(subject, voxel, clean_sh, pred_sh)

    def _show_fodf_for_voxel(self, voxel: tuple[int, int, int]) -> None:
        subject = self.current_subject
        window = self._ensure_fodf_window()
        clean_sh = self._get_clean_fodf_sh(subject, voxel)
        if clean_sh is None:
            window.show_message(subject, voxel, "Clean fODF SH coefficients are unavailable in this dataset.")
            return

        request = {
            "subject": subject,
            "voxel": voxel,
            "fodf_key": self._prediction_key(voxel[2]),
        }
        self.pending_fodf_request = request

        if self.fodf_model is None:
            window.update_plot(subject, voxel, clean_sh, pred_sh=None)
            return
        if self.fodf_channels <= 0:
            window.update_plot(subject, voxel, clean_sh, pred_sh=None)
            return
        if self._degradation_slider_active():
            window.show_message(subject, voxel, "Release the degradation slider to compute the predicted fODF.")
            return

        pred_result = self._fodf_prediction_result(voxel[2])
        if pred_result is not None:
            self._render_fodf_request(request)
            return

        window.show_message(subject, voxel, "Computing predicted fODF for the selected voxel…")
        self._start_fodf_prediction_worker(voxel[2])

    def _handle_fa_panel_click(self, panel_name: str, pos) -> None:
        voxel = self._panel_voxel_from_click(panel_name, pos)
        if voxel is None:
            return
        self._show_fodf_for_voxel(voxel)

    def _load_subject_by_name(self, subject_name: str) -> None:
        if subject_name not in self.subjects:
            return

        self.current_subject = subject_name
        self.current_group = self.store[subject_name]
        self.current_shape = _group_shape(self.current_group)
        self.current_bvals = np.asarray(self.current_group["bvals"][:], dtype=np.float32)
        self.current_bvecs = np.asarray(self.current_group["bvecs"][:], dtype=np.float32)
        self.current_brain_mask = self._get_brain_mask(subject_name)
        self.current_brain_mask_source = self.brain_mask_source_cache.get(subject_name, "")
        self.degraded_slice_cache.clear()
        self.noisy_metric_cache.clear()
        self.slice_metric_cache.clear()
        self.pred_cache.clear()
        self.fodf_pred_cache.clear()
        self.pred_tract_cache.clear()
        self.pending_fodf_request = None
        self._pending_noisy_metrics.clear()
        self._pending_metrics.clear()
        self._pending_predictions.clear()
        self._pending_fodf_predictions.clear()
        self._pending_pred_tracts.clear()

        subject_index = self.subjects.index(subject_name)
        if self.subject_combo.currentIndex() != subject_index:
            self.subject_combo.blockSignals(True)
            self.subject_combo.setCurrentIndex(subject_index)
            self.subject_combo.blockSignals(False)

        self._reset_slice_slider()
        self._update_subject_summary()
        self._update_view()

    def _get_brain_mask(self, subject_name: str) -> np.ndarray:
        if subject_name in self.brain_mask_cache:
            return self.brain_mask_cache[subject_name]

        group = self.store[subject_name]
        array_keys = set(group.array_keys())
        target_shape = tuple(group["target_dwi"].shape[:3])
        if "brain_mask" in array_keys:
            mask = np.asarray(group["brain_mask"][:], dtype=bool)
            if mask.shape == target_shape and mask.any():
                source = "stored Zarr brain_mask"
            else:
                print(
                    f"[visualizer] Ignoring invalid stored brain_mask for {subject_name}; "
                    "recomputing with DIPY median_otsu.",
                    file=sys.stderr,
                )
                target_dwi = np.asarray(group["target_dwi"][:], dtype=np.float32)
                bvals = np.asarray(group["bvals"][:], dtype=np.float32)
                mask = compute_brain_mask_from_dwi(target_dwi, bvals, cfg.B0_THRESHOLD)
                source = "fallback DIPY median_otsu"
        else:
            print(
                f"[visualizer] {subject_name} has no stored brain_mask; "
                "recomputing with DIPY median_otsu.",
                file=sys.stderr,
            )
            target_dwi = np.asarray(group["target_dwi"][:], dtype=np.float32)
            bvals = np.asarray(group["bvals"][:], dtype=np.float32)
            mask = compute_brain_mask_from_dwi(target_dwi, bvals, cfg.B0_THRESHOLD)
            source = "fallback DIPY median_otsu"

        self.brain_mask_cache[subject_name] = mask
        self.brain_mask_source_cache[subject_name] = source
        return mask

    @property
    def keep_fraction(self) -> float:
        return slider_to_float(
            self.keep_slider.value(),
            cfg.KEEP_FRACTION_MIN,
            cfg.KEEP_FRACTION_MAX,
        )

    @property
    def noise_level(self) -> float:
        return slider_to_float(
            self.noise_slider.value(),
            cfg.NOISE_MIN,
            cfg.NOISE_MAX,
        )

    def _degradation_key(self, plane: str, slice_idx: int) -> tuple:
        return (
            self.current_subject,
            plane,
            int(slice_idx),
            self.keep_slider.value(),
            self.noise_slider.value(),
        )

    def _prediction_key(self, slice_idx: int) -> tuple:
        return (
            self.current_subject,
            int(slice_idx),
            self.keep_slider.value(),
            self.noise_slider.value(),
        )

    def _pred_tract_key(self) -> tuple:
        return (
            self.current_subject,
            self.keep_slider.value(),
            self.noise_slider.value(),
        )

    def _get_degraded_slice_nhw(self, plane: str, slice_idx: int) -> np.ndarray:
        key = self._degradation_key(plane, slice_idx)
        cached = self.degraded_slice_cache.get(key)
        if cached is not None:
            return cached

        clean_nhw = extract_dwi_slice_nhw(self.current_group["target_dwi"], plane, slice_idx)
        rng = np.random.default_rng(
            stable_degrade_seed(
                self.current_subject,
                plane,
                slice_idx,
                self.keep_slider.value(),
                self.noise_slider.value(),
            )
        )
        degraded = degrade_dwi_slice(clean_nhw, self.keep_fraction, self.noise_level, rng)
        self.degraded_slice_cache[key] = degraded
        return degraded

    def _current_mask_slice(self) -> np.ndarray | None:
        if self.current_brain_mask is None:
            return None
        return extract_mask_slice(self.current_brain_mask, self.plane, self.slice_slider.value())

    def _handle_plane_change(self) -> None:
        self._reset_slice_slider()
        self._update_view()

    def _reset_slice_slider(self) -> None:
        axis = PLANE_TO_AXIS[self.plane]
        max_index = self.current_shape[axis] - 1
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max_index)
        self.slice_slider.setValue(max_index // 2)
        self.slice_slider.blockSignals(False)

    def _update_subject_summary(self) -> None:
        source = self.current_group.attrs.get("source_dwi", "unknown")
        shells = summarize_shells(self.current_bvals)
        mask_voxels = int(np.count_nonzero(self.current_brain_mask)) if self.current_brain_mask is not None else 0
        total_voxels = int(np.prod(self.current_shape[:3])) if self.current_shape[:3] else 0
        mask_fraction = 100.0 * mask_voxels / total_voxels if total_voxels else 0.0
        summary = (
            f"Subject: {self.current_subject}\n"
            f"Source: {source}\n"
            f"Shape: {self.current_shape[:3]}  Volumes: {self.current_shape[3]}\n"
            f"Brain mask: {mask_voxels} voxels ({mask_fraction:.1f}%)\n"
            f"Mask source: {self.current_brain_mask_source or 'unknown'}\n"
            f"Shell counts: {shells}"
        )
        self.subject_info_label.setText(summary)

    def _update_view(self) -> None:
        if self.current_group is None:
            return

        slice_idx = self.slice_slider.value()
        self.slice_label.setText(f"{slice_idx} / {self.slice_slider.maximum()}")

        # ── Noisy input DTI fit: degrade the clean slice using the slider values
        # Skip the (cache-miss-expensive) degradation while the keep/noise slider
        # is being dragged — the worker is gated below and would discard it anyway.
        if self._degradation_slider_active():
            input_nhw = None
        else:
            input_nhw = self._get_degraded_slice_nhw(self.plane, slice_idx)
        mask_slice = self._current_mask_slice()
        self.selection_info_label.setText(
            f"Plane: {self.plane}\n"
            f"Slice: {slice_idx}\n"
            f"K-space keep: {self.keep_fraction:.3f}\n"
            f"Noise level: {100.0 * self.noise_level:.1f}%\n"
            f"Noisy DTI fit: {cfg.DTI_FIT_METHOD}\n"
            f"Brain-mask voxels in slice: {int(np.count_nonzero(mask_slice)) if mask_slice is not None else 0}"
        )
        self._apply_brain_mask_panel(mask_slice)

        noisy_key = self._degradation_key(self.plane, slice_idx)
        if noisy_key in self.noisy_metric_cache:
            self._apply_noisy_metrics(self.noisy_metric_cache[noisy_key])
        elif self._degradation_slider_active():
            for p in ("Noisy FA", "Noisy MD", "Noisy Color FA"):
                self.panels[p].image_label.setPixmap(QPixmap())
                self.panels[p].image_label.setText("Release slider to update")
                self.panels[p].caption_label.setText("")
        elif noisy_key not in self._pending_noisy_metrics:
            self._pending_noisy_metrics.add(noisy_key)
            for p in ("Noisy FA", "Noisy MD", "Noisy Color FA"):
                self.panels[p].image_label.setPixmap(QPixmap())
                self.panels[p].image_label.setText("Computing DTI fit…")
                self.panels[p].caption_label.setText("")
            self.thread_pool.start(
                NoisyMetricsWorker(
                    noisy_key,
                    input_nhw,
                    self.current_bvals.copy(), self.current_bvecs.copy(),
                    cfg.B0_THRESHOLD,
                    self.worker_signals,
                )
            )

        # ── DTI metrics: serve from cache or queue background worker ───────
        metrics_key = (self.current_subject, self.plane, slice_idx)
        if metrics_key in self.slice_metric_cache:
            self._apply_metrics(self.slice_metric_cache[metrics_key])
        elif metrics_key not in self._pending_metrics:
            self._pending_metrics.add(metrics_key)
            for p in ("FA Map", "MD Map", "Color FA"):
                self.panels[p].image_label.setPixmap(QPixmap())
                self.panels[p].image_label.setText("Computing…")
                self.panels[p].caption_label.setText("")
            self.thread_pool.start(
                MetricsWorker(
                    self.current_subject, self.plane, slice_idx,
                    self.current_group, self.worker_signals,
                )
            )

        # ── NN predictions: serve from cache or queue background worker ────
        if self.model is not None:
            if self.plane != "Axial":
                for p in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                    self.panels[p].image_label.setPixmap(QPixmap())
                    self.panels[p].image_label.setText("Axial plane only")
                    self.panels[p].caption_label.setText("")
            else:
                pred_key = self._prediction_key(slice_idx)
                if pred_key in self.pred_cache:
                    self._apply_predictions(self.pred_cache[pred_key])
                elif self._degradation_slider_active():
                    for p in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                        self.panels[p].image_label.setPixmap(QPixmap())
                        self.panels[p].image_label.setText("Release slider to update")
                        self.panels[p].caption_label.setText("")
                elif pred_key not in self._pending_predictions:
                    self._start_prediction_worker(slice_idx, update_panels=True)

        # ── Fiber tracts: only compute when the user has the tract row visible
        self._update_tract_panels(slice_idx)

        self.statusBar().showMessage(
            f"{self.current_subject} | {self.plane} slice {slice_idx} | "
            f"keep={self.keep_fraction:.3f} noise={100.0 * self.noise_level:.1f}%"
        )

    def _tract_panel_visible(self, name: str) -> bool:
        cb = self.panel_checks.get(name)
        return cb is not None and cb.isChecked()

    def _underlay_fa_slice(self) -> np.ndarray:
        """FA slice used as a dim anatomical reference under tract overlays."""
        slice_idx = self.slice_slider.value()
        metrics_key = (self.current_subject, self.plane, slice_idx)
        cached = self.slice_metric_cache.get(metrics_key)
        if cached is not None:
            return cached["fa"]
        # Fallback: compute on demand from the target tensor slice.
        tensor6 = extract_tensor_slice(self.current_group["target_dti_6d"], self.plane, slice_idx)
        return np.asarray(compute_fa_from_tensor6(tensor6), dtype=np.float32)

    def _placeholder_panel(self, name: str, message: str) -> None:
        panel = self.panels.get(name)
        if panel is None:
            return
        panel.image_label.setPixmap(QPixmap())
        panel.image_label.setText(message)
        panel.caption_label.setText("")

    def _update_tract_panels(self, slice_idx: int) -> None:
        clean_visible = self._tract_panel_visible("Tracts (Clean)")
        pred_visible = self._tract_panel_visible("Tracts (Predicted)") and self.model is not None
        if not (clean_visible or pred_visible):
            return

        # ── Clean tracts: cached per subject; compute once
        if clean_visible:
            cached = self.clean_tract_cache.get(self.current_subject)
            if cached is not None:
                self._render_tract_panel("Tracts (Clean)", cached, slice_idx, kind="clean")
            elif self.current_subject not in self._pending_clean_tracts:
                self._pending_clean_tracts.add(self.current_subject)
                self._placeholder_panel("Tracts (Clean)", "Tracking…")
                tensor6_volume = np.asarray(
                    self.current_group["target_dti_6d"][:], dtype=np.float32
                )
                self.thread_pool.start(
                    CleanTractWorker(
                        self.current_subject, tensor6_volume,
                        self.current_brain_mask, self.worker_signals,
                    )
                )
            else:
                self._placeholder_panel("Tracts (Clean)", "Tracking…")

        # ── Predicted tracts: cached per (subject, keep, noise) and require model
        if pred_visible:
            key = self._pred_tract_key()
            cached = self.pred_tract_cache.get(key)
            if cached is not None:
                self._render_tract_panel("Tracts (Predicted)", cached, slice_idx, kind="pred")
            elif self._degradation_slider_active():
                self._placeholder_panel("Tracts (Predicted)", "Release slider to update")
            elif key not in self._pending_pred_tracts:
                self._pending_pred_tracts.add(key)
                self._placeholder_panel("Tracts (Predicted)", "Predicting volume + tracking…")
                target_volume = np.asarray(
                    self.current_group["target_dwi"][:], dtype=np.float32
                )
                seed = stable_degrade_seed(
                    self.current_subject, "Volume", -1,
                    self.keep_slider.value(), self.noise_slider.value(),
                )
                self.thread_pool.start(
                    PredTractWorker(
                        key, target_volume, self.current_brain_mask,
                        self.current_bvals.copy(), self.current_bvecs.copy(),
                        self.keep_fraction, self.noise_level, seed,
                        self.model, self.device,
                        self.dti_scale, self.max_bval, cfg.B0_THRESHOLD,
                        self.model_lock, self.worker_signals,
                    )
                )
            else:
                self._placeholder_panel("Tracts (Predicted)", "Predicting volume + tracking…")

    def _render_tract_panel(self, name: str, streamlines: list[np.ndarray],
                            slice_idx: int, kind: str) -> None:
        panel = self.panels.get(name)
        if panel is None:
            return
        try:
            underlay = self._underlay_fa_slice()
        except Exception as exc:
            print(f"[tract underlay] {exc}", file=sys.stderr)
            self._placeholder_panel(name, "Underlay unavailable")
            return
        overlay = render_tract_overlay(streamlines, self.plane, slice_idx, underlay)
        kind_label = "ground-truth" if kind == "clean" else "predicted"
        panel.set_pixmap(
            make_pixmap(overlay, width=TRACT_DISPLAY_WIDTH),
            f"{len(streamlines)} streamlines from {kind_label} tensors",
        )

    def _apply_brain_mask_panel(self, mask_slice: np.ndarray | None) -> None:
        panel = self.panels.get("Brain Mask")
        if panel is None:
            return
        if mask_slice is None:
            panel.image_label.setPixmap(QPixmap())
            panel.image_label.setText("No mask")
            panel.caption_label.setText("")
            return
        voxels = int(np.count_nonzero(mask_slice))
        total = int(mask_slice.size)
        fraction = 100.0 * voxels / total if total else 0.0
        panel.set_pixmap(
            make_pixmap(mask_slice.astype(np.float32), cmap="gray"),
            f"{self.current_brain_mask_source}: {voxels}/{total} voxels ({fraction:.1f}%)",
        )

    def _apply_noisy_metrics(self, metrics: dict) -> None:
        mask_slice = self._current_mask_slice()
        fa_display = apply_display_mask(metrics["fa"], mask_slice)
        md_display = apply_display_mask(metrics["md"], mask_slice)
        color_fa_display = apply_display_mask(metrics["color_fa"], mask_slice)
        fa_mean, fa_max = masked_stats(metrics["fa"], mask_slice)
        md_mean, md_max = masked_stats(metrics["md"], mask_slice)

        self.panels["Noisy FA"].set_pixmap(
            make_pixmap(fa_display, cmap="viridis"),
            f"brain mean={format_float(fa_mean)}  max={format_float(fa_max)}",
        )
        self.panels["Noisy MD"].set_pixmap(
            make_pixmap(md_display, cmap="plasma"),
            f"brain mean={format_float(md_mean)}  max={format_float(md_max)}",
        )
        self.panels["Noisy Color FA"].set_pixmap(
            make_pixmap(color_fa_display),
            "masked noisy principal direction RGB weighted by FA",
        )

    def _apply_metrics(self, metrics: dict) -> None:
        mask_slice = self._current_mask_slice()
        fa_display = apply_display_mask(metrics["fa"], mask_slice)
        md_display = apply_display_mask(metrics["md"], mask_slice)
        color_fa_display = apply_display_mask(metrics["color_fa"], mask_slice)
        fa_mean, fa_max = masked_stats(metrics["fa"], mask_slice)
        md_mean, md_max = masked_stats(metrics["md"], mask_slice)

        self.panels["FA Map"].set_pixmap(
            make_pixmap(fa_display, cmap="viridis"),
            f"brain mean={format_float(fa_mean)}  max={format_float(fa_max)}",
            data_shape=metrics["fa"].shape,
        )
        self.panels["MD Map"].set_pixmap(
            make_pixmap(md_display, cmap="plasma"),
            f"brain mean={format_float(md_mean)}  max={format_float(md_max)}",
        )
        self.panels["Color FA"].set_pixmap(
            make_pixmap(color_fa_display),
            "masked principal direction RGB weighted by FA",
        )

    def _apply_predictions(self, pred_metrics: dict) -> None:
        mask_slice = self._current_mask_slice()
        fa_display = apply_display_mask(pred_metrics["fa"], mask_slice)
        md_display = apply_display_mask(pred_metrics["md"], mask_slice)
        color_fa_display = apply_display_mask(pred_metrics["color_fa"], mask_slice)
        fa_mean, fa_max = masked_stats(pred_metrics["fa"], mask_slice)
        md_mean, md_max = masked_stats(pred_metrics["md"], mask_slice)

        self.panels["Predicted FA"].set_pixmap(
            make_pixmap(fa_display, cmap="viridis"),
            f"brain mean={format_float(fa_mean)}  max={format_float(fa_max)}",
            data_shape=pred_metrics["fa"].shape,
        )
        self.panels["Predicted MD"].set_pixmap(
            make_pixmap(md_display, cmap="plasma"),
            f"brain mean={format_float(md_mean)}  max={format_float(md_max)}",
        )
        self.panels["Predicted Color FA"].set_pixmap(
            make_pixmap(color_fa_display),
            "masked predicted principal direction RGB weighted by FA",
        )

    @pyqtSlot(object, object)
    def _on_noisy_metrics_ready(self, noisy_key: tuple, result: dict) -> None:
        self._pending_noisy_metrics.discard(noisy_key)
        self.noisy_metric_cache[noisy_key] = result
        if noisy_key == self._degradation_key(self.plane, self.slice_slider.value()):
            self._apply_noisy_metrics(result)

    @pyqtSlot(str, str, int, object)
    def _on_metrics_ready(self, subject: str, plane: str, slice_idx: int, result: dict) -> None:
        self._pending_metrics.discard((subject, plane, slice_idx))
        self.slice_metric_cache[(subject, plane, slice_idx)] = result
        if (subject == self.current_subject
                and plane == self.plane
                and slice_idx == self.slice_slider.value()):
            self._apply_metrics(result)

    @pyqtSlot(object, object)
    def _on_prediction_ready(self, pred_key: tuple, result: dict) -> None:
        self._pending_predictions.discard(pred_key)
        self.pred_cache[pred_key] = result
        if (pred_key == self._prediction_key(self.slice_slider.value())
                and self.plane == "Axial"):
            self._apply_predictions(result)

    @pyqtSlot(object, object)
    def _on_fodf_prediction_ready(self, pred_key: tuple, fodf_sh: np.ndarray) -> None:
        self._pending_fodf_predictions.discard(pred_key)
        self.fodf_pred_cache[pred_key] = np.asarray(fodf_sh, dtype=np.float32)
        if self.pending_fodf_request is not None and pred_key == tuple(self.pending_fodf_request["fodf_key"]):
            self._render_fodf_request(self.pending_fodf_request)

    @pyqtSlot(str, object)
    def _on_clean_tracts_ready(self, subject: str, streamlines: list) -> None:
        self._pending_clean_tracts.discard(subject)
        self.clean_tract_cache[subject] = list(streamlines)
        if subject == self.current_subject and self._tract_panel_visible("Tracts (Clean)"):
            self._render_tract_panel(
                "Tracts (Clean)", self.clean_tract_cache[subject],
                self.slice_slider.value(), kind="clean",
            )

    @pyqtSlot(object, object)
    def _on_pred_tracts_ready(self, pred_tract_key: tuple, streamlines: list) -> None:
        self._pending_pred_tracts.discard(pred_tract_key)
        self.pred_tract_cache[pred_tract_key] = list(streamlines)
        if (pred_tract_key == self._pred_tract_key()
                and self._tract_panel_visible("Tracts (Predicted)")):
            self._render_tract_panel(
                "Tracts (Predicted)", self.pred_tract_cache[pred_tract_key],
                self.slice_slider.value(), kind="pred",
            )


def main() -> None:
    args = parse_args()
    if args.summary_only:
        print(dataset_summary(args.zarr_path))
        return

    dti_checkpoint = args.dti_checkpoint
    fodf_checkpoint = args.fodf_checkpoint
    if args.checkpoint is not None and dti_checkpoint is None and fodf_checkpoint is None:
        dti_checkpoint = args.checkpoint
        fodf_checkpoint = args.checkpoint

    app = QApplication(sys.argv)
    viewer = DatasetViewer(
        args.zarr_path,
        initial_subject=args.subject,
        dti_checkpoint_path=dti_checkpoint,
        fodf_checkpoint_path=fodf_checkpoint,
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
