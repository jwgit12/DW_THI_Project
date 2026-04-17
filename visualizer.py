#!/usr/bin/env python3
"""Lightweight desktop viewer for DW_THI Zarr datasets.

Usage:
    python3 visualizer.py --zarr_path dataset/default_dataset.zarr --checkpoint research/runs/run_small/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import matplotlib
import numpy as np
import torch
import zarr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import colormaps
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as cfg
from functions import (
    compute_b0_norm,
    compute_brain_mask_from_dwi,
    compute_color_fa_from_tensor6,
    compute_fa_from_tensor6,
    compute_md_from_tensor6,
)
from research.model import QSpaceUNet

matplotlib.rcParams["font.family"] = "DejaVu Sans"


PLANE_TO_AXIS = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}

ROW_GROUPS: dict[str, tuple[str, ...]] = {
    "dwi": ("Input DWI", "Target DWI", "Absolute Difference"),
    "dti": ("FA Map", "MD Map", "Color FA"),
    "pred": ("Predicted FA", "Predicted MD", "Predicted Color FA"),
}
ROW_LABELS: dict[str, str] = {
    "dwi": "DWI Volumes",
    "dti": "DTI Maps (ground truth)",
    "pred": "NN Predictions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the DW_THI pretext Zarr dataset.")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="dataset/pretext_dataset_new.zarr",
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
        help="Path to a QSpaceUNet checkpoint (.pt). Enables live prediction panels.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print dataset summary and exit without opening the GUI.",
    )
    return parser.parse_args()


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


def shell_name(bval: float, b0_threshold: float = 50.0) -> str:
    return "b0" if bval < b0_threshold else f"b={int(round(float(bval)))}"


def summarize_shells(bvals: np.ndarray) -> str:
    rounded = np.rint(np.asarray(bvals, dtype=np.float32)).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    return ", ".join(f"{value} ({count})" for value, count in zip(values, counts))


def rotate_for_display(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 1)


def make_pixmap(arr: np.ndarray, cmap: str = "gray", symmetric: bool = False, width: int = 280) -> QPixmap:
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


def extract_volume_slice(array: zarr.Array, plane: str, slice_idx: int, volume_idx: int) -> np.ndarray:
    if plane == "Axial":
        return np.asarray(array[:, :, slice_idx, volume_idx], dtype=np.float32)
    if plane == "Coronal":
        return np.asarray(array[:, slice_idx, :, volume_idx], dtype=np.float32)
    return np.asarray(array[slice_idx, :, :, volume_idx], dtype=np.float32)


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


class ImagePanel(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)

        self.image_label = QLabel("No data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(280, 220)
        self.image_label.setStyleSheet(
            "background: #111; color: #ddd; border: 1px solid #333; font-size: 12px;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        self.caption_label = QLabel("")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.caption_label)

    def set_pixmap(self, pixmap: QPixmap, caption: str) -> None:
        self.image_label.setPixmap(pixmap)
        self.caption_label.setText(caption)


class BvalsCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(4.0, 2.3), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def update_plot(self, bvals: np.ndarray, current_idx: int) -> None:
        indices = np.arange(bvals.shape[0], dtype=int)
        colors = np.where(bvals < 50, "#4C97FF", "#F4A261")

        self.ax.clear()
        self.ax.scatter(indices, bvals, c=colors, s=22, linewidths=0)
        self.ax.axvline(current_idx, color="#C0392B", linewidth=1.5, alpha=0.95)
        self.ax.scatter([current_idx], [float(bvals[current_idx])], c="#C0392B", s=48, zorder=3)
        self.ax.set_title("b-values per volume")
        self.ax.set_xlabel("Volume index")
        self.ax.set_ylabel("b-value")
        self.ax.grid(alpha=0.25)
        self.ax.set_xlim(-1, max(1, len(indices)))
        self.draw_idle()


def dataset_summary(zarr_path: str) -> str:
    store = zarr.open_group(zarr_path, mode="r")
    subjects = sorted(store.group_keys())
    if not subjects:
        return f"{zarr_path}: no subjects found"

    spatial_shapes = []
    volume_counts = []
    for subject in subjects:
        shape = tuple(store[subject]["input_dwi"].shape)
        spatial_shapes.append(shape[:3])
        volume_counts.append(shape[3])

    unique_shapes = sorted(set(spatial_shapes))
    unique_counts = sorted(set(volume_counts))
    example = store[subjects[0]]
    source = example.attrs.get("source_dwi", "unknown")

    lines = [
        f"Dataset: {Path(zarr_path).resolve()}",
        f"Subjects: {len(subjects)}",
        f"Spatial shapes: {unique_shapes}",
        f"Volumes per subject: {unique_counts} (min={min(volume_counts)}, max={max(volume_counts)})",
        f"Example subject: {subjects[0]} -> source={source}",
    ]
    return "\n".join(lines)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Background worker infrastructure
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    metrics_ready = pyqtSignal(str, str, int, object)   # subject, plane, slice_idx, result
    prediction_ready = pyqtSignal(str, int, object)      # subject, slice_idx, result


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
            result = {
                "fa": np.asarray(compute_fa_from_tensor6(tensor6), dtype=np.float32),
                "md": np.asarray(compute_md_from_tensor6(tensor6), dtype=np.float32),
                "color_fa": np.asarray(compute_color_fa_from_tensor6(tensor6), dtype=np.float32),
            }
            self.signals.metrics_ready.emit(self.subject, self.plane, self.slice_idx, result)
        except Exception as exc:
            print(f"[MetricsWorker] {exc}", file=sys.stderr)


class PredictionWorker(QRunnable):
    """Runs QSpaceUNet inference for one axial slice off the main thread."""

    def __init__(self, subject: str, slice_idx: int,
                 zarr_group, bvals: np.ndarray, bvecs: np.ndarray,
                 model: QSpaceUNet, device: torch.device,
                 dti_scale: float, max_bval: float, b0_threshold: float,
                 model_lock: threading.Lock, signals: WorkerSignals):
        super().__init__()
        self.subject = subject
        self.slice_idx = slice_idx
        self.zarr_group = zarr_group
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

            signal_slice = np.asarray(
                self.zarr_group["input_dwi"][:, :, self.slice_idx, :], dtype=np.float32
            )
            if N < max_n:
                signal_slice = np.pad(signal_slice, ((0, 0), (0, 0), (0, max_n - N)))
            signal = signal_slice.transpose(2, 0, 1)  # (max_n, H, W)

            b0_idx = self.bvals < self.b0_threshold
            if b0_idx.any():
                b0_slice = np.asarray(
                    self.zarr_group["input_dwi"][:, :, self.slice_idx, :], dtype=np.float32
                )[..., b0_idx].mean(axis=-1)
            else:
                b0_slice = signal_slice[..., :N].mean(axis=-1)
            b0_norm = compute_b0_norm(b0_slice)
            if b0_norm > 0:
                signal = signal / b0_norm

            signal_t = torch.from_numpy(signal).unsqueeze(0).to(self.device)

            with self.model_lock:
                with torch.no_grad():
                    pred = self.model(signal_t, bvals_t, bvecs_t, vol_mask_t)

            pred_tensor6 = pred[0].permute(1, 2, 0).cpu().numpy() / self.dti_scale
            result = {
                "fa": np.asarray(compute_fa_from_tensor6(pred_tensor6), dtype=np.float32),
                "md": np.asarray(compute_md_from_tensor6(pred_tensor6), dtype=np.float32),
                "color_fa": np.asarray(compute_color_fa_from_tensor6(pred_tensor6), dtype=np.float32),
            }
            self.signals.prediction_ready.emit(self.subject, self.slice_idx, result)
        except Exception as exc:
            print(f"[PredictionWorker] {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------

class DatasetViewer(QMainWindow):
    def __init__(self, zarr_path: str, initial_subject: str | None = None, checkpoint_path: str | None = None):
        super().__init__()
        self.zarr_path = zarr_path
        self.store = zarr.open_group(zarr_path, mode="r")
        all_subjects = sorted(self.store.group_keys())
        if not all_subjects:
            raise RuntimeError(f"No subjects found in {zarr_path}")

        if checkpoint_path is not None:
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
        self.brain_mask_cache: dict[str, np.ndarray] = {}
        self.slice_metric_cache: dict[tuple, dict] = {}
        self.pred_cache: dict[tuple, dict] = {}
        self._pending_metrics: set[tuple] = set()
        self._pending_predictions: set[tuple] = set()

        self.model: QSpaceUNet | None = None
        self.dti_scale = 1.0
        self.max_bval = 1000.0
        self.device = torch.device("cpu")
        self.model_lock = threading.Lock()

        if checkpoint_path is not None:
            self.device = get_device()
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            max_n = ckpt["max_n"]
            feat_dim = ckpt.get("feat_dim", 64)
            channels = tuple(ckpt.get("channels", [64, 128, 256, 512]))
            cholesky = ckpt.get("cholesky", False)
            self.dti_scale = ckpt.get("dti_scale", 1.0)
            self.max_bval = ckpt.get("max_bval", 1000.0)
            self.model = QSpaceUNet(max_n=max_n, feat_dim=feat_dim, channels=channels, cholesky=cholesky).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            print(f"Model loaded from {checkpoint_path} (epoch {ckpt['epoch']}, device={self.device})")

        self.worker_signals = WorkerSignals()
        self.worker_signals.metrics_ready.connect(self._on_metrics_ready)
        self.worker_signals.prediction_ready.connect(self._on_prediction_ready)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        self.setWindowTitle("DW_THI Dataset Viewer")
        self.setMinimumSize(1550, 1200 if self.model is not None else 900)
        self._build_ui()
        self._load_subject_by_name(initial_subject or self.subjects[0])

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        page = QHBoxLayout(root)
        left = QVBoxLayout()
        right = QVBoxLayout()
        page.addLayout(left, stretch=4)
        page.addLayout(right, stretch=1)

        # ── Controls bar ──────────────────────────────────────────────────
        controls = QHBoxLayout()
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

        controls.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.valueChanged.connect(self._update_view)
        controls.addWidget(self.volume_slider, stretch=2)
        self.volume_label = QLabel("0 / 0")
        controls.addWidget(self.volume_label)

        # ── Image rows (QVBoxLayout so hidden rows collapse) ──────────────
        rows_container = QVBoxLayout()
        left.addLayout(rows_container, stretch=1)

        self.panels: dict[str, ImagePanel] = {}
        self.row_widgets: dict[str, QWidget] = {}

        for row_key, panel_names in ROW_GROUPS.items():
            if row_key == "pred" and self.model is None:
                continue
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for name in panel_names:
                panel = ImagePanel(name)
                row_layout.addWidget(panel)
                self.panels[name] = panel
            self.row_widgets[row_key] = row_widget
            rows_container.addWidget(row_widget, stretch=1)

        # ── Right panel ───────────────────────────────────────────────────
        info_box = QGroupBox("Subject Summary")
        info_layout = QVBoxLayout(info_box)
        self.subject_info_label = QLabel("")
        self.subject_info_label.setWordWrap(True)
        self.subject_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(self.subject_info_label)
        right.addWidget(info_box)

        volume_box = QGroupBox("Current Selection")
        volume_layout = QVBoxLayout(volume_box)
        self.volume_info_label = QLabel("")
        self.volume_info_label.setWordWrap(True)
        self.volume_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        volume_layout.addWidget(self.volume_info_label)
        right.addWidget(volume_box)

        bvals_box = QGroupBox("Acquisition")
        bvals_layout = QVBoxLayout(bvals_box)
        self.bvals_canvas = BvalsCanvas()
        bvals_layout.addWidget(self.bvals_canvas)
        right.addWidget(bvals_box, stretch=1)

        # ── Visibility checkboxes ─────────────────────────────────────────
        vis_box = QGroupBox("Visible Panels")
        vis_layout = QVBoxLayout(vis_box)
        vis_layout.setSpacing(2)
        self.panel_checks: dict[str, QCheckBox] = {}

        for row_key, panel_names in ROW_GROUPS.items():
            if row_key == "pred" and self.model is None:
                continue
            row_header = QLabel(f"<b>{ROW_LABELS[row_key]}</b>")
            vis_layout.addWidget(row_header)
            for name in panel_names:
                cb = QCheckBox(name)
                cb.setChecked(True)
                cb.toggled.connect(lambda checked, n=name: self._toggle_panel(n, checked))
                vis_layout.addWidget(cb)
                self.panel_checks[name] = cb

        right.addWidget(vis_box)

        help_box = QGroupBox("Viewer Notes")
        help_layout = QVBoxLayout(help_box)
        help_text = (
            "Row 1: DWI volumes for the selected diffusion volume. "
            "Row 2: DTI-derived maps computed from ground-truth tensors. "
        )
        if self.model is not None:
            help_text += "Row 3: live NN predictions (axial plane only). "
        help_text += "DWI, DTI, and NN panels are masked to the target-side brain mask. "
        help_text += "DTI and NN panels load in the background — the GUI stays responsive."
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        right.addWidget(help_box)

    def _toggle_panel(self, panel_name: str, visible: bool) -> None:
        if panel_name in self.panels:
            self.panels[panel_name].setVisible(visible)
        # Collapse row container when all its panels are hidden
        for row_key, panel_names in ROW_GROUPS.items():
            if panel_name in panel_names and row_key in self.row_widgets:
                row_visible = any(
                    self.panels[n].isVisible() for n in panel_names if n in self.panels
                )
                self.row_widgets[row_key].setVisible(row_visible)

    @property
    def plane(self) -> str:
        return self.plane_combo.currentText()

    def _load_subject_by_name(self, subject_name: str) -> None:
        if subject_name not in self.subjects:
            return

        self.current_subject = subject_name
        self.current_group = self.store[subject_name]
        self.current_shape = tuple(self.current_group["input_dwi"].shape)
        self.current_bvals = np.asarray(self.current_group["bvals"][:], dtype=np.float32)
        self.current_bvecs = np.asarray(self.current_group["bvecs"][:], dtype=np.float32)
        self.current_brain_mask = self._get_brain_mask(subject_name)
        self.slice_metric_cache.clear()
        self.pred_cache.clear()
        self._pending_metrics.clear()
        self._pending_predictions.clear()

        subject_index = self.subjects.index(subject_name)
        if self.subject_combo.currentIndex() != subject_index:
            self.subject_combo.blockSignals(True)
            self.subject_combo.setCurrentIndex(subject_index)
            self.subject_combo.blockSignals(False)

        self._reset_slice_slider()
        self._reset_volume_slider()
        self._update_subject_summary()
        self._update_view()

    def _get_brain_mask(self, subject_name: str) -> np.ndarray:
        if subject_name in self.brain_mask_cache:
            return self.brain_mask_cache[subject_name]

        group = self.store[subject_name]
        if "brain_mask" in group:
            mask = np.asarray(group["brain_mask"][:], dtype=bool)
        else:
            target_dwi = np.asarray(group["target_dwi"][:], dtype=np.float32)
            bvals = np.asarray(group["bvals"][:], dtype=np.float32)
            mask = compute_brain_mask_from_dwi(target_dwi, bvals, cfg.B0_THRESHOLD)

        self.brain_mask_cache[subject_name] = mask
        return mask

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

    def _reset_volume_slider(self) -> None:
        max_index = self.current_shape[3] - 1
        non_b0 = np.flatnonzero(self.current_bvals >= 50)
        initial_volume = int(non_b0[0]) if non_b0.size > 0 else 0

        self.volume_slider.blockSignals(True)
        self.volume_slider.setRange(0, max_index)
        self.volume_slider.setValue(initial_volume)
        self.volume_slider.blockSignals(False)

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
            f"Shell counts: {shells}"
        )
        self.subject_info_label.setText(summary)

    def _update_view(self) -> None:
        if self.current_group is None:
            return

        slice_idx = self.slice_slider.value()
        volume_idx = self.volume_slider.value()
        self.slice_label.setText(f"{slice_idx} / {self.slice_slider.maximum()}")
        self.volume_label.setText(f"{volume_idx} / {self.volume_slider.maximum()}")

        # ── DWI panels: fast zarr slicing, runs synchronously ─────────────
        input_slice = extract_volume_slice(self.current_group["input_dwi"], self.plane, slice_idx, volume_idx)
        target_slice = extract_volume_slice(self.current_group["target_dwi"], self.plane, slice_idx, volume_idx)
        diff_slice = np.abs(input_slice - target_slice)
        mask_slice = self._current_mask_slice()
        input_display = apply_display_mask(input_slice, mask_slice)
        target_display = apply_display_mask(target_slice, mask_slice)
        diff_display = apply_display_mask(diff_slice, mask_slice)
        input_mean, input_max = masked_stats(input_slice, mask_slice)
        target_mean, target_max = masked_stats(target_slice, mask_slice)
        diff_mean, diff_max = masked_stats(diff_slice, mask_slice)

        self.panels["Input DWI"].set_pixmap(
            make_pixmap(input_display, cmap="gray"),
            f"brain mean={format_float(input_mean)}  max={format_float(input_max)}",
        )
        self.panels["Target DWI"].set_pixmap(
            make_pixmap(target_display, cmap="gray"),
            f"brain mean={format_float(target_mean)}  max={format_float(target_max)}",
        )
        self.panels["Absolute Difference"].set_pixmap(
            make_pixmap(diff_display, cmap="magma"),
            f"brain mean={format_float(diff_mean)}  max={format_float(diff_max)}",
        )

        current_bval = float(self.current_bvals[volume_idx])
        current_bvec = self.current_bvecs[:, volume_idx]
        self.volume_info_label.setText(
            f"Plane: {self.plane}\n"
            f"Slice: {slice_idx}\n"
            f"Volume: {volume_idx}  ({shell_name(current_bval)})\n"
            f"bvec: [{current_bvec[0]:.3f}, {current_bvec[1]:.3f}, {current_bvec[2]:.3f}]\n"
            f"Brain-mask voxels in slice: {int(np.count_nonzero(mask_slice)) if mask_slice is not None else 0}\n"
            f"Input-target abs diff brain mean: {format_float(diff_mean)}"
        )
        self.bvals_canvas.update_plot(self.current_bvals, volume_idx)

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
                pred_key = (self.current_subject, slice_idx)
                if pred_key in self.pred_cache:
                    self._apply_predictions(self.pred_cache[pred_key])
                elif pred_key not in self._pending_predictions:
                    self._pending_predictions.add(pred_key)
                    for p in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                        self.panels[p].image_label.setPixmap(QPixmap())
                        self.panels[p].image_label.setText("Computing…")
                        self.panels[p].caption_label.setText("")
                    self.thread_pool.start(
                        PredictionWorker(
                            self.current_subject, slice_idx,
                            self.current_group,
                            self.current_bvals.copy(), self.current_bvecs.copy(),
                            self.model, self.device,
                            self.dti_scale, self.max_bval, cfg.B0_THRESHOLD,
                            self.model_lock, self.worker_signals,
                        )
                    )

        self.statusBar().showMessage(
            f"{self.current_subject} | {self.plane} slice {slice_idx} | volume {volume_idx} | {shell_name(current_bval)}"
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
        )
        self.panels["Predicted MD"].set_pixmap(
            make_pixmap(md_display, cmap="plasma"),
            f"brain mean={format_float(md_mean)}  max={format_float(md_max)}",
        )
        self.panels["Predicted Color FA"].set_pixmap(
            make_pixmap(color_fa_display),
            "masked predicted principal direction RGB weighted by FA",
        )

    @pyqtSlot(str, str, int, object)
    def _on_metrics_ready(self, subject: str, plane: str, slice_idx: int, result: dict) -> None:
        self._pending_metrics.discard((subject, plane, slice_idx))
        self.slice_metric_cache[(subject, plane, slice_idx)] = result
        if (subject == self.current_subject
                and plane == self.plane
                and slice_idx == self.slice_slider.value()):
            self._apply_metrics(result)

    @pyqtSlot(str, int, object)
    def _on_prediction_ready(self, subject: str, slice_idx: int, result: dict) -> None:
        self._pending_predictions.discard((subject, slice_idx))
        self.pred_cache[(subject, slice_idx)] = result
        if (subject == self.current_subject
                and slice_idx == self.slice_slider.value()
                and self.plane == "Axial"):
            self._apply_predictions(result)


def main() -> None:
    args = parse_args()
    if args.summary_only:
        print(dataset_summary(args.zarr_path))
        return

    app = QApplication(sys.argv)
    viewer = DatasetViewer(args.zarr_path, initial_subject=args.subject, checkpoint_path=args.checkpoint)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
