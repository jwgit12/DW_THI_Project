#!/usr/bin/env python3
"""Lightweight desktop viewer for DW_THI Zarr datasets.

Usage:
    python3 visualizer.py --zarr_path dataset/pretext_dataset_new.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import zarr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import colormaps
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
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
from functions import compute_color_fa_from_tensor6, compute_fa_from_tensor6, compute_md_from_tensor6
from research.model import QSpaceUNet

matplotlib.rcParams["font.family"] = "DejaVu Sans"


PLANE_TO_AXIS = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
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


class DatasetViewer(QMainWindow):
    def __init__(self, zarr_path: str, initial_subject: str | None = None, checkpoint_path: str | None = None):
        super().__init__()
        self.zarr_path = zarr_path
        self.store = zarr.open_group(zarr_path, mode="r")
        all_subjects = sorted(self.store.group_keys())
        if not all_subjects:
            raise RuntimeError(f"No subjects found in {zarr_path}")

        # When a checkpoint is provided, restrict to test and validation subjects only
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
        self.slice_metric_cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        self.pred_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

        # Load model if checkpoint provided
        self.model: QSpaceUNet | None = None
        self.dti_scale = 1.0
        self.max_bval = 1000.0
        self.device = torch.device("cpu")
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

        self.setWindowTitle("DW_THI Dataset Viewer")
        self.setMinimumSize(1550, 1300 if self.model is not None else 950)
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

        images = QGridLayout()
        left.addLayout(images, stretch=1)

        self.panels: dict[str, ImagePanel] = {}
        panel_specs = [
            ("Input DWI", 0, 0),
            ("Target DWI", 0, 1),
            ("Absolute Difference", 0, 2),
            ("FA Map", 1, 0),
            ("MD Map", 1, 1),
            ("Color FA", 1, 2),
            ("Predicted FA", 2, 0),
            ("Predicted MD", 2, 1),
            ("Predicted Color FA", 2, 2),
        ]
        for title, row, col in panel_specs:
            panel = ImagePanel(title)
            images.addWidget(panel, row, col)
            self.panels[title] = panel

        for key in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
            self.panels[key].setVisible(self.model is not None)

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

        help_box = QGroupBox("Viewer Notes")
        help_layout = QVBoxLayout(help_box)
        help_text = (
            "The first row follows the selected diffusion volume. "
            "The second row shows DTI-derived maps for the selected plane and slice."
        )
        if self.model is not None:
            help_text += " The third row shows neural network predictions (axial plane only)."
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        right.addWidget(help_box)

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
        self.slice_metric_cache.clear()
        self.pred_cache.clear()

        subject_index = self.subjects.index(subject_name)
        if self.subject_combo.currentIndex() != subject_index:
            self.subject_combo.blockSignals(True)
            self.subject_combo.setCurrentIndex(subject_index)
            self.subject_combo.blockSignals(False)

        self._reset_slice_slider()
        self._reset_volume_slider()
        self._update_subject_summary()
        self._update_view()

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

    def _get_slice_metrics(self, slice_idx: int) -> dict[str, np.ndarray]:
        cache_key = (self.current_subject, self.plane, slice_idx)
        if cache_key not in self.slice_metric_cache:
            tensor6 = extract_tensor_slice(self.current_group["target_dti_6d"], self.plane, slice_idx)
            self.slice_metric_cache[cache_key] = {
                "fa": np.asarray(compute_fa_from_tensor6(tensor6), dtype=np.float32),
                "md": np.asarray(compute_md_from_tensor6(tensor6), dtype=np.float32),
                "color_fa": np.asarray(compute_color_fa_from_tensor6(tensor6), dtype=np.float32),
            }
        return self.slice_metric_cache[cache_key]

    def _get_predicted_metrics(self, slice_idx: int) -> dict[str, np.ndarray] | None:
        if self.model is None or self.plane != "Axial":
            return None

        cache_key = (self.current_subject, slice_idx)
        if cache_key in self.pred_cache:
            return self.pred_cache[cache_key]

        bvals = self.current_bvals.copy()
        bvecs = self.current_bvecs.copy()
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

        # Extract axial slice and pad
        signal_slice = np.asarray(self.current_group["input_dwi"][:, :, slice_idx, :], dtype=np.float32)  # (X, Y, N)
        if N < max_n:
            signal_slice = np.pad(signal_slice, ((0, 0), (0, 0), (0, max_n - N)))
        signal = signal_slice.transpose(2, 0, 1)  # (max_n, H, W)

        # b0 normalization
        b0_idx = self.current_bvals < cfg.B0_THRESHOLD
        if b0_idx.any():
            b0_slice = np.asarray(self.current_group["input_dwi"][:, :, slice_idx, :], dtype=np.float32)[..., b0_idx].mean(axis=-1)
        else:
            b0_slice = signal_slice[..., :N].mean(axis=-1)
        b0_norm = float(b0_slice[b0_slice > 0.1 * b0_slice.max()].mean()) if (b0_slice > 0).any() else 1.0
        if b0_norm > 0:
            signal = signal / b0_norm

        signal_t = torch.from_numpy(signal).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(signal_t, bvals_t, bvecs_t, vol_mask_t)

        pred_tensor6 = pred[0].permute(1, 2, 0).cpu().numpy() / self.dti_scale

        result = {
            "fa": np.asarray(compute_fa_from_tensor6(pred_tensor6), dtype=np.float32),
            "md": np.asarray(compute_md_from_tensor6(pred_tensor6), dtype=np.float32),
            "color_fa": np.asarray(compute_color_fa_from_tensor6(pred_tensor6), dtype=np.float32),
        }
        self.pred_cache[cache_key] = result
        return result

    def _update_subject_summary(self) -> None:
        source = self.current_group.attrs.get("source_dwi", "unknown")
        shells = summarize_shells(self.current_bvals)
        summary = (
            f"Subject: {self.current_subject}\n"
            f"Source: {source}\n"
            f"Shape: {self.current_shape[:3]}  Volumes: {self.current_shape[3]}\n"
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

        input_slice = extract_volume_slice(self.current_group["input_dwi"], self.plane, slice_idx, volume_idx)
        target_slice = extract_volume_slice(self.current_group["target_dwi"], self.plane, slice_idx, volume_idx)
        diff_slice = np.abs(input_slice - target_slice)
        metrics = self._get_slice_metrics(slice_idx)

        self.panels["Input DWI"].set_pixmap(
            make_pixmap(input_slice, cmap="gray"),
            f"mean={format_float(float(np.mean(input_slice)))}  max={format_float(float(np.max(input_slice)))}",
        )
        self.panels["Target DWI"].set_pixmap(
            make_pixmap(target_slice, cmap="gray"),
            f"mean={format_float(float(np.mean(target_slice)))}  max={format_float(float(np.max(target_slice)))}",
        )
        self.panels["Absolute Difference"].set_pixmap(
            make_pixmap(diff_slice, cmap="magma"),
            f"mean={format_float(float(np.mean(diff_slice)))}  max={format_float(float(np.max(diff_slice)))}",
        )
        self.panels["FA Map"].set_pixmap(
            make_pixmap(metrics["fa"], cmap="viridis"),
            f"mean={format_float(float(np.mean(metrics['fa'])))}  max={format_float(float(np.max(metrics['fa'])))}",
        )
        self.panels["MD Map"].set_pixmap(
            make_pixmap(metrics["md"], cmap="plasma"),
            f"mean={format_float(float(np.mean(metrics['md'])))}  max={format_float(float(np.max(metrics['md'])))}",
        )
        self.panels["Color FA"].set_pixmap(
            make_pixmap(metrics["color_fa"]),
            "principal direction RGB weighted by FA",
        )

        current_bval = float(self.current_bvals[volume_idx])
        current_bvec = self.current_bvecs[:, volume_idx]
        volume_text = (
            f"Plane: {self.plane}\n"
            f"Slice: {slice_idx}\n"
            f"Volume: {volume_idx}  ({shell_name(current_bval)})\n"
            f"bvec: [{current_bvec[0]:.3f}, {current_bvec[1]:.3f}, {current_bvec[2]:.3f}]\n"
            f"Input-target abs diff mean: {format_float(float(np.mean(diff_slice)))}"
        )
        self.volume_info_label.setText(volume_text)

        self.bvals_canvas.update_plot(self.current_bvals, volume_idx)

        pred_metrics = self._get_predicted_metrics(slice_idx)
        if pred_metrics is not None:
            self.panels["Predicted FA"].set_pixmap(
                make_pixmap(pred_metrics["fa"], cmap="viridis"),
                f"mean={format_float(float(np.mean(pred_metrics['fa'])))}  max={format_float(float(np.max(pred_metrics['fa'])))}",
            )
            self.panels["Predicted MD"].set_pixmap(
                make_pixmap(pred_metrics["md"], cmap="plasma"),
                f"mean={format_float(float(np.mean(pred_metrics['md'])))}  max={format_float(float(np.max(pred_metrics['md'])))}",
            )
            self.panels["Predicted Color FA"].set_pixmap(
                make_pixmap(pred_metrics["color_fa"]),
                "predicted principal direction RGB weighted by FA",
            )
            for key in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                self.panels[key].setVisible(True)
        elif self.model is not None:
            for key in ("Predicted FA", "Predicted MD", "Predicted Color FA"):
                self.panels[key].image_label.setPixmap(QPixmap())
                self.panels[key].image_label.setText("Axial plane only")
                self.panels[key].caption_label.setText("")
                self.panels[key].setVisible(True)

        self.statusBar().showMessage(
            f"{self.current_subject} | {self.plane} slice {slice_idx} | volume {volume_idx} | {shell_name(current_bval)}"
        )


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
