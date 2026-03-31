"""
Qt6 desktop application to inspect the pretext Zarr dataset.

Usage:
    python visualizer.py --zarr_path dataset/pretext_dataset.zarr
"""

import argparse
import sys

import numpy as np
import zarr
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Project utilities
sys.path.insert(0, ".")
from functions import compute_fa_from_tensor6, norm


def _array_to_pixmap(arr: np.ndarray, width: int = 256) -> QPixmap:
    """Convert a 2-D float array (or H×W×3 RGB) to a QPixmap."""
    if arr.ndim == 2:
        arr = np.rot90(arr, 1)
        arr = norm(arr, pmin=1, pmax=99)
        arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        h, w = arr_u8.shape
        image = QImage(arr_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        arr = np.rot90(arr, 1)
        # Normalize per-channel
        for c in range(3):
            ch = arr[..., c]
            if ch.max() > 0:
                arr[..., c] = ch / (ch.max() + 1e-8)
        arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        h, w, _ = arr_u8.shape
        image = QImage(
            arr_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888
        )
    pixmap = QPixmap.fromImage(image)
    return pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


class DatasetViewer(QMainWindow):
    def __init__(self, zarr_path: str):
        super().__init__()
        self.setWindowTitle("DTI Pretext Dataset Viewer")
        self.setMinimumSize(1100, 500)

        self.store = zarr.open(zarr_path, mode="r")
        self.subjects = sorted(self.store.group_keys())
        if not self.subjects:
            raise RuntimeError(f"No subjects found in {zarr_path}")

        self._current_subject = None
        self._current_data = {}

        self._build_ui()
        self._load_subject(0)

    # ---- UI construction ----
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # --- Controls row ---
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        ctrl.addWidget(QLabel("Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(self.subjects)
        self.subject_combo.currentIndexChanged.connect(self._load_subject)
        ctrl.addWidget(self.subject_combo)

        ctrl.addWidget(QLabel("  Z-slice:"))
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._update_view)
        ctrl.addWidget(self.z_slider)
        self.z_label = QLabel("0")
        ctrl.addWidget(self.z_label)

        ctrl.addWidget(QLabel("  Direction:"))
        self.dir_slider = QSlider(Qt.Orientation.Horizontal)
        self.dir_slider.valueChanged.connect(self._update_view)
        ctrl.addWidget(self.dir_slider)
        self.dir_label = QLabel("0")
        ctrl.addWidget(self.dir_label)

        # --- Image panels ---
        panels = QHBoxLayout()
        layout.addLayout(panels)

        self.panels = {}
        for name in ["Noisy Input", "Clean DWI", "FA Map", "Difference"]:
            vbox = QVBoxLayout()
            title = QLabel(name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 13px;")
            vbox.addWidget(title)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(img_label)
            panels.addLayout(vbox)
            self.panels[name] = img_label

    # ---- Data loading ----
    def _load_subject(self, index: int):
        name = self.subjects[index]
        grp = self.store[name]
        self._current_data = {
            "input_dwi": grp["input_dwi"][:],
            "target_dwi": grp["target_dwi"][:],
            "target_dti_6d": grp["target_dti_6d"][:],
        }
        _, _, nz, n_dirs = self._current_data["input_dwi"].shape

        self.z_slider.setRange(0, nz - 1)
        self.z_slider.setValue(nz // 2)
        self.dir_slider.setRange(0, n_dirs - 1)
        self.dir_slider.setValue(0)
        self._update_view()

    # ---- Rendering ----
    def _update_view(self):
        z = self.z_slider.value()
        d = self.dir_slider.value()
        self.z_label.setText(str(z))
        self.dir_label.setText(str(d))

        noisy = self._current_data["input_dwi"][:, :, z, d]
        clean = self._current_data["target_dwi"][:, :, z, d]
        dti_6d = self._current_data["target_dti_6d"][:, :, z, :]

        # FA map
        fa = compute_fa_from_tensor6(dti_6d)

        # Difference
        diff = np.abs(noisy - clean)

        pw = 256
        self.panels["Noisy Input"].setPixmap(_array_to_pixmap(noisy, pw))
        self.panels["Clean DWI"].setPixmap(_array_to_pixmap(clean, pw))
        self.panels["FA Map"].setPixmap(_array_to_pixmap(fa, pw))
        self.panels["Difference"].setPixmap(_array_to_pixmap(diff, pw))


def main():
    parser = argparse.ArgumentParser(description="Inspect the pretext Zarr dataset.")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="dataset/pretext_dataset.zarr",
        help="Path to the Zarr store.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = DatasetViewer(args.zarr_path)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
