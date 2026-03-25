import sys
import os
import numpy as np
import cv2
import zarr
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QGroupBox, QRadioButton, QSlider, QGridLayout,
                             QScrollArea, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

import functions

def array_to_qimage_gray(arr):
    # Flip vertically to match matplotlib origin='lower' behavior
    arr = np.flip(arr, axis=0)
    
    arr_norm = arr - np.min(arr)
    arr_max = np.max(arr_norm)
    if arr_max > 0:
        arr_norm = arr_norm / arr_max
    
    img_8 = (arr_norm * 255.0).astype(np.uint8)
    if not img_8.flags['C_CONTIGUOUS']:
        img_8 = np.ascontiguousarray(img_8)
        
    h, w = img_8.shape
    qimg = QImage(img_8.data, w, h, w, QImage.Format.Format_Grayscale8)
    return qimg.copy()

def array_to_qimage_rgb(arr):
    # Flip vertically
    arr = np.flip(arr, axis=0)
    
    arr_norm = arr - np.min(arr)
    arr_max = np.max(arr_norm)
    if arr_max > 0:
        arr_norm = arr_norm / arr_max
        
    img_8 = (arr_norm * 255.0).astype(np.uint8)
    if not img_8.flags['C_CONTIGUOUS']:
        img_8 = np.ascontiguousarray(img_8)
        
    h, w, c = img_8.shape
    qimg = QImage(img_8.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return qimg.copy()

def array_to_qimage_magma(arr):
    # Flip vertically
    arr = np.flip(arr, axis=0)
    
    arr_norm = arr - np.min(arr)
    arr_max = np.max(arr_norm)
    if arr_max > 0:
        arr_norm = arr_norm / arr_max
    
    img_8 = (arr_norm * 255.0).astype(np.uint8)
    img_color = cv2.applyColorMap(img_8, cv2.COLORMAP_MAGMA)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    if not img_rgb.flags['C_CONTIGUOUS']:
        img_rgb = np.ascontiguousarray(img_rgb)
        
    h, w, c = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return qimg.copy()

class ImageLabel(QLabel):
    def __init__(self, title):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333333;")
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.title = title
        self._pixmap = None

    def set_image(self, qimage):
        self._pixmap = QPixmap.fromImage(qimage)
        self.update_image()

    def update_image(self):
        if self._pixmap is not None:
            # Scale pixmap to fit the label while keeping aspect ratio
            scaled_pixmap = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            
    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

class DTIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nexus DTI - Qt6 Migration")
        self.resize(1000, 700)
        
        # Data store
        self.zarr_root = None
        self.samples_meta = []
        
        # State
        self.current_sample = 0
        self.axis = 'z' # 'x', 'y', 'z'
        self.slice_indices = {'x': 0, 'y': 0, 'z': 0}
        self.metric = 'fa' # 'fa', 'md', 'cfa'
        
        self.init_ui()
        self.load_data()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Nexus DTI")
        title_font = QFont("Arial", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)
        sidebar_layout.addWidget(title_label)
        
        subtitle_label = QLabel("High-Performance DW-MRI Analysis")
        subtitle_label.setStyleSheet("color: #aaaaaa;")
        sidebar_layout.addWidget(subtitle_label)
        
        sidebar_layout.addSpacing(20)
        
        # Sample Selection
        sample_group = QGroupBox("Sample Selection")
        sample_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        sample_layout = QVBoxLayout(sample_group)
        self.sample_combo = QComboBox()
        self.sample_combo.setStyleSheet("background-color: #3b3b3b; padding: 5px;")
        self.sample_combo.currentIndexChanged.connect(self.on_sample_changed)
        sample_layout.addWidget(self.sample_combo)
        sidebar_layout.addWidget(sample_group)
        
        # Axis Selection
        axis_group = QGroupBox("View Plane / Axis")
        axis_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        axis_layout = QVBoxLayout(axis_group)
        
        self.radio_axial = QRadioButton("Axial (Z)")
        self.radio_coronal = QRadioButton("Coronal (Y)")
        self.radio_sagittal = QRadioButton("Sagittal (X)")
        self.radio_axial.setChecked(True)
        
        self.radio_axial.toggled.connect(lambda: self.on_axis_changed('z'))
        self.radio_coronal.toggled.connect(lambda: self.on_axis_changed('y'))
        self.radio_sagittal.toggled.connect(lambda: self.on_axis_changed('x'))
        
        axis_layout.addWidget(self.radio_axial)
        axis_layout.addWidget(self.radio_coronal)
        axis_layout.addWidget(self.radio_sagittal)
        sidebar_layout.addWidget(axis_group)
        
        # Slice Slider
        slice_group = QGroupBox("Slice Navigation")
        slice_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        slice_layout = QVBoxLayout(slice_group)
        self.slice_label = QLabel("Slice: 0 / 0")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        sidebar_layout.addWidget(slice_group)
        
        # Metric Selection
        metric_group = QGroupBox("Target Index Map")
        metric_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        metric_layout = QVBoxLayout(metric_group)
        self.radio_fa = QRadioButton("Fractional Anisotropy (FA)")
        self.radio_md = QRadioButton("Mean Diffusivity (MD)")
        self.radio_cfa = QRadioButton("Colored FA (RGB)")
        self.radio_fa.setChecked(True)
        
        self.radio_fa.toggled.connect(lambda: self.on_metric_changed('fa'))
        self.radio_md.toggled.connect(lambda: self.on_metric_changed('md'))
        self.radio_cfa.toggled.connect(lambda: self.on_metric_changed('cfa'))
        
        metric_layout.addWidget(self.radio_fa)
        metric_layout.addWidget(self.radio_md)
        metric_layout.addWidget(self.radio_cfa)
        sidebar_layout.addWidget(metric_group)
        
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)
        
        # Main Viewer Area
        viewer_area = QWidget()
        viewer_area.setStyleSheet("background-color: #121212;")
        viewer_layout = QGridLayout(viewer_area)
        
        self.lbl_input = ImageLabel("Input (Mean DWI)")
        self.lbl_target = ImageLabel("Target")
        self.lbl_k_input = ImageLabel("K-Space (Input)")
        self.lbl_k_target = ImageLabel("K-Space (Target)")
        
        # Add titles to grid
        def add_viewer_box(grid, row, col, title, img_label):
            box = QWidget()
            lo = QVBoxLayout(box)
            lo.setContentsMargins(5, 5, 5, 5)
            t = QLabel(title)
            t.setStyleSheet("color: white; font-weight: bold;")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lo.addWidget(t)
            lo.addWidget(img_label, stretch=1)
            grid.addWidget(box, row, col)
            
        add_viewer_box(viewer_layout, 0, 0, "Input (Mean DWI)", self.lbl_input)
        add_viewer_box(viewer_layout, 0, 1, "Target Map", self.lbl_target)
        add_viewer_box(viewer_layout, 1, 0, "K-Space (Input)", self.lbl_k_input)
        add_viewer_box(viewer_layout, 1, 1, "K-Space (Target)", self.lbl_k_target)
        
        main_layout.addWidget(viewer_area, stretch=1)
        
    def _get_subject_group(self, idx):
        """Return the zarr group for a subject (no data loaded into RAM)."""
        subj_grp = self.zarr_root[f"subject_{idx:03d}"]
        return subj_grp

    def _load_slice(self, axis, s_idx):
        """Lazily load a single 2D slice from the current subject's zarr store.

        For the z-axis this reads one zarr slice directly.
        For x/y axes it reads the required row/column from every zarr slice
        and stacks them, but never materialises the full volume.
        """
        subj_grp = self._get_subject_group(self.current_sample)
        slice_keys = sorted(k for k in subj_grp.keys() if k.startswith("slice_"))

        if axis == 'z':
            sl = subj_grp[slice_keys[s_idx]]
            return sl["input"][:], sl["target"][:]

        # For x / y we need to gather one row/column from each z-slice
        n_z = len(slice_keys)
        s0 = subj_grp[slice_keys[0]]
        inp0 = s0["input"][:]
        tgt0 = s0["target"][:]
        n_channels_inp = inp0.shape[2]
        n_channels_tgt = tgt0.shape[2]

        if axis == 'y':
            # slice along y -> shape (x, z, c)
            inp_out = np.empty((inp0.shape[0], n_z, n_channels_inp), dtype=np.float32)
            tgt_out = np.empty((tgt0.shape[0], n_z, n_channels_tgt), dtype=np.float32)
            inp_out[:, 0, :] = inp0[:, s_idx, :]
            tgt_out[:, 0, :] = tgt0[:, s_idx, :]
            for z in range(1, n_z):
                sl = subj_grp[slice_keys[z]]
                inp_out[:, z, :] = sl["input"][:, s_idx, :]
                tgt_out[:, z, :] = sl["target"][:, s_idx, :]
        else:  # axis == 'x'
            # slice along x -> shape (y, z, c)
            inp_out = np.empty((inp0.shape[1], n_z, n_channels_inp), dtype=np.float32)
            tgt_out = np.empty((tgt0.shape[1], n_z, n_channels_tgt), dtype=np.float32)
            inp_out[:, 0, :] = inp0[s_idx, :, :]
            tgt_out[:, 0, :] = tgt0[s_idx, :, :]
            for z in range(1, n_z):
                sl = subj_grp[slice_keys[z]]
                inp_out[:, z, :] = sl["input"][s_idx, :, :]
                tgt_out[:, z, :] = sl["target"][s_idx, :, :]

        return inp_out, tgt_out

    def load_data(self):
        path = 'dti_ml_dataset_v2.zarr'
        if not os.path.exists(path):
            print(f"Error: Dataset not found at {path}")
            return

        print("Loading dataset index...")
        self.zarr_root = zarr.open_group(path, mode="r")
        subjects = sorted(k for k in self.zarr_root.keys() if k.startswith("subject_"))
        print(f"Found {len(subjects)} subjects.")

        for i, subj_name in enumerate(subjects):
            subj_grp = self.zarr_root[subj_name]
            slices = sorted(k for k in subj_grp.keys() if k.startswith("slice_"))
            n_z = len(slices)
            s0 = subj_grp[slices[0]]
            x_dim, y_dim = s0["input"].shape[:2]
            self.samples_meta.append({
                "id": i,
                "shape": {"x": x_dim, "y": y_dim, "z": n_z}
            })
            self.sample_combo.addItem(f"Sample #{i + 1}")

        if self.samples_meta:
            shape = self.samples_meta[0]['shape']
            self.slice_indices = {
                'x': shape['x'] // 2,
                'y': shape['y'] // 2,
                'z': shape['z'] // 2,
            }
            self.update_slider_range()
            self.update_views()

    def update_slider_range(self):
        if not self.samples_meta: return
        shape = self.samples_meta[self.current_sample]['shape']
        max_idx = shape[self.axis] - 1
        
        # Temporarily block signals to avoid double updates
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max_idx)
        self.slice_slider.setValue(self.slice_indices[self.axis])
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f"Slice: {self.slice_indices[self.axis]} / {max_idx}")
        
    def on_sample_changed(self, idx):
        if self.current_sample == idx: return
        self.current_sample = idx
        shape = self.samples_meta[idx]['shape']
        # reset slice indices for new sample if out of bounds
        for ax in ['x', 'y', 'z']:
            self.slice_indices[ax] = min(self.slice_indices[ax], shape[ax] - 1)
        self.update_slider_range()
        self.update_views()

    def on_axis_changed(self, ax):
        # The radio buttons toggle sends two signals (one checked=False, one checked=True)
        # We only want to respond when it becomes active. But our lambda doesn't pass checked state.
        # We can check which button is active.
        if ax == 'z' and not self.radio_axial.isChecked(): return
        if ax == 'y' and not self.radio_coronal.isChecked(): return
        if ax == 'x' and not self.radio_sagittal.isChecked(): return
        
        if self.axis == ax: return
        self.axis = ax
        self.update_slider_range()
        self.update_views()

    def on_slice_changed(self, val):
        self.slice_indices[self.axis] = val
        max_idx = self.samples_meta[self.current_sample]['shape'][self.axis] - 1
        self.slice_label.setText(f"Slice: {val} / {max_idx}")
        self.update_views()

    def on_metric_changed(self, metric):
        if metric == 'fa' and not self.radio_fa.isChecked(): return
        if metric == 'md' and not self.radio_md.isChecked(): return
        if metric == 'cfa' and not self.radio_cfa.isChecked(): return
        
        if self.metric == metric: return
        self.metric = metric
        self.lbl_target.title = f"Target ({self.metric.upper()})"
        # The titles are external labels now, we could update them but they are static for simplicity as "Target Map"
        self.update_views()

    def update_views(self):
        if not self.samples_meta: return

        s_idx = self.slice_indices[self.axis]
        input_slice, tensor_slice = self._load_slice(self.axis, s_idx)

        # 1. Input Image
        input_mean = np.mean(input_slice, axis=-1)
        
        # 2. Target Image
        if self.metric == 'fa':
            target_img = functions.compute_fa_from_tensor6(tensor_slice)
            target_img = np.clip(target_img, 0, 1) # clamp for safety
        elif self.metric == 'md':
            target_img = functions.compute_md_from_tensor6(tensor_slice)
        elif self.metric == 'cfa':
            target_img = functions.compute_color_fa_from_tensor6(tensor_slice)
            target_img = np.clip(target_img, 0, 1) # clip [0, 1] color components
            
        # 3. K-space Input
        kspace_input = functions.show_kspace(input_mean)
        
        # 4. K-space Target
        # show_kspace expects 2D
        if target_img.ndim > 2:
            target_gray = np.mean(target_img, axis=-1)
            kspace_target = functions.show_kspace(target_gray)
        else:
            kspace_target = functions.show_kspace(target_img)
            
        # Update UI images
        self.lbl_input.set_image(array_to_qimage_gray(input_mean))
        
        if self.metric == 'cfa':
            self.lbl_target.set_image(array_to_qimage_rgb(target_img))
        else:
            self.lbl_target.set_image(array_to_qimage_gray(target_img))
            
        self.lbl_k_input.set_image(array_to_qimage_magma(kspace_input))
        self.lbl_k_target.set_image(array_to_qimage_magma(kspace_target))

if __name__ == "__main__":
    # Remove any potential backend/frontend paths or setup
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Dark theme fusion
    
    # Dark palette for overall app
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = DTIMainWindow()
    window.show()
    sys.exit(app.exec())
