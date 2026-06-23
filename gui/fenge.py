# -*- coding: utf-8 -*-
"""
GISeg-Bench  Unified Interactive Segmentation GUI
===================================================
**Single entry point** for all 24 models — no need to enter any model folder.

Features:
    - Dropdown model selector (all 24 models auto-detected)
    - Checkpoint file picker
    - Image upload (png / jpg / jpeg / bmp)
    - Mouse-drag ROI bounding box
    - Click "Run" → inference → 3-panel paper-grade visualization
    - Auto-saves: result.png, mask.png, overlay.png

Design:
    - Calls ``inference/loader.py`` → ``build_model()`` + ``load_checkpoint()``
    - Calls ``inference/predictor.py`` → ``_extract_prediction()``
    - Calls ``inference/postprocess.py`` → ``logits_to_mask()``
    - Zero modifications to existing modules — only imports their public APIs.

Usage::

    python fenge.py
"""

import os
import sys
import json
import threading
from datetime import datetime

# --- Project root ---
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize as sk_resize
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn.functional as F
from torchvision import transforms as T

# --- PyQt5 ---
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QFileDialog, QGraphicsScene, QGraphicsView,
    QGraphicsRectItem, QMessageBox, QStatusBar,
    QGroupBox, QSplitter, QFrame, QSizePolicy,
)

# --- Matplotlib (imported inside thread to avoid blocking GUI startup) ---
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for save; we use plt.show() separately
import matplotlib.pyplot as plt

# --- Project modules (read-only imports) ---
from inference.loader import build_model, load_checkpoint, list_available_models
from inference.predictor import _extract_prediction
from inference.postprocess import logits_to_mask
from configs.model_config import MODEL_REGISTRY
from configs.dataset_config import DATASET_REGISTRY
from utils import ensure_dir


# ===================================================================
#  Constants
# ===================================================================
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "infer_results")
ensure_dir(OUTPUT_DIR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_SIZE = 256


# ===================================================================
#  Metrics (standalone — same as existing gui.py logic)
# ===================================================================

def calculate_metrics(pred_mask, gt_mask):
    """Dice, IoU, Recall, Precision, HD95."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    inter = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    p_sum = np.sum(pred_mask)
    g_sum = np.sum(gt_mask)

    dice = 2 * inter / (p_sum + g_sum + 1e-6)
    iou = inter / (union + 1e-6)
    recall = inter / (g_sum + 1e-6) if g_sum > 0 else 0.0
    precision = inter / (p_sum + 1e-6) if p_sum > 0 else 0.0

    # HD95
    hd95 = 0.0
    if p_sum > 0 and g_sum > 0:
        try:
            dt_p = distance_transform_edt(~pred_mask)
            dt_g = distance_transform_edt(~gt_mask)
            s1 = dt_p[gt_mask]
            s2 = dt_g[pred_mask]
            if len(s1) > 0 and len(s2) > 0:
                hd95 = float(max(np.percentile(s1, 95), np.percentile(s2, 95)))
        except Exception:
            pass

    return dice, iou, recall, precision, hd95


# ===================================================================
#  Image helpers
# ===================================================================

def load_image_rgb(path):
    """Load any image as RGB uint8 numpy array."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def np_to_pixmap(arr):
    """uint8 [H, W, 3] → QPixmap."""
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ===================================================================
#  Inference Runner (QThread)
# ===================================================================

class InferenceRunner(QThread):
    """Run model inference in a background thread.

    Signals:
        result_ready(np.ndarray, tuple): emits (mask_uint8, bbox_xyxy).
        error_occurred(str): emits error message.
    """
    result_ready = pyqtSignal(np.ndarray, tuple)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, device, img_region, bbox, n_classes, image_size):
        super().__init__()
        self.model = model
        self.device = device
        self.img_region = img_region          # uint8 [H, W, 3] ROI crop
        self.bbox = bbox                       # (xmin, ymin, xmax, ymax)
        self.n_classes = n_classes
        self.image_size = image_size

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def run(self):
        try:
            h, w = self.img_region.shape[:2]
            im_size = self.image_size

            # 1. Resize ROI to model input size
            img_small = sk_resize(
                self.img_region, (im_size, im_size),
                preserve_range=True, anti_aliasing=True,
            ).astype(np.uint8)

            # 2. Preprocess → [1, 3, H, W]
            input_tensor = self.transform(img_small).unsqueeze(0).to(
                self.device, dtype=torch.float32
            )

            # 3. Forward
            with torch.no_grad():
                output = self.model(input_tensor)
                logits = _extract_prediction(output)   # normalize dict/tuple/list → tensor

            # 4. Post-process → [H, W] uint8 mask
            pred_mask = logits_to_mask(
                logits.cpu(), threshold=0.5
            ).squeeze(0).numpy().astype(np.uint8)

            # 5. Resize back to original ROI size
            if pred_mask.shape[:2] != (h, w):
                pred_mask = sk_resize(
                    pred_mask, (h, w), order=0,
                    preserve_range=True, anti_aliasing=False,
                ).astype(np.uint8)

            self.result_ready.emit(pred_mask, self.bbox)

        except Exception as e:
            import traceback
            msg = f"{e}\n{traceback.format_exc()}"
            self.error_occurred.emit(msg)


# ===================================================================
#  Visualization (threaded matplotlib)
# ===================================================================

def show_results_plot(full_image, mask_pred, bbox, gt_mask, metrics, save_dir):
    """Generate and display the 3-panel paper-grade figure.

    Panel 1: Full image + bbox rect + mask overlay
    Panel 2: Prediction mask (binary)
    Panel 3: Ground truth mask (binary, or blank)
    """
    xmin, ymin, xmax, ymax = bbox
    h_box, w_box = ymax - ymin, xmax - xmin

    # --- masks for display ---
    pred_display = (mask_pred * 255).astype(np.uint8)

    gt_display_crop = np.zeros_like(pred_display)
    gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)
    if gt_mask is not None:
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        if gt_mask.shape[0] >= ymax and gt_mask.shape[1] >= xmax:
            gt_region = gt_mask[ymin:ymax, xmin:xmax]
            gt_region_binary = (gt_region > 0).astype(np.uint8)
            gt_display_crop = gt_region_binary * 255
        else:
            try:
                gt_r = sk_resize(gt_mask, (h_box, w_box), preserve_range=True, order=0)
                gt_region_binary = (gt_r > 0).astype(np.uint8)
                gt_display_crop = gt_region_binary * 255
            except Exception:
                pass

    # --- compute metrics if GT available ---
    dice, iou, recall, precision, hd95 = 0, 0, 0, 0, 0
    if gt_region_binary.sum() > 0:
        dice, iou, recall, precision, hd95 = calculate_metrics(
            mask_pred, gt_region_binary
        )

    # --- create overlay image ---
    overlay_img = full_image.copy()
    roi = overlay_img[ymin:ymax, xmin:xmax]
    roi[mask_pred == 1] = [0, 255, 0]  # green prediction overlay
    overlay_img[ymin:ymax, xmin:xmax] = roi

    # --- draw bbox on overlay ---
    from PIL import Image, ImageDraw
    overlay_pil = Image.fromarray(overlay_img)
    draw = ImageDraw.Draw(overlay_pil)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    overlay_img = np.array(overlay_pil)

    # ================================================================
    #  3-panel figure
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Original + bbox + overlay
    axes[0].imshow(overlay_img)
    axes[0].set_title("Image + ROI + Overlay", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Prediction mask
    axes[1].imshow(pred_display, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Prediction Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Ground truth (or blank)
    if gt_region_binary.sum() > 0:
        axes[2].imshow(gt_display_crop, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Ground Truth", fontsize=13, fontweight="bold")
    else:
        axes[2].imshow(np.zeros_like(pred_display), cmap="gray")
        axes[2].set_title("Ground Truth (N/A)", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # --- metrics text ---
    metrics_str = (
        f"Dice: {dice:.4f}  |  IoU: {iou:.4f}  |  "
        f"Recall: {recall:.4f}  |  Precision: {precision:.4f}  |  "
        f"HD95: {hd95:.2f}"
    )
    fig.text(0.5, 0.02, metrics_str, ha="center", fontsize=12,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # --- save ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(save_dir, f"result_{ts}.png")
    mask_path   = os.path.join(save_dir, f"mask_{ts}.png")
    overlay_path = os.path.join(save_dir, f"overlay_{ts}.png")

    fig.savefig(result_path, dpi=150, bbox_inches="tight")
    Image.fromarray(pred_display).save(mask_path)
    Image.fromarray(overlay_img).save(overlay_path)

    # Save metrics
    metrics_json = {
        "Dice": dice, "IoU": iou, "Recall": recall,
        "Precision": precision, "HD95": hd95,
        "timestamp": ts,
    }
    with open(os.path.join(save_dir, f"metrics_{ts}.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[Results] Saved to {save_dir}/result_{ts}.png")

    # --- render figure to PNG bytes for in-GUI display ---
    import io as _io
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    buf.close()

    return png_bytes, dice, iou, recall, precision, hd95


# ===================================================================
#  Main GUI Window
# ===================================================================

class UnifiedSegmentationGUI(QMainWindow):
    """Unified single-image interactive segmentation GUI for all 24 models."""

    # Signal: emit the rendered 3-panel figure (PNG bytes) + status message
    result_figure_ready = pyqtSignal(bytes, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GISeg-Bench  —  Unified Interactive Segmentation")
        self.resize(1400, 950)

        # --- internal state ---
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = 1
        self.image_size = DEFAULT_IMAGE_SIZE

        self.img_3c = None           # full uint8 [H, W, 3]
        self.image_path = None
        self.is_mouse_down = False
        self.start_pos = None
        self.current_rect = None

        self._init_ui()
        self._populate_models()

        # Connect result display signal
        self.result_figure_ready.connect(self._display_result_figure)

        self.statusBar().showMessage(f"Device: {self.device}  |  Ready")

    # ------------------------------------------------------------------
    #  UI Construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ==================== Top control bar ====================
        top_bar = QHBoxLayout()

        # Model selector
        top_bar.addWidget(QLabel("<b>Model:</b>"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setToolTip("Select a segmentation model")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        top_bar.addWidget(self.model_combo)

        top_bar.addSpacing(12)

        # Weight selector
        self.weight_btn = QPushButton("Select Checkpoint (.pth)")
        self.weight_btn.setMinimumWidth(200)
        self.weight_btn.clicked.connect(self._select_weight)
        top_bar.addWidget(self.weight_btn)

        self.weight_label = QLabel("<i>No checkpoint selected</i>")
        self.weight_label.setWordWrap(True)
        top_bar.addWidget(self.weight_label)

        top_bar.addStretch()
        top_bar.addSpacing(12)

        # Image load
        self.img_btn = QPushButton("Load Image")
        self.img_btn.setMinimumWidth(120)
        self.img_btn.clicked.connect(self._load_image)
        top_bar.addWidget(self.img_btn)

        self.img_label = QLabel("<i>No image</i>")
        top_bar.addWidget(self.img_label)

        root_layout.addLayout(top_bar)

        # ==================== Center: graphics view ====================
        self.view = QGraphicsView()
        self.view.setMinimumHeight(500)
        self.view.setStyleSheet("background-color: #2b2b2b;")
        self.view.setRenderHints(self.view.renderHints())
        root_layout.addWidget(self.view, stretch=1)

        # ==================== Result display area ====================
        result_group = QGroupBox("Segmentation Result (3-Panel View)")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(180)
        self.result_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.result_label.setStyleSheet(
            "background-color: #ffffff; border: 2px solid #cccccc; border-radius: 4px;"
        )
        self.result_label.setText(
            "<i style='color: #888;'>Segmentation result (3-panel figure) "
            "will appear here after inference</i>"
        )
        result_layout.addWidget(self.result_label)
        root_layout.addWidget(result_group)

        # ==================== Bottom bar ====================
        bottom_bar = QHBoxLayout()

        self.info_label = QLabel(
            "<b>Instructions:</b> 1. Select model  2. Select checkpoint  "
            "3. Load image  4. Drag ROI box on image  5. Click Run"
        )
        bottom_bar.addWidget(self.info_label)

        bottom_bar.addStretch()

        # Run button
        self.run_btn = QPushButton("▶  Run Segmentation")
        self.run_btn.setMinimumHeight(42)
        self.run_btn.setMinimumWidth(200)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 15px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_btn.setEnabled(False)
        bottom_bar.addWidget(self.run_btn)

        root_layout.addLayout(bottom_bar)

        # ==================== Status bar ====================
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    #  Model population
    # ------------------------------------------------------------------

    def _populate_models(self):
        """Fill the model dropdown with all 24 registered models."""
        models = list_available_models()
        # Use MODEL_REGISTRY for metadata if available
        for name in models:
            info = MODEL_REGISTRY.get(name, {})
            desc = info.get("description", "")
            family = info.get("family", "")
            label = f"{name}  [{family}]" if family else name
            self.model_combo.addItem(label, userData=name)

    def _on_model_changed(self, text):
        """When model selection changes, update defaults."""
        if not text:
            return
        name = self.model_combo.currentData()
        if name and name in MODEL_REGISTRY:
            info = MODEL_REGISTRY[name]
            self.n_classes = info.get("n_classes", 1)
            self.image_size = info.get("image_size", DEFAULT_IMAGE_SIZE)
            self.statusBar().showMessage(
                f"Model: {name}  |  n_classes={self.n_classes}  "
                f"image_size={self.image_size}  |  {info.get('description', '')}"
            )
        self._update_run_button()

    # ------------------------------------------------------------------
    #  Weight selection
    # ------------------------------------------------------------------

    def _select_weight(self):
        """File dialog to pick a .pth checkpoint."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "outputs/",
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*)",
        )
        if path:
            self.weight_path = path
            self.weight_label.setText(os.path.basename(path))
            self.statusBar().showMessage(f"Checkpoint: {path}")
            self._update_run_button()

    # ------------------------------------------------------------------
    #  Image loading
    # ------------------------------------------------------------------

    def _load_image(self):
        """Open an image and display it in the QGraphicsView."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Medical Image", "data/",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        try:
            self.img_3c = load_image_rgb(path)
            self.image_path = path
            self.img_label.setText(os.path.basename(path))

            h, w = self.img_3c.shape[:2]
            self.scene = QGraphicsScene(0, 0, w, h)
            self.scene.addPixmap(np_to_pixmap(self.img_3c))

            # Bind mouse events for ROI drawing
            self.scene.mousePressEvent = self._mouse_press
            self.scene.mouseMoveEvent = self._mouse_move
            self.scene.mouseReleaseEvent = self._mouse_release

            self.view.setScene(self.scene)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

            self.statusBar().showMessage(
                f"Loaded: {path}  ({w}×{h}) — Drag to draw ROI"
            )
            self._update_run_button()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    # ------------------------------------------------------------------
    #  Mouse events (ROI drawing)
    # ------------------------------------------------------------------

    def _mouse_press(self, ev):
        if ev.button() == Qt.LeftButton:
            self.is_mouse_down = True
            self.start_pos = (int(ev.scenePos().x()), int(ev.scenePos().y()))

    def _mouse_move(self, ev):
        if not self.is_mouse_down:
            return
        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
        sx, sy = self.start_pos
        xmin, xmax = min(x, sx), max(x, sx)
        ymin, ymax = min(y, sy), max(y, sy)
        if self.current_rect:
            self.scene.removeItem(self.current_rect)
        self.current_rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin,
            pen=QPen(QColor("#00FF00"), 2),
        )

    def _mouse_release(self, ev):
        self.is_mouse_down = False
        if self.img_3c is None or self.start_pos is None:
            return

        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
        sx, sy = self.start_pos
        h_img, w_img = self.img_3c.shape[:2]

        xmin = max(0, min(x, sx))
        xmax = min(w_img, max(x, sx))
        ymin = max(0, min(y, sy))
        ymax = min(h_img, max(y, sy))

        if xmax - xmin > 5 and ymax - ymin > 5:
            self.bbox = (xmin, ymin, xmax, ymax)
            print(f"[ROI] {self.bbox}")
            self.statusBar().showMessage(
                f"ROI: ({xmin},{ymin}) → ({xmax},{ymax})  [{xmax-xmin}×{ymax-ymin}]"
            )
            self._update_run_button()

    # ------------------------------------------------------------------
    #  Run button state
    # ------------------------------------------------------------------

    def _update_run_button(self):
        ready = (
            self.model_combo.currentData() is not None
            and hasattr(self, "weight_path")
            and self.img_3c is not None
            and hasattr(self, "bbox")
        )
        self.run_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    #  Segmentation execution
    # ------------------------------------------------------------------

    def _run_segmentation(self):
        """Build model, load checkpoint, crop ROI, run inference."""
        model_name = self.model_combo.currentData()
        if not model_name:
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Running...")
        self.statusBar().showMessage("Building model & loading checkpoint...")

        try:
            # --- Build model ---
            self.model = build_model(
                model_name=model_name,
                n_classes=self.n_classes,
                image_size=self.image_size,
            )
            self.model = load_checkpoint(
                self.model, self.weight_path, device=self.device
            )
            self.model.to(self.device)
            self.model.eval()

            # --- Crop ROI ---
            xmin, ymin, xmax, ymax = self.bbox
            roi = self.img_3c[ymin:ymax, xmin:xmax].copy()

            # --- Launch inference thread ---
            self.infer_thread = InferenceRunner(
                model=self.model,
                device=self.device,
                img_region=roi,
                bbox=self.bbox,
                n_classes=self.n_classes,
                image_size=self.image_size,
            )
            self.infer_thread.result_ready.connect(self._on_result)
            self.infer_thread.error_occurred.connect(self._on_error)
            self.infer_thread.start()

        except Exception as e:
            self._on_error(str(e))

    # ------------------------------------------------------------------
    #  Result handling
    # ------------------------------------------------------------------

    def _on_result(self, mask_pred, bbox):
        """Called from inference thread with prediction mask."""
        self.run_btn.setText("▶  Run Segmentation")
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Inference complete — generating visualization...")

        xmin, ymin, xmax, ymax = bbox

        # --- Update overlay on the displayed image ---
        overlay_region = self.img_3c[ymin:ymax, xmin:xmax].copy()
        overlay_region[mask_pred == 1] = [0, 255, 0]  # green
        display_img = self.img_3c.copy()
        display_img[ymin:ymax, xmin:xmax] = overlay_region
        self.scene.clear()
        self.scene.addPixmap(np_to_pixmap(display_img))

        # --- Find GT ---
        gt_mask = self._find_ground_truth()

        # --- Show 3-panel figure in a thread (matplotlib) ---
        def _show():
            png_bytes, dice, iou, recall, precision, hd95 = show_results_plot(
                full_image=self.img_3c,
                mask_pred=mask_pred,
                bbox=bbox,
                gt_mask=gt_mask,
                metrics={},
                save_dir=OUTPUT_DIR,
            )
            # Emit PNG bytes + status (main-thread safe via queued signal)
            status_msg = (
                f"Done!  Dice={dice:.4f}  IoU={iou:.4f}  "
                f"Results saved to {OUTPUT_DIR}"
            )
            self.result_figure_ready.emit(png_bytes, status_msg)
        threading.Thread(target=_show, daemon=True).start()

    def _display_result_figure(self, png_bytes, status_msg=""):
        """Display the rendered 3-panel result figure in the GUI.

        Called on the main thread via ``result_figure_ready`` signal.
        ``png_bytes`` is the PNG-encoded figure as raw bytes.
        ``status_msg`` is an optional status-bar message.
        """
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes, "PNG")
        if pixmap.isNull():
            self.statusBar().showMessage("Warning: failed to render result figure")
            return
        # Scale to fit the label width while keeping aspect ratio
        lbl_w = self.result_label.width()
        if lbl_w > 10:
            scaled = pixmap.scaledToWidth(
                min(lbl_w - 20, pixmap.width()),
                Qt.SmoothTransformation,
            )
        else:
            scaled = pixmap
        self.result_label.setPixmap(scaled)
        self.result_label.setMinimumHeight(scaled.height() + 10)
        # Update status bar (main-thread safe)
        if status_msg:
            self.statusBar().showMessage(status_msg)

    def _on_error(self, msg):
        self.run_btn.setText("▶  Run Segmentation")
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Error during inference")
        print(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Inference Error", msg[:500])

    # ------------------------------------------------------------------
    #  Ground truth discovery
    # ------------------------------------------------------------------

    def _find_ground_truth(self):
        """Auto-discover ground-truth mask for the loaded image.

        Search order:
            1. ../masks/<basename>.png  (standard layout)
            2. ../masks/<basename>.jpg
            3. ./<basename>_mask.png    (same-dir layout)
            4. ../masktest/<basename>.png  (Kvasir test layout)
        """
        if not self.image_path:
            return None

        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        parent_dir = os.path.dirname(base_dir)

        candidates = [
            os.path.join(parent_dir, "masks", base_name + ".png"),
            os.path.join(parent_dir, "masks", base_name + ".jpg"),
            os.path.join(parent_dir, "masks", base_name + ".jpeg"),
            os.path.join(parent_dir, "masktest", base_name + ".png"),
            os.path.join(parent_dir, "masktest", base_name + ".jpg"),
            os.path.join(base_dir, base_name + "_mask.png"),
            os.path.join(base_dir, base_name + "_mask.jpg"),
        ]

        for p in candidates:
            if os.path.isfile(p):
                try:
                    gt = io.imread(p)
                    print(f"[GT] Found: {p}")
                    return gt
                except Exception:
                    continue

        print("[GT] No ground-truth mask found")
        return None


# ===================================================================
#  Entry point
# ===================================================================

def main():
    print("=" * 60)
    print("  GISeg-Bench  —  Unified Interactive Segmentation GUI")
    print("=" * 60)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Available models: {len(list_available_models())}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global stylesheet
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QGroupBox { font-weight: bold; }
        QPushButton { padding: 6px 14px; }
        QComboBox { padding: 4px 8px; min-height: 28px; }
    """)

    gui = UnifiedSegmentationGUI()
    gui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
