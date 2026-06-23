# -*- coding: utf-8 -*-

import sys, os, threading

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transnuseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QGraphicsView, QHBoxLayout, QVBoxLayout, \
    QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from skimage import io
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt

# ===================== 模型 =====================
from models.transnuseg import TransNuSeg

# ================= 指标计算函数 =================
def calculate_metrics(pred_mask, gt_mask):
    """计算 Dice, IoU, Recall, Precision, HD95"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # 计算交集和并集
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    # Dice 系数
    dice = 2 * intersection / (pred_sum + gt_sum + 1e-6)

    # IoU
    iou = intersection / (union + 1e-6)

    # Recall (Sensitivity)
    recall = intersection / (gt_sum + 1e-6)

    # Precision
    precision = intersection / (pred_sum + 1e-6)

    # HD95 (Hausdorff Distance 95%)
    def compute_hd95(mask1, mask2):
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            return 0.0

        # 计算边界距离
        dist1 = distance_transform_edt(~mask1)
        dist2 = distance_transform_edt(~mask2)

        surface1 = dist1[mask2]
        surface2 = dist2[mask1]

        if len(surface1) == 0 or len(surface2) == 0:
            return 0.0

        # 95百分位数
        hd95_1 = np.percentile(surface1, 95)
        hd95_2 = np.percentile(surface2, 95)
        hd95 = max(hd95_1, hd95_2)

        return hd95

    hd95 = compute_hd95(pred_mask, gt_mask)

    return dice, iou, recall, precision, hd95

# ===================== 工具函数 =====================
def np2pixmap(np_img):
    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)
    h, w, c = np_img.shape
    bytesPerLine = 3 * w
    qImg = QImage(np_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

# ===================== 推理线程 =====================
class InferenceThread(QThread):
    result_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self, model, device, img_region, box):
        super().__init__()
        self.model = model
        self.device = device
        self.img_region = img_region
        self.box = box
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self):
        h, w, _ = self.img_region.shape
        img_small = resize(self.img_region, (256, 256), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        input_tensor = self.transform(img_small).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            seg_mask, edge_mask, cluster_edge = self.model(input_tensor)
            # 如果输出尺寸不是256x256，插值到256x256
            if seg_mask.shape[-1] != 256:
                seg_mask = torch.nn.functional.interpolate(seg_mask, size=(256, 256), mode='bilinear',
                                                           align_corners=True)
            probs = torch.softmax(seg_mask, dim=1)
            mask_pred = probs[0, 1, :, :].cpu().numpy()
            mask_pred = (mask_pred > 0.5).astype(np.uint8)
            mask_pred = resize(mask_pred, (h, w), order=0, preserve_range=True).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== GUI =====================
class InteractiveSegmentationGUI(QWidget):
    def __init__(self, model, device='cpu'):
        super().__init__()
        self.setWindowTitle("TransNuSeg GUI")
        self.resize(1200, 900)
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.is_mouse_down = False
        self.start_pos = None
        self.rect = None
        self.bg_img = None
        self.image_path = None
        self.img_3c = None

        # --- GUI布局 ---
        self.view = QGraphicsView()
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)
        hbox = QHBoxLayout()
        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Result")
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)
        vbox.addLayout(hbox)
        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_image)

    # -------- 加载图像 --------
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", ".", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if not file_path:
            return
        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
        self.img_3c = img_3c
        self.image_path = file_path
        pixmap = np2pixmap(self.img_3c)
        H, W, _ = self.img_3c.shape
        self.scene = QGraphicsScene(0, 0, W, H)
        self.bg_img = self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release
        print(f"Loaded: {file_path}")

    # -------- 鼠标事件 --------
    def mouse_press(self, ev):
        self.is_mouse_down = True
        self.start_pos = int(ev.scenePos().x()), int(ev.scenePos().y())

    def mouse_move(self, ev):
        if not self.is_mouse_down: return
        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
        sx, sy = self.start_pos
        xmin, xmax = min(x, sx), max(x, sx)
        ymin, ymax = min(y, sy), max(y, sy)
        if self.rect: self.scene.removeItem(self.rect)
        self.rect = self.scene.addRect(xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"), 2))

    def mouse_release(self, ev):
        self.is_mouse_down = False
        if self.img_3c is None: return
        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
        sx, sy = self.start_pos
        h_img, w_img = self.img_3c.shape[:2]
        xmin, xmax = max(0, min(x, sx)), min(w_img, max(x, sx))
        ymin, ymax = max(0, min(y, sy)), min(h_img, max(y, sy))
        if xmax - xmin > 5 and ymax - ymin > 5:
            print(f"Region: [{xmin}:{xmax},{ymin}:{ymax}]")
            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()
            self.thread = InferenceThread(self.model, self.device, img_region, (xmin, ymin, xmax, ymax))
            self.thread.result_signal.connect(self.update_overlay)
            self.thread.start()

    # -------- 更新显示 --------
    def update_overlay(self, mask_pred, box):
        xmin, ymin, xmax, ymax = box
        overlay_region = self.img_3c[ymin:ymax, xmin:xmax]
        overlay_region[mask_pred == 1] = [255, 255, 255]
        self.img_3c[ymin:ymax, xmin:xmax] = overlay_region
        self.bg_img.setPixmap(np2pixmap(self.img_3c))
        self.view.viewport().update()
        self.show_plots_in_thread(mask_pred, box)

    def show_plots_in_thread(self, mask_pred, box):
        def _worker():
            xmin, ymin, xmax, ymax = box
            h_box, w_box = ymax - ymin, xmax - xmin
            mask_display = (mask_pred * 255).astype(np.uint8)

            # 尝试寻找金标准
            gt = None
            try_paths = []
            if self.image_path:
                base_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                parent_dir = os.path.dirname(base_dir)
                try_paths.append(os.path.join(parent_dir, "label", base_name + ".jpg"))
                try_paths.append(os.path.join(parent_dir, "label", base_name + ".png"))
                try_paths.append(os.path.join(base_dir, base_name + "_mask.jpg"))
                try_paths.append(os.path.join(base_dir, base_name + "_mask.png"))

            for p in try_paths:
                if os.path.exists(p):
                    try:
                        gt = io.imread(p)
                        print(f"Found GT at: {p}")
                        break
                    except:
                        continue

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)
            gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)

            if gt is not None:
                if gt.ndim == 3: gt = gt[:, :, 0]
                try:
                    from skimage.transform import resize as sk_resize
                    gt_r = sk_resize(gt, (h_box, w_box), preserve_range=True, order=0)
                    gt_region_binary = (gt_r > 0).astype(np.uint8)
                    gt_display_crop = gt_region_binary * 255
                except:
                    pass

            # ====== 计算指标 ======
            dice, iou, recall, precision, hd95 = calculate_metrics(mask_pred, gt_region_binary)

            # ====== 绘图 ======
            plt.figure(figsize=(14, 7))

            plt.subplot(1, 3, 1)
            plt.imshow(self.img_3c)
            plt.title("Full Image (with Result)")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask_display, cmap="gray")
            plt.title("Prediction Mask (ROI)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(gt_display_crop, cmap="gray")
            plt.title("Ground Truth (ROI)")
            plt.axis('off')

            # ================== 添加指标 ==================
            metrics_text = (f'Dice: {dice:.4f}  |  IoU: {iou:.4f}  |  '
                            f'Recall: {recall:.4f}  |  Precision: {precision:.4f}  |  '
                            f'HD95: {hd95:.2f}')
            plt.figtext(0.5, 0.02, metrics_text, wrap=True,
                        horizontalalignment='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            plt.show()

        threading.Thread(target=_worker, daemon=True).start()

    def save_image(self):
        if self.image_path:
            out_path = f"{os.path.splitext(self.image_path)[0]}_result.png"
            io.imsave(out_path, self.img_3c)
            QMessageBox.information(self, "Success", f"Saved to: {out_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    weight_path = "transnuseg_kvasir_epoch_15.pth"
    if not os.path.exists(weight_path):
        print(f"Weight not found at {weight_path}")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TransNuSeg(img_size=256, num_classes=2, depths=[2, 2, 2, 2], embed_dim=96)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    gui = InteractiveSegmentationGUI(model, device=device)
    gui.show()
    sys.exit(app.exec())
