# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import threading

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/h2former')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
from torchvision import transforms

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,
    QVBoxLayout, QPushButton, QWidget
)

from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

# ================= 路径 =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.H2Former import Res34_Swin_MS, BasicBlock

# ================= 工具 =================
def np2pixmap(img):
    img = img.astype(np.uint8)
    h, w, _ = img.shape
    return QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_RGB888))

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

# ================= GUI =================
class SegGUI(QWidget):
    def __init__(self, model, device):
        super().__init__()
        self.setWindowTitle("H2Former Segmentation (Final)")
        self.resize(1000, 800)

        self.model = model
        self.device = device
        self.image_path = None
        self.img_raw = None

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        btn = QPushButton("Load Image & Segment")

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addWidget(btn)

        btn.clicked.connect(self.load_and_segment)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_and_segment(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", ".", "*.jpg *.png"
        )
        if not path:
            return

        self.image_path = path

        # ================= 1. 读取图像 =================
        img = io.imread(path)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        self.img_raw = img.copy()

        # ================= 2. 模型输入 =================
        img_resized = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        # ====== 4 通道输入（和训练一致）======
        extra = tensor.mean(dim=1, keepdim=True)
        tensor = torch.cat([tensor, extra], dim=1)

        # ================= 3. 推理 =================
        with torch.no_grad():
            logits = self.model(tensor)
            pred_mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

        # resize 回原图尺寸
        pred_mask = resize(
            pred_mask,
            (img.shape[0], img.shape[1]),
            order=0,
            preserve_range=True
        ).astype(np.uint8)

        # ================= 4. 白色覆盖 =================
        overlay = img.copy()
        overlay[pred_mask == 1] = [255, 255, 255]

        # ================= 5. 更新 GUI =================
        self.scene.clear()
        self.scene.addPixmap(np2pixmap(overlay))
        self.view.fitInView(self.scene.itemsBoundingRect())

        # ================= 6. 三图显示 =================
        self.show_three_results(overlay, pred_mask)

    def show_three_results(self, overlay_img, pred_mask):
        def _worker():
            # ========== 预测 Mask（黑底白） ==========
            pred_vis = pred_mask * 255

            # ========== 查找 GT ==========
            gt_vis = np.zeros_like(pred_vis, dtype=np.uint8)
            gt = None

            base_dir = os.path.dirname(self.image_path)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            parent_dir = os.path.dirname(base_dir)

            try_paths = [
                os.path.join(parent_dir, "masks", base_name + ".png"),
                os.path.join(parent_dir, "masks", base_name + ".jpg"),
                os.path.join(base_dir, base_name + "_mask.png"),
                os.path.join(base_dir, base_name + "_mask.jpg"),
            ]

            for p in try_paths:
                if os.path.exists(p):
                    try:
                        gt = io.imread(p)
                        break
                    except:
                        pass

            if gt is not None:
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
                gt_vis = resize(
                    gt,
                    pred_vis.shape,
                    order=0,
                    preserve_range=True
                )
                gt_vis = (gt_vis > 0).astype(np.uint8) * 255

            # ========== 计算指标 ==========
            gt_binary = (gt_vis > 0).astype(np.uint8)
            dice, iou, recall, precision, hd95 = calculate_metrics(pred_mask, gt_binary)

            # ========== 绘图 ==========
            plt.figure(figsize=(14, 7))

            plt.subplot(1, 3, 1)
            plt.imshow(overlay_img)
            plt.title("Image + White Mask Overlay")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(pred_vis, cmap="gray")
            plt.title("Prediction Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(gt_vis, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

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

# ================= 主程序 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = Res34_Swin_MS(
        image_size=224,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=1
    ).to(device)

    model.load_state_dict(torch.load(
        r"./weights/h2former_kvasir_epoch_1.pth",
        map_location=device
    ))
    model.eval()

    app = QApplication(sys.argv)
    gui = SegGUI(model, device)
    gui.show()
    sys.exit(app.exec())
