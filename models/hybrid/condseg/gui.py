# -*- coding: utf-8 -*-

import sys

import os

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/hybrid/condseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QPen, QColor

from PyQt5.QtWidgets import (

    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,

    QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QMessageBox

)

from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib import pyplot as plt

from skimage import io

from skimage.transform import resize

import torch

import torch.nn as nn

from torchvision import transforms

import threading

import numpy as np

from scipy.spatial.distance import cdist

def compute_metrics(pred, gt):

    """

    pred, gt: 二值 mask，shape = (H, W)，值为 0/1

    """

    pred = pred.astype(bool)

    gt = gt.astype(bool)

    eps = 1e-7

    # TP / FP / FN

    tp = np.logical_and(pred, gt).sum()

    fp = np.logical_and(pred, ~gt).sum()

    fn = np.logical_and(~pred, gt).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    iou = tp / (tp + fp + fn + eps)

    recall = tp / (tp + fn + eps)

    precision = tp / (tp + fp + eps)

    # ---------- HD95 ----------

    def hd95(a, b):

        a_pts = np.argwhere(a)

        b_pts = np.argwhere(b)

        if len(a_pts) == 0 or len(b_pts) == 0:

            return np.nan

        dists = cdist(a_pts, b_pts)

        return np.percentile(dists.min(axis=1), 95)

    hd95_val = hd95(pred, gt)

    return dice, iou, recall, precision, hd95_val

# ============ ★★★ 从 ConDSeg-main 加载模型，并做一层封装 ★★★

# 把 ConDSeg 所在目录加入 sys.path

sys.path.append(r"./network")

from network.model import ConDSeg  # 确保 model.py 中类名为 ConDSeg

class ConDSegWrapper(nn.Module):

    """

    包一层，把 ConDSeg 的输出变成：

        {"out": (B, 2, H, W)}  # [bg, fg]

    这样就能完全兼容原来 DenseNetSeg 的使用方式，

    不用改 InferenceThread 里对 model 的调用逻辑。

    """

    def __init__(self):

        super().__init__()

        self.model = ConDSeg()  # 和你训练时一致

    def forward(self, x):

        # 训练代码里： outputs, mask_fg, mask_bg, mask_uc = model(images)

        out = self.model(x)

        if isinstance(out, (tuple, list)):

            out = out[0]  # 取 outputs: [B, 1, H, W]，模型里已经做了 Sigmoid

        # out 范围在 [0,1]，视为前景概率

        # 构造两通道：第 0 通道为背景概率，第 1 通道为前景概率

        out_fg = out

        out_bg = 1.0 - out_fg

        out_2ch = torch.cat([out_bg, out_fg], dim=1)  # [B, 2, H, W]

        return {"out": out_2ch}

# ===================== 工具函数 =====================

def np2pixmap(np_img):

    """将 numpy 数组转换为 QPixmap 用于显示"""

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

        # 预处理：使用 ImageNet 的均值和方差

        self.transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def run(self):

        h, w, _ = self.img_region.shape

        # 1. 缩放

        img_small = resize(self.img_region, (256, 256), preserve_range=True, anti_aliasing=True).astype(np.uint8)

        # 2. 转 Tensor

        input_tensor = self.transform(img_small).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():

            # 3. 推理 (保持原逻辑：model(...)['out'])

            output = self.model(input_tensor)['out']  # (B,2,H,W)，来自 ConDSegWrapper

            # 4. 获取概率 -> 取通道1 -> 阈值化

            probs = torch.softmax(output, dim=1)

            mask_pred = probs[0, 1, :, :].cpu().numpy()

            mask_pred = (mask_pred > 0.5).astype(np.uint8)

            # 5. 还原回框选区域大小

            mask_pred = resize(mask_pred, (h, w), order=0, preserve_range=True).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== GUI =====================

class InteractiveSegmentationGUI(QWidget):

    def __init__(self, model, device='cpu'):

        super().__init__()

        self.setWindowTitle("DeepLabV3 Medical Segmentation - Final")

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

        # --- GUI 布局 ---

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

        # 处理通道

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

        if not self.is_mouse_down:

            return

        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())

        sx, sy = self.start_pos

        xmin, xmax = min(x, sx), max(x, sx)

        ymin, ymax = min(y, sy), max(y, sy)

        if self.rect:

            self.scene.removeItem(self.rect)

        self.rect = self.scene.addRect(xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"), 2))

    def mouse_release(self, ev):

        self.is_mouse_down = False

        if self.img_3c is None:

            return

        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())

        sx, sy = self.start_pos

        h_img, w_img = self.img_3c.shape[:2]

        xmin, xmax = max(0, min(x, sx)), min(w_img, max(x, sx))

        ymin, ymax = max(0, min(y, sy)), min(h_img, max(y, sy))

        if xmax - xmin > 5 and ymax - ymin > 5:

            print(f"Region: [{xmin}:{xmax}, {ymin}:{ymax}]")

            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

            self.thread = InferenceThread(self.model, self.device, img_region, (xmin, ymin, xmax, ymax))

            self.thread.result_signal.connect(self.update_overlay)

            self.thread.start()

    # -------- 更新与显示 --------

    def update_overlay(self, mask_pred, box):

        xmin, ymin, xmax, ymax = box

        # 1. 修改原图内存数据 (涂白)

        overlay_region = self.img_3c[ymin:ymax, xmin:xmax]

        overlay_region[mask_pred == 1] = [255, 255, 255]

        self.img_3c[ymin:ymax, xmin:xmax] = overlay_region

        # 2. 更新 GUI

        self.bg_img.setPixmap(np2pixmap(self.img_3c))

        self.view.viewport().update()

        # 3. 弹窗显示结果（包含金标准对比）

        self.show_plots_in_thread(mask_pred, box)

    def show_plots_in_thread(self, mask_pred, box):

        def _worker():

            import os

            import numpy as np

            from skimage import io

            from skimage.transform import resize as sk_resize

            import matplotlib.pyplot as plt

            # ====== 解包 box ======

            xmin, ymin, xmax, ymax = box

            h_box, w_box = ymax - ymin, xmax - xmin

            # ====== 预测 Mask (ROI) ======

            mask_display = (mask_pred > 0).astype(np.uint8) * 255

            # ====== 寻找 Ground Truth ======

            gt = None

            try_paths = []

            if self.image_path:

                base_dir = os.path.dirname(self.image_path)

                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

                parent_dir = os.path.dirname(base_dir)

                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".jpg"))

                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".png"))

                try_paths.append(os.path.join(base_dir, base_name + "_mask.jpg"))

                try_paths.append(os.path.join(base_dir, base_name + "_mask.png"))

            for p in try_paths:

                if os.path.exists(p):

                    try:

                        gt = io.imread(p)

                        print(f"[INFO] Found GT at: {p}")

                        break

                    except Exception as e:

                        print(f"[WARN] Failed to read GT: {p}, {e}")

            # ====== 处理 GT ======

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)

            if gt is not None:

                if gt.ndim == 3:

                    gt = gt[:, :, 0]

                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:

                    gt_region = gt[ymin:ymax, xmin:xmax]

                    gt_display_crop = (gt_region > 0).astype(np.uint8) * 255

                else:

                    try:

                        gt_r = sk_resize(

                            gt,

                            (h_box, w_box),

                            preserve_range=True,

                            order=0,

                            anti_aliasing=False

                        )

                        gt_display_crop = (gt_r > 0).astype(np.uint8) * 255

                    except Exception as e:

                        print(f"[WARN] GT resize failed: {e}")

            # ====== 计算指标（保证 GT 存在才算）=====

            if gt is not None and gt_display_crop.sum() > 0:

                dice, iou, recall, precision, hd95_val = compute_metrics(

                    mask_pred,

                    gt_display_crop // 255

                )

            else:

                dice = iou = recall = precision = hd95_val = np.nan

            # ====== 绘图 ======

            plt.figure(figsize=(12, 6))

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

            # ====== 指标文本（安全格式化）=====

            metrics_text = (

                f"Dice: {dice:.4f}    "

                f"IoU: {iou:.4f}    "

                f"Recall: {recall:.4f}    "

                f"Precision: {precision:.4f}    "

                f"HD95: {hd95_val:.2f}"

                if not np.isnan(dice)

                else "Ground Truth not found / invalid"

            )

            plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=11)

            plt.tight_layout(rect=[0, 0.05, 1, 1])

            plt.show()

        threading.Thread(target=_worker, daemon=True).start()

    def save_image(self):

        if self.image_path:

            out_path = f"{os.path.splitext(self.image_path)[0]}_result.png"

            io.imsave(out_path, self.img_3c)

            QMessageBox.information(self, "Success", f"Saved to: {out_path}")

# ===================== 主程序 =====================

if __name__ == "__main__":

    # 配置模型路径（使用你指定的 ConDSeg 权重）

    weight_path = r"./weights/con_dseg_epoch_20.pth"

    if not os.path.exists(weight_path):

        print(f"Error: Weight file not found at {weight_path}")

        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    # 构建 ConDSeg Wrapper 模型

    model = ConDSegWrapper().to(device)

    # 加载权重（注意加载到 wrapper.model 里）

    try:

        state_dict = torch.load(weight_path, map_location=device)

        model.model.load_state_dict(state_dict)

        print("ConDSeg model loaded successfully!")

    except Exception as e:

        print(f"Error loading weights: {e}")

        sys.exit(1)

    app = QApplication(sys.argv)

    gui = InteractiveSegmentationGUI(model, device=device)

    gui.show()

    sys.exit(app.exec())
