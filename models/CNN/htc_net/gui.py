# -*- coding: utf-8 -*-

# 文件名: run_gui.py

# 请将此文件放在项目根目录下运行

import sys

import os

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/htc_net')
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

import ml_collections  # HTC-Net 需要这个库来创建 config

import threading

# ============ ★★★ 1. 导入你训练的 HTC-Net 模型 ★★★ ============

try:

    from network.Net import model as HTCNetWrapper

    print("成功从 'network.Net' 导入模型。")

except ImportError as e:

    print(f"模型导入失败: {e}")

    print("错误：请确保此脚本与 'network' 文件夹在同一目录下（即项目根目录）。")

    sys.exit(1)

# ===================== 工具函数 =====================

def np2pixmap(np_img):

    if np_img.dtype != np.uint8: np_img = np_img.astype(np.uint8)

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

        # ★★★ 2. 适配 HTC-Net 的输入尺寸 (224x224) ★★★

        self.INPUT_SIZE = 224

        self.transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def run(self):

        h, w, _ = self.img_region.shape

        # 1. 缩放至模型输入尺寸 224x224

        img_small = resize(self.img_region, (self.INPUT_SIZE, self.INPUT_SIZE), preserve_range=True,

                           anti_aliasing=True).astype(np.uint8)

        # 2. 转 Tensor

        input_tensor = self.transform(img_small).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():

            # ★★★ 3. 推理 (HTC-Net 直接输出 Tensor) ★★★

            output = self.model(input_tensor)  # (B, 1, H, W)

            # 4. 获取概率 -> 阈值化

            probs = torch.sigmoid(output)

            mask_pred = probs[0, 0, :, :].cpu().numpy()  # 取第0个batch和第0个通道

            mask_pred = (mask_pred > 0.5).astype(np.uint8)

            # 5. 还原回框选区域大小

            mask_pred = resize(mask_pred, (h, w), order=0, preserve_range=True).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== GUI (这部分几乎保持不变) =====================

class InteractiveSegmentationGUI(QWidget):

    def __init__(self, model, device='cpu'):

        super().__init__()

        self.setWindowTitle("HTC-Net Medical Segmentation")

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

    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", ".", "Image Files (*.png *.jpg *.bmp *.jpeg)")

        if not file_path: return

        img_np = io.imread(file_path)

        if len(img_np.shape) == 2:

            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)

        elif img_np.shape[2] == 4:

            img_3c = img_np[:, :, :3]

        else:

            img_3c = img_np

        self.img_3c, self.image_path = img_3c, file_path

        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)

        self.bg_img = self.scene.addPixmap(pixmap)

        self.view.setScene(self.scene)

        self.scene.mousePressEvent = self.mouse_press

        self.scene.mouseMoveEvent = self.mouse_move

        self.scene.mouseReleaseEvent = self.mouse_release

        print(f"Loaded: {file_path}")

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

            print(f"Region: [{xmin}:{xmax}, {ymin}:{ymax}]")

            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

            self.thread = InferenceThread(self.model, self.device, img_region, (xmin, ymin, xmax, ymax))

            self.thread.result_signal.connect(self.update_overlay)

            self.thread.start()

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

            mask_display = (mask_pred * 255).astype(np.uint8)

            gt = None

            if self.image_path:

                base_dir = os.path.dirname(self.image_path)

                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

                parent_dir = os.path.dirname(base_dir)

                try_paths = [

                    os.path.join(parent_dir, "masks", base_name + ".jpg"),

                    os.path.join(parent_dir, "masks", base_name + ".png"),

                    os.path.join(base_dir, base_name + "_mask.jpg"),

                    os.path.join(base_dir, base_name + "_mask.png")

                ]

                for p in try_paths:

                    if os.path.exists(p):

                        try:

                            gt = io.imread(p)

                            print(f"Found GT at: {p}")

                            break

                        except:

                            continue

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)

            if gt is not None:

                if gt.ndim == 3: gt = gt[:, :, 0]

                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:

                    gt_region = gt[ymin:ymax, xmin:xmax]

                    gt_display_crop = (gt_region > 127).astype(np.uint8) * 255

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)

            plt.imshow(self.img_3c);

            plt.title("Full Image (with Result)");

            plt.axis('off')

            plt.subplot(1, 3, 2)

            plt.imshow(mask_display, cmap="gray");

            plt.title("Prediction Mask (ROI)");

            plt.axis('off')

            plt.subplot(1, 3, 3)

            plt.imshow(gt_display_crop, cmap="gray");

            plt.title("Ground Truth (ROI)");

            plt.axis('off')

            plt.tight_layout()

            plt.show()

        threading.Thread(target=_worker, daemon=True).start()

    def save_image(self):

        if self.image_path:

            out_path = f"{os.path.splitext(self.image_path)[0]}_result.png"

            io.imsave(out_path, self.img_3c)

            QMessageBox.information(self, "Success", f"Saved to: {out_path}")

# ===================== 主程序 =====================

if __name__ == "__main__":

    # ★★★ 4. 配置 HTC-Net 权重路径 ★★★

    weight_path = r"./weights/htc_net_best_model.pth"

    if not os.path.exists(weight_path):

        print(f"错误: 权重文件未找到于 {weight_path}")

        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    # ★★★ 5. 构建 HTC-Net 模型实例 ★★★

    # 为模型创建必要的 config 对象

    cfg = ml_collections.config_dict.ConfigDict()

    cfg.n_classes = 1

    cfg.decoder_channels = (128, 64, 32, 16)

    cfg.n_skip = 3

    # 实例化模型 (输入尺寸为 224, 类别为 1)

    model = HTCNetWrapper(config=cfg, img_size=224, num_classes=1)

    # ★★★ 6. 加载微调后的权重 ★★★

    try:

        model.load_state_dict(torch.load(weight_path, map_location=device))

        print("HTC-Net 微调权重加载成功!")

    except Exception as e:

        print(f"错误：加载权重失败: {e}")

        print("请检查权重文件是否与模型结构匹配。")

        sys.exit(1)

    app = QApplication(sys.argv)

    gui = InteractiveSegmentationGUI(model, device=device)

    gui.show()

    sys.exit(app.exec())
