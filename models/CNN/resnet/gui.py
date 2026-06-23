# -*- coding: utf-8 -*-

import sys

import os

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/resnet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import math

import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QPen, QColor

from PyQt5.QtWidgets import (

    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,

    QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QMessageBox

)

from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib import pyplot as plt

from skimage import io

from skimage.transform import resize as skimage_resize

from scipy.ndimage import distance_transform_edt

from PIL import Image

import torch

import torch.nn as nn

from torchvision import transforms

import threading

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

# ============ 1. ResNet 模型定义 (保持不变) ============

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,

                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.downsample is not None:

            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self):

        self.inplanes = 64

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)

        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)

        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)

        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x

class ResNetSeg(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.backbone = ResNet()

        out_channels = 2048

        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):

        features = self.backbone(x)

        out = self.classifier(features)

        out = self.upsample(out)

        return {"out": out}

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

            transforms.Resize((256, 256)),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406],

                                 [0.229, 0.224, 0.225])

        ])

    def run(self):

        h_orig, w_orig = self.img_region.shape[:2]

        pil_img = Image.fromarray(self.img_region)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(input_tensor)['out']

            pred_mask = torch.argmax(output, dim=1).squeeze(0)

            pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

            mask_final = skimage_resize(pred_mask_np, (h_orig, w_orig), order=0, preserve_range=True,

                                        anti_aliasing=False).astype(np.uint8)

        self.result_signal.emit(mask_final, self.box)

# ===================== GUI =====================

class InteractiveSegmentationGUI(QWidget):

    def __init__(self, model, device='cpu'):

        super().__init__()

        self.setWindowTitle("Medical Segmentation System (ResNet50)")

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

        if not file_path:

            return

        img_np = io.imread(file_path)

        if len(img_np.shape) == 2:

            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)

        elif img_np.shape[2] == 4:

            img_3c = img_np[:, :, :3]

        else:

            img_3c = img_np

        if img_3c.dtype != np.uint8:

            img_3c = (img_3c * 255).astype(np.uint8) if img_3c.max() <= 1.0 else img_3c.astype(np.uint8)

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

        print(f"Loaded: {file_path}, Shape: {self.img_3c.shape}")

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

            print(f"Region Selected: [{xmin}:{xmax}, {ymin}:{ymax}]")

            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

            self.thread = InferenceThread(self.model, self.device, img_region, (xmin, ymin, xmax, ymax))

            self.thread.result_signal.connect(self.update_overlay)

            self.thread.start()

    def update_overlay(self, mask_pred, box):

        xmin, ymin, xmax, ymax = box

        if np.sum(mask_pred) == 0:

            print("Warning: Model predicted background only (all zeros).")

        else:

            print(f"Prediction successful. Mask area: {np.sum(mask_pred)} pixels.")

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

                        if len(gt.shape) == 3:

                            gt = gt[:, :, 0]

                        print(f"Found GT at: {p}")

                        break

                    except:

                        continue

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)

            gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)

            if gt is not None:

                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:

                    gt_crop = gt[ymin:ymax, xmin:xmax]

                    gt_region_binary = (gt_crop > 0).astype(np.uint8)

                    gt_display_crop = gt_region_binary * 255

                else:

                    try:

                        gt_resized = skimage_resize(gt, (h_box, w_box), preserve_range=True, order=0)

                        gt_region_binary = (gt_resized > 0).astype(np.uint8)

                        gt_display_crop = gt_region_binary * 255

                    except:

                        pass

            # ====== 计算指标 ======

            dice, iou, recall, precision, hd95 = calculate_metrics(mask_pred, gt_region_binary)

            # ====== 绘图 ======

            plt.figure(figsize=(14, 7))

            plt.subplot(1, 3, 1)

            plt.imshow(self.img_3c)

            plt.title("Full Image (Prediction Overlay)")

            plt.axis('off')

            plt.subplot(1, 3, 2)

            plt.imshow(mask_display, cmap="gray", vmin=0, vmax=255)

            plt.title("Prediction (ROI)")

            plt.axis('off')

            plt.subplot(1, 3, 3)

            plt.imshow(gt_display_crop, cmap="gray", vmin=0, vmax=255)

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

    weight_path = r"./weights/resnet_medical_model.pth"

    if not os.path.exists(weight_path):

        print(f"Error: Weight file not found at {weight_path}")

        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    model = ResNetSeg(num_classes=2)

    try:

        model.load_state_dict(torch.load(weight_path, map_location=device))

        print("Model loaded successfully!")

    except Exception as e:

        print(f"Error loading weights: {e}")

        sys.exit(1)

    app = QApplication(sys.argv)

    gui = InteractiveSegmentationGUI(model, device=device)

    gui.show()

    sys.exit(app.exec())
