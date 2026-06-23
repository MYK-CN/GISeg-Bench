import sys

import os

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/hiformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QPen, QColor

from PyQt5.QtWidgets import (

    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,

    QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QMessageBox

)

from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib import pyplot as plt

from skimage.transform import resize

from skimage import io

from scipy.ndimage import distance_transform_edt

import torch

import threading

# ====== 引入 HiFormer 模型和配置（按你训练时的路径来） ======

from models.HiFormer import HiFormer

import configs.HiFormer_configs as hcfg

# ===================== 小工具：numpy 转 QPixmap =====================

def np2pixmap(np_img: np.ndarray) -> QPixmap:

    """将 numpy 数组(H,W,3)转换为 QPixmap 用于显示"""

    if np_img.dtype != np.uint8:

        np_img = np_img.astype(np.uint8)

    h, w, c = np_img.shape

    bytes_per_line = 3 * w

    q_img = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(q_img)

# ===================== 指标计算函数 =====================

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

        # 计算边界

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

# ===================== 推理线程 =====================

class InferenceThread(QThread):

    # 发出：预测的二值掩膜 + 框框坐标

    result_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self, model, device, img_region, box,

                 img_size=224, num_classes=9):

        """

        img_region: 框选出来的彩色图像 (H,W,3)，uint8

        box: (xmin, ymin, xmax, ymax)

        """

        super().__init__()

        self.model = model

        self.device = device

        self.img_region = img_region

        self.box = box

        self.img_size = img_size

        self.num_classes = num_classes

    def run(self):

        h, w, _ = self.img_region.shape

        # 1. 缩放到与训练一致的分辨率 (img_size, img_size)

        img_small = resize(

            self.img_region,

            (self.img_size, self.img_size),

            preserve_range=True,

            anti_aliasing=True

        ).astype(np.float32)

        # 2. 简单标准化（与训练脚本类似：减均值 / 除标准差）

        img_mean = img_small.mean()

        img_std = img_small.std() + 1e-8

        img_small = (img_small - img_mean) / img_std

        # 3. HWC -> CHW -> Tensor

        img_small = np.transpose(img_small, (2, 0, 1))  # (C, H, W)

        input_tensor = torch.from_numpy(img_small).unsqueeze(0).to(

            self.device, dtype=torch.float32

        )

        with torch.no_grad():

            # HiFormer 直接输出 logits: (B, num_classes, H, W)

            output = self.model(input_tensor)  # (1, C, img_size, img_size)

            # softmax -> argmax 得到类别图

            probs = torch.softmax(output, dim=1)

            pred_label = torch.argmax(probs, dim=1)[0].cpu().numpy()  # (img_size, img_size)

            # ✅ 不分器官，只要 >0 就当前景

            mask_pred = (pred_label > 0).astype(np.uint8)

            # 4. 还原回框选区域大小

            mask_pred = resize(

                mask_pred,

                (h, w),

                order=0,

                preserve_range=True

            ).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== 主 GUI 类 =====================

class InteractiveSegmentationGUI(QWidget):

    def __init__(self, model, device="cpu", img_size=224, num_classes=9):

        super().__init__()

        self.setWindowTitle("HiFormer Segmentation - JPG/PNG")

        self.resize(1200, 900)

        self.model = model

        self.device = device

        self.img_size = img_size

        self.num_classes = num_classes

        self.model.to(device)

        self.model.eval()

        # 交互状态

        self.is_mouse_down = False

        self.start_pos = None

        self.rect = None

        self.bg_img = None

        # 图像相关

        self.image_path = None

        self.img_3c = None  # 当前显示的 2D RGB 图像 (H,W,3)

        self.img_3c_original = None  # 原始图像备份

        self.gt_mask = None  # GT mask

        # -------- 布局 --------

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

    # ================== 加载图像 ==================

    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(

            self, "Choose Image", ".", "Image Files (*.jpg *.png *.jpeg *.bmp)"

        )

        if not file_path:

            return

        self.image_path = file_path

        # 读取图像

        img = io.imread(file_path)

        # 处理灰度图

        if img.ndim == 2:

            img = np.repeat(img[:, :, None], 3, axis=-1)

        # 处理RGBA

        if img.shape[2] == 4:

            img = img[:, :, :3]

        self.img_3c = img.copy()

        self.img_3c_original = img.copy()

        # 尝试加载对应的GT mask

        self.gt_mask = self.find_gt_mask(file_path)

        if self.gt_mask is not None:

            print(f"Found GT mask, unique values: {np.unique(self.gt_mask)}")

        else:

            print("GT mask not found")

        # 显示到 QGraphicsScene

        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)

        self.bg_img = self.scene.addPixmap(pixmap)

        self.view.setScene(self.scene)

        # 绑定鼠标事件

        self.scene.mousePressEvent = self.mouse_press

        self.scene.mouseMoveEvent = self.mouse_move

        self.scene.mouseReleaseEvent = self.mouse_release

        print(f"Loaded image: {file_path}")

    # ================== 查找GT Mask ==================

    def find_gt_mask(self, image_path):

        """尝试在多个位置查找对应的GT mask"""

        base_dir = os.path.dirname(image_path)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        parent_dir = os.path.dirname(base_dir)

        # 可能的mask路径

        try_paths = [

            os.path.join(parent_dir, "masks", base_name + ".png"),

            os.path.join(parent_dir, "masks", base_name + ".jpg"),

            os.path.join(parent_dir, "mask", base_name + ".png"),

            os.path.join(parent_dir, "mask", base_name + ".jpg"),

            os.path.join(base_dir, "masks", base_name + ".png"),

            os.path.join(base_dir, "masks", base_name + ".jpg"),

            os.path.join(base_dir, base_name + "_mask.png"),

            os.path.join(base_dir, base_name + "_mask.jpg"),

            os.path.join(parent_dir, "labels", base_name + ".png"),

            os.path.join(parent_dir, "labels", base_name + ".jpg"),

        ]

        for p in try_paths:

            if os.path.exists(p):

                try:

                    gt = io.imread(p)

                    # 转为二值

                    if gt.ndim == 3:

                        gt = gt[:, :, 0]

                    gt = (gt > 0).astype(np.uint8)

                    return gt

                except:

                    pass

        return None

    # ================== 鼠标事件 ==================

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

        self.rect = self.scene.addRect(

            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"), 2)

        )

    def mouse_release(self, ev):

        self.is_mouse_down = False

        if self.img_3c is None:

            return

        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())

        sx, sy = self.start_pos

        h_img, w_img = self.img_3c.shape[:2]

        xmin, xmax = max(0, min(x, sx)), min(w_img, max(x, sx))

        ymin, ymax = max(0, min(y, sy)), min(h_img, max(y, sy))

        if xmax - xmin <= 5 or ymax - ymin <= 5:

            return

        print(f"Region: [{xmin}:{xmax}, {ymin}:{ymax}]")

        img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

        # 启动推理线程

        self.thread = InferenceThread(

            self.model,

            self.device,

            img_region,

            (xmin, ymin, xmax, ymax),

            img_size=self.img_size,

            num_classes=self.num_classes

        )

        self.thread.result_signal.connect(self.update_overlay)

        self.thread.start()

    # ================== 更新可视化 ==================

    def update_overlay(self, mask_pred: np.ndarray, box: tuple):

        xmin, ymin, xmax, ymax = box

        # 1. 用预测掩膜把原图 ROI 内对应位置涂白

        overlay_region = self.img_3c[ymin:ymax, xmin:xmax]

        overlay_region[mask_pred == 1] = [255, 255, 255]

        self.img_3c[ymin:ymax, xmin:xmax] = overlay_region

        # 2. 更新 GUI 显示

        self.bg_img.setPixmap(np2pixmap(self.img_3c))

        self.view.viewport().update()

        # 3. 弹出三图：原图+掩膜 / 预测掩膜 / GT掩膜

        self.show_plots_in_thread(mask_pred, box)

    def show_plots_in_thread(self, mask_pred: np.ndarray, box: tuple):

        def _worker():

            xmin, ymin, xmax, ymax = box

            h_box, w_box = ymax - ymin, xmax - xmin

            # (1) 预测掩膜: 黑底白色

            mask_display_pred = (mask_pred * 255).astype(np.uint8)

            # (2) 从GT mask取同一 ROI 作为金标准

            if self.gt_mask is None:

                gt_display = np.zeros_like(mask_display_pred, dtype=np.uint8)

                gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)

            else:

                # 确保GT mask和原图尺寸一致

                h_img, w_img = self.img_3c.shape[:2]

                if self.gt_mask.shape[0] != h_img or self.gt_mask.shape[1] != w_img:

                    gt_resized = resize(

                        self.gt_mask, (h_img, w_img),

                        order=0, preserve_range=True

                    ).astype(np.uint8)

                else:

                    gt_resized = self.gt_mask

                # 裁剪 ROI

                gt_region = gt_resized[ymin:ymax, xmin:xmax]

                gt_region_binary = (gt_region > 0).astype(np.uint8)

                gt_display = (gt_region_binary * 255).astype(np.uint8)

                print("GT ROI unique labels:", np.unique(gt_region))

            # ====== 计算指标 ======

            dice, iou, recall, precision, hd95 = calculate_metrics(

                mask_pred, gt_region_binary

            )

            # ====== 绘图：三张图 + 指标 ======

            plt.figure(figsize=(14, 7))

            # 图1: 完整原图（已叠加预测掩膜）

            plt.subplot(1, 3, 1)

            plt.imshow(self.img_3c)

            plt.title("Full Image + Predicted Mask")

            plt.axis("off")

            # 图2: 预测掩膜（黑底白色）

            plt.subplot(1, 3, 2)

            plt.imshow(mask_display_pred, cmap="gray", vmin=0, vmax=255)

            plt.title("Predicted Mask")

            plt.axis("off")

            # 图3: 金标准掩膜（黑底白色）

            plt.subplot(1, 3, 3)

            plt.imshow(gt_display, cmap="gray", vmin=0, vmax=255)

            plt.title("Ground Truth Mask")

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

    # ================== 保存结果图 ==================

    def save_image(self):

        if self.image_path and self.img_3c is not None:

            base = os.path.splitext(os.path.basename(self.image_path))[0]

            out_path = f"{base}_result.png"

            io.imsave(out_path, self.img_3c)

            QMessageBox.information(self, "Success", f"Saved to: {out_path}")

# ===================== 主入口 =====================

if __name__ == "__main__":

    # 1. 替换为你的 HiFormer 预训练权重路径

    weight_path = r"./weights/hiformer_final.pth"

    if not os.path.exists(weight_path):

        print(f"Error: Weight file not found at {weight_path}")

        sys.exit(1)

    # 2. 设备

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # 3. 构建 HiFormer 模型（与训练脚本保持一致）

    img_size = 224

    num_classes = 9

    config = hcfg.get_hiformer_b_configs()

    model = HiFormer(

        config=config,

        img_size=img_size,

        in_chans=3,

        n_classes=num_classes

    )

    # 4. 加载权重

    try:

        state = torch.load(weight_path, map_location=device)

        model.load_state_dict(state)

        print("HiFormer model loaded successfully!")

    except Exception as e:

        print(f"Error loading weights: {e}")

        sys.exit(1)

    # 5. 启动 Qt 应用

    app = QApplication(sys.argv)

    gui = InteractiveSegmentationGUI(

        model,

        device=device,

        img_size=img_size,

        num_classes=num_classes

    )

    gui.show()

    sys.exit(app.exec())
