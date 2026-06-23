# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import io
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
import threading
from matplotlib import pyplot as plt

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/sam2_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
# 1. 动态导入 SAM2UNet 模型
def import_sam2unet_class():
    # 你的 SAM2-UNet 项目根目录
    project_root = r"."

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from SAM2UNet import SAM2UNet
        return SAM2UNet
    except ImportError as e:
        print("【错误】无法导入 SAM2UNet，请检查路径配置。")
        print(f"尝试搜索的路径: {project_root}")
        raise e

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
    """将 numpy 数组转换为 QPixmap 用于显示"""

    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)

    h, w, c = np_img.shape
    bytesPerLine = 3 * w
    qImg = QImage(np_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

# ===================== 推理线程 (适配 SAM2UNet) =====================
class InferenceThread(QThread):
    result_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self, model, device, img_region, box):
        super().__init__()
        self.model = model
        self.device = device
        self.img_region = img_region
        self.box = box

        # 预处理：训练代码中使用的是 mean=[0.5]*3, std=[0.5]*3
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # 训练时统一 Resize 大小
        self.input_size = 352

    def run(self):
        h_orig, w_orig, _ = self.img_region.shape

        # 1. 缩放 (Resize to 352x352)
        img_small = resize(self.img_region, (self.input_size, self.input_size),
                           preserve_range=True, anti_aliasing=True).astype(np.uint8)

        # 2. 转 Tensor
        input_tensor = self.transform(img_small).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # 3. 推理
            # SAM2UNet forward 返回三个输出: out, out1, out2
            # 我们只需要主输出 out
            outs = self.model(input_tensor)

            if isinstance(outs, tuple):
                logits = outs[0]  # out
            else:
                logits = outs

            # 4. 二分类处理: Sigmoid -> 阈值 0.5
            prob = torch.sigmoid(logits)  # [1, 1, 352, 352]
            mask_pred = prob.squeeze().cpu().numpy()  # [352, 352]
            mask_pred = (mask_pred > 0.5).astype(np.uint8)

            # 5. 还原回框选区域大小 (Nearest Neighbor 插值保持二值特性)
            mask_pred = resize(mask_pred, (h_orig, w_orig), order=0, preserve_range=True).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== GUI =====================
class InteractiveSegmentationGUI(QWidget):
    def __init__(self, model, device='cpu'):
        super().__init__()
        self.setWindowTitle("SAM2-UNet Interactive Segmentation")
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
            xmin, ymin, xmax, ymax = box
            h_box, w_box = ymax - ymin, xmax - xmin

            mask_display = (mask_pred * 255).astype(np.uint8)

            # ====== 寻找金标准 (Ground Truth) ======
            gt = None
            try_paths = []

            if self.image_path:
                base_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                parent_dir = os.path.dirname(base_dir)

                # 常见结构适配
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".jpg"))
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".png"))
                try_paths.append(os.path.join(parent_dir, "maskval", base_name + ".jpg"))
                try_paths.append(os.path.join(parent_dir, "maskval", base_name + ".png"))
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
                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:
                    gt_region = gt[ymin:ymax, xmin:xmax]
                    gt_region_binary = (gt_region > 0).astype(np.uint8)
                    gt_display_crop = gt_region_binary * 255
                else:
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
    # 1. 微调后的权重文件路径
    finetuned_weight_path = r"./weights/sam2_unet_best.pth"

    # 2. SAM2 backbone 预训练权重 (初始化模型需要)
    sam2_backbone_path = r"./pretrained_ckpt/sam2.1_hiera_large.pt"

    if not os.path.exists(finetuned_weight_path):
        print(f"【严重错误】找不到微调权重文件: {finetuned_weight_path}")
        sys.exit(1)

    if not os.path.exists(sam2_backbone_path):
        print(f"【严重错误】找不到SAM2 Backbone权重文件: {sam2_backbone_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 3. 构建 SAM2UNet 模型
    try:
        SAM2UNetClass = import_sam2unet_class()
        # 初始化模型，传入 backbone 路径
        model = SAM2UNetClass(checkpoint_path=sam2_backbone_path)
    except Exception as e:
        print(f"【初始化错误】模型构建失败: {e}")
        sys.exit(1)

    # 4. 加载微调权重
    try:
        checkpoint = torch.load(finetuned_weight_path, map_location=device)

        # 兼容性处理
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print("SAM2-UNet weights loaded successfully!")
    except Exception as e:
        print(f"【加载权重错误】无法加载权重: {e}")
        sys.exit(1)

    # 5. 启动 GUI
    app = QApplication(sys.argv)
    gui = InteractiveSegmentationGUI(model, device=device)
    gui.show()
    sys.exit(app.exec())
