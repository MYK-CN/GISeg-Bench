# -*- coding: utf-8 -*-
import sys
import os
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
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
from torchvision import models, transforms

import threading

# ====== 新增：导入 SAM 所需的模块 ======
from types import SimpleNamespace
import torch.nn.functional as F

# 请按你的实际路径修改这个仓库根目录
SAM_REPO_ROOT = r"."
if SAM_REPO_ROOT not in sys.path:
    sys.path.append(SAM_REPO_ROOT)

from models.sam.build_sam import build_sam_vit_b

# ====== 新增：基础 SAM 权重路径（和你训练脚本里的一致） ======
BASE_SAM_PATH = r"./pretrained_ckpt/sam_vit_b_01ec64.pth"


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


# ============ ★★★ 把原来的 DenseNetSeg 改成 Med-SAM 包装模型 ★★★
class DenseNetSeg(nn.Module):
    """
    这里虽然还叫 DenseNetSeg，
    但内部已经换成你训练的 Med-SAM 模型，
    并且 forward 返回 {'out': logits}，保持与原 GUI 代码完全兼容。
    """

    def __init__(self, num_classes=2, image_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # 构造与训练时一致的 args
        self.args = SimpleNamespace(
            image_size=image_size,
            mod='sam_adpt',
            sam_checkpoint=BASE_SAM_PATH,
            type='map',
            encoder_adapter=True,
            mid_dim=None,
            up_dim=None,
            multimask_output=1,
            vit_out_dim=256,
            thd=False,
            chunk=None,
            num_sample=1,
            evl_chunk=None
        )

        # 构建 Med-SAM 模型（ViT-B）
        self.sam = build_sam_vit_b(checkpoint=BASE_SAM_PATH, args=self.args)

    def forward(self, x):
        """
        x: (B,3,256,256) ，已经经过 ToTensor + Normalize(mean,std)

        输出：
            {'out': (B,2,256,256)}  2 通道，用于后面 softmax 做前景/背景分割
        """
        B, C, H, W = x.shape

        # 1. 反归一化，恢复到 [0,255]，与训练时保持一致
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        img = x * std + mean
        img = torch.clamp(img, 0.0, 1.0)
        img_255 = img * 255.0

        # 2. 构建 Med-SAM 所需的 batched_input
        batched_input = []
        for i in range(B):
            inp_dict = {
                "image": img_255[i],
                "original_size": (self.image_size, self.image_size)
            }
            batched_input.append(inp_dict)

        # 3. 复用训练脚本里的 custom_forward 逻辑（这里写在 forward 里）
        # 3.1 预处理
        input_images = torch.stack([self.sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.sam.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None,
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # low_res_masks: (1,1,h,w)
            outputs.append(low_res_masks)

        # 4. 拼成 batch，并上采样到 256x256
        preds = torch.cat(outputs, dim=0)  # (B,1,h,w)
        preds = F.interpolate(
            preds,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )  # (B,1,256,256)

        # 5. 为了兼容原来 GUI 中的 softmax 通道 2 类，这里人为构造 2 通道 logits:
        #    通道0：背景 = -preds
        #    通道1：前景 =  preds
        #    这样 softmax 后通道1 的概率单调等价于 sigmoid(preds)，阈值0.5位置一致
        bg = -preds
        fg = preds
        out = torch.cat([bg, fg], dim=1)  # (B,2,256,256)

        return {"out": out}


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
            # 3. 推理 (模型输出字典，键 'out')
            output = self.model(input_tensor)['out']

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
        self.setWindowTitle("Med-SAM Medical Segmentation - GUI")
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
            # ====== 关键：先解包坐标，防止后续报错 ======
            xmin, ymin, xmax, ymax = box
            h_box, w_box = ymax - ymin, xmax - xmin

            # 预测的 Mask (局部)
            mask_display = (mask_pred * 255).astype(np.uint8)

            # ====== 寻找金标准 (Ground Truth) ======
            gt = None
            try_paths = []

            if self.image_path:
                # 假设 image_path = D:/Dataset/images/pic1.jpg
                base_dir = os.path.dirname(self.image_path)  # D:/Dataset/images
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]  # pic1
                parent_dir = os.path.dirname(base_dir)  # D:/Dataset

                # 优先级 1: ../masks/pic1.jpg (标准结构)
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".jpg"))
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".png"))

                # 优先级 2: ./pic1_mask.jpg (同目录结构)
                try_paths.append(os.path.join(base_dir, base_name + "_mask.jpg"))
                try_paths.append(os.path.join(base_dir, base_name + "_mask.png"))

            # 遍历寻找
            for p in try_paths:
                if os.path.exists(p):
                    try:
                        gt = io.imread(p)
                        print(f"Found GT at: {p}")
                        break
                    except:
                        continue

            # 处理金标准显示 (裁剪到 Box 区域以便对比)
            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)
            gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)

            if gt is not None:
                if gt.ndim == 3:
                    gt = gt[:, :, 0]  # 转灰度

                # 检查尺寸是否匹配原图
                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:
                    # 裁剪出对应的局部 GT
                    gt_region = gt[ymin:ymax, xmin:xmax]
                    gt_region_binary = (gt_region > 0).astype(np.uint8)
                    gt_display_crop = gt_region_binary * 255
                else:
                    # 如果尺寸不对，尝试强行缩放
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

            # 图1: 显示修改后的完整原图
            plt.subplot(1, 3, 1)
            plt.imshow(self.img_3c)
            plt.title("Full Image (with Result)")
            plt.axis('off')

            # 图2: 预测的 Mask (局部)
            plt.subplot(1, 3, 2)
            plt.imshow(mask_display, cmap="gray")
            plt.title("Prediction Mask (ROI)")
            plt.axis('off')

            # 图3: 金标准 (局部)
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
    # ⚠ 这里换成你训练好模型保存的路径（latest_model.pth）
    weight_path = r"./weights/medical_sam_adapter_latest_model.pth"

    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        sys.exit(1)

    if not os.path.exists(BASE_SAM_PATH):
        print(f"Error: Base SAM weight file not found at {BASE_SAM_PATH}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 构建 Med-SAM 模型（包装成 DenseNetSeg 接口，方便复用 GUI）
    model = DenseNetSeg(num_classes=2, image_size=256)

    # 加载你训练好的 Med-SAM 权重（整个 SAM 模型的 state_dict）
    try:
        sam_state = torch.load(weight_path, map_location=device)
        # 只把权重加载到内部 self.sam 上
        model.sam.load_state_dict(sam_state, strict=True)
        print("Med-SAM model loaded successfully!")
    except Exception as e:
        print(f"Error loading Med-SAM weights: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    gui = InteractiveSegmentationGUI(model, device=device)
    gui.show()
    sys.exit(app.exec())
