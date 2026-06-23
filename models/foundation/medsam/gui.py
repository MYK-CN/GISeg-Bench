# -*- coding: utf-8 -*-
import sys
import os
import time
import threading

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medsam（wu-test）')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize
import nibabel as nib  # 读取 NII / NII.gz

import torch
import torch.nn.functional as F

from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal

from segment_anything import sam_model_registry

# ================== 配置 & 设备 ==================
# 固定随机种子
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = r"./pretrained_ckpt/medsam_point_prompt_flare22.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

# ================== 工具函数 ==================
def np2pixmap(np_img):
    """将 numpy 数组转换为 QPixmap 用于显示"""
    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)

    h, w, c = np_img.shape
    bytesPerLine = 3 * w
    qImg = QImage(np_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    """
    MedSAM 推理函数：
      - img_embed: (1,256,64,64)
      - box_1024: (1,4) or (B,4) in 1024 坐标系
      - height, width: 原图大小
    返回：full_size_mask: (H, W) 的 0/1 masks
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)

    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

# ================== 推理线程 ==================
class InferenceThread(QThread):
    """
    单独的推理线程：
      - 用 MedSAM + 已算好的 embedding + 框，输出 ROI 区域的 masks
    """

    result_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self, model, device, img_region, box, img_embed, full_shape):
        super().__init__()
        self.model = model
        self.device = device
        self.img_region = img_region         # [h,w,3], 仅用于可视化
        self.box = box                       # (xmin, ymin, xmax, ymax) in 原图坐标
        self.img_embed = img_embed           # (1,256,64,64)
        self.full_shape = full_shape         # (H_img, W_img)

    def run(self):
        xmin, ymin, xmax, ymax = self.box
        H_img, W_img = self.full_shape

        # 将 bbox 映射到 1024 坐标系
        box_np = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
        box_1024 = box_np / np.array([W_img, H_img, W_img, H_img], dtype=np.float32) * MEDSAM_IMG_INPUT_SIZE

        with torch.no_grad():
            full_mask = medsam_inference(self.model, self.img_embed, box_1024, H_img, W_img)
            # 只取 ROI 部分
            mask_pred = full_mask[int(ymin):int(ymax), int(xmin):int(xmax)]

        self.result_signal.emit(mask_pred, self.box)

# ================== GUI ==================
class InteractiveSegmentationGUI(QWidget):
    def __init__(self, model, device='cpu'):
        super().__init__()
        self.setWindowTitle("MedSAM Interactive Segmentation (NII + Image)")
        self.resize(1200, 900)

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # 状态变量
        self.is_mouse_down = False
        self.start_pos = None
        self.rect = None
        self.bg_img = None
        self.image_path = None
        self.img_3c = None           # 当前显示用 RGB 图
        self.img_embed = None        # MedSAM embedding
        self.mask_c = None           # 叠加颜色用 masks

        # --- GUI 布局 ---
        self.view = QGraphicsView()
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        hbox = QHBoxLayout()
        load_button = QPushButton("Load Image / NII")
        save_button = QPushButton("Save Result")
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)
        vbox.addLayout(hbox)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_image)

    # -------- 加载图像 --------
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image / NII",
            ".",
            "Image Files (*.png *.jpg *.bmp *.jpeg *.nii *.nii.gz)"
        )
        if not file_path:
            return

        # 支持 NII / NII.gz
        if file_path.endswith(('.nii', '.nii.gz')):
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()

            # 取中间切片（你也可以改成别的策略）
            slice_idx = nii_data.shape[-1] // 2
            img_np = nii_data[..., slice_idx]

            # 归一化到 [0,255]
            img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8)
            img_np = (img_np * 255).astype(np.uint8)

            img_3c = np.repeat(img_np[..., None], 3, axis=-1)
        else:
            img_np = io.imread(file_path)
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            elif img_np.shape[2] == 4:
                img_3c = img_np[:, :, :3]
            else:
                img_3c = img_np

        self.img_3c = img_3c
        self.image_path = file_path

        # 计算 MedSAM embedding
        self.compute_embedding()

        pixmap = np2pixmap(self.img_3c)
        H, W, _ = self.img_3c.shape
        self.scene = QGraphicsScene(0, 0, W, H)
        self.bg_img = self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)

        # 叠加 masks 的彩色图（和原图同大小）
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype=np.uint8)

        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

        print(f"Loaded: {file_path}")

    # -------- 计算 MedSAM embedding --------
    @torch.no_grad()
    def compute_embedding(self):
        print("Calculating MedSAM embedding, GUI 可能短暂无响应...")
        img_1024 = resize(
            self.img_3c,
            (MEDSAM_IMG_INPUT_SIZE, MEDSAM_IMG_INPUT_SIZE),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # [0,1]

        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )  # [1,3,1024,1024]

        with torch.no_grad():
            self.img_embed = self.model.image_encoder(img_1024_tensor)
        print("Embedding done.")

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
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"), 2)
        )

    def mouse_release(self, ev):
        self.is_mouse_down = False
        if self.img_3c is None or self.img_embed is None:
            return

        x, y = int(ev.scenePos().x()), int(ev.scenePos().y())
        sx, sy = self.start_pos
        h_img, w_img = self.img_3c.shape[:2]

        xmin, xmax = max(0, min(x, sx)), min(w_img, max(x, sx))
        ymin, ymax = max(0, min(y, sy)), min(h_img, max(y, sy))

        if xmax - xmin > 5 and ymax - ymin > 5:
            print(f"Region: [{xmin}:{xmax}, {ymin}:{ymax}]")
            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

            # 开线程做推理
            self.thread = InferenceThread(
                self.model,
                self.device,
                img_region,
                (xmin, ymin, xmax, ymax),
                self.img_embed,
                (h_img, w_img),
            )
            self.thread.result_signal.connect(self.update_overlay)
            self.thread.start()

    # -------- 更新与显示 --------
    def update_overlay(self, mask_pred, box):
        xmin, ymin, xmax, ymax = box

        # 1. 在原图上把 ROI 内 masks=1 的地方染白
        overlay_region = self.img_3c[ymin:ymax, xmin:xmax]
        overlay_region[mask_pred == 1] = [255, 255, 255]
        self.img_3c[ymin:ymax, xmin:xmax] = overlay_region

        # 2. 更新 GUI
        self.bg_img.setPixmap(np2pixmap(self.img_3c))
        self.view.viewport().update()

        # 3. 弹窗显示结果（和金标准对比）
        self.show_plots_in_thread(mask_pred, box)

    def show_plots_in_thread(self, mask_pred, box):
        def _worker():
            xmin, ymin, xmax, ymax = box
            h_box, w_box = ymax - ymin, xmax - xmin

            # 预测的 Mask (ROI)
            mask_display = (mask_pred * 255).astype(np.uint8)

            # 寻找金标准 (GT)，和你之前那份 GUI 一样逻辑
            gt = None
            try_paths = []

            if self.image_path:
                base_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                parent_dir = os.path.dirname(base_dir)

                # ../masks/base_name.xxx
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".jpg"))
                try_paths.append(os.path.join(parent_dir, "masks", base_name + ".png"))

                # ./base_name_mask.xxx
                try_paths.append(os.path.join(base_dir, base_name + "_mask.jpg"))
                try_paths.append(os.path.join(base_dir, base_name + "_mask.png"))

            for p in try_paths:
                if os.path.exists(p):
                    try:
                        gt = io.imread(p)
                        print(f"Found GT at: {p}")
                        break
                    except Exception:
                        continue

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)

            if gt is not None:
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
                if gt.shape[0] >= ymax and gt.shape[1] >= xmax:
                    gt_region = gt[ymin:ymax, xmin:xmax]
                    gt_display_crop = (gt_region > 0).astype(np.uint8) * 255
                else:
                    try:
                        from skimage.transform import resize as sk_resize
                        gt_r = sk_resize(gt, (h_box, w_box), preserve_range=True, order=0)
                        gt_display_crop = (gt_r > 0).astype(np.uint8) * 255
                    except Exception:
                        pass

            # ====== 绘图 ======
            plt.figure(figsize=(12, 5))

            # 图1: 完整原图（叠加结果）
            plt.subplot(1, 3, 1)
            plt.imshow(self.img_3c)
            plt.title("Full Image (with Result)")
            plt.axis('off')

            # 图2: 预测 Mask (ROI)
            plt.subplot(1, 3, 2)
            plt.imshow(mask_display, cmap="gray")
            plt.title("Prediction Mask (ROI)")
            plt.axis('off')

            # 图3: GT (ROI)
            plt.subplot(1, 3, 3)
            plt.imshow(gt_display_crop, cmap="gray")
            plt.title("Ground Truth (ROI)")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        threading.Thread(target=_worker, daemon=True).start()

    def save_image(self):
        if self.image_path:
            out_path = f"{os.path.splitext(self.image_path)[0]}_result.png"
            io.imsave(out_path, self.img_3c)
            QMessageBox.information(self, "Success", f"Saved to: {out_path}")

# ================== 主程序 ==================
if __name__ == "__main__":
    print("Loading MedSAM model, please wait...")
    tic = time.perf_counter()

    medsam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=MedSAM_CKPT_PATH).to(device)
    medsam_model.eval()

    print(f"MedSAM loaded, took {time.perf_counter() - tic:.2f}s")

    app = QApplication(sys.argv)
    gui = InteractiveSegmentationGUI(medsam_model, device=device)
    gui.show()
    sys.exit(app.exec())
