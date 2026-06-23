# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/universeg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
from torch import nn

import threading

# ==== 引入 UniverSeg ====
from universeg.model import universeg

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

# ===================== Kvasir-SEG 数据集（做 support set） =====================
class KvasirSegDataset:
    """
    简化版 Dataset，只用来在 GUI 启动时准备 few-shot support images / masks
    """

    def __init__(self, root_dir: str, image_subdir: str = "images", mask_subdir: str = "masks"):
        self.root_dir = root_dir
        img_dir = os.path.join(root_dir, image_subdir)
        mask_dir = os.path.join(root_dir, mask_subdir)

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"找不到图像目录: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"找不到标注目录: {mask_dir}")

        # 建立 masks 索引
        mask_index = {}
        for name in os.listdir(mask_dir):
            m_path = os.path.join(mask_dir, name)
            if not os.path.isfile(m_path):
                continue
            stem, _ = os.path.splitext(name)
            mask_index[stem] = m_path

        self.samples = []
        for name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, name)
            if not os.path.isfile(img_path):
                continue
            stem, _ = os.path.splitext(name)

            # 如果你的 masks 命名是 xxx_mask.png，可以改这里：
            # mask_stem = stem + "_mask"
            # mask_path = mask_index.get(mask_stem)
            mask_path = mask_index.get(stem)

            if mask_path is None:
                continue
            self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("没有找到任何 image-masks 对，检查 Kvasir-SEG 命名是否一致。")

        print(f"[KvasirSegDataset] 找到 {len(self.samples)} 个 image-masks 对.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_image_as_tensor(path: str, is_mask: bool = False):
        """
        读一张图，转为 1x128x128 的 tensor，像素归一化到 [0, 1]。
        masks 二值化到 {0,1}
        """
        img = Image.open(path).convert("L")
        img = img.resize((128, 128), Image.BILINEAR if not is_mask else Image.NEAREST)

        arr = np.array(img).astype(np.float32)
        if is_mask:
            arr = (arr > 127).astype(np.float32)
        else:
            arr = arr / 255.0

        tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W]
        return tensor

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = self._load_image_as_tensor(img_path, is_mask=False)  # [1, H, W]
        mask = self._load_image_as_tensor(mask_path, is_mask=True)  # [1, H, W]
        return image, mask

def prepare_support_tensors(dataset: KvasirSegDataset, support_size: int, device):
    """
    从 Kvasir-SEG 中取若干张图片作为 few-shot support set
    返回:
      support_images: [1, S, 1, 128, 128]
      support_masks : [1, S, 1, 128, 128]
    """

    support_size = min(support_size, len(dataset))
    indices = list(range(support_size))

    imgs = []
    masks = []
    for idx in indices:
        img, msk = dataset[idx]  # [1,128,128]
        imgs.append(img)
        masks.append(msk)

    imgs = torch.stack(imgs, dim=0)  # [S,1,128,128]
    masks = torch.stack(masks, dim=0)  # [S,1,128,128]

    imgs = imgs.unsqueeze(0).to(device)  # [1,S,1,128,128]
    masks = masks.unsqueeze(0).to(device)  # [1,S,1,128,128]

    print(f"[Support] support_images shape: {imgs.shape}, support_masks shape: {masks.shape}")
    return imgs, masks

# ===================== 推理线程 =====================
class InferenceThread(QThread):
    result_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self, model, device, img_region, box, support_images, support_labels):
        super().__init__()
        self.model = model
        self.device = device
        self.img_region = img_region
        self.box = box
        self.support_images = support_images  # [1,S,1,128,128]
        self.support_labels = support_labels  # [1,S,1,128,128]

    def run(self):
        h, w, _ = self.img_region.shape

        # 1. 将选中的区域转换为灰度，并缩放到 128x128（和训练保持一致）
        img_pil = Image.fromarray(self.img_region).convert("L")
        img_pil = img_pil.resize((128, 128), Image.BILINEAR)
        arr = np.array(img_pil).astype(np.float32) / 255.0  # [H,W]
        input_tensor = torch.from_numpy(arr)[None, None, :, :]  # [1,1,128,128]
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # 2. UniverSeg 前向：输出 [1,1,128,128] 的 logits
            logits = self.model(input_tensor, self.support_images, self.support_labels)

            # 3. Sigmoid -> 概率 -> 阈值化
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [128,128]
            mask_pred = (probs > 0.5).astype(np.uint8)

            # 4. 还原回框选区域大小
            mask_pred = resize(mask_pred, (h, w), order=0, preserve_range=True).astype(np.uint8)

        self.result_signal.emit(mask_pred, self.box)

# ===================== GUI =====================
class InteractiveSegmentationGUI(QWidget):
    def __init__(self, model, support_images, support_labels, device='cpu'):
        super().__init__()
        self.setWindowTitle("UniverSeg Kvasir-SEG Interactive Segmentation")
        self.resize(1200, 900)

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.support_images = support_images
        self.support_labels = support_labels

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose Image", ".", "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
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

        if xmax - xmin > 5 and ymax - ymin > 5:
            print(f"Region: [{xmin}:{xmax}, {ymin}:{ymax}]")
            img_region = self.img_3c[ymin:ymax, xmin:xmax].copy()

            self.thread = InferenceThread(
                self.model,
                self.device,
                img_region,
                (xmin, ymin, xmax, ymax),
                self.support_images,
                self.support_labels
            )
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

            # 预测的 Mask (局部)
            mask_display = (mask_pred * 255).astype(np.uint8)

            # ====== 寻找金标准 (Ground Truth) ======
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
                    except:
                        continue

            gt_display_crop = np.zeros_like(mask_display, dtype=np.uint8)
            gt_region_binary = np.zeros_like(mask_pred, dtype=np.uint8)

            if gt is not None:
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
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

            # 图1: 完整原图（已叠加预测结果）
            plt.subplot(1, 3, 1)
            plt.imshow(self.img_3c)
            plt.title("Full Image (with Result)")
            plt.axis('off')

            # 图2: 预测的 ROI masks
            plt.subplot(1, 3, 2)
            plt.imshow(mask_display, cmap="gray")
            plt.title("Prediction Mask (ROI)")
            plt.axis('off')

            # 图3: 金标准 ROI
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
    # ----- 路径配置 -----
    data_root = r"./data/Kvasir-SEG"
    pretrain_path = r"./pretrained_ckpt/universeg_v1_nf64_ss64_STA.pt"
    finetune_path = r"./weights/universeg_kvasir_best_epoch24.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ----- 构建 Kvasir-SEG Dataset，并准备 support set -----
    dataset = KvasirSegDataset(data_root)
    support_images, support_labels = prepare_support_tensors(dataset, support_size=8, device=device)

    # ----- 构建 UniverSeg 模型，并加载预训练 + 微调权重 -----
    model = universeg(version="v1", pretrained=False)

    # 先尝试加载官方预训练
    if os.path.exists(pretrain_path):
        print(f"加载预训练权重: {pretrain_path}")
        state_pre = torch.load(pretrain_path, map_location=device)
        if isinstance(state_pre, dict) and "state_dict" in state_pre:
            state_pre = state_pre["state_dict"]
        model.load_state_dict(state_pre, strict=False)
    else:
        print(f"警告: 未找到预训练权重 {pretrain_path}，将只加载微调权重（如果存在）。")

    # 再加载你自己的微调权重（覆盖相应参数）
    if os.path.exists(finetune_path):
        print(f"加载微调权重: {finetune_path}")
        state_ft = torch.load(finetune_path, map_location=device)
        if isinstance(state_ft, dict) and "state_dict" in state_ft:
            state_ft = state_ft["state_dict"]
        model.load_state_dict(state_ft, strict=False)
    else:
        print(f"警告: 未找到微调权重 {finetune_path}，仅使用预训练模型。")

    print("UniverSeg model loaded successfully!")

    app = QApplication(sys.argv)
    gui = InteractiveSegmentationGUI(model, support_images, support_labels, device=device)
    gui.show()
    sys.exit(app.exec())
