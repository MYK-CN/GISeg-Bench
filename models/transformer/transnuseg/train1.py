import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transnuseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 导入你的模型文件 (请确保将原文件重命名为 transnuseg.py)
from models.transnuseg import TransNuSeg

# ================= 配置参数 =================
# ================= 配置参数 (修改后) =================
CONFIG = {
    "train_img_path": r"./data/fluorescence/data",
    "train_mask_path": r"./data/fluorescence/label",

    # 核心修改：将 224 改为 256，以适配 window_size=8
    "img_size": 256,

    # 注意：尺寸变大后显存占用会增加，如果报错 OOM (Out of Memory)，请将 batch_size 改小 (如改为 2 或 1)
    "batch_size": 4,

    "lr": 1e-4,
    "epochs": 15,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ================= 数据集加载类 =================
class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # 获取所有图片路径
        self.img_list = glob.glob(os.path.join(img_dir, "*.*"))
        self.img_list = [x for x in self.img_list if x.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(self.img_list)} images.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_list)

    def generate_edge(self, mask):
        """通过形态学操作从Mask生成边缘"""
        kernel = np.ones((3, 3), np.uint8)
        mask_erosion = cv2.erode(mask, kernel, iterations=1)
        edge = mask - mask_erosion
        return edge

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        file_name = os.path.basename(img_path)

        # 假设Mask文件名与图片一致，或者是png格式
        # Kvasir-SEG通常文件名一致
        mask_path = os.path.join(self.mask_dir, file_name)
        if not os.path.exists(mask_path):
            # 尝试替换扩展名，有些数据集mask是png
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, file_name_png)

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        # 读取Mask (转为灰度)
        mask = Image.open(mask_path).convert('L')

        # 预处理 Mask 用于生成 Edge
        mask_np = np.array(mask.resize((self.img_size, self.img_size), Image.NEAREST))
        # 二值化
        _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # 生成边缘 (Ground Truth for Edge Head)
        edge_np = self.generate_edge(mask_np)

        # 转换 Image
        image = self.transform(image)

        # 转换 Mask 和 Edge 为 Tensor
        # Mask 归一化到 [0, 1], LongTensor用于CrossEntropy
        mask_tensor = torch.from_numpy(mask_np // 255).long()
        edge_tensor = torch.from_numpy(edge_np // 255).long()

        return image, mask_tensor, edge_tensor

# ================= 简单的损失函数 =================
def structure_loss(pred, mask):
    """
    结合 CrossEntropy 和 Dice Loss
    """

    ce_loss = F.cross_entropy(pred, mask)

    pred = torch.softmax(pred, dim=1)
    # 取前景类 (假设 class 1 是前景)
    pred = pred[:, 1, :, :]
    mask = mask.float()

    intersection = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2. * intersection + 1e-5) / (union + 1e-5)

    return ce_loss + dice_loss.mean()

# ================= 训练主流程 =================
def train():
    print(f"Using device: {CONFIG['device']}")

    # 1. 准备数据
    dataset = KvasirDataset(CONFIG['train_img_path'], CONFIG['train_mask_path'], CONFIG['img_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    # 2. 初始化模型
    # 注意：Kvasir是2分类(背景/息肉)，img_size需与Dataset一致
    model = TransNuSeg(
        img_size=CONFIG['img_size'],
        num_classes=CONFIG['num_classes'],
        depths=[2, 2, 2, 2],  # 默认参数
        embed_dim=96  # 默认参数
    )
    model.to(CONFIG['device'])

    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)

    # 4. 训练循环
    print("Start Training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0

        for i, (images, masks, edges) in enumerate(dataloader):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])
            edges = edges.to(CONFIG['device'])

            optimizer.zero_grad()

            # 前向传播
            # 模型输出: seg_mask, edge_mask, cluster_edge
            out_seg, out_edge, out_cluster = model(images)

            # 计算尺寸匹配 (如果模型输出尺寸和GT不一致，进行插值)
            if out_seg.shape[-1] != CONFIG['img_size']:
                out_seg = F.interpolate(out_seg, size=(CONFIG['img_size'], CONFIG['img_size']), mode='bilinear',
                                        align_corners=True)
                out_edge = F.interpolate(out_edge, size=(CONFIG['img_size'], CONFIG['img_size']), mode='bilinear',
                                         align_corners=True)
                out_cluster = F.interpolate(out_cluster, size=(CONFIG['img_size'], CONFIG['img_size']), mode='bilinear',
                                            align_corners=True)

            # 计算损失
            # 语义分割损失
            loss_s = structure_loss(out_seg, masks)
            # 边缘损失 (使用生成的边缘GT)
            loss_e = structure_loss(out_edge, edges)
            # 聚类边缘损失 (暂时也使用边缘GT监督)
            loss_c = structure_loss(out_cluster, edges)

            # 总损失 (权重可以根据论文调整，这里简单设为 1:1:1)
            loss = loss_s + loss_e + loss_c

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{CONFIG['epochs']}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f} (Seg:{loss_s.item():.3f}, Edge:{loss_e.item():.3f})")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] Finished. Avg Loss: {epoch_loss:.4f}")

        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"transnuseg_kvasir_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # Windows下使用多进程DataLoader需要保护主入口
    try:
        train()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
