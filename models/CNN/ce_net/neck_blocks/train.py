# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/ce_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 添加项目根目录到 sys.path =====================
# 假设 train1.py 位于 src/lib/models/
# 根目录 = src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ===================== 导入 CE-Net 模型 =====================
from lib.models.model import create_model, load_model, save_model

# ===================== 配置参数 =====================
DATA_DIR = os.path.join(root_dir, "dataset")  # 你的 dataset 文件夹
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "CE_Net_weights.pth")

# ===================== 自定义数据集 =====================
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = [x for x in sorted(os.listdir(self.images_dir))
                       if x.endswith('.jpg') or x.endswith('.png')]

        self.img_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        image = self.img_trans(image)
        mask = self.mask_trans(mask)
        mask = (mask > 0).long().squeeze(0)
        return image, mask

# ===================== Dice 系数 =====================
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# ===================== 主训练 =====================
if __name__ == '__main__':
    # 加载数据
    dataset = SimpleDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"加载数据集成功，共 {len(dataset)} 张图片。")

    # 创建 CE-Net 模型
    model = create_model('cenet').to(DEVICE)
    print(f"模型加载完成，设备: {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(dataloader):.4f} - Dice: {running_dice/len(dataloader):.4f}")

    # 保存模型权重
    save_model(MODEL_SAVE_PATH, EPOCHS, model, optimizer)
    print(f"训练完成，模型已保存到 {MODEL_SAVE_PATH}")

