import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 导入 PraNet 模型
from PraNet_ResNet import CRANet  # ←←← 你必须确保 PraNet_ResNet.py 文件存在并可 import

# =================配置参数=================
DATA_DIR = r"./data/Kvasir-SEG"  # 数据集路径
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 定义数据集加载器（与你原脚本一致）
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = [
            x for x in sorted(os.listdir(self.images_dir))
            if x.endswith('.jpg') or x.endswith('.png')
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        # Image transform
        img_t = transforms.Resize((256, 256))(image)
        img_t = transforms.ToTensor()(img_t)
        img_t = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )(img_t)

        # Mask transform
        mask_t = transforms.Resize((256, 256),
                                   interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_t = transforms.ToTensor()(mask_t)
        mask_t = (mask_t > 0).long().squeeze(0)

        return img_t, mask_t

# 2. 加载 PraNet 模型（替换你原来的 get_fcn_model）
def get_pranet_model():
    print("正在加载 PraNet-ResNet 模型...")
    model = CRANet()  # 默认使用 Res2Net 作为 backbone
    return model

# 3. Dice 计算函数（保持不变）
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)

    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 4. 主训练流程
if __name__ == '__main__':

    # 加载数据
    try:
        dataset = SimpleDataset(DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
        print(f"成功加载数据集，共 {len(dataset)} 张图片。")
    except Exception as e:
        print("数据加载失败：", e)
        exit()

    # 加载 PraNet
    model = get_pranet_model().to(DEVICE)

    # PraNet 输出是二分类（1 通道），但我们用 CrossEntropyLoss，因此改成 2 通道
    model.seg_head = nn.Conv2d(1, 2, 1) if hasattr(model, 'seg_head') else None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始训练，设备: {DEVICE}，批大小: {BATCH_SIZE}")

    # =================== 训练 ===================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # PraNet 输出多个 map (r1, r2, r3, r4)
            outputs = model(images)
            if isinstance(outputs, tuple):
                pred = outputs[0]  # 取主输出
            else:
                pred = outputs

            # 如果只有 1 通道，则升到 2 通道用于 CE Loss
            if pred.shape[1] == 1:
                pred = torch.cat([1 - pred, pred], dim=1)

            loss = criterion(pred, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(pred, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f}")

    # 保存结果
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pranet_model.pth"))
    print("训练完成！模型已保存为 pranet_model.pth")
