import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import os

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/ce_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 当前文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
# networks 文件夹加入搜索路径
sys.path.append(current_dir)
# models 文件夹加入搜索路径
sys.path.append(os.path.join(current_dir, ".."))
# lib 文件夹加入搜索路径
sys.path.append(os.path.join(current_dir, "..", ".."))
# src 文件夹加入搜索路径
sys.path.append(os.path.join(current_dir, "..", "..", ".."))
from cenet import CE_Net_

# =================配置参数=================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
# 1. 自定义数据集
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

# 2. 导入 CE-Net 模型
from cenet import CE_Net_  # 根据你的文件路径修改 import

def get_cenet_model():
    print("正在加载 CE-Net 分割模型...")
    return CE_Net_(num_classes=2)  # 二分类任务

# 3. Dice 系数
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 4. 主训练
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

    # 初始化 CE-Net 模型
    model = get_cenet_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始训练，设备: {DEVICE}，批大小: {BATCH_SIZE}")

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)  # CE-Net 返回 sigmoid 输出
            # CE-Net 输出 (B, 2, H, W)，CrossEntropy 需要 raw logits
            # 如果 CE-Net 返回 sigmoid，需要用 BCEWithLogitsLoss 或转换
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f}")

    # 保存权重
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "cenet_medical_model.pth"))
    print("训练完成！模型已保存为 cenet_medical_model.pth")
