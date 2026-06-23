import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/densenet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

# 2. DenseNet Segmentation 模型
class DenseNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 加载 torchvision DenseNet121
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # DenseNet features 输出: (B, 1024, H/32, W/32)
        self.backbone = backbone.features

        # 1×1 conv 变成 2 类
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1)

        # 上采样回输入大小
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        feat = self.backbone(x)            # (B, 1024, 8, 8)
        out = self.classifier(feat)        # (B, 2, 8, 8)
        out = self.upsample(out)           # (B, 2, 256, 256)
        return {"out": out}

def get_densenet_model():
    print("正在加载 DenseNet121 分割模型（ImageNet 预训练）...")
    return DenseNetSeg(num_classes=2)

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

    # 初始化 DenseNet 模型
    model = get_densenet_model().to(DEVICE)

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

            outputs = model(images)["out"]

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f}")

    # 保存权重
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "densenet_medical_model1.pth"))
    print("训练完成！模型已保存为 densenet_medical_model1.pth")
