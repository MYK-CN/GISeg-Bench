import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from cenet import CE_Net_  # 导入预训练的模型

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/ce_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = r"./data/Kvasir-SEG"  # 测试数据集路径
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
PRETRAINED_MODEL_PATH = r"./weights/cenet_medical_model.pth"  # 预训练权重路径

# 数据集类
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir, image_subdir="test", mask_subdir="masktest"):
        self.img_dir = os.path.join(root_dir, image_subdir)
        self.mask_dir = os.path.join(root_dir, mask_subdir)

        exts = (".png", ".jpg", ".jpeg")

        # 1️⃣ 建立 mask 索引（按 stem）
        mask_index = {}
        for f in os.listdir(self.mask_dir):
            if f.lower().endswith(exts):
                stem = os.path.splitext(f)[0]
                mask_index[stem] = f

        # 2️⃣ 只保留“有 mask 的 image”
        self.samples = []
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            if stem in mask_index:
                self.samples.append(
                    (
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem])
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError("❌ test / masktest 中没有匹配的 image-mask 对")

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

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
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_trans(image)
        mask = self.mask_trans(mask)

        mask = (mask > 0).long().squeeze(0)

        return image, mask

# 导入模型
def get_cenet_model():
    print("加载 CE-Net 模型...")
    return CE_Net_(num_classes=2)  # 二分类

# 计算Dice系数
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 计算IoU
def calculate_iou(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = (pred_flat + target_flat).sum() - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()

# 测试函数
def test():
    # 加载测试数据集
    test_dataset = KvasirTestDataset(DATA_DIR, image_subdir="test", mask_subdir="masktest")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 加载模型
    model = get_cenet_model().to(DEVICE)

    # 加载预训练权重
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"加载预训练模型权重: {PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"未找到预训练权重: {PRETRAINED_MODEL_PATH}")
        return

    model.eval()  # 切换到评估模式

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)  # CE-Net 模型输出 (B, 2, H, W)

            # 计算损失
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # 计算Dice和IoU指标
            total_dice += calculate_dice(outputs, masks)
            total_iou += calculate_iou(outputs, masks)

    avg_loss = total_loss / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)

    print(f"测试结果: Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

if __name__ == '__main__':
    test()
