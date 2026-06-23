import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/fcn')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================配置参数=================
DATA_DIR = './Kvasir-SEG'  # 数据集路径，请修改为你实际的路径
BATCH_SIZE = 8  # 如果显存不够(报OOM错误)，请改小，比如 4 或 2
LR = 1e-4  # 学习率
EPOCHS = 20  # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 定义数据集加载器
class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        # 读取文件名列表
        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)  # 假设mask文件名和原图一致

        # 打开图片
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Mask转为灰度图

        if self.transform:
            # 应用变换 (Resize + ToTensor)
            image = self.transform(image)
            mask = self.transform(mask)

        # 处理 Mask: 像素值 > 0 的设为 1 (目标), 否则 0 (背景)
        mask = (mask > 0).float()

        # FCN 需要的 Mask 形状通常是 (H, W)，而不是 (1, H, W) 用于 CrossEntropyLoss
        # 但这里我们为了方便计算 Dice，先保持 (1, H, W)
        return image, mask

# 2. 定义数据变换
# FCN 模型通常需要 ImageNet 的标准化参数
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一大小，必须做
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  # Mask 必须用最近邻插值，防止产生非0/1的小数
    transforms.ToTensor()
])

# 自定义 Dataset 类需要分别传入不同的 transform
# 这里为了代码简洁，我们稍微修改一下 Dataset 类里的逻辑，或者像下面这样简单处理
# 为了演示简单，我们在 getitem 里手动做 transform，不传入参数了

class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = [x for x in sorted(os.listdir(self.images_dir)) if x.endswith('.jpg') or x.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # 读取
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        # 变换
        # Image
        img_t = transforms.Resize((256, 256))(image)
        img_t = transforms.ToTensor()(img_t)
        img_t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_t)

        # Mask
        mask_t = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_t = transforms.ToTensor()(mask_t)
        # 转换为 LongTensor (H, W) 用于 CrossEntropyLoss，且值为 0 或 1
        mask_t = (mask_t > 0).long().squeeze(0)

        return img_t, mask_t

# 3. 准备模型
def get_fcn_model():
    print("正在加载预训练 FCN ResNet50 模型...")
    # weights='DEFAULT' 自动下载 COCO 预训练权重
    model = models.segmentation.fcn_resnet50(weights='DEFAULT')

    # 修改分类头：输入通道不变，输出改为 2 类 (背景 vs 息肉)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

    # 为了省显存，我们可以禁用辅助分类头 (Auxiliary Classifier)
    model.aux_classifier = None

    return model

# 4. Dice 系数计算函数 (评估指标)
def calculate_dice(pred, target):
    # pred: (B, 2, H, W) -> argmax -> (B, H, W)
    pred_mask = torch.argmax(pred, dim=1)

    # 展平
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 5. 主训练流程
if __name__ == '__main__':
    # 加载数据
    try:
        dataset = SimpleDataset(R"./data/Kvasir-SEG")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        print(f"成功加载数据集，共 {len(dataset)} 张图片。")
    except Exception as e:
        print(f"数据加载失败，请检查路径: {e}")
        exit()

    # 初始化模型、损失函数、优化器
    model = get_fcn_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 适用于多分类（这里是2类）
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始训练，设备: {DEVICE}，批次大小: {BATCH_SIZE}")

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            # 注意：Torchvision 的分割模型返回的是一个字典 {'out': tensor, 'aux': tensor}
            outputs = model(images)['out']

            # 计算损失
            loss = criterion(outputs, masks)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录数据
            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice Score: {epoch_dice:.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fcn_medical_model.pth"))
    print("训练完成！模型已保存为 fcn_medical_model.pth")
