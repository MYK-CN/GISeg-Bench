import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math  # ResNet 初始化需要用到

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/resnet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================配置参数=================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
# 1. 自定义数据集 (保持不变)
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

# 2. ResNet 模型定义 (基于你提供的代码，修复了 forward 部分)
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # Standard ResNet50 structure based on provided __init__
    def __init__(self):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 修正：使用 __init__ 中定义的标准层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 256 channels
        x = self.layer2(x)  # 512 channels
        x = self.layer3(x)  # 1024 channels
        x = self.layer4(x)  # 2048 channels

        return x

# 3. ResNet 分割模型包装器
class ResNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 实例化 ResNet 骨干
        self.backbone = ResNet()

        # ResNet50 的 Layer4 输出通道数是 2048 (512 * Bottleneck expansion 4)
        out_channels = 2048

        # 简单的分割头：1x1 卷积将通道数降为类别数
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        # 上采样：ResNet 下采样了 32 倍 (stride 2 在 conv1, maxpool, layer2, layer3, layer4)
        # 所以这里需要上采样 32 倍恢复到原图大小
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)

        # 分类
        out = self.classifier(features)

        # 上采样
        out = self.upsample(out)

        return {"out": out}

def get_resnet_model():
    print("正在加载 ResNet50 分割模型...")
    return ResNetSeg(num_classes=2)

# 4. Dice 系数 (保持不变)
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 5. 主训练
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

    # 初始化 ResNet 模型
    model = get_resnet_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # 可以给 ResNet 加个预热或者调整 LR，这里保持原样
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

            # 前向传播
            outputs = model(images)["out"]

            # 计算 Loss
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f}")

    # 保存权重
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "resnet_medical_model.pth"))
    print("训练完成！模型已保存为 resnet_medical_model.pth")
