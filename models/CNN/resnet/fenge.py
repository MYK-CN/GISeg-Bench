import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

# ======================
# 模型定义（与你训练时完全一致）
# ======================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet()
        self.classifier = nn.Conv2d(2048, num_classes, 1)
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        out = self.upsample(out)
        return {"out": out}


# ======================================================
# 1. 基本配置
# ======================================================
MODEL_NAME = "ResNetSeg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/resnet_medical_model.pth"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. 图像预处理（与你测试代码一致）
# ======================================================
img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img)
    return img.unsqueeze(0)  # [1,3,H,W]

# ======================================================
# 3. 构建模型并加载权重
# ======================================================
def build_model():
    print("[INFO] Building ResNetSeg...")
    model = ResNetSeg(num_classes=2)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()

# ======================================================
# 4. 主推理流程
# ======================================================
@torch.no_grad()
def main():
    model = build_model()

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        img_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if len(img_paths) == 0:
        raise RuntimeError("❌ 推理目录中未找到图片")

    print(f"[INFO] 待推理图片数: {len(img_paths)}")
    print("===== Start Inferencing =====")

    for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
        img = preprocess(img_path).to(DEVICE)

        outputs = model(img)["out"]          # [1,2,H,W]
        prob = torch.softmax(outputs, dim=1)[0, 1]  # 前景概率

        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")

# ======================================================
# 5. 程序入口
# ======================================================
if __name__ == "__main__":
    main()
