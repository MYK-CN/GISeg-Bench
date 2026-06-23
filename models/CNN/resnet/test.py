import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/resnet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型定义（必须与你训练时完全一致）
import math
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
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
        out = self.relu(out)
        return out

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
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
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
        self.upsample = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        out = self.upsample(out)
        return {"out": out}
# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMG_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./weights/resnet_medical_model.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
# Dataset（stem 精确匹配）
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, TEST_IMG_DIR)
        self.mask_dir = os.path.join(root_dir, TEST_MASK_DIR)

        exts = (".png", ".jpg", ".jpeg")

        mask_index = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith(exts)
        }

        self.samples = []
        for f in os.listdir(self.img_dir):
            if f.lower().endswith(exts):
                stem = os.path.splitext(f)[0]
                if stem in mask_index:
                    self.samples.append((
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem])
                    ))

        print(f"[INFO] Test samples: {len(self.samples)}")

        self.img_tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = self.img_tf(Image.open(img_path).convert("RGB"))
        mask = self.mask_tf(Image.open(mask_path).convert("L"))
        mask = (mask > 0).long().squeeze(0)
        return img, mask
# Metrics
def dice_iou(pred, target, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    inter = (pred * target).sum()
    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    iou = (inter + eps) / (pred.sum() + target.sum() - inter + eps)
    return dice.item(), iou.item()
# Test
@torch.no_grad()
def main():
    dataset = KvasirTestDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetSeg(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = total_dice = total_iou = 0.0

    print("===== Start Testing =====")
    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        outputs = model(imgs)["out"]
        loss = criterion(outputs, masks)
        dice, iou = dice_iou(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
