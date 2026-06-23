import argparse
import importlib.util

def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=8):
    """Dynamically load an external data loader"""
    spec = importlib.util.spec_from_file_location(
        "universal_data_loader", data_loader_path
    )
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        num_workers=0,
    )
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/resnet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
# 1. Dataset (kept unchanged)
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        self.images = [
            x for x in sorted(os.listdir(self.images_dir))
            if x.endswith('.jpg') or x.endswith('.png')
        ]

        self.img_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.mask_trans = transforms.Compose([
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = Image.open(
            os.path.join(self.images_dir, img_name)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.masks_dir, img_name)
        ).convert("L")

        image = self.img_trans(image)
        mask = self.mask_trans(mask)

        mask = (mask > 0).long().squeeze(0)
        return image, mask

# 2. ResNet architecture (fully preserving your implementation)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
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
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# 3. ResNet segmentation wrapper
class ResNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet()
        self.classifier = nn.Conv2d(2048, num_classes, 1)
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        out = self.classifier(self.backbone(x))
        out = self.upsample(out)
        return {"out": out}

def get_resnet_model():
    print("Loading ResNet50 segmentation model...")
    return ResNetSeg(num_classes=2)

# 4. Dice
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)

# 5. Main training entry
if __name__ == '__main__':

    # ===== Argument parsing (added) =====
    parser = argparse.ArgumentParser(
        description='ResNet Medical Image Segmentation Training'
    )
    parser.add_argument('--data_loader', type=str, default=None, help='External data loader path')
    parser.add_argument('--image_folder', type=str, default=None, help='Image folder path')
    parser.add_argument('--mask_folder', type=str, default=None, help='Mask folder path')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    # ===== Data loading (auto switch) =====
    try:
        if args.data_loader and args.image_folder and args.mask_folder:
            print('[Train] Using external data loader')
            dataloader = load_external_dataloader(
                args.data_loader,
                args.image_folder,
                args.mask_folder,
                args.batch_size
            )
        else:
            print('[Train] Using built-in SimpleDataset')
            dataset = SimpleDataset(DATA_DIR)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2
            )
            print(f"Dataset loaded successfully, total {len(dataset)} images.")
    except Exception as e:
        print("Data loading failed:", e)
        exit()
    model = get_resnet_model().to(DEVICE)

    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        model.load_state_dict(torch.load(args.pretrain, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training, device: {DEVICE}, batch size: {args.batch_size}")

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        dice_sum = 0.0

        for images, masks in dataloader:
            if masks.dim() == 4:   # [B,1,H,W]
                masks = masks.squeeze(1)
            masks = masks.long()
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            dice_sum += calculate_dice(outputs, masks).item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"- Loss: {loss_sum/len(dataloader):.4f} "
            f"- Dice: {dice_sum/len(dataloader):.4f}"
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "resnet_medical_model.pth"))
    print("Training completed! Model saved as resnet_medical_model.pth")
