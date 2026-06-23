import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 1. 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

DATA_ROOT = r"./data/Kvasir-SEG"

TRAIN_IMG_DIR = "images"
TRAIN_MASK_DIR = "masks"
VAL_IMG_DIR   = "val"
VAL_MASK_DIR  = "maskval"
TEST_IMG_DIR  = "test"
TEST_MASK_DIR = "masktest"

SAVE_PATH = "outputs/cnn/unet"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
# 2. Dataset（严格 stem 匹配）
class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

        mask_index = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(mask_dir)
            if f.lower().endswith(exts)
        }

        self.samples = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(exts):
                stem = os.path.splitext(f)[0]
                if stem in mask_index:
                    self.samples.append((
                        os.path.join(img_dir, f),
                        os.path.join(mask_dir, mask_index[stem])
                    ))

        if len(self.samples) == 0:
            raise RuntimeError(f"❌ No data in {img_dir}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {img_dir}")

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

        mask = (mask > 0).float()   # [1,H,W]

        return img, mask
# 3. UNet 模型（医学分割标准）
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        return self.outc(x)  # logits
# 4. Dice
def dice_score(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + eps)
# 5. Train / Val / Test
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_dice = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_dice += dice_score(preds, masks).item()

    return total_dice / len(loader)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_dice = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_dice += dice_score(preds, masks).item()

    return total_dice / len(loader)
# 6. Main
def main():
    train_ds = KvasirDataset(
        os.path.join(DATA_ROOT, TRAIN_IMG_DIR),
        os.path.join(DATA_ROOT, TRAIN_MASK_DIR)
    )
    val_ds = KvasirDataset(
        os.path.join(DATA_ROOT, VAL_IMG_DIR),
        os.path.join(DATA_ROOT, VAL_MASK_DIR)
    )
    test_ds = KvasirDataset(
        os.path.join(DATA_ROOT, TEST_IMG_DIR),
        os.path.join(DATA_ROOT, TEST_MASK_DIR)
    )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0.0

    print("\n===== Start Training =====")
    for epoch in range(EPOCHS):
        tr_dice = train_epoch(model, train_loader, optimizer, criterion)
        va_dice = eval_epoch(model, val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Dice {tr_dice:.4f} | Val Dice {va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saved best model (Val Dice={best_dice:.4f})")

    print("\n===== Start Testing =====")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)

    test_dice = eval_epoch(model, test_loader)

    print("\n========== Final Test Result ==========")
    print(f"Test Dice : {test_dice:.4f}")
    print("======================================")

if __name__ == "__main__":
    main()
