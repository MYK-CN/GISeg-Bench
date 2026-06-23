import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

# ======================
# 1. 模型定义（与你训练时完全一致）
# ======================
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
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
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

        self.outc = nn.Conv2d(64, out_ch, 1)

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


# ======================
# 2. 配置（全部写死）
# ======================
MODEL_NAME = "UNet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/unet_kvasir_best.pth"  # 你的训练输出权重

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 3. 图像预处理（与你训练一致）
# ======================
img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img)
    return img.unsqueeze(0)  # [1,3,H,W]


# ======================
# 4. 主推理流程
# ======================
@torch.no_grad()
def main():
    # -------- Model --------
    model = UNet(in_ch=3, out_ch=1)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    # -------- Images --------
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        img_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if len(img_paths) == 0:
        raise RuntimeError("❌ 推理目录中未找到图片")

    print(f"[INFO] 待推理图片数: {len(img_paths)}")
    print("===== Start Inferencing =====")

    for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
        img = preprocess(img_path).to(DEVICE)

        logits = model(img)                 # [1,1,H,W]
        prob = torch.sigmoid(logits)[0, 0]  # 前景概率

        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")


# ======================
# 5. 程序入口
# ======================
if __name__ == "__main__":
    main()
