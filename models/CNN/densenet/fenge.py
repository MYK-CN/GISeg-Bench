import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models

# ======================================================
# 1. 基本配置
# ======================================================
MODEL_NAME = "DenseNet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/DenseNet-Kvasir-SEG-best.pth"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. 图像预处理（与训练 / 测试保持一致）
# ======================================================
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

# ======================================================
# 3. DenseNet 分割模型（与训练一致）
# ======================================================
class DenseNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        self.backbone = backbone.features
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=32,
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        out = self.upsample(out)
        return {"out": out}

# ======================================================
# 4. 构建模型并加载权重
# ======================================================
def build_model():
    print("[INFO] Building DenseNetSeg...")
    model = DenseNetSeg(num_classes=2)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()

# ======================================================
# 5. 主推理流程
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
# 6. 程序入口
# ======================================================
if __name__ == "__main__":
    main()
