import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
from torchvision import transforms

from cenet import CE_Net_   # CE-Net 模型

# ======================================================
# 1. 基本配置
# ======================================================
MODEL_NAME = "CENet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

PRETRAINED_MODEL_PATH = r"./weights/cenet_medical_model.pth"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. 图像预处理（与测试代码保持一致）
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
# 3. 构建模型
# ======================================================
def build_model():
    print("[INFO] 加载 CE-Net 模型...")
    model = CE_Net_(num_classes=2)
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
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

        outputs = model(img)              # [1,2,H,W]
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
