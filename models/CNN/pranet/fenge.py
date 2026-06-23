import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# ========== PraNet ==========
from PraNet_ResNet import CRANet   # 必须与你训练时一致

# ======================================================
# 1. 基本配置
# ======================================================
MODEL_NAME = "PraNet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/pranet_model.pth"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. 图像预处理（与你测试代码一致）
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
# 3. 构建模型并加载权重
# ======================================================
def build_model():
    print("[INFO] Building PraNet (CRANet)...")
    model = CRANet().to(DEVICE)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.eval()

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

        outputs = model(img)

        # PraNet 返回多个输出 (r1, r2, r3, r4)，取主输出
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]
        else:
            pred = outputs

        # 如果是 1 通道，升为 2 通道（与你测试代码一致）
        if pred.shape[1] == 1:
            pred = torch.cat([1 - pred, pred], dim=1)

        # 前景概率
        prob = torch.softmax(pred, dim=1)[0, 1]

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
