import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

# ======================================================
# 1. 模型导入
# ======================================================
from models.transnuseg import TransNuSeg

# ======================================================
# 2. 路径与参数（全部写死）
# ======================================================
MODEL_NAME = "TransNuSeg"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/transnuseg_kvasir_epoch_15.pth"

IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 3. 图像预处理
# ======================================================
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img)
    return img.unsqueeze(0)  # [1,3,H,W]

# ======================================================
# 4. 主推理流程
# ======================================================
@torch.no_grad()
def main():
    print(f"[INFO] 使用设备: {DEVICE}")

    # ---------- 构建模型 ----------
    print("[INFO] 构建 TransNuSeg...")
    model = TransNuSeg(
        img_size=IMG_SIZE,
        num_classes=2,
        depths=[2, 2, 2, 2],
        embed_dim=96
    )

    print("[INFO] 加载训练好的权重...")
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    # ---------- 获取图片列表 ----------
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if len(img_paths) == 0:
        raise RuntimeError("❌ 推理目录中未找到任何图片")

    print(f"[INFO] 待推理图片数: {len(img_paths)}")

    # ---------- 推理并保存 ----------
    for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
        img = preprocess(img_path).to(DEVICE)

        out_seg, _, _ = model(img)   # [1,2,H,W]

        if out_seg.shape[-1] != IMG_SIZE:
            out_seg = F.interpolate(
                out_seg,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=True
            )

        prob = torch.softmax(out_seg, dim=1)[0, 1]  # 前景概率 [H,W]
        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)        # 0 / 255

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")

# ======================================================
# 5. 程序入口
# ======================================================
if __name__ == "__main__":
    main()
