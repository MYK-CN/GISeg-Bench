import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ======================================================
# 1. 导入 Swin-Unet
# ======================================================
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# ======================================================
# 2. 路径与参数（全部写死）
# ======================================================
MODEL_NAME = "SwinUNet"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./weights/swin_unet_kvasir_finetune.pth"

IMG_SIZE = 224
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 3. 图像预处理
# ======================================================
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
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
    print("[INFO] 构建 Swin-Unet...")
    model = SwinTransformerSys(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        depths_decoder=[1, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        final_upsample="expand_first"
    )

    print("[INFO] 加载训练好的权重...")
    state_dict = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=False  # ⭐ 关键
    )

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    # ---------- 获取图片列表 ----------
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        img_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if len(img_paths) == 0:
        raise RuntimeError("❌ 推理目录中未找到任何图片")

    print(f"[INFO] 待推理图片数: {len(img_paths)}")

    # ---------- 推理并保存 ----------
    for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
        img = preprocess(img_path).to(DEVICE)

        logits = model(img)          # [1,1,H,W]
        prob = torch.sigmoid(logits)[0, 0]

        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)  # 0 / 255

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")

# ======================================================
# 5. 程序入口
# ======================================================
if __name__ == "__main__":
    main()
