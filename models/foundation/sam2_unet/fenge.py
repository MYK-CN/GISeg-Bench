import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ======================================================
# 1. 路径与参数配置（写死）
# ======================================================
MODEL_NAME = "SAM2UNet"

# 已训练好的权重
FINETUNED_WEIGHT = r"./weights/sam2_unet_best.pth"
SAM2_BACKBONE    = r"./pretrained_ckpt/sam2.1_hiera_large.pt"

# 推理图片目录 & 保存目录
IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

IMG_SIZE = 352
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. 导入模型
# ======================================================
try:
    from SAM2UNet import SAM2UNet
except ImportError:
    from model.SAM2UNet import SAM2UNet

# ======================================================
# 3. 图像预处理
# ======================================================
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return img.unsqueeze(0)  # [1,3,H,W]

# ======================================================
# 4. 主推理流程
# ======================================================
def main():
    print(f"[INFO] Device: {DEVICE}")

    # ---------- 构建模型 ----------
    print("[INFO] Building model...")
    model = SAM2UNet(checkpoint_path=SAM2_BACKBONE)
    model.to(DEVICE)

    # ---------- 加载权重 ----------
    print("[INFO] Loading finetuned weights...")
    ckpt = torch.load(FINETUNED_WEIGHT, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]

    model_dict = model.state_dict()
    load_dict = {k: v for k, v in ckpt.items()
                 if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(load_dict, strict=False)
    print(f"[INFO] Loaded {len(load_dict)} layers")

    model.eval()

    # ---------- 获取图片列表 ----------
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    print(f"[INFO] Total images: {len(img_paths)}")

    # ---------- 推理 ----------
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
            img = preprocess(img_path).to(DEVICE)

            out, _, _ = model(img)

            if out.shape[2:] != (IMG_SIZE, IMG_SIZE):
                out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE),
                                    mode="bilinear", align_corners=False)

            prob = torch.sigmoid(out)[0, 0]          # [H,W]
            mask = (prob > THRESHOLD).cpu().numpy()  # bool
            mask = (mask * 255).astype(np.uint8)     # 0/255

            save_name = f"{idx+1}-{MODEL_NAME}-xirou.png"
            save_path = os.path.join(SAVE_DIR, save_name)

            Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] Done! Results saved to: {SAVE_DIR}")

# ======================================================
# 5. 入口
# ======================================================
if __name__ == "__main__":
    main()
