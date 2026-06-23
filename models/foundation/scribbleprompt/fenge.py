import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ====== 1. ScribblePrompt 模型导入 ======
from scribbleprompt.models.unet import ScribblePromptUNet, prepare_inputs

# ====== 2. 路径与参数（全部写死） ======
MODEL_NAME = "ScribblePromptUNet"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

WEIGHT_PATH = r"./pretrained_ckpt/ScribblePrompt_unet_finetuned_kvasir.pt"

IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 3. 图像预处理 ======
def preprocess(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img

# ====== 4. 主推理流程 ======
@torch.no_grad()
def main():
    print(f"[INFO] 使用设备: {DEVICE}")

    # -------- 构建模型 --------
    print("[INFO] 初始化 ScribblePromptUNet...")
    model = ScribblePromptUNet(version="v1", device=DEVICE)

    print("[INFO] 加载微调权重...")
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.model.load_state_dict(state_dict, strict=True)

    model.model.to(DEVICE)
    model.model.eval()

    # -------- 获取图片列表 --------
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        img_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
    img_paths.sort()

    if len(img_paths) == 0:
        raise RuntimeError("❌ 推理目录中未找到图片")

    print(f"[INFO] 待推理图片数: {len(img_paths)}")

    # -------- 推理并保存 --------
    for idx, img_path in enumerate(tqdm(img_paths, desc="Inferencing")):
        img = preprocess(img_path).to(DEVICE)

        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        x = prepare_inputs(prompts, device=DEVICE)
        logits = model.model(x)  # [1,1,H,W]

        prob = torch.sigmoid(logits)[0, 0]          # [H,W]
        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)        # 0/255

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")

# ====== 5. 程序入口 ======
if __name__ == "__main__":
    main()
