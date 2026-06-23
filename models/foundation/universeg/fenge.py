import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from universeg import universeg

# ======================================================
# 1. 基本配置
# ======================================================
MODEL_NAME = "UniverSeg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# -------- Support（训练集）--------
SUPPORT_IMG_DIR  = r"./data/Kvasir-SEG/images"
SUPPORT_MASK_DIR = r"./data/Kvasir-SEG/masks"

# -------- Query（待推理）--------
QUERY_IMG_DIR = r"./data/Kvasir-SEG/xirou/images"

# -------- Output --------
SAVE_DIR = r"./results"

# -------- Weights --------
PRETRAIN_PATH = r"./pretrained_ckpt/universeg_v1_nf64_ss64_STA.pt"
FINETUNE_PATH = r"./weights/universeg_kvasir_best_epoch24.pth"

IMG_SIZE = 128
SUPPORT_SIZE = 16          # few-shot 数量
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# 2. Dataset：Support（image + mask）
# ======================================================
class KvasirSupportDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        mask_index = {m.stem: m for m in self.mask_dir.glob("*")}

        self.samples = []
        for img_path in self.img_dir.glob("*"):
            if img_path.stem in mask_index:
                self.samples.append((img_path, mask_index[img_path.stem]))

        if len(self.samples) == 0:
            raise RuntimeError("❌ Support 数据集为空")

        print(f"[SupportDataset] 样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(path, is_mask=False):
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE),
                         Image.NEAREST if is_mask else Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

        if is_mask:
            arr = (arr > 127).astype(np.float32)
        else:
            arr = arr / 255.0

        return torch.from_numpy(arr)[None, ...]  # [1,H,W]

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = self.load_image(img_path, is_mask=False)
        mask = self.load_image(mask_path, is_mask=True)
        return img, mask


# ======================================================
# 3. Dataset：Query（仅 image）
# ======================================================
class XirouQueryDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = sorted([
            p for p in Path(img_dir).glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        ])

        if len(self.img_paths) == 0:
            raise RuntimeError("❌ Query 目录中未找到图片")

        print(f"[QueryDataset] 推理图片数: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def load_image(path):
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def __getitem__(self, idx):
        return self.load_image(self.img_paths[idx])


# ======================================================
# 4. 构建模型（pretrain + finetune）
# ======================================================
def build_model():
    model = universeg(version="v1", pretrained=False)

    print("[INFO] Load pretrain:", PRETRAIN_PATH)
    pre = torch.load(PRETRAIN_PATH, map_location="cpu")
    if isinstance(pre, dict) and "state_dict" in pre:
        pre = pre["state_dict"]
    model.load_state_dict(pre, strict=False)

    print("[INFO] Load finetune:", FINETUNE_PATH)
    fin = torch.load(FINETUNE_PATH, map_location="cpu")
    model.load_state_dict(fin, strict=False)

    return model.to(DEVICE).eval()


# ======================================================
# 5. 构建 Support Set（真实 train mask）
# ======================================================
def build_support_set(dataset, support_size):
    support_size = min(support_size, len(dataset))

    imgs, masks = [], []
    for i in range(support_size):
        img, mask = dataset[i]
        imgs.append(img)
        masks.append(mask)

    imgs = torch.stack(imgs).unsqueeze(0).to(DEVICE)   # [1,S,1,H,W]
    masks = torch.stack(masks).unsqueeze(0).to(DEVICE)

    print(f"[Support] images: {imgs.shape}, masks: {masks.shape}")
    return imgs, masks


# ======================================================
# 6. 主推理流程
# ======================================================
@torch.no_grad()
def main():
    support_dataset = KvasirSupportDataset(
        SUPPORT_IMG_DIR,
        SUPPORT_MASK_DIR
    )
    query_dataset = XirouQueryDataset(QUERY_IMG_DIR)

    model = build_model()

    support_images, support_masks = build_support_set(
        support_dataset, SUPPORT_SIZE
    )

    print("===== Start Inferencing =====")
    for idx, img in enumerate(tqdm(query_dataset, desc="Inferencing")):
        img = img.unsqueeze(0).to(DEVICE)  # [1,1,H,W]

        logits = model(img, support_images, support_masks)
        prob = torch.sigmoid(logits)[0, 0]

        mask = (prob > THRESHOLD).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        Image.fromarray(mask).save(save_path)

    print(f"\n[INFO] 推理完成，结果已保存至：{SAVE_DIR}")


# ======================================================
# 7. 程序入口
# ======================================================
if __name__ == "__main__":
    main()
