import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.DAEFormer import DAEFormer


# =========================
# 基本配置（直接改这里即可）
# =========================
MODEL_NAME = "DAEFormer"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

MODEL_PATH = r"./weights/daeformer_kvasir_best.pth"

IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 1
NUM_WORKERS = 0


# =========================
# 推理数据集（无 mask）
# =========================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, img_size=224):
        self.img_dir = img_dir
        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, name


# =========================
# 推理主流程
# =========================
def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] device:", device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = XirouInferenceDataset(IMAGE_DIR, IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # -------- 加载模型 --------
    model = DAEFormer(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False  # ⭐ 必须显式指定
    )

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("[info] Model loaded:", MODEL_PATH)

    # -------- 推理并保存 --------
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(loader, desc="Inferencing")):
            imgs = imgs.to(device)

            logits = model(imgs)                  # [1, 2, H, W]
            preds = torch.argmax(logits, dim=1)   # [1, H, W]

            pred_np = preds[0].cpu().numpy().astype(np.uint8)
            pred_np = pred_np * 255               # 0 / 255

            pred_img = Image.fromarray(pred_np)

            save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
            save_path = os.path.join(SAVE_DIR, save_name)
            pred_img.save(save_path)

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


# =========================
# 直接运行
# =========================
if __name__ == "__main__":
    inference()
