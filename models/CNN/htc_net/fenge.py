import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import ml_collections

# =========================
# 基本配置
# =========================
MODEL_NAME = "HTCNet"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

CHECKPOINT_PATH = r"./weights/HTC-Net-master-Kvasir-SEG-best.pth"

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 推理数据集（无 mask）
# =========================
class XirouInferenceDataset(Dataset):
    def __init__(self, images_dir, img_size=224):
        self.images = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"在 {images_dir} 中未找到图片")

        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img, os.path.basename(img_path)


# =========================
# 推理主流程
# =========================
def inference():
    print("[info] Device:", DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = XirouInferenceDataset(IMAGE_DIR, IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # =========================
    # 构建模型（与你训练完全一致）
    # =========================
    from network.Net import model as SwinModelWrapper

    cfg = ml_collections.config_dict.ConfigDict()
    cfg.n_classes = 1
    cfg.decoder_channels = (128, 64, 32, 16)
    cfg.n_skip = 3

    model = SwinModelWrapper(
        config=cfg,
        img_size=IMG_SIZE,
        num_classes=1
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"模型权重未找到: {CHECKPOINT_PATH}")

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    )
    model.eval()

    print("[info] Model loaded:", CHECKPOINT_PATH)

    # =========================
    # 推理并保存结果
    # =========================
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(loader, desc="Inferencing")):
            imgs = imgs.to(DEVICE)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
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
