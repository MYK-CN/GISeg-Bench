import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from models.HiFormer import HiFormer
import configs.HiFormer_configs as hcfg


# ======================
# 基本配置
# ======================
MODEL_NAME = "HiFormer"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

MODEL_PATH = r"./weights/HiFormer-main-Kvasir-SEG-best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0


# ======================
# 推理数据集（无 mask）
# ======================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, img_size=224):
        self.img_size = img_size
        self.image_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        img = np.asarray(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5   # 与 HiFormer 训练一致
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).float(), os.path.basename(img_path)


# ======================
# 推理主流程
# ======================
def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] Device:", device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = XirouInferenceDataset(IMAGE_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # -------- 构建模型 --------
    config = hcfg.get_hiformer_b_configs()
    model = HiFormer(
        config=config,
        img_size=IMAGE_SIZE,
        in_chans=3,
        n_classes=2
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    print("[info] Model loaded:", MODEL_PATH)

    # -------- 推理并保存 --------
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(loader, desc="Inferencing")):
            imgs = imgs.to(device)

            outputs = model(imgs)              # [1, 2, H, W]
            preds = torch.argmax(outputs, dim=1)  # [1, H, W]

            pred_np = preds[0].cpu().numpy().astype(np.uint8)
            pred_np = pred_np * 255            # 0 / 255

            pred_img = Image.fromarray(pred_np)

            save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
            save_path = os.path.join(SAVE_DIR, save_name)
            pred_img.save(save_path)

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


# ======================
# 直接运行
# ======================
if __name__ == "__main__":
    inference()
