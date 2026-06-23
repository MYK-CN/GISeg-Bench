import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================
# 基本配置
# =========================
MODEL_NAME = "MTUNet"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

MODEL_PATH = r"./weights/MT-UNet-main-Kvasir-SEG-best.pth"  # ←你的新权重

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 模型导入
# =========================
try:
    from model.MTUNet import MTUNet
except ImportError:
    from MTUNet import MTUNet


# =========================
# 推理数据集
# =========================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, img_size=224):
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, os.path.basename(img_path)


# =========================
# 推理主流程
# =========================
def inference():
    print("[info] Device:", DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    loader = DataLoader(
        XirouInferenceDataset(IMAGE_DIR, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ★ 关键：out_ch = 1，与权重一致
    model = MTUNet(out_ch=1).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("[info] Model loaded:", MODEL_PATH)

    with torch.no_grad():
        for idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)

            logits = model(imgs)                 # [B,1,H,W]
            preds = (torch.sigmoid(logits) > 0.5).float()

            pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(pred_np).save(
                os.path.join(SAVE_DIR, f"{idx+1}-{MODEL_NAME}-xirou.png")
            )

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    inference()
