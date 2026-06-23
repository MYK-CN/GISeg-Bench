import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ======================
# 模型导入路径
# ======================
sys.path.append(r"./models")
from models.H2Former import Res34_Swin_MS, BasicBlock


# ======================
# 基本配置
# ======================
MODEL_NAME = "H2Former"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

MODEL_PATH = r"./weights/h2former_kvasir_epoch_4.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0


# ======================
# 推理数据集（无 mask）
# ======================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, image_size=224):
        self.img_dir = img_dir
        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, name


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
    model = Res34_Swin_MS(
        image_size=IMAGE_SIZE,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=1
    )

    # ======= 关键修复点：安全加载 checkpoint =======
    ckpt = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False   # ★ PyTorch 2.6+ 必须显式关闭
    )

    # 兼容不同保存方式
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    print("[info] Model loaded:", MODEL_PATH)

    # -------- 推理并保存 --------
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(loader, desc="Inferencing")):
            imgs = imgs.to(device)

            # ===== 3 → 4 通道（与训练保持一致）=====
            extra_channel = imgs.mean(dim=1, keepdim=True)
            imgs = torch.cat([imgs, extra_channel], dim=1)

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


# ======================
# 直接运行
# ======================
if __name__ == "__main__":
    inference()
