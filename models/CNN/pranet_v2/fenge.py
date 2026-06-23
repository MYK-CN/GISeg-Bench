import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================
# 基本配置（直接改这里）
# =========================
MODEL_NAME = "PraNetV2"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

CKPT_PATH = r"./weights/pranetv2_pvt_kvasirseg_best.pth"

IMG_SIZE = 352
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 模型导入（与你训练/测试一致）
# =========================
def import_model():
    from lib.pranet import PVT_PraNet_V2
    return PVT_PraNet_V2


def load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型权重未找到: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    print(f"[info] Loaded checkpoint: {ckpt_path}")


# =========================
# 推理数据集（无 mask）
# =========================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, img_size=352):
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到图片")

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
@torch.no_grad()
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

    # -------- 构建模型（与训练一致）--------
    Model = import_model()
    model = Model(
        channel=32,
        num_class=1,
        sem_downsample=1,
        use_softmax=False
    )

    load_checkpoint(model, CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    # -------- 推理并保存 --------
    for idx, (imgs, _) in enumerate(tqdm(loader, desc="Inferencing")):
        imgs = imgs.to(DEVICE)

        logits = model(imgs)[0]              # PraNet-V2 前景输出
        preds = (torch.sigmoid(logits) > 0.5).float()

        pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
        pred_img = Image.fromarray(pred_np)

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        pred_img.save(os.path.join(SAVE_DIR, save_name))

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


# =========================
# 直接运行（无命令行）
# =========================
if __name__ == "__main__":
    inference()
