import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/sam2_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
# 1. 路径配置（按你的要求写死）
finetuned_weight_path = r"./weights/sam2_unet_best.pth"
sam2_backbone_path   = r"./pretrained_ckpt/sam2.1_hiera_large.pt"

test_img_dir  = r"./data/Kvasir-SEG/test"
test_mask_dir = r"./data/Kvasir-SEG/masktest"

IMG_SIZE = 352
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 2. 导入 SAM2-UNet
try:
    from SAM2UNet import SAM2UNet
except ImportError:
    from model.SAM2UNet import SAM2UNet
# 3. 测试数据集
class KvasirTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=352):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = (img_size, img_size)

        self.ids = []
        for f in os.listdir(img_dir):
            name = os.path.splitext(f)[0]
            if any(os.path.exists(os.path.join(mask_dir, name + e))
                   for e in [".png", ".jpg", ".jpeg", ".bmp"]):
                self.ids.append(name)

        self.ids.sort()
        print(f"[INFO] Test samples: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        def find(root):
            for e in [".png", ".jpg", ".jpeg", ".bmp"]:
                p = os.path.join(root, name + e)
                if os.path.exists(p):
                    return p
            return None

        img = Image.open(find(self.img_dir)).convert("RGB")
        mask = Image.open(find(self.mask_dir)).convert("L")

        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        img = TF.normalize(TF.to_tensor(img), [0.5]*3, [0.5]*3)
        mask = (TF.to_tensor(mask) > 0.5).float()

        return img, mask
# 4. 指标函数
def dice_iou(pred, target, eps=1e-7):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter

    dice = (2 * inter + eps) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean(), iou.mean()
# 5. 主测试流程
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # ---------- Dataset ----------
    test_loader = DataLoader(
        KvasirTestDataset(test_img_dir, test_mask_dir, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ---------- Model ----------
    print("[INFO] Building SAM2-UNet...")
    model = SAM2UNet(checkpoint_path=sam2_backbone_path)
    model.to(DEVICE)

    print("[INFO] Loading finetuned weights...")
    ckpt = torch.load(finetuned_weight_path, map_location="cpu")
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

    # ---------- Metrics ----------
    bce = nn.BCEWithLogitsLoss()
    total_loss, total_dice, total_iou = 0, 0, 0

    with torch.no_grad():
        for img, mask in tqdm(test_loader, desc="Testing"):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            out, _, _ = model(img)

            if out.shape[2:] != mask.shape[2:]:
                out = F.interpolate(out, size=mask.shape[2:], mode="bilinear", align_corners=False)

            loss = bce(out, mask)
            dice, iou = dice_iou(out, mask)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

    n = len(test_loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("==================================")

if __name__ == "__main__":
    main()
