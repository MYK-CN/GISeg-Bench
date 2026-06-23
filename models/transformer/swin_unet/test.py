import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/swin_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 1. Import Swin-Unet Model
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# 2. Test Configuration
TEST_IMAGE_DIR = r"./data/Kvasir-SEG/test"
TEST_MASK_DIR  = r"./data/Kvasir-SEG/masktest"

WEIGHT_PATH = r"./weights/swin_unet_kvasir_finetune_v2.pth"

IMG_SIZE = 224
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 3. Test Dataset
class KvasirTestDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        # 支持多种后缀
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

        img_files = []
        for e in exts:
            img_files.extend(
                [f for f in os.listdir(images_dir) if f.lower().endswith(e)]
            )

        self.samples = []
        for img_name in img_files:
            stem = os.path.splitext(img_name)[0]
            mask_path = None
            for e in exts:
                cand = os.path.join(masks_dir, stem + e)
                if os.path.exists(cand):
                    mask_path = cand
                    break

            if mask_path is not None:
                self.samples.append(
                    (os.path.join(images_dir, img_name), mask_path)
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found in:\n"
                f"{images_dir}\n{masks_dir}"
            )

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_tf(img)

        mask = self.mask_tf(mask)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = (mask > 0.5).float()

        return img, mask
# 4. Metrics
@torch.no_grad()
def dice_iou_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter

    dice = (2 * inter + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()
# 5. Main Test Logic
@torch.no_grad()
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # ---------- Data ----------
    test_dataset = KvasirTestDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------- Model ----------
    print("[INFO] Building Swin-Unet...")
    model = SwinTransformerSys(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        depths_decoder=[1, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        final_upsample="expand_first"
    )

    print("[INFO] Loading finetuned weights...")
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    # ---------- Loss ----------
    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for imgs, masks in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(imgs)

        loss = bce_loss_fn(logits, masks)
        dice, iou = dice_iou_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(test_loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
