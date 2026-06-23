import os
import sys
import argparse
import importlib.util
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/mt_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 0. [NEW] Universal Data Loader Utility =====================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size):
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=img_size,
        num_workers=0
    )

# ===================== 1. Model Import =====================
try:
    from model.MTUNet import MTUNet, configs
except ImportError:
    try:
        from MTUNet import MTUNet, configs
    except ImportError:
        print("[Error] MTUNet model definition not found.", flush=True)
        raise

# ===================== 2. Default Configuration =====================
DEFAULT_TRAIN_IMG = r"./data/Kvasir-SEG/images"
DEFAULT_TRAIN_MASK = r"./data/Kvasir-SEG/masks"
DEFAULT_PRETRAIN = r"./pretrained_ckpt/mtunet_pretrain.pth"
SAVE_DIR = "outputs/transformer/mt_unet"

# ===================== 3. Dataset =====================
class KvasirSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.img_files[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, self.mask_files[idx])).convert("L")

        img = self.img_transform(img)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)

        return img, mask

# ===================== 4. Loss =====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        dice = (2 * inter + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

# ===================== [NEW] Metrics =====================
def dice_metric(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2 * inter + eps) / (union + eps)).item()

def iou_metric(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    return ((inter + eps) / (union + eps)).item()
# ===================== [NEW END] =====================

# ===================== 5. Weight Loading =====================
def load_pretrained(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"[Warn] Pretrained weights not found: {ckpt_path}", flush=True)
        return model

    print(f"===> Loading weights from: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    model_dict = model.state_dict()
    matched = {k: v for k, v in new_state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    print(f"   Loaded {len(matched)} layers.", flush=True)
    return model

# ===================== 6. Training =====================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    img_size = 224
    num_epochs = 15
    lr = 1e-4
    batch_size = args.batch_size if args.batch_size else 4

    # -------- Data --------
    if args.data_loader and args.image_folder and args.mask_folder:
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size=batch_size, img_size=img_size
        )
    else:
        dataset = KvasirSegDataset(DEFAULT_TRAIN_IMG, DEFAULT_TRAIN_MASK, img_size)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------- Model --------
    model = MTUNet(out_ch=1).to(device)

    pretrained_path = args.pretrain if args.pretrain else DEFAULT_PRETRAIN
    model = load_pretrained(model, pretrained_path, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    print(f"Start training for {num_epochs} epochs...", flush=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = total_dice = total_iou = 0.0

        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss_b = bce_loss(logits, masks)
            loss_d = dice_loss(logits, masks)
            loss = 0.5 * loss_b + 0.5 * loss_d

            loss.backward()
            optimizer.step()

            # ===================== [NEW] Metrics =====================
            with torch.no_grad():
                d = dice_metric(logits, masks)
                iou = iou_metric(logits, masks)

            total_loss += loss.item()
            total_dice += d
            total_iou += iou

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Step [{i+1}/{len(train_loader)}] "
                    f"Loss {loss.item():.4f} Dice {d:.4f} IoU {iou:.4f}",
                    flush=True
                )

        n = len(train_loader)
        print(
            f"==> Epoch [{epoch}/{num_epochs}] "
            f"Avg Loss {total_loss/n:.4f} "
            f"Dice {total_dice/n:.4f} "
            f"IoU {total_iou/n:.4f}",
            flush=True
        )

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"mtunet_epoch_{epoch}.pth"))

    print("Training Finished!", flush=True)

# ===================== 7. Entry =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loader", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--mask_folder", type=str)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    train(args)
