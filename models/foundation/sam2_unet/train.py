import os
import sys
import argparse
import random
import importlib.util
from typing import Tuple
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/sam2_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

# ================= 0. Environment Setup =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Try to import model
try:
    from SAM2UNet import SAM2UNet
except ImportError:
    # Compatibility fallback
    try:
        from model.SAM2UNet import SAM2UNet
    except ImportError:
        print("[Error] SAM2UNet model definition not found. Please ensure SAM2UNet.py exists.", flush=True)

# ================= 1. [NEW] Universal DataLoader Utility =================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size):
    """Dynamically load DataLoader from an external script (GUI mode)"""
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    # Note: SAM2-UNet usually uses 352x352 or 1024x1024, here we accept img_size from args
    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=img_size,
        num_workers=0,
    )

# ================= 2. Default Configuration (Local Fallback) =================
class DefaultConfig:
    # SAM2 base checkpoint (Backbone) - required to build the model
    SAM2_CHECKPOINT = r"./pretrained_ckpt/sam2.1_hiera_large.pt"

    # Local dataset path
    KVASIR_ROOT = r"./data/Kvasir-SEG"

    SAVE_DIR = "outputs/foundation/sam2_unet"
    IMG_SIZE = 352
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LR = 1e-4
    NUM_WORKERS = 0  # More stable on Windows
    SEED = 42

# ================= 3. Utility Functions =================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = num / den
    return 1 - dice.mean()

def compute_dice_iou(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * target).sum(dim=(2, 3))
    union = (pred_bin + target - pred_bin * target).sum(dim=(2, 3)) + eps
    dice = 2 * inter / (pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)
    iou = inter / union
    return dice.mean(), iou.mean()

# ================= 4. Dataset (Local Mode) =================
class KvasirPolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True, img_size=352):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.img_size = (img_size, img_size)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"[Warn] Dataset path not found: {img_dir}", flush=True)
            self.ids = []
            return

        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))
        img_set = {os.path.splitext(f)[0] for f in img_files}
        mask_set = {os.path.splitext(f)[0] for f in mask_files}
        self.ids = sorted(list(img_set & mask_set))

    def __len__(self):
        return len(self.ids)

    def _load_pair(self, idx):
        id_ = self.ids[idx]

        def find_file(root, stem):
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                p = os.path.join(root, stem + ext)
                if os.path.exists(p):
                    return p
            return None

        img_path = find_file(self.img_dir, id_)
        mask_path = find_file(self.mask_dir, id_)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_pair(idx)
        img = img.resize(self.img_size, resample=Image.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)

        if self.is_train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask

# ================= 5. Training Loop =================
def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch}/{total_epochs}", file=sys.stdout)

    for i, (imgs, masks) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # SAM2UNet returns 3 outputs for deep supervision
        out, out1, out2 = model(imgs)

        # Align spatial size (prevent output-mask mismatch)
        if out.shape[2:] != masks.shape[2:]:
            out = F.interpolate(out, size=masks.shape[2:], mode="bilinear", align_corners=False)
            out1 = F.interpolate(out1, size=masks.shape[2:], mode="bilinear", align_corners=False)
            out2 = F.interpolate(out2, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss_main = criterion_bce(out, masks) + dice_loss(out, masks)
        loss_side1 = criterion_bce(out1, masks) + dice_loss(out1, masks)
        loss_side2 = criterion_bce(out2, masks) + dice_loss(out2, masks)

        loss = loss_main + 0.5 * (loss_side1 + loss_side2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        dice, _ = compute_dice_iou(out, masks)
        total_dice += dice.item()

        if (i + 1) % 5 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

    return total_loss / len(loader), total_dice / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    if loader is None:
        return 0.0, 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for imgs, masks in tqdm(loader, desc="Val", file=sys.stdout):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out, out1, out2 = model(imgs)

        if out.shape[2:] != masks.shape[2:]:
            out = F.interpolate(out, size=masks.shape[2:], mode="bilinear", align_corners=False)
            out1 = F.interpolate(out1, size=masks.shape[2:], mode="bilinear", align_corners=False)
            out2 = F.interpolate(out2, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss = criterion_bce(out, masks) + dice_loss(out, masks)

        total_loss += loss.item()
        dice, iou = compute_dice_iou(out, masks)
        total_dice += dice.item()
        total_iou += iou.item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# ================= 6. Main Function =================
def main(args):
    set_seed(DefaultConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- Configuration ---
    save_dir = DefaultConfig.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    batch_size = args.batch_size if args.batch_size else DefaultConfig.BATCH_SIZE
    img_size = DefaultConfig.IMG_SIZE

    # --- Data Loading Branch ---
    val_loader = None
    if args.data_loader and args.image_folder and args.mask_folder:
        print("[GUI Mode] Loading external dataloader...", flush=True)
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size, img_size
        )
        print("[Info] GUI Mode: Validation skipped (using full data for training).", flush=True)
    else:
        print("[Local Mode] Loading local Kvasir dataset...", flush=True)
        train_ds = KvasirPolypDataset(
            os.path.join(DefaultConfig.KVASIR_ROOT, "images"),
            os.path.join(DefaultConfig.KVASIR_ROOT, "masks"),
            is_train=True, img_size=img_size
        )
        val_ds = KvasirPolypDataset(
            os.path.join(DefaultConfig.KVASIR_ROOT, "val"),
            os.path.join(DefaultConfig.KVASIR_ROOT, "maskval"),
            is_train=False, img_size=img_size
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=DefaultConfig.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=DefaultConfig.NUM_WORKERS)

    # --- Model Building ---
    print("Building SAM2UNet...", flush=True)
    # 1. Base Backbone checkpoint (must exist, otherwise SAM2 cannot initialize)
    if not os.path.exists(DefaultConfig.SAM2_CHECKPOINT):
        print(f"[Warn] SAM2 checkpoint not found at: {DefaultConfig.SAM2_CHECKPOINT}", flush=True)

    # Initialize model (loads backbone internally)
    model = SAM2UNet(checkpoint_path=DefaultConfig.SAM2_CHECKPOINT)

    # 2. Finetune / resume checkpoint (passed from GUI)
    if args.pretrain and os.path.exists(args.pretrain):
        print(f"Loading finetuned weights from: {args.pretrain}", flush=True)
        try:
            state_dict = torch.load(args.pretrain, map_location='cpu')
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Smart matching load
            model_dict = model.state_dict()
            new_dict = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model.load_state_dict(new_dict, strict=False)
            print(f"Loaded {len(new_dict)} layers.", flush=True)
        except Exception as e:
            print(f"Load weights failed: {e}", flush=True)

    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=DefaultConfig.LR, weight_decay=1e-4
    )

    epochs = DefaultConfig.NUM_EPOCHS
    best_dice = 0.0

    print(f"Start training for {epochs} epochs...", flush=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, epoch, epochs
        )

        log_str = f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} Dice: {train_dice:.4f}"

        if val_loader:
            val_loss, val_dice, val_iou = validate(model, val_loader, device)
            log_str += f" | Val Dice: {val_dice:.4f} IoU: {val_iou:.4f}"

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                log_str += " [Best Saved]"
        else:
            # Save every epoch in GUI mode
            save_path = os.path.join(save_dir, f"sam2unet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            log_str += f" | Saved: {save_path}"

        print(log_str, flush=True)

    print("Training Finished!", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str, help='Data loader path passed from GUI')
    parser.add_argument('--image_folder', type=str, help='Image folder passed from GUI')
    parser.add_argument('--mask_folder', type=str, help='Mask folder passed from GUI')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights passed from GUI')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')

    args = parser.parse_args()
    main(args)
