import os
import sys
import glob
import time
import random
import argparse
import importlib.util
from PIL import Image
import numpy as np
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/htc_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import ml_collections  # Keep original dependency

# -------------------------------------------------------------
# [NEW 1] Universal Data Loader Utility Function (for GUI)
# -------------------------------------------------------------
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size):
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=img_size,
        num_workers=0,
    )

# ---------------- Default Configuration (Fallback) ----------------
DEFAULT_DATA_ROOT = r"./data/Kvasir-SEG"
DEFAULT_PRETRAINED_PATH = r"./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
LOG_DIR = os.path.join(OUTPUT_DIR, "test_log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

IMG_SIZE = 224
NUM_EPOCHS = 16
NUM_WORKERS = 0 if os.name == 'nt' else 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 20
RANDOM_SEED = 42

# ---------------- Dataset ----------------
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=IMG_SIZE, mode="train"):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.masks = []

        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            found = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                cand = os.path.join(masks_dir, name + ext)
                if os.path.exists(cand):
                    found = cand
                    break
            self.masks.append(found)

        pairs = [(i, m) for i, m in zip(self.images, self.masks) if m is not None]
        self.images, self.masks = [p[0] for p in pairs], [p[1] for p in pairs]

        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.mode == "train":
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = (np.array(mask) > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

# ---------------- Metrics ----------------
def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2.0 * intersection + eps) / (union + eps)).item()

# ======================= [NEW] IoU Metric =======================
def iou_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()
# ======================= [NEW END] =======================

# ---------------- Model Import ----------------
from network.Net import model as SwinModelWrapper

# ---------------- Training & Validation ----------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Train Epoch {epoch}",
        leave=False,
        file=sys.stdout
    )

    for i, (imgs, masks) in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = dice_coeff(outputs, masks)
            iou = iou_coeff(outputs, masks)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou

        if (i + 1) % PRINT_FREQ == 0:
            pbar.set_postfix({
                "loss": f"{running_loss / (i + 1):.4f}",
                "dice": f"{running_dice / (i + 1):.4f}",
                "iou": f"{running_iou / (i + 1):.4f}"
            })

    n = len(dataloader)
    return running_loss / n, running_dice / n, running_iou / n

def validate(model, dataloader, criterion, device):
    if dataloader is None:
        return 0.0, 0.0, 0.0

    model.eval()
    losses, dices, ious = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False, file=sys.stdout)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            losses.append(criterion(outputs, masks).item())
            dices.append(dice_coeff(outputs, masks))
            ious.append(iou_coeff(outputs, masks))

    return np.mean(losses), np.mean(dices), np.mean(ious)

# ---------------- Main ----------------
def main(args):
    print(f"Using Device: {DEVICE}", flush=True)
    batch_size = args.batch_size if args.batch_size else 8
    val_loader = None

    if args.data_loader and args.image_folder and args.mask_folder:
        print("[GUI Mode] Using Universal Loader...", flush=True)
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size=batch_size, img_size=IMG_SIZE
        )
    else:
        print("[Local Mode] Using Hardcoded Paths...", flush=True)
        images_dir = os.path.join(DEFAULT_DATA_ROOT, "images")
        masks_dir = os.path.join(DEFAULT_DATA_ROOT, "masks")

        dataset = SegmentationDataset(images_dir, masks_dir, img_size=IMG_SIZE, mode="train")
        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)
        split = int(n * 0.8)

        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(dataset, indices[:split]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            Subset(dataset, indices[split:]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

    cfg = ml_collections.config_dict.ConfigDict()
    cfg.n_classes = 1
    cfg.decoder_channels = (128, 64, 32, 16)
    cfg.n_skip = 3

    net_wrapper = SwinModelWrapper(config=cfg, img_size=IMG_SIZE, num_classes=1)
    net = net_wrapper.to(DEVICE)

    pretrained_path = args.pretrain if (args.pretrain and os.path.exists(args.pretrain)) else DEFAULT_PRETRAINED_PATH
    if os.path.exists(pretrained_path):
        net.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    print(f"Start training for {NUM_EPOCHS} epochs...", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            net, train_loader, criterion, optimizer, DEVICE, epoch
        )

        if val_loader:
            val_loss, val_dice, val_iou = validate(net, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)
            info = (
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"TrainLoss {train_loss:.4f} | "
                f"TrainDice {train_dice:.4f} | "
                f"TrainIoU {train_iou:.4f} | "
                f"ValLoss {val_loss:.4f} | "
                f"ValDice {val_dice:.4f} | "
                f"ValIoU {val_iou:.4f}"
            )
        else:
            info = (
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"TrainLoss {train_loss:.4f} | "
                f"TrainDice {train_dice:.4f} | "
                f"TrainIoU {train_iou:.4f}"
            )

        print(info, flush=True)
        with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
            f.write(info + "\n")

        torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth"))

    print("Training Complete.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--mask_folder', type=str)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    main(args)
