import os
import sys
import argparse
import importlib.util
import random

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/daeformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.DAEFormer import DAEFormer
# [NEW] Dice & IoU metric (for CE-based segmentation)
def dice_coefficient(pred, target, smooth=1e-6):
    """
    pred: [B, H, W] (long)
    target: [B, H, W] (long)
    """

    pred = (pred > 0).float()
    target = (target > 0).float()

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0).float()
    target = (target > 0).float()

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
# ====================== [NEW END] ======================
# [NEW] Unified mask processing function (key fix)
def normalize_mask(mask):
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    mask = (mask > 0).long()
    return mask
# [NEW] Universal data loader utility function (GUI)
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=4):
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=224,
        num_workers=0
    )
# Default configuration
DATA_ROOT = r"./data/Kvasir-SEG"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "masks")

PRETRAIN_PATH_DEFAULT = r"./pretrained_ckpt/synapse_epoch_399.pth"
SAVE_DIR = r"exp/kvasir_daeformer"

IMG_SIZE = 224
NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 1e-4
SEED = 42
# Utility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class KvasirSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224):
        self.img_names = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])
        self.mask_transform = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.uint8)).long()
        return img, mask

def dice_loss(pred, target, num_classes, smooth=1.0):
    pred_soft = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(pred_soft * target, dims)
    cardinality = torch.sum(pred_soft + target, dims)

    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return 1. - dice.mean()

def load_pretrained(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[warn] Pretrained weights not found: {ckpt_path}")
        return model

    print(f"[info] Loading pretrained weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    for k in ["decoder_0.last_layer.weight", "decoder_0.last_layer.bias"]:
        new_state.pop(k, None)

    model.load_state_dict(new_state, strict=False)
    return model
# Training
def train(args):
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    if args.data_loader and args.image_folder and args.mask_folder:
        print("[GUI Mode] Using universal data loader")
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            args.batch_size
        )
    else:
        print("[Local Mode] Using local dataset")
        train_loader = DataLoader(
            KvasirSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

    model = DAEFormer(num_classes=NUM_CLASSES)
    model = load_pretrained(model, args.pretrain or PRETRAIN_PATH_DEFAULT)
    model.to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NUM_EPOCHS)

    print("Start training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        # ====================== [NEW] ======================
        epoch_dice = 0.0
        epoch_iou = 0.0
        # ====================== [NEW END] ======================

        for i, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = normalize_mask(masks).to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss = ce(logits, masks) + dice_loss(logits, masks, NUM_CLASSES)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

            # ====================== [NEW] Dice / IoU ======================
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                dice = dice_coefficient(preds, masks)
                iou = iou_score(preds, masks)

            epoch_dice += dice
            epoch_iou += iou
            # ====================== [NEW END] ======================

            if i % 10 == 0:
                print(
                    f"Epoch {epoch} Step {i} "
                    f"Loss {loss.item():.4f} "
                    f"Dice {dice:.4f} "
                    f"IoU {iou:.4f}",
                    flush=True
                )

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)

        print(
            f"[Epoch {epoch}] "
            f"Loss: {avg_loss:.4f} | "
            f"Dice: {avg_dice:.4f} | "
            f"IoU: {avg_iou:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))
        sched.step()

    print("Training finished")
# Argument entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--mask_folder', type=str)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()
    train(args)
