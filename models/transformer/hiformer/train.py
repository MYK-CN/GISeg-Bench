import os
import sys
import glob
import random
import argparse
import importlib.util
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/hiformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Try importing medpy; if not installed, define a fallback metric
try:
    from medpy import metric
except ImportError:
    print("[Warning] medpy not found. Dice metric will be approximated.", flush=True)
    metric = None

from models.HiFormer import HiFormer
import configs.HiFormer_configs as hcfg

# ======================= 0. Universal Data Loader Utility =======================
def load_external_dataloader(data_loader_path, image_folder, mask_folder,
                             batch_size=4, img_size=224):
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

# ======================= 1. Pretrained Weight Loader =======================
def load_pretrained_weights(model, weight_path):
    if not os.path.exists(weight_path):
        print(f"[Warn] Weight file not found: {weight_path}", flush=True)
        return model

    print(f"Loading weights from: {weight_path}", flush=True)
    checkpoint = torch.load(weight_path, map_location="cpu")

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and v.shape == model_dict[k].shape:
            new_dict[k] = v

    model.load_state_dict(new_dict, strict=False)
    return model

# ======================= 2. Dataset (Fallback: Synapse NPZ) =======================
class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = sorted(os.listdir(self.images_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        mask = np.array(mask)
        mask = (mask > 0).astype(np.int64)
        mask = torch.from_numpy(mask).long()  # [H, W]

        return img, mask

# ======================= 3. Dice =======================
def compute_mean_dice(preds, labels, num_classes, ignore_index=0):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    dices = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = preds_np == c
        label_c = labels_np == c
        if pred_c.sum() + label_c.sum() == 0:
            continue

        if metric is not None:
            try:
                d = metric.dc(pred_c, label_c)
            except:
                d = 0.0
        else:
            inter = np.logical_and(pred_c, label_c).sum()
            d = (2 * inter) / (pred_c.sum() + label_c.sum() + 1e-8)

        dices.append(d)

    return float(np.mean(dices)) if dices else 0.0

# ======================= 4. IoU =======================
def compute_mean_iou(preds, labels, num_classes, ignore_index=0):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = preds_np == c
        label_c = labels_np == c
        if pred_c.sum() + label_c.sum() == 0:
            continue

        inter = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        ious.append(inter / (union + 1e-8))

    return float(np.mean(ious)) if ious else 0.0

# ======================= 5. Training =======================
def train_hiformer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    img_size = 224
    num_epochs = 15
    lr = 1e-4

    default_data_dir = r"./data/Kvasir-SEG"
    save_dir = "outputs/transformer/hiformer"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Data --------
    if args.data_loader and args.image_folder and args.mask_folder:
        num_classes = 2
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size=args.batch_size, img_size=img_size
        )
    else:
        num_classes = 2
        train_loader = DataLoader(
            KvasirSegDataset(default_data_dir, img_size=img_size),
            batch_size=args.batch_size, shuffle=True
        )

    # -------- Model --------
    config = hcfg.get_hiformer_b_configs()
    model = HiFormer(
        config=config,
        img_size=img_size,
        in_chans=3,
        n_classes=num_classes
    ).to(device)

    if args.pretrain:
        model = load_pretrained_weights(model, args.pretrain)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Start training for {num_epochs} epochs", flush=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_dice, epoch_iou = 0, 0, 0

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # ======================= FIX: CE target shape =======================
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                dice = compute_mean_dice(preds, labels, num_classes)
                iou = compute_mean_iou(preds, labels, num_classes)

            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{step+1}/{len(train_loader)}] "
                    f"Loss {loss.item():.4f} Dice {dice:.4f} IoU {iou:.4f}",
                    flush=True
                )

        n = len(train_loader)
        print(
            f"==> Epoch [{epoch+1}] "
            f"Loss {epoch_loss/n:.4f} "
            f"Dice {epoch_dice/n:.4f} "
            f"IoU {epoch_iou/n:.4f}",
            flush=True
        )

        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"hiformer_epoch{epoch+1}.pth")
        )

    print("Training Complete", flush=True)

# ======================= Entry =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--mask_folder', type=str)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    train_hiformer(args)
