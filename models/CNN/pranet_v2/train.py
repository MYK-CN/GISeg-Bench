import sys
import os
import argparse
import importlib.util
import time
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ========================= 0. Path and Environment Configuration =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY_SEG_ROOT = os.path.join(PROJECT_ROOT, "binary_seg")

if BINARY_SEG_ROOT not in sys.path:
    sys.path.insert(0, BINARY_SEG_ROOT)

# ========================= 1. Universal Data Loader Utility =========================
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

# ========================= 2. Default Configuration =========================
CFG = {
    "project_root": r".",
    "data_root": r"./data/Kvasir-SEG",
    "pvt_backbone_weight": r".\pvt_v2_b2.pth",
    "img_size": 352,
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "save_dir": "outputs/cnn/pranet_v2",
    "use_pvt": True,
    "num_class": 1,
    "use_softmax": False,
    "ds_weights": [1.0, 0.6, 0.4, 0.2],
}

# ========================= 3. Dataset =========================
def list_images(folder):
    exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

class KvasirSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=352):
        self.img_paths = list_images(img_dir)
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(mask_dir)}

        pairs = []
        for ip in self.img_paths:
            k = os.path.splitext(os.path.basename(ip))[0]
            if k in mask_map:
                pairs.append((ip, mask_map[k]))

        self.img_paths = [p[0] for p in pairs]
        self.mask_paths = [p[1] for p in pairs]
        self.img_size = img_size

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        return img, mask

# ========================= 4. Metrics =========================
@torch.no_grad()
def dice_iou_from_logits(logits, target, eps=1e-7):
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp_max(1).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

# ========================= 5. Model Import =========================
def import_model_classes():
    from lib.pranet import PVT_PraNet_V2, PraNet_V2
    return PVT_PraNet_V2, PraNet_V2

def load_pvt_backbone_weight(model, weight_path):
    if not weight_path or (not os.path.exists(weight_path)):
        print("[INFO] Backbone not loaded (train from scratch)")
        return

    print(f"[INFO] Loading backbone: {weight_path}")

    state = torch.load(weight_path, map_location="cpu")

    # 兼容 checkpoint / state_dict 两种格式
    if "state_dict" in state:
        state = state["state_dict"]

    if "model" in state:
        state = state["model"]

    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        new_state[k] = v

    msg = model.backbone.load_state_dict(new_state, strict=False)

    print("[INFO] Backbone load result:", msg)

def load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items() if k.replace("module.", "") in model.state_dict()},
        strict=False
    )

# ========================= 6. Training =========================
def train_one_epoch(model, loader, optimizer, scaler, device, bce_loss, ds_weights, epoch, total_epochs):
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch}/{total_epochs}", file=sys.stdout)

    for i, (img, mask) in pbar:
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outs = model(img)
            map2_fg, map3_fg, map4_fg, map5_fg = outs[:4]

            loss = (ds_weights[0] * bce_loss(map2_fg, mask) +
                    ds_weights[1] * bce_loss(map3_fg, mask) +
                    ds_weights[2] * bce_loss(map4_fg, mask) +
                    ds_weights[3] * bce_loss(map5_fg, mask))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ===================== [NEW] Train Dice / IoU =====================
        d, iou = dice_iou_from_logits(map2_fg, mask)
        total_loss += loss.item()
        total_dice += d
        total_iou += iou

        if (i + 1) % 10 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{d:.4f}",
                iou=f"{iou:.4f}"
            )

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n

@torch.no_grad()
def validate(model, loader, device, bce_loss):
    model.eval()
    total_loss, dices, ious = 0.0, [], []

    for img, mask in tqdm(loader, desc="Validating", file=sys.stdout):
        img = img.to(device)
        mask = mask.to(device)

        map2_fg = model(img)[0]
        total_loss += bce_loss(map2_fg, mask).item()

        d, i = dice_iou_from_logits(map2_fg, mask)
        dices.append(d)
        ious.append(i)

    return total_loss / max(len(loader), 1), float(np.mean(dices)), float(np.mean(ious))

# ========================= 7. Main =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or CFG["batch_size"]

    if args.data_loader:
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size, CFG["img_size"]
        )
        val_loader = None
    else:
        train_loader = DataLoader(
            KvasirSegDataset(os.path.join(CFG["data_root"], "images"),
                             os.path.join(CFG["data_root"], "masks"), CFG["img_size"]),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            KvasirSegDataset(os.path.join(CFG["data_root"], "val"),
                             os.path.join(CFG["data_root"], "maskval"), CFG["img_size"]),
            batch_size=batch_size, shuffle=False
        )

    PVT_PraNet_V2, PraNet_V2 = import_model_classes()
    model = PVT_PraNet_V2(channel=32, num_class=CFG["num_class"],
                          sem_downsample=1, use_softmax=CFG["use_softmax"])

    if os.path.exists(CFG["pvt_backbone_weight"]):
        load_pvt_backbone_weight(model, CFG["pvt_backbone_weight"])
    if args.pretrain:
        load_checkpoint(model, args.pretrain)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            bce_loss, CFG["ds_weights"], epoch, CFG["epochs"]
        )

        log = f"Epoch {epoch}/{CFG['epochs']} | Train Loss {train_loss:.4f} Dice {train_dice:.4f} IoU {train_iou:.4f}"

        if val_loader:
            val_loss, val_dice, val_iou = validate(model, val_loader, device, bce_loss)
            log += f" | Val Loss {val_loss:.4f} Dice {val_dice:.4f} IoU {val_iou:.4f}"

        print(log, flush=True)

    print("Training Finished!", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loader", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--mask_folder", type=str)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    main(args)
