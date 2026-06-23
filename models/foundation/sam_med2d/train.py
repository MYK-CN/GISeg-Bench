import os
import sys
import glob
import argparse
import importlib.util
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/sam_med2d')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================= 0. Environment Setup =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# ================= 1. Universal Data Loader Utility =================
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

# ================= 2. Default Configuration =================
class DefaultConfig:
    data_root =  r"./data/Kvasir-SEG"
    image_dir = "images"
    mask_dir = "masks"
    ckpt = r"./pretrained_ckpt/sam-med2d_b.pth"
    model_type = "vit_b"
    image_size = 256
    out_dir = OUTPUT_DIR
    batch_size = 2
    epochs = 20
    lr = 1e-4
    weight_decay = 0.01

# ================= 3. Utility Functions =================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def dice_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean()

# ================= [NEW] IoU Metric =================
def iou_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()

def mask_to_bbox(mask_tensor):
    B, _, H, W = mask_tensor.shape
    bboxes = []
    for i in range(B):
        m = mask_tensor[i, 0] > 0.5
        if m.sum() == 0:
            bboxes.append([0, 0, W - 1, H - 1])
        else:
            ys, xs = torch.where(m)
            bboxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return torch.tensor(bboxes, dtype=torch.float32, device=mask_tensor.device)

# ================= 4. Dataset =================
class SegDataset(Dataset):
    def __init__(self, root, image_dir="test", mask_dir="masktest", image_size=256):
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.image_size = image_size

        self.image_paths = []
        for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
            self.image_paths += glob.glob(os.path.join(self.image_dir, e))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        mask_path = None
        for e in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
            p = os.path.join(self.mask_dir, name + e)
            if os.path.exists(p):
                mask_path = p
                break

        img = Image.open(img_path).convert("RGB")
        img = img.resize((DefaultConfig.image_size, DefaultConfig.image_size))
        img = np.asarray(img).astype(np.float32) / 255.0

        if mask_path:
            msk = Image.open(mask_path).convert("L")
            msk = msk.resize((DefaultConfig.image_size, DefaultConfig.image_size))
            msk = (np.array(msk) > 0).astype(np.float32)
        else:
            msk = np.zeros((DefaultConfig.image_size, DefaultConfig.image_size), dtype=np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        msk = torch.from_numpy(msk)[None]

        return img, msk

# ================= 5. Build Model =================
def build_model(ckpt_path, model_type, device, image_size):
    from segment_anything import sam_model_registry
    class SAMArgs:
        def __init__(self):
            self.image_size = image_size
            self.sam_checkpoint = ckpt_path
            self.encoder_adapter = True
            self.sam_type = model_type

    sam = sam_model_registry[model_type](SAMArgs())
    sam.to(device)
    return sam

# ================= 6. Main Training =================
def main(args):
    os.makedirs(DefaultConfig.out_dir, exist_ok=True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or DefaultConfig.batch_size
    img_size = DefaultConfig.image_size

    if args.data_loader:
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size, img_size
        )
    else:
        ds = SegDataset(DefaultConfig.data_root, DefaultConfig.image_dir, DefaultConfig.mask_dir, img_size)
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    sam = build_model(DefaultConfig.ckpt, DefaultConfig.model_type, device, img_size)

    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad],
        lr=DefaultConfig.lr,
        weight_decay=DefaultConfig.weight_decay
    )

    bce = nn.BCEWithLogitsLoss()
    sam.train()

    for epoch in range(1, DefaultConfig.epochs + 1):
        total_loss, total_dice, total_iou = 0, 0, 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{DefaultConfig.epochs}", file=sys.stdout)

        for i, (img, gt_mask) in pbar:
            img, gt_mask = img.to(device), gt_mask.to(device)
            bbox = mask_to_bbox(gt_mask)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                img_emb = sam.image_encoder(img)

            sparse, dense = sam.prompt_encoder(None, bbox, None)
            low_res, _ = sam.mask_decoder(
                img_emb, sam.prompt_encoder.get_dense_pe(),
                sparse, dense, multimask_output=False
            )

            pred = F.interpolate(low_res, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

            loss = bce(pred, gt_mask) + dice_loss(pred, gt_mask)
            loss.backward()
            optimizer.step()

            d = dice_score(pred.detach(), gt_mask)
            iou = iou_score(pred.detach(), gt_mask)

            total_loss += loss.item()
            total_dice += d.item()
            total_iou += iou.item()

            if (i + 1) % 5 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{d.item():.4f}",
                    iou=f"{iou.item():.4f}"
                )

        print(
            f"Epoch {epoch}/{DefaultConfig.epochs} | "
            f"Loss {total_loss/len(train_loader):.4f} | "
            f"Dice {total_dice/len(train_loader):.4f} | "
            f"IoU {total_iou/len(train_loader):.4f}",
            flush=True
        )

        torch.save(
            sam.state_dict(),
            os.path.join(DefaultConfig.out_dir, f"sam_med2d_epoch_{epoch}.pth")
        )

    print("Training Done.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loader", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--mask_folder", type=str)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    main(args)
