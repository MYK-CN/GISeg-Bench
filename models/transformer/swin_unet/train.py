import os
import argparse
import importlib.util
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/swin_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Import Swin-Unet Model
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# External DataLoader Support
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=4):
    spec = importlib.util.spec_from_file_location(
        "universal_data_loader", data_loader_path
    )
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        num_workers=0,
        image_size=224
    )
# 2. Dataset Definition (Unchanged)
class KvasirSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks does not match!"

        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]

        img = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")

        img = self.img_transform(img)

        mask = self.mask_transform(mask)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
# 3. Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()
# 4. Pretrained Weight Loading (Unchanged)
def load_pretrained_backbone(model, ckpt_path):
    print(f"[INFO] Loading pretrained weights: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model_dict = model.state_dict()
    matched, unmatched = 0, 0
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            matched += 1
        else:
            unmatched += 1
    model.load_state_dict(model_dict, strict=False)
    print(f"   Matched: {matched}, Unmatched: {unmatched}")
    return model
# 5. Training & Validation (Unchanged)
def train_one_epoch(model, loader, optimizer, bce_loss_fn, dice_loss_fn, device):
    model.train()
    total_bce, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        bce = bce_loss_fn(logits, masks)
        dice = dice_loss_fn(logits, masks)
        loss = bce + dice

        loss.backward()
        optimizer.step()

        total_bce += bce.item()
        total_dice += dice.item()

    n = len(loader)
    return total_bce / n, total_dice / n

@torch.no_grad()
def validate(model, loader, bce_loss_fn, dice_loss_fn, device):
    model.eval()
    total_bce, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        total_bce += bce_loss_fn(logits, masks).item()
        total_dice += dice_loss_fn(logits, masks).item()

    n = len(loader)
    return total_bce / n, total_dice / n
# 6. Main Function (Adapted)
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- DataLoader ----------
    if args.data_loader and args.image_folder and args.mask_folder:
        print("[INFO] Using external data loader")
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            args.batch_size
        )
        val_loader = None
    else:
        print("[INFO] Using built-in Kvasir-SEG dataset")
        train_dataset = KvasirSegDataset(
            args.train_images, args.train_masks, img_size=args.img_size
        )
        val_dataset = KvasirSegDataset(
            args.val_images, args.val_masks, img_size=args.img_size
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

    # ---------- Model ----------
    model = SwinTransformerSys(
        img_size=args.img_size,
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

    if args.pretrain:
        model = load_pretrained_backbone(model, args.pretrain)

    model = model.to(device)

    # ---------- Optim ----------
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        train_bce, train_dice = train_one_epoch(
            model, train_loader, optimizer, bce_loss_fn, dice_loss_fn, device
        )

        if val_loader is not None:
            val_bce, val_dice = validate(
                model, val_loader, bce_loss_fn, dice_loss_fn, device
            )
            print(
                f"[Epoch {epoch}] "
                f"Train DiceLoss: {train_dice:.4f} | Val DiceLoss: {val_dice:.4f}"
            )
            if val_dice < best_val:
                best_val = val_dice
                torch.save(model.state_dict(), args.save_path)
        else:
            print(f"[Epoch {epoch}] Train DiceLoss: {train_dice:.4f}")
# 7. Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_loader", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)

    parser.add_argument("--train_images", type=str, default=r"./data/Kvasir-SEG/images")
    parser.add_argument("--train_masks", type=str, default=r"./data/Kvasir-SEG/masks")
    parser.add_argument("--val_images", type=str, default=r"./data/Kvasir-SEG/val")
    parser.add_argument("--val_masks", type=str, default=r"./data/Kvasir-SEG/maskval")

    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_path", type=str, default="outputs/transformer/swin_unet")

    args = parser.parse_args()
    main(args)
