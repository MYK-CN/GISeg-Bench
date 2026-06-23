import os
import random
import argparse
import importlib.util
from pathlib import Path

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/universeg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Try to import the model to avoid path errors
try:
    from universeg import universeg
except ImportError:
    print("[Error] Failed to import universeg. Please check installation or path.")
# 0. Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
# 1. [New] External DataLoader Support
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size=128):
    """
    Standard GUI interface: dynamically load DataLoader from an external script
    """
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    print(f"[Info] External Loader: Force img_size={img_size}, num_workers=0")

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=img_size,  # Universeg defaults to 128
        num_workers=0,        # Must be 0 to fix Windows PicklingError
        shuffle=True
    )
# 2. Built-in Dataset (unchanged, for local testing)
class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, image_subdir="images", mask_subdir="masks",
                 indices=None, augment=False):
        self.root_dir = root_dir
        self.img_dir = Path(root_dir) / image_subdir
        self.mask_dir = Path(root_dir) / mask_subdir
        self.augment = augment

        # Simple fault tolerance
        if not self.mask_dir.exists():
            print(f"[Warn] Mask dir not found: {self.mask_dir}")
            self.samples = []
            return

        mask_index = {p.stem: p for p in self.mask_dir.glob("*")}

        pairs = []
        for img in self.img_dir.glob("*"):
            if img.stem in mask_index:
                pairs.append((img, mask_index[img.stem]))

        if indices is not None:
            self.samples = [pairs[i] for i in indices]
        else:
            self.samples = pairs

        print(f"[Dataset] {len(self.samples)} samples | augment={self.augment}")

    def __len__(self):
        return len(self.samples)

    def _load(self, path, is_mask=False):
        img = Image.open(path).convert("L")
        img = img.resize((128, 128), Image.NEAREST if is_mask else Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        if is_mask:
            arr = (arr > 127).astype(np.float32)
        else:
            arr /= 255.0
        return arr

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = np.flip(img, 1), np.flip(mask, 1)
        if random.random() < 0.5:
            img, mask = np.flip(img, 0), np.flip(mask, 0)
        k = random.randint(0, 3)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        return img.copy(), mask.copy()

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = self._load(img_path, False)
        mask = self._load(mask_path, True)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Return shape: [1, H, W]
        return (
            torch.from_numpy(img)[None, ...],
            torch.from_numpy(mask)[None, ...]
        )
# 3. Dice Loss / Metric
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice

def dice_coef(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2 * inter + 1) / (union + 1)
    return dice.item()
# 4. Build Model
def build_model(pretrain):
    model = universeg(version="v1", pretrained=False)
    if pretrain and os.path.exists(pretrain):
        print("[Load] Pretrain:", pretrain)
        try:
            state = torch.load(pretrain, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        if pretrain:
            print(f"[Warn] Pretrain path does not exist: {pretrain}")

    return model.to(DEVICE)
# 5. Support Set (DataLoader-compatible)
def get_support_from_loader(loader, size, device):
    """Fetch a batch from DataLoader as Support Set"""

    try:
        imgs, masks = next(iter(loader))
    except StopIteration:
        return None, None

    if imgs.size(0) < size:
        repeat_times = (size // imgs.size(0)) + 1
        imgs = imgs.repeat(repeat_times, 1, 1, 1)
        masks = masks.repeat(repeat_times, 1, 1, 1)

    imgs = imgs[:size]
    masks = masks[:size]

    # Key: external loader usually returns RGB, convert to grayscale
    if imgs.shape[1] == 3:
        imgs = 0.299 * imgs[:, 0:1, :, :] + \
               0.587 * imgs[:, 1:2, :, :] + \
               0.114 * imgs[:, 2:3, :, :]

    if masks.shape[1] == 3:
        masks = masks[:, 0:1, :, :]

    # Universeg expects support shape: [1, S, C, H, W]
    imgs = imgs.unsqueeze(0).to(device)
    masks = masks.unsqueeze(0).to(device)

    return imgs, masks
# 6. Train / Val (DataLoader-adapted)
def train_epoch(model, dataloader, support_i, support_m, opt, bce, dice_fn, ep):
    model.train()
    steps = len(dataloader)

    for i, (img, msk) in enumerate(dataloader):
        img = img.to(DEVICE)
        msk = msk.to(DEVICE)

        # Key: RGB to grayscale for Universeg
        if img.shape[1] == 3:
            img = 0.299 * img[:, 0:1, :, :] + \
                  0.587 * img[:, 1:2, :, :] + \
                  0.114 * img[:, 2:3, :, :]

        if msk.shape[1] > 1:
            msk = msk[:, 0:1, :, :]

        B = img.shape[0]
        curr_sup_i = support_i.expand(B, -1, -1, -1, -1)
        curr_sup_m = support_m.expand(B, -1, -1, -1, -1)

        opt.zero_grad()
        out = model(img, curr_sup_i, curr_sup_m)

        loss = bce(out, msk) + dice_fn(out, msk)
        loss.backward()
        opt.step()

        if (i + 1) % 10 == 0:
            print(f"[Train] Ep{ep} Step [{i + 1}/{steps}] Loss: {loss.item():.4f}")
# 7. Main
def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.data_loader and args.image_folder and args.mask_folder:
        print("[INFO] Using external DataLoader (GUI Mode)")
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            batch_size=args.batch_size if args.batch_size else 1,
            img_size=128
        )
        val_loader = None
    else:
        print("[INFO] Using built-in Kvasir (Local Mode)")
        full_ds = KvasirSegDataset(args.data_root)
        if len(full_ds) == 0:
            print("[Error] Empty dataset")
            return
        train_loader = DataLoader(full_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = None

    model = build_model(args.pretrain)

    print("[INFO] Building Support Set...")
    support_i, support_m = get_support_from_loader(train_loader, args.support_size, DEVICE)

    if support_i is None:
        print("[Error] Failed to obtain Support Set")
        return

    print(f"[Support Set Ready] Shape: {support_i.shape}")

    opt = optim.AdamW(model.parameters(), lr=5e-5)
    bce = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "universeg_finetuned.pth")

    print(f"Start Training for {args.epochs} epochs...")

    for ep in range(1, args.epochs + 1):
        train_epoch(model, train_loader, support_i, support_m, opt, bce, dice_fn, ep)

        if ep % 5 == 0 or ep == args.epochs:
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Saved model to: {save_path}")

    print("Training Finished.")
# 8. Entry (modified argument definitions)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GUI-compatible arguments
    parser.add_argument("--data_loader", type=str, default=None, help="External data loader path")
    parser.add_argument("--image_folder", type=str, default=None, help="Image folder")
    parser.add_argument("--mask_folder", type=str, default=None, help="Mask folder")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size")

    # Original arguments
    parser.add_argument("--data_root", type=str, default=r"./data/Kvasir-SEG")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/foundation/universeg")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--support_size", type=int, default=16)

    # Deprecated arguments kept for compatibility
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--val_steps", type=int, default=100)
    parser.add_argument("--dataset_py", type=str, default=None)

    args = parser.parse_args()
    main(args)
