import os
import sys
import argparse
import importlib.util
from glob import glob
from typing import List, Tuple

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/scribbleprompt')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ====== 0. Model Import ======
try:
    from scribbleprompt.models.unet import ScribblePromptUNet, prepare_inputs
except ImportError:
    print("[Error] Cannot find ScribblePromptUNet definition. Please check the environment.", flush=True)

# ===================== 1. [Modified] Universal DataLoader Utility =====================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size):
    """
    Standard GUI interface: dynamically load DataLoader from an external script
    """
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    # Pass image_size parameter (use the first value if tuple)
    size = img_size[0] if isinstance(img_size, tuple) else img_size

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=size,
        num_workers=0
    )

# ===================== 2. Safe Weight Loading Utility =====================
def safe_load_weights(model, ckpt_path, device="cpu"):
    """
    Safely load weights:
    - Automatically handle dict formats
    - Remove 'module.' prefix
    - Skip layers with shape mismatch
    """
    if not os.path.exists(ckpt_path):
        print(f"[Warn] Weight file not found: {ckpt_path}", flush=True)
        return model

    print(f"Loading weights from: {ckpt_path}", flush=True)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        model_dict = model.state_dict()
        matched_dict = {}
        for k, v in new_state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    print(f" -> Skipping layer {k}: shape mismatch {v.shape} vs {model_dict[k].shape}", flush=True)

        model.load_state_dict(matched_dict, strict=False)
        print(f"✅ Successfully loaded {len(matched_dict)} layers.", flush=True)

    except Exception as e:
        print(f"❌ Weight loading failed: {e}", flush=True)

    return model

# ====== 3. Hyperparameter Settings ======
DATA_ROOT = r"./data/Kvasir-SEG"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")

NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 4. Kvasir-SEG Dataset Definition ======
class KvasirSegDataset(Dataset):
    """Simple Kvasir-SEG dataset (grayscale, 1 channel)"""

    def __init__(self, image_dir: str, mask_dir: str, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"]
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(image_dir, ext)))
        image_paths = sorted(image_paths)

        self.samples = []
        for img_path in image_paths:
            name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, name)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        print(f"Found {len(self.samples)} samples in total.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(*self.image_size)
             .numpy())
        ).float() / 255.0
        mask = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
             .view(*self.image_size)
             .numpy())
        ).float() / 255.0

        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()

        return img, mask

# ====== 5. Dice Loss ======
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()

# ====== 6. [Core Modification] RGB to Grayscale Preprocessing ======
def convert_rgb_to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale
    Args:
        img: B x 3 x H x W or B x 1 x H x W
    Returns:
        B x 1 x H x W
    """

    if img.size(1) == 1:
        # Already grayscale
        return img
    elif img.size(1) == 3:
        # RGB -> Grayscale (standard weights)
        # Note: external loader already normalized the image, so we denormalize first
        # Standard ImageNet normalization:
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)

        img_denorm = img * std + mean

        grayscale = (
            0.299 * img_denorm[:, 0:1, :, :] +
            0.587 * img_denorm[:, 1:2, :, :] +
            0.114 * img_denorm[:, 2:3, :, :]
        )

        return grayscale
    else:
        raise ValueError(f"Unexpected number of channels: {img.size(1)}")

# ====== 7. [Modified] Training Function ======
def train_one_epoch(model: ScribblePromptUNet,
                    loader: DataLoader,
                    bce_loss_fn,
                    dice_loss_fn,
                    optimizer,
                    device: str):
    model.model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    steps = len(loader)

    for i, (img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)

        # [Key Modification] Convert to grayscale
        img = convert_rgb_to_grayscale(img)

        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        x = prepare_inputs(prompts, device=device)

        # [Debug Output] Check channel number
        if i == 0:
            print(f"[Debug] Input shape after prepare_inputs: {x.shape}", flush=True)

        logits = model.model(x)

        bce = bce_loss_fn(logits, mask)
        dice_loss = dice_loss_fn(logits, mask)
        loss = bce + dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_dice = 1.0 - dice_loss.item()

        total_loss += loss.item()
        total_dice += batch_dice
        total_batches += 1

        if (i + 1) % 10 == 0:
            print(f"   Step [{i + 1}/{steps}] Loss: {loss.item():.4f} Dice: {batch_dice:.4f}", flush=True)

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = total_dice / max(total_batches, 1)
    return avg_loss, avg_dice

# ====== 8. [Modified] Validation Function ======
@torch.no_grad()
def validate(model: ScribblePromptUNet,
             loader: DataLoader,
             bce_loss_fn,
             dice_loss_fn,
             device: str):
    if loader is None:
        return 0.0, 0.0

    model.model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        img = convert_rgb_to_grayscale(img)

        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        x = prepare_inputs(prompts, device=device)
        logits = model.model(x)

        bce = bce_loss_fn(logits, mask)
        dice_loss = dice_loss_fn(logits, mask)
        loss = bce + dice_loss

        batch_dice = 1.0 - dice_loss.item()

        total_loss += loss.item()
        total_dice += batch_dice
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = total_dice / max(total_batches, 1)
    return avg_loss, avg_dice

# ====== 9. Main Function ======
def main(args):
    print("==== Initializing Configuration ====", flush=True)

    batch_size = args.batch_size if args.batch_size else BATCH_SIZE

    val_loader = None

    if args.data_loader and args.image_folder and args.mask_folder:
        print("[GUI Mode] Using external data loader...", flush=True)
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            batch_size=batch_size,
            img_size=(128, 128)
        )
        print(f"[GUI Mode] Data loader created successfully. Batch size: {batch_size}", flush=True)
    else:
        print("[Local Mode] Using local Kvasir dataset...", flush=True)
        dataset = KvasirSegDataset(IMAGE_DIR, MASK_DIR, image_size=(128, 128))
        if len(dataset) > 0:
            val_len = int(len(dataset) * VAL_RATIO)
            train_len = len(dataset) - val_len
            train_set, val_set = random_split(dataset, [train_len, val_len])

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
            print(f"Training samples: {train_len}, Validation samples: {val_len}", flush=True)
        else:
            print("Local dataset is empty. Training aborted.", flush=True)
            return

    print("==== Initializing Model ====", flush=True)
    model = ScribblePromptUNet(version="v1", device=DEVICE)

    if args.pretrain:
        model.model = safe_load_weights(model.model, args.pretrain, DEVICE)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ScribblePrompt_unet_finetuned.pt")

    print(f"==== Start Training (Epochs: {NUM_EPOCHS}) ====", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, bce_loss_fn, dice_loss_fn, optimizer, DEVICE
        )

        log_str = f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}"

        if val_loader:
            val_loss, val_dice = validate(
                model, val_loader, bce_loss_fn, dice_loss_fn, DEVICE
            )
            log_str += f" || Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.model.state_dict(), save_path)
                log_str += f" [✓ Best]"
        else:
            gui_save_path = os.path.join(save_dir, f"scribble_epoch_{epoch}.pth")
            torch.save(model.model.state_dict(), gui_save_path)
            log_str += f" [Saved]"

        print(log_str, flush=True)

    print("✅ Training completed!", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str, help='Path to GUI-provided data loader script')
    parser.add_argument('--image_folder', type=str, help='GUI-provided training image folder')
    parser.add_argument('--mask_folder', type=str, help='GUI-provided training mask folder')
    parser.add_argument('--pretrain', type=str, default=None, help='GUI-provided pretrained weights path')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size')

    args = parser.parse_args()

    main(args)

