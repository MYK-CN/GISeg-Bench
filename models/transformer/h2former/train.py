import os
import sys
import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/h2former')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 0. [NEW] Universal Data Loader Utility =================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=16):
    """Dynamically load DataLoader from an external script"""
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    # H2Former input size is 224x224
    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=224,
        num_workers=0
    )

# ================= 1. Path Configuration =================
# Keep original path settings
sys.path.append(r"./models")

# Default local paths (fallback)
DEFAULT_IMAGES_DIR = r"./data/Kvasir-SEG/images"
DEFAULT_MASKS_DIR = r"./data/Kvasir-SEG/masks"
DEFAULT_WEIGHT_PATH = r"./pretrained_ckpt/resnet34-333f7ec4.pth"

# Try importing the model
try:
    from models.H2Former import Res34_Swin_MS, BasicBlock
except ImportError:
    # Compatibility handling: try importing directly from current directory
    try:
        from H2Former import Res34_Swin_MS, BasicBlock
    except ImportError:
        print("Error: Cannot find models.H2Former, please check sys.path settings")
        raise

# ================= 2. Dice Metric =================
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

# ================= 3. IoU Metric =================
def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

# ================= 4. Smart Weight Loading (Compatible with 3-channel & 4-channel) =================
def load_resnet34_weights(model, weight_path):
    if not os.path.exists(weight_path):
        print(f"[Warn] Weight file not found: {weight_path}")
        return model

    print(f"Loading weights from: {weight_path}")

    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return model

    # Handle checkpoint containing 'state_dict'
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_dict = model.state_dict()
    new_dict = {}

    for k, v in checkpoint.items():
        # Remove possible 'module.' prefix
        if k.startswith('module.'):
            k = k[7:]

        if k in model_dict:
            # Special handling for conv1.weight
            if k == "conv1.weight":
                if v.shape == model_dict[k].shape:
                    # Case 1: Shape fully matches (already trained H2Former weights)
                    print(f" -> Loading {k} directly (4-channel match)")
                    new_dict[k] = v
                else:
                    # Case 2: Shape mismatch (standard ResNet34 3-channel weights)
                    print(f" -> Adapting {k} from 3-channels to 4-channels")
                    new_weight = torch.zeros_like(model_dict[k])
                    # Copy first 3 channels
                    new_weight[:, :3, :, :] = v
                    # Initialize the 4th channel with small random noise
                    new_weight[:, 3:, :, :] = torch.randn_like(new_weight[:, 3:, :, :]) * 0.01
                    new_dict[k] = new_weight

            elif v.shape == model_dict[k].shape:
                new_dict[k] = v

    model.load_state_dict(new_dict, strict=False)
    print("Weights loaded successfully.")
    return model

# ================= 5. Local Dataset Class (Fallback) =================
class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

# ================= 6. Main Training Pipeline =================
if __name__ == "__main__":

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str, help='Data loader path passed from GUI')
    parser.add_argument('--image_folder', type=str, help='Image folder passed from GUI')
    parser.add_argument('--mask_folder', type=str, help='Mask folder passed from GUI')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights passed from GUI')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    # Configuration
    image_size = 224
    epochs = 10
    lr = 1e-4

    # Use batch_size from GUI if provided, otherwise default to 16
    batch_size = args.batch_size if args.batch_size else 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # -------- Data Loading Branch --------
    if args.data_loader and args.image_folder and args.mask_folder:
        # === GUI Mode ===
        print("[GUI Mode] Using universal data loader...", flush=True)
        dataloader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            batch_size=batch_size
        )
    else:
        # === Local Mode ===
        print("[Local Mode] Using hardcoded paths...", flush=True)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        dataset = PolypDataset(DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # -------- Model Initialization --------
    model = Res34_Swin_MS(
        image_size=image_size,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=1
    )

    # -------- Weight Loading --------
    # Use GUI-provided weight path if available, otherwise default
    target_weight = args.pretrain if (args.pretrain and os.path.exists(args.pretrain)) else DEFAULT_WEIGHT_PATH

    if target_weight:
        model = load_resnet34_weights(model, target_weight)

    model.to(device)

    # -------- Loss & Optimizer --------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n===== Start Training =====\n", flush=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0

        # file=sys.stdout ensures GUI can capture progress bar output
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]", file=sys.stdout)

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # === H2Former-specific logic: 3-channel -> 4-channel ===
            # Compute RGB mean as the 4th channel (Input Adaptation)
            extra_channel = imgs.mean(dim=1, keepdim=True)
            imgs = torch.cat([imgs, extra_channel], dim=1)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # ================== [NEW] Dice & IoU Calculation ==================
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "iou": f"{iou:.4f}"
            })

        epoch_loss /= len(dataloader)
        epoch_dice /= len(dataloader)
        epoch_iou /= len(dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {epoch_loss:.4f} | "
            f"Dice: {epoch_dice:.4f} | "
            f"IoU: {epoch_iou:.4f}",
            flush=True
        )

        # -------- Save Model --------
        # Save a checkpoint every epoch so the GUI can stop anytime safely
        save_path = os.path.join(OUTPUT_DIR, f"h2former_kvasir_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}", flush=True)

    print("\nTraining Finished!", flush=True)
