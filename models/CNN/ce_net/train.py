import argparse
import importlib.util

import argparse
import importlib.util
import os
import sys

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/ce_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= Path configuration (robust) =================
current_dir = os.path.dirname(os.path.abspath(__file__))

# networks/
sys.path.insert(0, current_dir)
# models/
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))
# lib/
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "..")))
# src/
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "..", "..")))

# ================= Safe import CE-Net =================
try:
    # Package mode
    from .cenet import CE_Net_
except ImportError:
    # Script mode
    from cenet import CE_Net_

# ================= Torch imports =================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ================= Path configuration (prevent CE-Net module not found) =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(current_dir, "..", ".."))
sys.path.append(os.path.join(current_dir, "..", "..", ".."))
# try:
#     from .cenet import CE_Net_
# except ImportError:
#     from cenet import CE_Net_
#Try to import the model
# try:
#
#     from .cenet import CE_Net_
# except ImportError:
#     # Fallback: try importing from an alternative path
#     try:
#         from src.lib.models.networks.cenet import CE_Net_
#     except ImportError:
#         print("[Error] Unable to find CE_Net_ model definition. Please check file paths.")

# ================= 1. External DataLoader support =================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=8):
    """Dynamically load an external DataLoader"""
    spec = importlib.util.spec_from_file_location(
        "universal_data_loader", data_loader_path
    )
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    print(f"[Info] External Loader: Force img_size=256, num_workers=0")

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=256,  # CE-Net requires a fixed input size
        num_workers=0,   # Fix Windows multiprocessing errors
        shuffle=True
    )

# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)

# ================= 2. Built-in dataset (kept) =================
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        self.images = [x for x in sorted(os.listdir(self.images_dir))
                       if x.endswith('.jpg') or x.endswith('.png')]

        self.img_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        image = self.img_trans(image)
        mask = self.mask_trans(mask)

        # Built-in dataset already squeezes, so mask shape is [H, W]
        mask = (mask > 0).long().squeeze(0)
        return image, mask

# ================= 3. Helper functions =================
def get_cenet_model():
    print("Loading CE-Net segmentation model...")
    # num_classes=2 means background + foreground
    return CE_Net_(num_classes=2)

def calculate_dice(pred, target):
    # pred: [B, 2, H, W] -> argmax -> [B, H, W]
    pred_mask = torch.argmax(pred, dim=1)

    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# ================= 4. Main training entry =================
if __name__ == '__main__':

    # ====== Argument parsing ======
    parser = argparse.ArgumentParser(description='CE-Net Medical Segmentation Training')
    parser.add_argument('--data_loader', type=str, default=None, help='External DataLoader path')
    parser.add_argument('--image_folder', type=str, default=None, help='Image folder path')
    parser.add_argument('--mask_folder', type=str, default=None, help='Mask folder path')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')

    # Parse known arguments to avoid errors from unknown ones
    args, unknown = parser.parse_known_args()

    # ====== Data loading ======
    try:
        if args.data_loader and args.image_folder and args.mask_folder:
            print('[Train] Using external DataLoader (GUI Mode)')
            dataloader = load_external_dataloader(
                args.data_loader,
                args.image_folder,
                args.mask_folder,
                args.batch_size
            )
        else:
            print('[Train] Using built-in SimpleDataset (Local Mode)')
            if not os.path.exists(DATA_DIR):
                print(f"[Error] Data path does not exist: {DATA_DIR}")
                exit()
            dataset = SimpleDataset(DATA_DIR)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2
            )
            print(f"Dataset loaded successfully, total {len(dataset)} images.")
    except Exception as e:
        print("Data loading failed:", e)
        exit()

    # Initialize model
    model = get_cenet_model().to(DEVICE)

    # Load pretrained weights
    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        if os.path.exists(args.pretrain):
            model.load_state_dict(torch.load(args.pretrain, map_location=DEVICE))
        else:
            print(f"[Warn] Pretrained weights file not found: {args.pretrain}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training, device: {DEVICE}, batch size: {args.batch_size}")

    # ====== Training loop ======
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for i, (images, masks) in enumerate(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # ================= [Core Fix Start] =================
            # 1. Dimension fix: [B, 1, H, W] -> [B, H, W]
            # CrossEntropyLoss does not accept a channel dimension
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            # 2. Type fix: must be Long (int64)
            if masks.dtype != torch.long:
                masks = masks.long()
            # ================= [Core Fix End] =================

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)

            # Print every 10 steps to monitor progress
            if (i + 1) % 10 == 0:
                print(f"  Step [{i + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(len(dataloader), 1)
        epoch_dice = running_dice / max(len(dataloader), 1)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f}")

    # Save model
    save_path = "cenet_medical_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished! Model saved as {save_path}")

