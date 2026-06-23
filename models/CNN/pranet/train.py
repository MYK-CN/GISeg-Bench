import argparse
import importlib.util

def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=8):
    """Dynamically load an external data loader"""
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
    )
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Import PraNet model
from PraNet_ResNet import CRANet

# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. Dataset (kept unchanged)
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = [
            x for x in sorted(os.listdir(self.images_dir))
            if x.endswith('.jpg') or x.endswith('.png')
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = Image.open(
            os.path.join(self.images_dir, img_name)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.masks_dir, img_name)
        ).convert("L")

        # Image transform
        img_t = transforms.Resize((256, 256))(image)
        img_t = transforms.ToTensor()(img_t)
        img_t = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )(img_t)

        # Mask transform
        mask_t = transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.NEAREST
        )(mask)
        mask_t = transforms.ToTensor()(mask_t)
        mask_t = (mask_t > 0).long().squeeze(0)

        return img_t, mask_t

# 2. PraNet model
def get_pranet_model():
    print("Loading PraNet-ResNet (CRANet) model...")
    model = CRANet()
    return model

# 3. Dice calculation
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 4. Main training entry
if __name__ == '__main__':

    # ===== Argument parsing (added) =====
    parser = argparse.ArgumentParser(
        description='PraNet Medical Image Segmentation Training'
    )
    parser.add_argument('--data_loader', type=str, default=None, help='External data loader path')
    parser.add_argument('--image_folder', type=str, default=None, help='Image folder path')
    parser.add_argument('--mask_folder', type=str, default=None, help='Mask folder path')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    # ===== Data loading (auto switch) =====
    try:
        if args.data_loader and args.image_folder and args.mask_folder:
            print('[Train] Using external data loader')
            dataloader = load_external_dataloader(
                args.data_loader,
                args.image_folder,
                args.mask_folder,
                args.batch_size
            )
        else:
            print('[Train] Using built-in SimpleDataset')
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
    model = get_pranet_model().to(DEVICE)

    # Load pretrained weights (if provided)
    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        model.load_state_dict(
            torch.load(args.pretrain, map_location=DEVICE)
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training, device: {DEVICE}, batch size: {args.batch_size}")

    # =================== Training ===================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            if masks.dim() == 4:   # [B,1,H,W]
                masks = masks.squeeze(1)
            masks = masks.long()
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs

            # Single-channel → two-channel (for CrossEntropyLoss compatibility)
            if pred.shape[1] == 1:
                pred = torch.cat([1 - pred, pred], dim=1)

            loss = criterion(pred, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += calculate_dice(pred, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Loss: {epoch_loss:.4f} "
            f"- Dice: {epoch_dice:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pranet_model.pth"))
    print("Training completed! Model saved as pranet_model.pth")
