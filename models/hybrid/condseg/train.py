import os
import sys
import argparse
import importlib.util
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from network.model import ConDSeg  # Your model file

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/hybrid/condseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== [NEW] Dice & IoU Metric Utilities ==================
def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice coefficient for binary segmentation
    pred, target: (B, 1, H, W)
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1e-6):
    """
    IoU score for binary segmentation
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()
# ================== [NEW] END ==================

# ================== [NEW] Universal Data Loader Utility Function ==================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=8):
    """Dynamically load DataLoader from an external script"""

    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        num_workers=0
    )

# ================== Acceleration-related Settings ==================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== Image & Mask Preprocessing (Unchanged) ==================
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ================== Original Dataset Definition (Fallback Retained) ==================
class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()
        return image, mask

# ================== Main Entry ==================
if __name__ == '__main__':
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Training')
    parser.add_argument('--data_loader', type=str, help='Data loader path passed from GUI')
    parser.add_argument('--image_folder', type=str, help='Image folder passed from GUI')
    parser.add_argument('--mask_folder', type=str, help='Mask folder passed from GUI')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights passed from GUI')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    # 2. Initialize Model
    print("Initializing model...")
    model = ConDSeg().to(device)

    # 3. Load Pretrained Weights
    if args.pretrain and os.path.exists(args.pretrain):
        print(f"Loading pretrain weights from: {args.pretrain}")
        try:
            state_dict = torch.load(args.pretrain, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Pretrain weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")

    # 4. Loss & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Data Loader Logic
    if args.data_loader and args.image_folder and args.mask_folder:
        print("[GUI Mode] Using external data loading logic...")
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            batch_size=args.batch_size
        )
    else:
        print("[Local Mode] Using hardcoded paths inside script...")
        image_dir = r"./data/Kvasir-SEG/images"
        mask_dir = r"./data/Kvasir-SEG/masks"

        dataset = KvasirSegDataset(
            image_dir,
            mask_dir,
            image_transform=image_transform,
            mask_transform=mask_transform
        )
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    # 6. Training Loop
    num_epochs = 30
    print(f"Start training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # ================== [NEW] Metric Accumulators ==================
        epoch_dice = 0.0
        epoch_iou = 0.0
        # ================== [NEW] END ==================

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            file=sys.stdout
        )

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs, mask_fg, mask_bg, mask_uc = model(images)

            loss_main = criterion(outputs.float(), masks.float())

            scaler.scale(loss_main).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_main.item()

            # ================== [NEW] Dice & IoU Calculation ==================
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)

            epoch_dice += dice.item()
            epoch_iou += iou.item()
            # ================== [NEW] END ==================

            pbar.set_postfix(
                loss=loss_main.item(),
                dice=f"{dice.item():.4f}",   # [NEW]
                iou=f"{iou.item():.4f}"      # [NEW]
            )

        epoch_loss = running_loss / len(train_loader)

        # ================== [NEW] Epoch-level Metrics ==================
        avg_dice = epoch_dice / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        # ================== [NEW] END ==================

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Dice: {avg_dice:.4f} | "
            f"IoU: {avg_iou:.4f}"
        )

        # Save model
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"con_dseg_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print("Model saved to:", save_path)

    print("Training complete!")
