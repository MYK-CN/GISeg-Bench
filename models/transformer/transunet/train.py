# ========== External data loader support (unified interface) ==========
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

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transunet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from vit_seg_modeling import VisionTransformer
from vit_seg_configs import get_r50_b16_config

# ================= Configuration =================
DATA_DIR = r'./data/Kvasir-SEG'
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. Image and mask transforms (kept exactly as designed)
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transforms = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.NEAREST
    ),
    transforms.ToTensor()
])

# 2. Dataset (kept unchanged)
class MedicalDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.images = sorted(os.listdir(self.images_dir))

        self.image_transform = image_transforms
        self.mask_transform = mask_transforms

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

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).long().squeeze(0)
        return image, mask

# 3. Vision Transformer model
def get_vit_model():
    print("Loading Vision Transformer segmentation model...")
    config = get_r50_b16_config()
    model = VisionTransformer(
        config=config,
        img_size=256,
        num_classes=2
    )
    return model

# 4. Dice calculation
def calculate_dice(pred, target):
    pred_mask = torch.argmax(pred, dim=1)
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

# 5. Main training entry
if __name__ == '__main__':

    # ===== Argument parsing (added, unified interface) =====
    parser = argparse.ArgumentParser(
        description='Vision Transformer Medical Segmentation Training'
    )
    parser.add_argument('--data_loader', type=str, default=None, help='External data loader path')
    parser.add_argument('--image_folder', type=str, default=None, help='Image folder path')
    parser.add_argument('--mask_folder', type=str, default=None, help='Mask folder path')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    # ===== Data loading (automatic switch) =====
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
            print('[Train] Using built-in MedicalDataset')
            dataset = MedicalDataset(DATA_DIR)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )
            print(f"Dataset loaded successfully, total {len(dataset)} images.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit()
    # Initialize model
    model = get_vit_model().to(DEVICE)

    # Load pretrained weights (if provided)
    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        model.load_state_dict(
            torch.load(args.pretrain, map_location=DEVICE)
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training, device: {DEVICE}, batch size: {args.batch_size}")

    # ===== Training loop (kept unchanged) =====
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        dice_sum = 0.0

        for images, masks in dataloader:
            if masks.dim() == 4:   # [B,1,H,W]
                masks = masks.squeeze(1)
            masks = masks.long()
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            dice_sum += calculate_dice(outputs.detach(), masks.detach())

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Loss: {loss_sum / len(dataloader):.4f} "
            f"- Dice: {dice_sum / len(dataloader):.4f}"
        )

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vit_medical_model.pth"))
    print("Training completed! Model saved as vit_medical_model.pth")
