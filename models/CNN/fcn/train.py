import argparse
import importlib.util
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/fcn')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
# ========= External data loader =========
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=8):
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

# ========= Built-in dataset =========
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        self.images = sorted([
            x for x in os.listdir(self.images_dir)
            if x.endswith('.jpg') or x.endswith('.png')
        ])

        self.img_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.mask_trans = transforms.Compose([
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

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

        image = self.img_trans(image)
        mask = self.mask_trans(mask)

        mask = (mask > 0).long().squeeze(0)  # [H, W]
        return image, mask

# ========= FCN model =========
def get_fcn_model():
    print("Loading FCN-ResNet50 (ImageNet pretrained)...")
    model = models.segmentation.fcn_resnet50(weights='DEFAULT')

    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    model.aux_classifier = None

    return model

# ========= Dice =========
def calculate_dice(pred, target):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

# ========= Main entry =========
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='FCN Medical Image Segmentation Training'
    )
    parser.add_argument('--data_loader', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--mask_folder', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # ===== Data loading =====
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
            num_workers=2,
            pin_memory=True
        )

    # ===== Model =====
    model = get_fcn_model().to(DEVICE)

    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        model.load_state_dict(
            torch.load(args.pretrain, map_location=DEVICE)
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training | Device: {DEVICE}")

    # ===== Training loop =====
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)

            # ✅ Core fix (shared by FCN / DenseNet / DenseUNet)
            if masks.dim() == 4:   # [B,1,H,W]
                masks = masks.squeeze(1)
            masks = masks.long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += calculate_dice(outputs, masks)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {total_loss/len(dataloader):.4f} "
            f"Dice: {total_dice/len(dataloader):.4f}"
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fcn_medical_model.pth"))
    print("Training finished, model saved ✅")
