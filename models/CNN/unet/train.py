# ========== External data loader support (unified standard) ==========
import argparse
import importlib.util
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
VAL_SPLIT = 0.2
PATIENCE = 10
# ========= External data loader =========
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
                interpolation=InterpolationMode.NEAREST
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

        mask = (mask > 0).float()   # [1,H,W] float

        return image, mask

# ========= U-Net =========
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        return self.outc(x)

# ========= Mask normalization (BCE only) =========
def normalize_mask_for_bce(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.max() > 1:
        mask = mask / 255.0
    return mask.float()

# ========= Training / validation =========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0

    for images, masks in loader:
        images = images.to(device)
        masks = normalize_mask_for_bce(masks).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_dice += dice_score(preds, masks)

    return total_loss / len(loader), total_dice / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = normalize_mask_for_bce(masks).to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_dice += dice_score(preds, masks)

    return total_loss / len(loader), total_dice / len(loader)

def dice_score(pred, target):
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-6)

# ========= Main entry =========
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--mask_folder', type=str, default=None)
    args = parser.parse_args()

    dataset = SimpleDataset(DATA_DIR)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if args.data_loader:
        print('[Train] Using external data loader')
        train_loader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder, BATCH_SIZE
        )
    else:
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best = 0
    for epoch in range(EPOCHS):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_dice = validate_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1} | Train Dice {tr_dice:.4f} | Val Dice {va_dice:.4f}")

        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "unet_medical_best-k.pth"))

    print("Training finished")
