# ========== External data loader support (unified standard) ==========
import argparse
import importlib.util
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/viewpoint_aware_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from VANet import VANet

# ================= Configuration =================
DATA_DIR = r"./data/Kvasir-SEG"
PRETRAINED_PATH = r"./weights/CvT-13-224x224-IN-1k.pth"
CONFIG_PATH = r"./experiments/imagenet/cvt/cvt-13-224x224.yaml"

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
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

        self.images = [
            x for x in sorted(os.listdir(self.images_dir))
            if x.endswith('.jpg') or x.endswith('.png')
        ]

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

        mask = (mask > 0.5).float()   # [1,H,W]
        return image, mask

# ========= VANet =========
def get_vanet_model():
    print("Loading VANet model...")
    return VANet(
        num_class=1,
        cfg=CONFIG_PATH if os.path.exists(CONFIG_PATH) else None,
        weights=PRETRAINED_PATH if os.path.exists(PRETRAINED_PATH) else None
    )

# ========= Deep supervision loss =========
def structure_loss(preds, target):
    bce = nn.BCEWithLogitsLoss()
    loss = 0.0

    for pred in preds:
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        loss += bce(pred, target)

    return loss

# ========= Dice =========
def calculate_dice(pred_logits, target):
    pred_probs = torch.sigmoid(pred_logits)
    pred_mask = (pred_probs > 0.5).float()

    smooth = 1e-5
    inter = (pred_mask * target).sum()
    return (2 * inter + smooth) / (pred_mask.sum() + target.sum() + smooth)

# ========= BCE mask normalization =========
def normalize_mask_for_bce(mask):
    if mask.dim() == 3:          # [B,H,W]
        mask = mask.unsqueeze(1)
    if mask.max() > 1:
        mask = mask / 255.0
    return mask.float()

# ========= Main entry =========
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='VANet Medical Image Segmentation Training'
    )
    parser.add_argument('--data_loader', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--mask_folder', type=str, default=None)
    args = parser.parse_args()

    # ===== Data loading =====
    if args.data_loader and args.image_folder and args.mask_folder:
        print("[Train] Using external data loader")
        dataloader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            BATCH_SIZE
        )
    else:
        dataset = SimpleDataset(DATA_DIR)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    model = get_vanet_model().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    print(f"Start training, device: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        dice_sum = 0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = normalize_mask_for_bce(masks).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)      # list of logits

            loss = structure_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            final_pred = outputs[0]
            if final_pred.shape[2:] != masks.shape[2:]:
                final_pred = F.interpolate(
                    final_pred,
                    size=masks.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            dice_sum += calculate_dice(final_pred, masks).item()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {loss_sum/len(dataloader):.4f} "
            f"Dice: {dice_sum/len(dataloader):.4f} "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vanet_medical_model.pth"))
    print("Training completed, model saved ✅")
