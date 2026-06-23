import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transunet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from vit_seg_modeling import VisionTransformer
from vit_seg_configs import get_r50_b16_config
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMG_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./weights/vit_medical_model.pth"

BATCH_SIZE = 4
IMAGE_SIZE = (256, 256)
# Dataset (严格 stem 匹配，不加 _result)
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, TEST_IMG_DIR)
        self.mask_dir = os.path.join(root_dir, TEST_MASK_DIR)

        exts = (".png", ".jpg", ".jpeg")

        mask_index = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith(exts)
        }

        self.samples = []
        for f in os.listdir(self.img_dir):
            if f.lower().endswith(exts):
                stem = os.path.splitext(f)[0]
                if stem in mask_index:
                    self.samples.append((
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem])
                    ))

        print(f"[INFO] Test samples: {len(self.samples)}")

        self.img_tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = self.img_tf(Image.open(img_path).convert("RGB"))
        mask = self.mask_tf(Image.open(mask_path).convert("L"))
        mask = (mask > 0.5).long().squeeze(0)

        return image, mask
# Model
def get_vit_model():
    config = get_r50_b16_config()
    model = VisionTransformer(
        config=config,
        img_size=256,
        num_classes=2
    )
    return model
# Metrics
def dice_iou(pred, target, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    inter = (pred * target).sum()
    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    iou = (inter + eps) / (pred.sum() + target.sum() - inter + eps)
    return dice.item(), iou.item()
# Test
@torch.no_grad()
def main():
    dataset = KvasirTestDataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = get_vit_model().to(DEVICE)
    print(f"[INFO] Loading weights: {WEIGHT_PATH}")
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    print("========== Start Testing ==========")
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        dice, iou = dice_iou(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
