import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== PraNet ==========
from PraNet_ResNet import CRANet   # 必须与你训练时一致
# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMAGE_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./weights/pranet_model.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
# 测试数据集（stem 匹配，彻底避免 _result）
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, TEST_IMAGE_DIR)
        self.mask_dir = os.path.join(root_dir, TEST_MASK_DIR)

        exts = (".png", ".jpg", ".jpeg")

        # 建立 mask 索引
        mask_index = {}
        for f in os.listdir(self.mask_dir):
            if f.lower().endswith(exts):
                mask_index[os.path.splitext(f)[0]] = f

        self.samples = []
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            if stem in mask_index:
                self.samples.append(
                    (
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem])
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError("❌ test / masktest 中没有匹配的 image-mask 对")

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

        self.img_tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
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

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_tf(image)
        mask = self.mask_tf(mask)

        mask = (mask > 0).long().squeeze(0)
        return image, mask
# Dice / IoU（hard）
def dice_iou(pred, target, eps=1e-6):
    pred_mask = torch.argmax(pred, dim=1)

    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    inter = (pred_flat * target_flat).sum()

    dice = (2 * inter + eps) / (pred_flat.sum() + target_flat.sum() + eps)
    iou = (inter + eps) / (pred_flat.sum() + target_flat.sum() - inter + eps)

    return dice.item(), iou.item()
# 测试主流程
@torch.no_grad()
def main():
    dataset = KvasirTestDataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------- Model --------
    model = CRANet().to(DEVICE)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    print("===== Start Testing =====")
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # PraNet forward
        outputs = model(images)

        # PraNet 返回多个输出 (r1, r2, r3, r4)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            pred = outputs[0]
        else:
            pred = outputs

        # 如果是 1 通道，升为 2 通道（与你训练一致）
        if pred.shape[1] == 1:
            pred = torch.cat([1 - pred, pred], dim=1)

        loss = criterion(pred, masks)
        dice, iou = dice_iou(pred, masks)

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
