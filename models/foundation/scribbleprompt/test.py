import os
from glob import glob
from typing import List, Tuple

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/scribbleprompt')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ====== 1. 根据你的工程结构导入 ======
from scribbleprompt.models.unet import ScribblePromptUNet, prepare_inputs

# ====== 2. 路径与参数 ======
TEST_ROOT = r"./data/Kvasir-SEG"
TEST_IMAGE_DIR = os.path.join(TEST_ROOT, "test")
TEST_MASK_DIR = os.path.join(TEST_ROOT, "masktest")

WEIGHT_PATH = r"./pretrained_ckpt/ScribblePrompt_unet_finetuned_kvasir.pt"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 3. 测试数据集 ======
class KvasirTestDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size=(128, 128)):
        super().__init__()
        self.image_size = image_size

        exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"]
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(image_dir, ext)))
        image_paths = sorted(image_paths)

        self.samples = []
        for img_path in image_paths:
            name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, name)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("❌ 测试集为空，请检查 test / masktest 路径")

        print(f"[INFO] 测试样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(*self.image_size)
             .numpy())
        ).float() / 255.0

        mask = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
             .view(*self.image_size)
             .numpy())
        ).float() / 255.0

        img = img.unsqueeze(0)
        mask = (mask > 0.5).float().unsqueeze(0)

        return img, mask

# ====== 4. 指标函数 ======
def dice_iou_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter

    dice = (2 * inter + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()

# ====== 5. 主测试流程 ======
@torch.no_grad()
def main():
    print(f"[INFO] 使用设备: {DEVICE}")

    # Dataset & Loader
    test_dataset = KvasirTestDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    print("[INFO] 初始化 ScribblePromptUNet...")
    model = ScribblePromptUNet(version="v1", device=DEVICE)

    print("[INFO] 加载微调权重...")
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.model.load_state_dict(state_dict, strict=True)

    model.model.to(DEVICE)
    model.model.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for img, mask in test_loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        x = prepare_inputs(prompts, device=DEVICE)
        logits = model.model(x)

        loss = bce_loss_fn(logits, mask)
        dice, iou = dice_iou_from_logits(logits, mask)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(test_loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
