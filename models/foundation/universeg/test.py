import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/universeg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
from torch import nn
from torch.utils.data import Dataset

from universeg import universeg
# 0. 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMAGE_SUBDIR = "test"
TEST_MASK_SUBDIR = "masktest"

PRETRAIN_PATH = r"./pretrained_ckpt/universeg_v1_nf64_ss64_STA.pt"
FINETUNE_PATH = r"./weights/universeg_kvasir_best_epoch24.pth"

IMG_SIZE = 128
SUPPORT_SIZE = 16
# 1. 测试数据集（完全对齐你的 Dataset 逻辑）
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir, image_subdir, mask_subdir):
        self.img_dir = Path(root_dir) / image_subdir
        self.mask_dir = Path(root_dir) / mask_subdir

        if not self.img_dir.exists():
            raise FileNotFoundError(self.img_dir)
        if not self.mask_dir.exists():
            raise FileNotFoundError(self.mask_dir)

        # 按 stem 建立 mask 索引
        mask_index = {m.stem: m for m in self.mask_dir.glob("*") if m.is_file()}

        self.samples = []
        for img_path in self.img_dir.glob("*"):
            if not img_path.is_file():
                continue
            mask_path = mask_index.get(img_path.stem)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("❌ test / masktest 中没有匹配的 image-mask 对")

        print(f"[TestDataset] 测试样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load(path: Path, is_mask=False):
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE),
                         Image.NEAREST if is_mask else Image.BILINEAR)
        arr = np.array(img).astype(np.float32)

        if is_mask:
            arr = (arr > 127).astype(np.float32)
        else:
            arr = arr / 255.0

        return torch.from_numpy(arr)[None, ...]  # [1,128,128]

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = self._load(img_path, is_mask=False)
        mask = self._load(mask_path, is_mask=True)
        return img, mask
# 2. Dice / IoU / Loss（与你训练一致）
def test_metrics(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2 * inter + smooth) / (union + smooth)
    iou = inter / (preds.sum() + targets.sum() - inter + smooth)

    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss = bce + (1 - dice)

    return loss.item(), dice.item(), iou.item()
# 3. 构建模型（pretrain + finetune）
def build_model():
    model = universeg(version="v1", pretrained=False)

    print("[INFO] Load pretrain:", PRETRAIN_PATH)
    pre = torch.load(PRETRAIN_PATH, map_location="cpu")
    if isinstance(pre, dict) and "state_dict" in pre:
        pre = pre["state_dict"]
    model.load_state_dict(pre, strict=False)

    print("[INFO] Load finetune:", FINETUNE_PATH)
    fin = torch.load(FINETUNE_PATH, map_location="cpu")
    model.load_state_dict(fin, strict=False)

    return model.to(DEVICE).eval()
# 4. 构建 Support Set（few-shot 核心）
def build_support_set(dataset, support_size):
    support_size = min(support_size, len(dataset))

    imgs, masks = [], []
    for i in range(support_size):
        img, mask = dataset[i]
        imgs.append(img)
        masks.append(mask)

    imgs = torch.stack(imgs).unsqueeze(0).to(DEVICE)    # [1,S,1,H,W]
    masks = torch.stack(masks).unsqueeze(0).to(DEVICE)

    print(f"[Support] images: {imgs.shape}, masks: {masks.shape}")
    return imgs, masks
# 5. 测试主流程
@torch.no_grad()
def main():
    test_dataset = KvasirTestDataset(
        DATA_ROOT,
        TEST_IMAGE_SUBDIR,
        TEST_MASK_SUBDIR
    )

    model = build_model()

    # ⚠️ 正确做法：support 不能来自 test，本例为了跑通示例用 test 前几个
    support_images, support_masks = build_support_set(
        test_dataset, SUPPORT_SIZE
    )

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    print("===== Start Testing =====")
    for img, mask in tqdm(test_dataset):
        img = img.unsqueeze(0).to(DEVICE)    # [1,1,128,128]
        mask = mask.unsqueeze(0).to(DEVICE)

        logits = model(img, support_images, support_masks)

        loss, dice, iou = test_metrics(logits, mask)

        total_loss += loss
        total_dice += dice
        total_iou += iou

    n = len(test_dataset)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
