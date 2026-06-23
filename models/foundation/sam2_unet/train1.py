import os
import random
from typing import Tuple

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/sam2_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
# 1. 导入你的 SAM2UNet
# !!! 把 sam2_unet_model 改成你自己保存 SAM2UNet 的文件名 !!!
from SAM2UNet import SAM2UNet
# 2. 路径和超参数
SAM2_CHECKPOINT = r"./pretrained_ckpt/sam2.1_hiera_large.pt"
KVASIR_ROOT = r"./data/Kvasir-SEG"

TRAIN_IMG_DIR = os.path.join(KVASIR_ROOT, "images")
TRAIN_MASK_DIR = os.path.join(KVASIR_ROOT, "masks")
VAL_IMG_DIR = os.path.join(KVASIR_ROOT, "val")
VAL_MASK_DIR = os.path.join(KVASIR_ROOT, "maskval")

SAVE_PATH = "outputs/foundation/sam2_unet"

IMG_SIZE = (352, 352)  # 和你模型 forward 里的 352x352 对齐
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
SEED = 42
# 3. 工具函数
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(pred, target, eps=1e-7):
    """
    pred: logits, shape [B, 1, H, W]
    target: 0/1 float mask, shape [B, 1, H, W]
    """
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = num / den
    return 1 - dice.mean()

def compute_dice_iou(pred, target, threshold=0.5, eps=1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 batch 内平均 dice, iou
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * target).sum(dim=(2, 3))
    union = (pred_bin + target - pred_bin * target).sum(dim=(2, 3)) + eps
    dice = 2 * inter / (pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)
    iou = inter / union
    return dice.mean(), iou.mean()
# 4. 数据集定义
class KvasirPolypDataset(Dataset):
    """
    假设：
    - images 目录下是 RGB 图片
    - masks / maskval 中是对应的单通道掩码（前景为白/非零）
    - 文件名一一对应
    """

    def __init__(self, img_dir, mask_dir, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train

        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))
        # 只保留两边都存在的文件名
        img_set = {os.path.splitext(f)[0] for f in img_files}
        mask_set = {os.path.splitext(f)[0] for f in mask_files}
        common = sorted(img_set & mask_set)
        self.ids = common
        if len(self.ids) == 0:
            raise RuntimeError(f"No matching images and masks in {img_dir} and {mask_dir}")

        print(f"{'Train' if is_train else 'Val'} dataset: {len(self.ids)} samples")

    def __len__(self):
        return len(self.ids)

    def _load_pair(self, idx):
        id_ = self.ids[idx]
        # 找到实际文件（兼容 jpg/png 等）
        def find_file(root, stem):
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                p = os.path.join(root, stem + ext)
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"{stem} not found in {root}")

        img_path = find_file(self.img_dir, id_)
        mask_path = find_file(self.mask_dir, id_)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_pair(idx)

        # 统一 resize 到 352x352
        img = img.resize(IMG_SIZE, resample=Image.BILINEAR)
        mask = mask.resize(IMG_SIZE, resample=Image.NEAREST)

        # 转成 tensor 前的数据增强（图像和 mask 必须同步）
        if self.is_train:
            # 随机水平翻转
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # 随机垂直翻转
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            # 随机轻微旋转
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)

        # 转 tensor & 归一化
        img = TF.to_tensor(img)  # [0,1]
        # 简单归一化（也可以改成 imagenet 均值/方差）
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        mask = TF.to_tensor(mask)  # [1,H,W], 0~1
        mask = (mask > 0.5).float()  # 二值化

        return img, mask
# 5. 训练 & 验证
def train_one_epoch(model, loader, optimizer, bce_weight=0.5, device="cuda"):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        out, out1, out2 = model(imgs)
        # 确保输出和 mask 尺寸一致（保险起见，再插值一次）
        def resize_to_mask(x):
            if x.shape[2:] != masks.shape[2:]:
                x = F.interpolate(x, size=masks.shape[2:], mode="bilinear", align_corners=False)
            return x

        out = resize_to_mask(out)
        out1 = resize_to_mask(out1)
        out2 = resize_to_mask(out2)

        # 三个输出做深监督
        loss_main = criterion_bce(out, masks) + dice_loss(out, masks)
        loss_side1 = criterion_bce(out1, masks) + dice_loss(out1, masks)
        loss_side2 = criterion_bce(out2, masks) + dice_loss(out2, masks)
        loss = loss_main + 0.5 * (loss_side1 + loss_side2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        dice, iou = compute_dice_iou(out, masks)
        total_dice += dice.item() * imgs.size(0)
        total_iou += iou.item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n

@torch.no_grad()
def validate(model, loader, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out, out1, out2 = model(imgs)

        def resize_to_mask(x):
            if x.shape[2:] != masks.shape[2:]:
                x = F.interpolate(x, size=masks.shape[2:], mode="bilinear", align_corners=False)
            return x

        out = resize_to_mask(out)
        out1 = resize_to_mask(out1)
        out2 = resize_to_mask(out2)

        loss_main = criterion_bce(out, masks) + dice_loss(out, masks)
        loss_side1 = criterion_bce(out1, masks) + dice_loss(out1, masks)
        loss_side2 = criterion_bce(out2, masks) + dice_loss(out2, masks)
        loss = loss_main + 0.5 * (loss_side1 + loss_side2)

        total_loss += loss.item() * imgs.size(0)
        dice, iou = compute_dice_iou(out, masks)
        total_dice += dice.item() * imgs.size(0)
        total_iou += iou.item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n
# 6. 主函数：初始化 & 开始训练
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 数据集 & DataLoader
    train_dataset = KvasirPolypDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, is_train=True)
    val_dataset = KvasirPolypDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 模型
    print("Building SAM2UNet...")
    model = SAM2UNet(checkpoint_path=SAM2_CHECKPOINT)
    model = model.to(device)

    # 只训练可学习参数（encoder 冻结在你模型里已经处理过）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    best_dice = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, device=device)
        val_loss, val_dice, val_iou = validate(model, val_loader, device=device)

        print(f"Train: loss={train_loss:.4f}, dice={train_dice:.4f}, iou={train_iou:.4f}")
        print(f"Val  : loss={val_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}")

        # 保存最佳模型（按验证 Dice）
        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  >>> New best dice={best_dice:.4f}, saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
