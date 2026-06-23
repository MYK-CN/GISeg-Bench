import os
import random

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/daeformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.DAEFormer import DAEFormer   # 确保这里能导入到你那份 DAEFormer 模型代码
# 配置区域
DATA_ROOT = r"./data/Kvasir-SEG"

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "masks")

VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "maskval")

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "masktest")

PRETRAIN_PATH = r"./pretrained_ckpt/synapse_epoch_399.pth"
SAVE_DIR = r"./outputs/daeformer"

IMG_SIZE = 224
NUM_CLASSES = 2          # 背景 + 息肉
BATCH_SIZE = 4
NUM_EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 0          # ★★ 已改为 0，避免 Windows 多进程问题 ★★
SEED = 42
# 工具函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class KvasirSegDataset(Dataset):
    """
    通用数据集类：
      img_dir  : 图像文件夹
      mask_dir : masks 文件夹
    假设同名文件一一对应：xxx.png / xxx.png
    """

    def __init__(self, img_dir, mask_dir, img_size=224):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.img_names = sorted(os.listdir(img_dir))
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        self.mask_transform = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 如果后缀不一样，这里尝试匹配
        if not os.path.exists(mask_path):
            base, _ = os.path.splitext(img_name)
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                alt = base + ext
                alt_path = os.path.join(self.mask_dir, alt)
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"找不到对应 masks：{mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # 转成二值 masks：>0 为前景 1，=0 为背景 0
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask).long()  # [H, W]

        return img, mask

def dice_loss(pred, target, num_classes, smooth=1.0):
    """
    pred: [B, C, H, W] (logits)
    target: [B, H, W]  (long)
    """
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes)  # [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    dims = (0, 2, 3)
    intersection = torch.sum(pred_soft * target_one_hot, dims)
    cardinality = torch.sum(pred_soft + target_one_hot, dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    loss = 1.0 - dice.mean()
    return loss

def load_pretrained(model, ckpt_path):
    """
    加载预训练权重，自动跳过最后的分类头（原始是 9 类，你现在是 2 类）。
    """
    if not os.path.exists(ckpt_path):
        print(f"[warn] 预训练权重不存在：{ckpt_path}")
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 去掉可能的 "module." 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    # 删掉分类头（9 类 → 2 类，不加载这部分）
    for bad_key in [
        "decoder_0.last_layer.weight",
        "decoder_0.last_layer.bias",
    ]:
        if bad_key in new_state:
            print(f"[info] 删除预训练中的 {bad_key}（类别数不同）")
            del new_state[bad_key]

    msg = model.load_state_dict(new_state, strict=False)
    print("[info] 加载预训练权重完成：", msg)

    return model

def evaluate(model, data_loader, device, num_classes=2):
    """
    在给定 data_loader 上做一次评估，返回平均 ce_loss, dice 指标
    """

    ce_loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_num = 0

    with torch.no_grad():
        for imgs, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = dice_loss(logits, masks, num_classes=num_classes)
            loss = loss_ce + loss_dice

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_dice += (1.0 - loss_dice.item()) * bs
            total_num += bs

    if total_num == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_num
    avg_dice = total_dice / total_num
    return avg_loss, avg_dice

def train():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] 使用设备: {device}")

    # ============= 1. 构建 train/val/test 数据集 =============
    train_dataset = KvasirSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE)
    val_dataset = KvasirSegDataset(VAL_IMG_DIR, VAL_MASK_DIR, IMG_SIZE)
    test_dataset = KvasirSegDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[info] 训练样本数: {len(train_dataset)}")
    print(f"[info] 验证样本数: {len(val_dataset)}")
    print(f"[info] 测试样本数: {len(test_dataset)}")

    # ============= 2. 模型 =============
    model = DAEFormer(num_classes=NUM_CLASSES)
    model = load_pretrained(model, PRETRAIN_PATH)
    model = model.to(device)

    # ============= 3. 损失 & 优化器 =============
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_dice = 0.0
    best_model_path = os.path.join(SAVE_DIR, "daeformer_kvasir_best.pth")

    # ============= 4. 训练循环 =============
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = dice_loss(logits, masks, num_classes=NUM_CLASSES)
            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        val_loss, val_dice = evaluate(model, val_loader, device, num_classes=NUM_CLASSES)

        lr_scheduler.step()

        print(
            f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # 保存最优模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                best_model_path,
            )
            print(f"[info] 保存最佳模型到: {best_model_path} (Val Dice={best_val_dice:.4f})")

    # ============= 5. 训练结束后，在 test 集上评估一次 =============
    print("[info] 训练结束，加载最佳模型做测试集评估...")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    test_loss, test_dice = evaluate(model, test_loader, device, num_classes=NUM_CLASSES)
    print(f"[Test] Loss: {test_loss:.4f} | Dice: {test_dice:.4f}")

if __name__ == "__main__":
    train()
