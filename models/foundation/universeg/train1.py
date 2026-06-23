# train_universeg_kvasir_strong.py
import os
import random
from pathlib import Path

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/universeg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset

from universeg import universeg   # 你的 universeg(...) 函数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
# 1. 数据集：Kvasir-SEG
class KvasirSegDataset(Dataset):
    """
    读取 Kvasir-SEG 的 images / masks，
    每个样本返回单通道 1x128x128 的图像和 masks（0/1）。
    """

    def __init__(self, root_dir: str, image_subdir: str = "images", mask_subdir: str = "masks", indices=None, augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = Path(root_dir) / image_subdir
        self.mask_dir = Path(root_dir) / mask_subdir
        self.augment = augment

        if not self.img_dir.exists():
            raise FileNotFoundError(f"找不到图像目录: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"找不到标注目录: {self.mask_dir}")

        # 构建 masks 索引
        mask_index = {}
        for m in self.mask_dir.glob("*"):
            if m.is_file():
                mask_index[m.stem] = m

        all_pairs = []
        for img_path in self.img_dir.glob("*"):
            if not img_path.is_file():
                continue
            stem = img_path.stem
            # 如果你的 masks 命名是 xxx_mask.png，改成：
            # mask_stem = stem + "_mask"
            # mask_path = mask_index.get(mask_stem)
            mask_path = mask_index.get(stem)
            if mask_path is None:
                continue
            all_pairs.append((img_path, mask_path))

        if len(all_pairs) == 0:
            raise RuntimeError("没有找到 image-masks 对，检查 Kvasir-SEG 命名。")

        if indices is not None:
            # 用外部给定的子集索引
            self.samples = [all_pairs[i] for i in indices]
        else:
            self.samples = all_pairs

        print(f"[KvasirSegDataset] {root_dir} -> {len(self.samples)} 个样本 (augment={self.augment})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_image_as_array(path: Path, is_mask: bool = False):
        img = Image.open(path).convert("L")  # 单通道
        img = img.resize((128, 128), Image.BILINEAR if not is_mask else Image.NEAREST)
        arr = np.array(img).astype(np.float32)

        if is_mask:
            arr = (arr > 127).astype(np.float32)
        else:
            arr = arr / 255.0

        return arr  # [H, W]

    def _augment(self, img_arr, mask_arr):
        """简单 2D 数据增强：随机翻转 & 旋转 90/180/270"""
        # 随机水平翻转
        if random.random() < 0.5:
            img_arr = np.flip(img_arr, axis=1)
            mask_arr = np.flip(mask_arr, axis=1)

        # 随机垂直翻转
        if random.random() < 0.5:
            img_arr = np.flip(img_arr, axis=0)
            mask_arr = np.flip(mask_arr, axis=0)

        # 随机旋转 k * 90 度
        k = random.randint(0, 3)
        if k > 0:
            img_arr = np.rot90(img_arr, k, axes=(0, 1))
            mask_arr = np.rot90(mask_arr, k, axes=(0, 1))

        img_arr = img_arr.copy()
        mask_arr = mask_arr.copy()
        return img_arr, mask_arr

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img_arr = self._load_image_as_array(img_path, is_mask=False)  # [H,W]
        mask_arr = self._load_image_as_array(mask_path, is_mask=True)  # [H,W]

        if self.augment:
            img_arr, mask_arr = self._augment(img_arr, mask_arr)

        img_t = torch.from_numpy(img_arr)[None, ...]   # [1,128,128]
        mask_t = torch.from_numpy(mask_arr)[None, ...] # [1,128,128]

        return img_t, mask_t
# 2. Dice Loss & Dice metric
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, 1, H, W]
        targets: [B, 1, H, W] (0/1)
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()
# 3. 构建 / 加载 UniverSeg
def build_model(pretrained_weight_path: str):
    model = universeg(version="v1", pretrained=False)

    if pretrained_weight_path is not None and os.path.isfile(pretrained_weight_path):
        print(f"加载预训练权重: {pretrained_weight_path}")
        state = torch.load(pretrained_weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("【Warning】缺失参数:", missing)
        if unexpected:
            print("【Warning】多余参数:", unexpected)
    else:
        print("未找到预训练权重，将从随机初始化开始。")

    return model.to(DEVICE)
# 4. 固定 Support Set
def prepare_support_set(dataset: KvasirSegDataset, support_size: int, device):
    """
    从训练集前 support_size 个样本构建一个固定 support set：
    support_images: [1, S, 1, 128, 128]
    support_masks : [1, S, 1, 128, 128]
    """

    n = len(dataset)
    support_size = min(support_size, n)
    indices = list(range(support_size))

    imgs = []
    masks = []
    for idx in indices:
        img, msk = dataset[idx]   # [1,128,128]
        imgs.append(img)
        masks.append(msk)

    imgs = torch.stack(imgs, dim=0)   # [S,1,H,W]
    masks = torch.stack(masks, dim=0) # [S,1,H,W]

    imgs = imgs.unsqueeze(0).to(device)   # [1,S,1,H,W]
    masks = masks.unsqueeze(0).to(device) # [1,S,1,H,W]

    print(f"[Support] support_images: {imgs.shape}, support_masks: {masks.shape}")
    return imgs, masks
# 5. 训练 / 验证
def train_one_epoch(
    model,
    train_dataset: KvasirSegDataset,
    support_images,
    support_masks,
    optimizer,
    bce_loss_fn,
    dice_loss_fn,
    epoch: int,
    n_steps: int = 200,
):
    model.train()
    running_loss = 0.0

    n = len(train_dataset)

    for step in range(1, n_steps + 1):
        # 随机抽一个 target 样本
        target_idx = random.randint(0, n - 1)
        target_image, target_mask = train_dataset[target_idx]  # [1,H,W]

        target_image = target_image.unsqueeze(0).to(DEVICE)  # [1,1,H,W]
        target_mask = target_mask.unsqueeze(0).to(DEVICE)    # [1,1,H,W]

        optimizer.zero_grad()

        logits = model(target_image, support_images, support_masks)  # [1,1,H,W]

        loss_bce = bce_loss_fn(logits, target_mask)
        loss_dice = dice_loss_fn(logits, target_mask)
        loss = loss_bce + loss_dice

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if step % 20 == 0:
            print(
                f"[Train] Epoch {epoch} | Step {step}/{n_steps} | "
                f"Loss: {loss.item():.4f} (BCE {loss_bce.item():.4f}, Dice {loss_dice.item():.4f})"
            )

    avg_loss = running_loss / n_steps
    print(f"[Train] Epoch {epoch} 完成，平均 loss = {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def validate(
    model,
    val_dataset: KvasirSegDataset,
    support_images,
    support_masks,
    bce_loss_fn,
    dice_loss_fn,
    n_steps: int = 100,
):
    model.eval()
    n = len(val_dataset)
    total_loss = 0.0
    total_dice = 0.0

    steps = min(n_steps, n)

    for _ in range(steps):
        idx = random.randint(0, n - 1)
        target_image, target_mask = val_dataset[idx]

        target_image = target_image.unsqueeze(0).to(DEVICE)  # [1,1,H,W]
        target_mask = target_mask.unsqueeze(0).to(DEVICE)    # [1,1,H,W]

        logits = model(target_image, support_images, support_masks)

        loss_bce = bce_loss_fn(logits, target_mask)
        loss_dice = dice_loss_fn(logits, target_mask)
        loss = loss_bce + loss_dice

        total_loss += loss.item()
        total_dice += dice_coefficient(logits, target_mask)

    avg_loss = total_loss / steps
    avg_dice = total_dice / steps
    print(f"[Val] avg_loss = {avg_loss:.4f} | avg_dice = {avg_dice:.4f}")
    return avg_loss, avg_dice
# 6. 主程序
def main():
    # ---------- 路径配置 ----------
    data_root = r"./data/Kvasir-SEG"
    pretrained_weight_path = r"./pretrained_ckpt/universeg_v1_nf64_ss64_STA.pt"
    save_dir = "outputs/foundation/universeg"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 固定随机种子 ----------
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---------- 构建全集数据集 ----------
    full_dataset = KvasirSegDataset(data_root, augment=False)
    n_all = len(full_dataset)
    indices = list(range(n_all))
    random.shuffle(indices)

    # train:val = 8:2
    split = int(0.8 * n_all)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = KvasirSegDataset(data_root, indices=train_indices, augment=True)
    val_dataset = KvasirSegDataset(data_root, indices=val_indices, augment=False)

    # ---------- 构建模型 + 预训练 ----------
    model = build_model(pretrained_weight_path)

    # ---------- 固定 support set（从 train_dataset 抽） ----------
    support_size = 16   # 可以根据显存调大一点，比如 32
    support_images, support_masks = prepare_support_set(train_dataset, support_size, DEVICE)

    # ---------- 损失 & 优化器 ----------
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # ---------- 训练配置 ----------
    num_epochs = 40
    steps_per_epoch = 200      # 每个 epoch 的训练 steps
    val_steps = 100            # 每次验证的样本数上限

    best_val_dice = 0.0
    best_model_path = None

    for epoch in range(1, num_epochs + 1):
        print("=" * 60)
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model,
            train_dataset,
            support_images,
            support_masks,
            optimizer,
            bce_loss,
            dice_loss,
            epoch,
            n_steps=steps_per_epoch,
        )

        val_loss, val_dice = validate(
            model,
            val_dataset,
            support_images,
            support_masks,
            bce_loss,
            dice_loss,
            n_steps=val_steps,
        )

        # 保存最佳模型（按验证 Dice）
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_path = os.path.join(save_dir, f"universeg_kvasir_best_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 新 best 模型已保存: {best_model_path} | val_dice = {best_val_dice:.4f}")

    print("=" * 60)
    print("训练结束。")
    if best_model_path is not None:
        print(f"最优模型: {best_model_path} | 最优验证 Dice: {best_val_dice:.4f}")
    else:
        print("未保存任何 best 模型（说明验证 Dice 一直没提升）。")

if __name__ == "__main__":
    main()
