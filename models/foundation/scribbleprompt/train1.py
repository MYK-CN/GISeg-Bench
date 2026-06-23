import os
from glob import glob
from typing import List, Tuple

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/scribbleprompt')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image

# ====== 1. 根据你自己的工程结构修改这里的 import ======
# 假设你的 ScribblePromptUNet 代码在：
# ScribblePrompt-main/scribbleprompt/models/unet.py
# 文件里定义了 class ScribblePromptUNet, prepare_inputs
from scribbleprompt.models.unet import ScribblePromptUNet, prepare_inputs

# ====== 2. 一些超参数 & 路径设置（你只需要改这里就行） ======
DATA_ROOT = r"./data/Kvasir-SEG"  # Kvasir-SEG 根目录
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")

# 训练相关
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2  # 训练/验证划分比例

# 使用 GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 3. Kvasir-SEG 数据集定义 ======
class KvasirSegDataset(Dataset):
    """
    简单的 Kvasir-SEG 数据集:
    - 读取 image 和 mask
    - 转成灰度 1 通道
    - resize 到 128x128
    - 归一化到 [0,1]
    """

    def __init__(self, image_dir: str, mask_dir: str, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # 假设 image 和 mask 同名
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
            raise RuntimeError(f"在 {image_dir} / {mask_dir} 下没有找到有效的 image-mask 对！")

        print(f"共找到 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        # 读图像 -> 灰度
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # resize
        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        # 转 tensor，归一化到 [0,1]
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

        # 添加通道维度 -> 1 x H x W
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # 有些 mask 不是纯 0/1，这里阈值一下
        mask = (mask > 0.5).float()

        return img, mask

# ====== 4. Dice Loss（和 BCE 一起用，效果更好） ======
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: B x 1 x H x W （未过 sigmoid）
        targets: B x 1 x H x W （0/1）
        """

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) /(probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()  # 返回的是 1 - Dice

# ====== 5. 训练 & 验证函数 ======
def train_one_epoch(model: ScribblePromptUNet,
                    loader: DataLoader,
                    bce_loss_fn,
                    dice_loss_fn,
                    optimizer,
                    device: str):
    model.model.train()  # 注意：真正的 UNet 在 model.model 里

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    for img, mask in loader:
        img = img.to(device)   # B x 1 x H x W
        mask = mask.to(device) # B x 1 x H x W

        # 构造 prompts（这里不使用点/框/涂鸦，全部为 None）
        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        # 准备网络输入 B x 5 x H x W
        x = prepare_inputs(prompts, device=device)
        # 前向
        logits = model.model(x)  # 不做 sigmoid

        # 计算损失：BCE + DiceLoss（1 - Dice）
        bce = bce_loss_fn(logits, mask)
        dice_loss = dice_loss_fn(logits, mask)
        loss = bce + dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 当前 batch 的 Dice 系数 = 1 - DiceLoss
        batch_dice = 1.0 - dice_loss.item()

        total_loss += loss.item()
        total_dice += batch_dice
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = total_dice / max(total_batches, 1)
    return avg_loss, avg_dice

@torch.no_grad()
def validate(model: ScribblePromptUNet,
             loader: DataLoader,
             bce_loss_fn,
             dice_loss_fn,
             device: str):
    model.model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        prompts = {
            "img": img,
            "point_coords": None,
            "point_labels": None,
            "scribbles": None,
            "box": None,
            "mask_input": None,
        }

        x = prepare_inputs(prompts, device=device)
        logits = model.model(x)

        bce = bce_loss_fn(logits, mask)
        dice_loss = dice_loss_fn(logits, mask)
        loss = bce + dice_loss

        batch_dice = 1.0 - dice_loss.item()

        total_loss += loss.item()
        total_dice += batch_dice
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = total_dice / max(total_batches, 1)
    return avg_loss, avg_dice

# ====== 6. 主函数：组装一切，开始微调 ======
def main():
    print("==== 加载数据集 ====")
    dataset = KvasirSegDataset(IMAGE_DIR, MASK_DIR, image_size=(128, 128))

    # 划分训练/验证
    val_len = int(len(dataset) * VAL_RATIO)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练样本: {train_len}, 验证样本: {val_len}")

    print("==== 初始化模型（加载预训练 UNet 权重） ====")
    model = ScribblePromptUNet(version="v1", device=DEVICE)  # 会自动在 checkpoints 里找权重

    # 损失函数 & 优化器
    bce_loss_fn = nn.BCEWithLogitsLoss()  # 注意：我们直接用 logits
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    save_path = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
        "ScribblePrompt_unet_finetuned_kvasir1.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("==== 开始训练 ====")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, bce_loss_fn, dice_loss_fn, optimizer, DEVICE
        )
        val_loss, val_dice = validate(
            model, val_loader, bce_loss_fn, dice_loss_fn, DEVICE
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        # 保存最优模型（按验证 loss）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.model.state_dict(), save_path)
            print(f"  -> 最佳模型更新，已保存到: {save_path}")

    print("训练完成！最佳验证损失: ", best_val_loss)

if __name__ == "__main__":
    main()
