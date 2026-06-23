import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/mt_unet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 把你给的模型代码粘到这里 =====================
# 为了方便，你可以把下面这行改成 from mtunet_model import MTUNet, configs
# 然后把你那整段模型代码放到 mtunet_model.py 里。
# 这里先假设当前文件里已经有 MTUNet 和 configs 定义。

from model.MTUNet import MTUNet, configs
# 上面这行需要你改成你实际的文件名：
# 例如你把模型那一大段代码存成 mtunet_model.py，就写：
# from mtunet_model import MTUNet, configs

# ===================== Kvasir-SEG 数据集类 =====================
class KvasirSegDataset(Dataset):
    """
    Kvasir-SEG:
        images_dir: 图像路径
        masks_dir : 掩码路径
        图像和掩码文件名一一对应
    """
    def __init__(self, images_dir, masks_dir, img_size=224):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        assert len(self.img_files) == len(self.mask_files), "图像和掩码数量不一致！"

        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),          # [0,1]
        ])

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # 读取图像
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道

        img = self.img_transform(img)  # [3,H,W]

        mask = self.mask_resize(mask)
        mask = np.array(mask, dtype=np.float32)
        # 假设前景为255，背景为0，归一化为0/1
        mask = mask / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

        return img, mask

# ===================== Dice Loss =====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B,1,H,W] (未过sigmoid)
        targets: [B,1,H,W] (0/1)
        """

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()

# ===================== 加载预训练权重（尽量兼容各种保存方式） =====================
def load_pretrained(model, ckpt_path, device):
    print(f"===> 从预训练权重加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容几种常见格式
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        # 去掉可能存在的 "module." 前缀
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    model_dict = model.state_dict()
    matched, unmatched = 0, 0
    for k, v in new_state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            matched += 1
        else:
            unmatched += 1

    model.load_state_dict(model_dict, strict=False)
    print(f"   已匹配参数: {matched} 个，未匹配: {unmatched} 个（比如最后输出头尺寸不一样会被忽略）")
    return model

# ===================== 训练与验证 =====================
def train_one_epoch(model, loader, optimizer, bce_loss_fn, dice_loss_fn, device):
    model.train()
    total_bce, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(imgs)      # [B,1,H,W]
        bce = bce_loss_fn(logits, masks)
        dice = dice_loss_fn(logits, masks)
        loss = bce + dice

        loss.backward()
        optimizer.step()

        total_bce += bce.item()
        total_dice += dice.item()

    n = len(loader)
    return total_bce / n, total_dice / n

@torch.no_grad()
def validate(model, loader, bce_loss_fn, dice_loss_fn, device):
    model.eval()
    total_bce, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)

        bce = bce_loss_fn(logits, masks)
        dice = dice_loss_fn(logits, masks)

        total_bce += bce.item()
        total_dice += dice.item()

    n = len(loader)
    return total_bce / n, total_dice / n

# ===================== 主函数 =====================
def main():
    # ---------- 路径配置（按你实际情况改一下就行） ----------
    pretrained_ckpt = r"./pretrained_ckpt/mtunet_pretrain.pth"

    train_images_dir = r"./data/Kvasir-SEG/images"
    train_masks_dir = r"./data/Kvasir-SEG/masks"
    val_images_dir = r"./data/Kvasir-SEG/val"
    val_masks_dir = r"./data/Kvasir-SEG/maskval"

    save_model_path = "./mtunet_kvasir_finetune.pth"

    # ---------- 超参数 ----------
    img_size = 224
    num_classes = 1          # 息肉分割：前景/背景
    batch_size = 4
    num_epochs = 15
    learning_rate = 1e-4
    weight_decay = 1e-5

    # ⚠️ 你的模型代码里写死了 .cuda()，这里强制用 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("当前模型实现中包含 .cuda() 调用，请在有 GPU 的环境下运行，否则需要先修改模型源码。")

    device = torch.device("cuda:0")
    print("使用设备:", device)

    # ---------- 数据集 & DataLoader ----------
    train_dataset = KvasirSegDataset(train_images_dir, train_masks_dir, img_size=img_size)
    val_dataset = KvasirSegDataset(val_images_dir, val_masks_dir, img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------- 构建 MTUNet 模型 ----------
    model = MTUNet(out_ch=num_classes)   # 最后一层 Conv2d(64,1,1) 输出单通道
    model = model.to(device)

    # ---------- 加载预训练权重 ----------
    if os.path.exists(pretrained_ckpt):
        model = load_pretrained(model, pretrained_ckpt, device)
    else:
        print(f"警告：未找到预训练权重：{pretrained_ckpt}，将从头训练。")

    # ---------- 损失函数 & 优化器 ----------
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_dice_loss = 1e9  # DiceLoss 越低越好

    # ---------- 训练循环 ----------
    for epoch in range(1, num_epochs + 1):
        train_bce, train_dice = train_one_epoch(
            model, train_loader, optimizer, bce_loss_fn, dice_loss_fn, device
        )
        val_bce, val_dice = validate(
            model, val_loader, bce_loss_fn, dice_loss_fn, device
        )

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train BCE: {train_bce:.4f}, Train DiceLoss: {train_dice:.4f} | "
              f"Val BCE: {val_bce:.4f}, Val DiceLoss: {val_dice:.4f}")

        # 按验证 DiceLoss 最小保存最好模型
        if val_dice < best_val_dice_loss:
            best_val_dice_loss = val_dice
            torch.save(model.state_dict(), save_model_path)
            print(f"  >>> 更新最优模型，已保存到: {save_model_path}")

    print("训练结束！最优验证 DiceLoss:", best_val_dice_loss)

if __name__ == "__main__":
    main()
