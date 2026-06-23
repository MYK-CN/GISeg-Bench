import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ==========================
# 1. 导入你的 Swin-Unet 模型
# ==========================
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


# ==========================
# 2. 数据集定义
# ==========================
class KvasirSegDataset(Dataset):
    """
    简单的 Kvasir-SEG 数据集读取:
    images_dir: 图像路径
    masks_dir: 掩码路径
    图像和掩码按文件名一一对应
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
            transforms.ToTensor(),  # [0,1]
        ])
        # 掩码也 resize 到同样大小
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道

        img = self.img_transform(img)  # [3,H,W]

        mask = self.mask_transform(mask)
        mask = np.array(mask, dtype=np.float32)
        # 假设前景为255，背景为0 -> 归一化为0/1
        mask = mask / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

        return img, mask


# ==========================
# 3. Dice Loss（用于分割）
# ==========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B,1,H,W]，未过sigmoid
        targets: [B,1,H,W]，0/1
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
                probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


# ==========================
# 4. 载入预训练权重（只加载能对上的部分）
# ==========================
def load_pretrained_backbone(model, ckpt_path):
    print(f"===> 从预训练权重加载: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # 官方 Swin 的权重一般在 'model' 这个 key 里
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model_dict = model.state_dict()
    matched, unmatched = 0, 0
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            matched += 1
        else:
            unmatched += 1
    model.load_state_dict(model_dict, strict=False)
    print(f"   已匹配权重参数: {matched}，未匹配: {unmatched}")
    return model


# ==========================
# 5. 训练与验证
# ==========================
def train_one_epoch(model, loader, optimizer, bce_loss_fn, dice_loss_fn, device):
    model.train()
    total_bce, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # [B,1,H,W]

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


# ==========================
# 6. 主函数：微调 Swin-UNet
# ==========================
def main():
    # ---------- 路径配置（按你的实际情况改） ----------
    pretrained_ckpt = r"./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"

    train_images_dir = r"./data/Kvasir-SEG/images"
    train_masks_dir = r"./data/Kvasir-SEG/masks"
    val_images_dir = r"./data/Kvasir-SEG/val"
    val_masks_dir = r"./data/Kvasir-SEG/maskval"

    save_model_path = "./swin_unet_kvasir_finetune.pth"

    # ---------- 训练超参数 ----------
    img_size = 224
    num_classes = 1          # Kvasir-SEG 是二分类（前景/背景）
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ---------- 数据集 & DataLoader ----------
    train_dataset = KvasirSegDataset(train_images_dir, train_masks_dir, img_size=img_size)
    val_dataset = KvasirSegDataset(val_images_dir, val_masks_dir, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- 初始化模型 ----------
    model = SwinTransformerSys(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,       # 输出通道=1，用于二分类 mask
        embed_dim=96,
        depths=[2, 2, 2, 2],
        depths_decoder=[1, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first"
    )

    # 载入预训练 Swin-T 权重（只加载编码器等能对上的部分）
    model = load_pretrained_backbone(model, pretrained_ckpt)
    model = model.to(device)

    # ---------- 损失函数 & 优化器 ----------
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_dice = 1e9  # 这里因为 dice_loss 越低越好，所以用 loss 记录

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

        # 保存最优模型（按验证DiceLoss最小）
        if val_dice < best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_model_path)
            print(f"  >>> 更新最优模型，已保存到: {save_model_path}")

    print("训练结束！最终最佳验证 DiceLoss:", best_val_dice)


if __name__ == "__main__":
    main()
