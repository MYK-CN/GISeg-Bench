import os
import glob
import random
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/hiformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from medpy import metric  # 用 medpy 来算 Dice

from models.HiFormer import HiFormer
import configs.HiFormer_configs as hcfg  # 如果这里报错，就改成你原来 train1.py 里用的那个 import

# ======================= 数据集 ==========================
class SynapseNPZ2DDataset(Dataset):
    """
    读取 Synapse 的 train_npz：
    - 如果 image 是 3D (N, H, W)，按体数据逐 slice 展开为 N 个 2D 样本
    - 如果 image 是 2D (H, W)，当作 1 个样本
    """
    def __init__(self, root_npz_dir):
        super().__init__()
        self.root = root_npz_dir
        self.items = []  # [(npz_path, slice_idx or None), ...]

        npz_files = sorted(glob.glob(os.path.join(self.root, "*.npz")))
        if len(npz_files) == 0:
            raise RuntimeError(f"在 {self.root} 没找到 .npz 文件，请检查路径是否正确。")

        print(f"找到 {len(npz_files)} 个 npz 文件，正在扫描切片数量...")

        total_slices = 0
        for npz_path in npz_files:
            data = np.load(npz_path)

            if "image" in data.files:
                vol_img = data["image"]
            else:
                # 如果 key 不是 image，就默认取第一个
                first_key = data.files[0]
                vol_img = data[first_key]

            if "label" not in data.files:
                raise RuntimeError(f"{npz_path} 中没有 'label'，这个数据集不包含标签，无法监督训练。")

            # 根据维度决定如何展开
            if vol_img.ndim == 3:
                # (N, H, W)
                num_slices = vol_img.shape[0]
                for i in range(num_slices):
                    self.items.append((npz_path, i))
                total_slices += num_slices
            elif vol_img.ndim == 2:
                # (H, W) —— 只当 1 个样本
                self.items.append((npz_path, None))
                total_slices += 1
            else:
                raise RuntimeError(f"{npz_path} 中 image 维度为 {vol_img.shape}，不属于 (N,H,W) 或 (H,W)。")

        print(f"总共 {total_slices} 个 2D 样本。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, slice_idx = self.items[idx]
        data = np.load(npz_path)

        if "image" in data.files:
            vol_img = data["image"]
        else:
            first_key = data.files[0]
            vol_img = data[first_key]

        if "label" in data.files:
            vol_lab = data["label"]
        else:
            raise RuntimeError(f"{npz_path} 中没有 'label'，请确认数据是否正确。")

        # ---------- 根据维度取出 2D ----------
        if vol_img.ndim == 3:
            # (N, H, W) 一定要有 slice_idx
            assert slice_idx is not None
            img2d = vol_img[slice_idx]      # (H, W)
            lab2d = vol_lab[slice_idx]      # (H, W)
        elif vol_img.ndim == 2:
            # (H, W) 直接用
            img2d = vol_img                  # (H, W)
            lab2d = vol_lab
        else:
            raise RuntimeError(f"{npz_path} 中 image 维度为 {vol_img.shape}，不属于 (N,H,W) 或 (H,W)。")

        # ==== 简单预处理 ====
        img2d = img2d.astype(np.float32)
        img_mean = img2d.mean()
        img_std = img2d.std() + 1e-8
        img2d = (img2d - img_mean) / img_std

        # 扩成 3 通道： (3, H, W)
        img2d = np.expand_dims(img2d, axis=0)    # (1, H, W)
        img2d = np.repeat(img2d, 3, axis=0)      # (3, H, W)

        img_tensor = torch.from_numpy(img2d)                     # float32
        lab_tensor = torch.from_numpy(lab2d.astype(np.int64))    # long, (H, W)

        return img_tensor, lab_tensor

# ======================= Dice 计算函数 ==========================
def compute_mean_dice(preds: torch.Tensor,
                      labels: torch.Tensor,
                      num_classes: int,
                      ignore_index: int = 0) -> float:
    """
    用 medpy 计算多类 Dice：
    - preds: (B, H, W) 预测类别（整型）
    - labels: (B, H, W) GT
    - 对每个前景类 c=1..num_classes-1 计算 binary Dice，再取平均
    """

    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    dices = []
    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (preds_np == c)
        label_c = (labels_np == c)

        # 如果该类在 GT 和 pred 里都没出现，跳过
        if label_c.sum() == 0 and pred_c.sum() == 0:
            continue

        d = metric.dc(pred_c, label_c)
        dices.append(d)

    if len(dices) == 0:
        return 0.0
    return float(np.mean(dices))

# ======================= 训练函数 ==========================
def train_hiformer_standalone():
    # ---------- 路径 & 超参数 ----------
    train_npz_dir = r"./data/Synapse/train_npz"
    save_dir = "outputs/transformer/hiformer"
    os.makedirs(save_dir, exist_ok=True)

    img_size = 224
    num_classes = 9        # Synapse: 背景 + 8 器官
    batch_size = 4
    num_epochs = 15
    base_lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ---------- 随机种子 ----------
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---------- 数据 ----------
    train_dataset = SynapseNPZ2DDataset(train_npz_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # Windows 建议先用 0
        pin_memory=True
    )

    # ---------- 模型 ----------
    # 根据你仓库的配置文件函数名调整，如果不是这个，就改成对应名字
    config = hcfg.get_hiformer_b_configs()
    model = HiFormer(
        config=config,
        img_size=img_size,
        in_chans=3,
        n_classes=num_classes
    ).to(device)

    # ---------- 损失 & 优化器 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    print("开始训练 HiFormer...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)   # (B, 3, H, W 或 H0, W0)
            labels = labels.to(device)   # (B, H, W)

            # 确保图像调整到统一分辨率
            if images.dim() != 4:
                raise RuntimeError(f"images 形状异常: {images.shape}，期望 (B,3,H,W)")

            images = F.interpolate(
                images,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False
            )

            # 确保标签尺寸也匹配
            if labels.shape[-2:] != (img_size, img_size):
                labels = labels.unsqueeze(1).float()  # (B,1,H,W)
                labels = F.interpolate(
                    labels,
                    size=(img_size, img_size),
                    mode="nearest"
                )
                labels = labels.squeeze(1).long()     # (B,H,W)

            optimizer.zero_grad()
            outputs = model(images)   # (B, num_classes, H, W)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ---------- 计算 Dice ----------
            with torch.no_grad():
                # 取 argmax 得到类别预测
                preds = torch.argmax(outputs, dim=1)      # (B,H,W)
                batch_dice = compute_mean_dice(
                    preds=preds,
                    labels=labels,
                    num_classes=num_classes,
                    ignore_index=0
                )

            epoch_loss += loss.item()
            epoch_dice += batch_dice
            num_batches += 1

            if (step + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  "
                      f"Dice: {batch_dice:.4f}")

        avg_loss = epoch_loss / max(1, num_batches)
        avg_dice = epoch_dice / max(1, num_batches)
        print(f"==> Epoch [{epoch+1}/{num_epochs}] "
              f"平均 Loss: {avg_loss:.4f}  平均 Dice: {avg_dice:.4f}")

        # 每轮保存一次模型
        ckpt_path = os.path.join(save_dir, f"hiformer_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"已保存模型: {ckpt_path}")

    final_path = os.path.join(save_dir, "hiformer_final-xirou2.pth")
    torch.save(model.state_dict(), final_path)
    print("训练完成，最终模型保存到:", final_path)

if __name__ == "__main__":
    train_hiformer_standalone()
