# train_finetune.py
# 放在项目根，例如:
# Original: train_finetune.py

import os
import glob
import time
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/htc_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# ---- 修改下面的路径为你的实际路径（脚本默认已按你给的路径） ----
DATA_ROOT = r"./data/Kvasir-SEG"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
MASKS_DIR = os.path.join(DATA_ROOT, "masks")
PRETRAINED_PATH = r"./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"

# 项目根（脚本所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
LOG_DIR = os.path.join(OUTPUT_DIR, "test_log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- 超参数（按需修改） ----
IMG_SIZE = 224
BATCH_SIZE = 8  # 如果显存不足（OOM），可以从 8 减小到 4 或 2
NUM_EPOCHS = 16
# 在 Windows 上，num_workers > 0 可能需要放在 if __name__ == "__main__": 块中，这里设为 0 以保证兼容性
NUM_WORKERS = 0 if os.name == 'nt' else 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 20
RANDOM_SEED = 42

# 固定随机种子
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------------- Dataset ----------------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=IMG_SIZE, mode="train"):
        super().__init__()
        self.images = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.masks = []
        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            found = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                cand = os.path.join(masks_dir, name + ext)
                if os.path.exists(cand):
                    found = cand
                    break
            self.masks.append(found)
        pairs = [(i, m) for i, m in zip(self.images, self.masks) if m is not None]
        self.images, self.masks = [p[0] for p in pairs], [p[1] for p in pairs]
        if not self.images:
            raise RuntimeError(f"在 {images_dir} 和 {masks_dir} 中未找到匹配的图像-掩码对。请检查路径和文件名。")
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        if self.mode == "train":
            if random.random() > 0.5: img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5: img, mask = TF.vflip(img), TF.vflip(mask)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = (np.array(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# ---------------- metrics ----------------
def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()

# ---------------- model import ----------------
try:
    from network.Net import model as SwinModelWrapper
except Exception as e:
    print(f"自动导入模型失败：{e}\n请检查 network/Net.py 是否存在且其中定义了 class model(nn.Module)。")
    raise

# ---------------- training & validation functions ----------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}", leave=False)
    for i, (imgs, masks) in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % PRINT_FREQ == 0:
            pbar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, dices = 0.0, []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            running_loss += criterion(outputs, masks).item()
            dices.append(dice_coeff(outputs, masks))
    return running_loss / len(dataloader), np.mean(dices)

# ---------------- main ----------------
def main():
    print(f"设备: {DEVICE}")
    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, img_size=IMG_SIZE, mode="train")
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(SegmentationDataset(IMAGES_DIR, MASKS_DIR, img_size=IMG_SIZE, mode="val"), val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 关键修复：创建正确的 config ---
    import ml_collections
    cfg = ml_collections.config_dict.ConfigDict()
    cfg.n_classes = 1
    cfg.decoder_channels = (128, 64, 32, 16)
    cfg.n_skip = 3
    # ---------------------------------
    net_wrapper = SwinModelWrapper(config=cfg, img_size=IMG_SIZE, num_classes=1)
    net = net_wrapper.to(DEVICE)

    # --- 权重加载（已修复匹配问题） ---
    if os.path.exists(PRETRAINED_PATH):
        print(f"开始从 {PRETRAINED_PATH} 加载预训练权重...")
        pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        if "model" in pretrained_dict: pretrained_dict = pretrained_dict["model"]

        model_to_load = net_wrapper.swin_unet if hasattr(net_wrapper, 'swin_unet') else net_wrapper
        model_dict = model_to_load.state_dict()

        full_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        for k, v in pretrained_dict.items():
            if "layers." in k:
                try:
                    parts = k.split('.')
                    if len(parts) > 1 and parts[1].isdigit():
                        current_layer_num = len(model_to_load.depths) - 1 - int(parts[1])
                        if current_layer_num >= 0:
                            current_k = f"layers_up.{current_layer_num}.{'.'.join(parts[2:])}"
                            if current_k in model_dict and v.shape == model_dict[current_k].shape:
                                full_dict[current_k] = v
                except Exception:
                    pass

        model_dict.update(full_dict)
        msg = model_to_load.load_state_dict(model_dict, strict=False)
        print("权重加载完成。")
        if msg.missing_keys: print(" - 缺失的键:",
                                   [k for k in msg.missing_keys if 'layers_up' not in k and 'concat_back_dim' not in k])
        if msg.unexpected_keys: print(" - 未预期的键:", msg.unexpected_keys)
    else:
        print(f"警告：预训练权重文件未找到于 {PRETRAINED_PATH}，将从头开始训练。")
    # ---------------------------------

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_dice = 0.0
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_dice = validate(net, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        epoch_info = f"Epoch {epoch}/{NUM_EPOCHS} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValDice: {val_dice:.4f}"
        print(epoch_info)
        with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {epoch_info}\n")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(net.state_dict(), best_path)
            print(f"  -> 新的最优模型已保存至 {best_path} (Dice: {best_val_dice:.4f})")

    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")
    print(f"最优 Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
