import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY_SEG_ROOT = os.path.join(PROJECT_ROOT, "binary_seg")

if BINARY_SEG_ROOT not in sys.path:
    sys.path.insert(0, BINARY_SEG_ROOT)

import os
import glob
import time
import math
from typing import Tuple, List

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
# 1) 你只需要改这里的配置
CFG = {
    # 项目根目录（你的 PraNet-V2-main\PraNet-V2-main 这一层）
    "project_root": r".",

    # 数据集根目录（包含 images/masks/val/maskval）
    "data_root": r"./data/Kvasir-SEG",

    # ✅ PVT backbone 权重（你刚刚少写了这个 key）
    "pvt_backbone_weight": r"./pretrained_ckpt/pvt_v2_b2.pth",

    # Res2Net backbone（本脚本默认不用）
    "res2net_backbone_weight": r"./pretrained_ckpt/res2net50_v1b_26w_4s-3cf99910.pth",

    # ================= 训练参数 =================
    "img_size": 352,
    "batch_size": 4,
    "num_workers": 0,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,

    # ================= 保存 =================
    "save_dir": "outputs/cnn/pranet_v2",
    "save_name": "pranetv2_pvt_kvasirseg_best.pth",

    # ================= 模型选项 =================
    "use_pvt": True,

    # 二分类
    "num_class": 1,
    "use_softmax": False,

    # deep supervision 权重
    "ds_weights": [1.0, 0.6, 0.4, 0.2],
}
# 2) Dataset
def list_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    return files

class KvasirSegDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 352):
        self.img_paths = list_images(img_dir)
        self.mask_paths = []
        self.img_size = img_size

        # 假设 mask 与 image 同名（常见 Kvasir-SEG 格式）
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(mask_dir)}

        for ip in self.img_paths:
            key = os.path.splitext(os.path.basename(ip))[0]
            if key not in mask_map:
                raise FileNotFoundError(f"找不到对应 mask：{key}，请检查 mask 文件名是否与 image 对齐")
            self.mask_paths.append(mask_map[key])

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.mask_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))

        # mask: [H,W] -> [1,H,W]，并把 0/255 变成 0/1 float
        mask = (mask > 0).float().unsqueeze(0)
        return img, mask
# 3) 指标：Dice / IoU
@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-7) -> Tuple[float, float]:
    """
    logits: [B,1,H,W] raw logits
    target: [B,1,H,W] 0/1
    """
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()

    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).clamp_max(1).sum(dim=(1,2,3))
    pred_sum = pred.sum(dim=(1,2,3))
    tgt_sum = target.sum(dim=(1,2,3))

    dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()
# 4) 载入你的模型源代码
def import_model_classes(project_root: str):
    """
    你的目录是：
    PraNet-V2-main\PraNet-V2-main\binary_seg\lib\pranet.py (或类似文件)
    这里用 sys.path 让你直接 import。
    """
    import sys
    lib_dir = os.path.join(project_root, "binary_seg", "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    # ⚠️ 这里假设你的类在 pranet.py 里
    # 如果你的文件名不是 pranet.py（比如 PraNet_Res2Net.py），把下面 import 改成对应文件名即可
    from lib.pranet import PVT_PraNet_V2, PraNet_V2
    return PVT_PraNet_V2, PraNet_V2

def load_pvt_backbone_weight(model, weight_path: str):
    """
    你的 PVT_PraNet_V2 __init__ 里写死了 path='./models/pvt_v2_b2.pth'
    为了不改你源代码，这里直接再 load 一次，覆盖进去。
    """

    if not weight_path or not os.path.exists(weight_path):
        print("[INFO] PVT backbone weight not found, train from scratch")
        return

    state = torch.load(weight_path, map_location="cpu")
    model_dict = model.backbone.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict.keys()}
    model_dict.update(filtered)
    model.backbone.load_state_dict(model_dict, strict=False)
    print(f"[OK] 已加载 PVT backbone 权重：{weight_path}，匹配参数数={len(filtered)}")
# 5) 训练
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def train_one_epoch(model, loader, optimizer, scaler, device, bce_loss, ds_weights):
    model.train()
    total_loss = 0.0

    for step, (img, mask) in enumerate(loader, start=1):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # 输出：map2_fg,map3_fg,map4_fg,map5_fg, map2_bg,map3_bg,map4_bg,map5_bg
            outs = model(img)

            map2_fg, map3_fg, map4_fg, map5_fg = outs[0], outs[1], outs[2], outs[3]

            # 二分类训练只用 fg 分支（bg 分支可忽略）
            loss2 = bce_loss(map2_fg, mask)
            loss3 = bce_loss(map3_fg, mask)
            loss4 = bce_loss(map4_fg, mask)
            loss5 = bce_loss(map5_fg, mask)

            loss = (ds_weights[0] * loss2 +
                    ds_weights[1] * loss3 +
                    ds_weights[2] * loss4 +
                    ds_weights[3] * loss5)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def validate(model, loader, device, bce_loss):
    model.eval()
    total_loss = 0.0
    dices, ious = [], []

    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        outs = model(img)
        map2_fg = outs[0]  # 用最高分辨率 map2 做评估

        loss = bce_loss(map2_fg, mask)
        total_loss += loss.item()

        d, i = dice_iou_from_logits(map2_fg, mask)
        dices.append(d)
        ious.append(i)

    return total_loss / max(len(loader), 1), float(np.mean(dices)), float(np.mean(ious))

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_dir(CFG["save_dir"])

    # 数据集
    train_img_dir = os.path.join(CFG["data_root"], "images")
    train_mask_dir = os.path.join(CFG["data_root"], "masks")
    val_img_dir = os.path.join(CFG["data_root"], "val")
    val_mask_dir = os.path.join(CFG["data_root"], "maskval")

    train_ds = KvasirSegDataset(train_img_dir, train_mask_dir, CFG["img_size"])
    val_ds = KvasirSegDataset(val_img_dir, val_mask_dir, CFG["img_size"])

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True
    )

    # 模型导入
    PVT_PraNet_V2, PraNet_V2 = import_model_classes(CFG["project_root"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CFG["use_pvt"]:
        model = PVT_PraNet_V2(channel=32, num_class=CFG["num_class"],
                              sem_downsample=1, use_softmax=CFG["use_softmax"])
        # 覆盖加载你本地的 pvt backbone 权重（避免源码里路径写死）
        load_pvt_backbone_weight(model, CFG["pvt_backbone_weight"])
    else:
        # Res2Net 版：如果它内部 pretrained=True 会尝试在线下载，你最好改源码为 pretrained=False 再手动 load
        model = PraNet_V2(channel=32, num_class=CFG["num_class"],
                          sem_downsample=1, use_softmax=CFG["use_softmax"])

    model = model.to(device)

    # Loss/Optimizer
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    # AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_dice = -1.0
    save_path = os.path.join(CFG["save_dir"], CFG["save_name"])

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Save to: {save_path}")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, bce_loss, CFG["ds_weights"]
        )

        val_loss, val_dice, val_iou = validate(model, val_loader, device, bce_loss)

        dt = time.time() - t0
        print(f"Epoch [{epoch:03d}/{CFG['epochs']}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"dice={val_dice:.4f}  iou={val_iou:.4f}  time={dt:.1f}s")

        # 保存最优
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "cfg": CFG,
            }, save_path)
            print(f"[SAVE] best_dice={best_dice:.4f} -> {save_path}")

    print("训练结束。")

if __name__ == "__main__":
    main()
