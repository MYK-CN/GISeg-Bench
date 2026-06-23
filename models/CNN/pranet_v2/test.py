import os
import sys
import numpy as np
from sympy import false
from tqdm import tqdm
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/pranet_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ========================= 0. Path Configuration =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY_SEG_ROOT = os.path.join(PROJECT_ROOT, "binary_seg")
if BINARY_SEG_ROOT not in sys.path:
    sys.path.insert(0, BINARY_SEG_ROOT)

# ========================= 1. Config =========================
CFG = {
    "data_root": r"./data/Kvasir-SEG",   # 数据根目录
    "ckpt_path": r"./weights/pranetv2_pvt_kvasirseg_best.pth",  # 训练好的模型
    "img_size": 352,
    "batch_size": 1,
    "num_class": 1,
    "use_softmax": False,
    "save_pred": False,
    "save_dir": "test_results",
}

# ========================= 2. Dataset（直接复用你训练时的） =========================
from train import KvasirSegDataset, dice_iou_from_logits

# ========================= 3. Model =========================
def import_model():
    from lib.pranet import PVT_PraNet_V2, PraNet_V2
    return PVT_PraNet_V2

def load_checkpoint(model, ckpt_path):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()
         if k.replace("module.", "") in model.state_dict()},
        strict=False
    )
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

# ========================= 4. Test =========================
@torch.no_grad()
def test(model, loader, device, save_dir=None):
    model.eval()
    dices, ious = [], []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (img, mask) in enumerate(tqdm(loader, desc="Testing")):
        img = img.to(device)
        mask = mask.to(device)

        logits = model(img)[0]   # map2_fg
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        d, i = dice_iou_from_logits(logits, mask)
        dices.append(d)
        ious.append(i)

        # ================= 保存预测结果 =================
        if save_dir:
            pred_np = pred[0, 0].cpu().numpy() * 255
            gt_np = mask[0, 0].cpu().numpy() * 255

            Image.fromarray(pred_np.astype(np.uint8)).save(
                os.path.join(save_dir, f"{idx:04d}_pred.png")
            )
            Image.fromarray(gt_np.astype(np.uint8)).save(
                os.path.join(save_dir, f"{idx:04d}_gt.png")
            )

    return float(np.mean(dices)), float(np.mean(ious))

# ========================= 5. Main =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = KvasirSegDataset(
        img_dir=os.path.join(CFG["data_root"], "test"),
        mask_dir=os.path.join(CFG["data_root"], "masktest"),
        img_size=CFG["img_size"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False
    )

    Model = import_model()
    model = Model(
        channel=32,
        num_class=CFG["num_class"],
        sem_downsample=1,
        use_softmax=CFG["use_softmax"]
    )

    load_checkpoint(model, CFG["ckpt_path"])
    model.to(device)

    dice, iou = test(
        model,
        test_loader,
        device,
        save_dir=CFG["save_dir"] if CFG["save_pred"] else None
    )

    print("=" * 60)
    print(f"[TEST RESULT] Dice: {dice:.4f} | IoU: {iou:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
