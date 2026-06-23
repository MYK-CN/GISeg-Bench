import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transnuseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 模型导入 =================
from models.transnuseg import TransNuSeg

# ================= 测试配置 =================
TEST_CONFIG = {
    "test_img_path": r"./data/Kvasir-SEG/test",
    "test_mask_path": r"./data/Kvasir-SEG/masktest",
    "weight_path": r"./weights/transnuseg_kvasir_epoch_15.pth",
    "img_size": 256,
    "batch_size": 4,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ================= 测试数据集（文件名匹配，防炸） =================
class KvasirTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_size = img_size
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

        self.samples = []
        for e in exts:
            for img_path in glob.glob(os.path.join(img_dir, f"*{e}")):
                name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = None
                for me in exts:
                    cand = os.path.join(mask_dir, name + me)
                    if os.path.exists(cand):
                        mask_path = cand
                        break
                if mask_path is not None:
                    self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("❌ No valid image-mask pairs found!")

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform(img)

        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = torch.from_numpy(mask).long()

        return img, mask

# ================= 指标计算 =================
def dice_iou_from_logits(logits, target, eps=1e-7):
    """
    logits: [B, 2, H, W]
    target: [B, H, W]
    """

    prob = torch.softmax(logits, dim=1)[:, 1, :, :]
    pred = (prob > 0.5).float()
    target = target.float()

    inter = (pred * target).sum(dim=(1, 2))
    union = (pred + target - pred * target).sum(dim=(1, 2))

    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()

def structure_loss(pred, mask):
    ce = F.cross_entropy(pred, mask)

    prob = torch.softmax(pred, dim=1)[:, 1, :, :]
    mask = mask.float()

    inter = (prob * mask).sum(dim=(1, 2))
    union = prob.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)

    return ce + dice.mean()

# ================= 测试主流程 =================
@torch.no_grad()
def main():
    device = TEST_CONFIG["device"]
    print(f"Using device: {device}")

    # 数据
    test_dataset = KvasirTestDataset(
        TEST_CONFIG["test_img_path"],
        TEST_CONFIG["test_mask_path"],
        TEST_CONFIG["img_size"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # 模型
    model = TransNuSeg(
        img_size=TEST_CONFIG["img_size"],
        num_classes=TEST_CONFIG["num_classes"],
        depths=[2, 2, 2, 2],
        embed_dim=96
    )
    model.load_state_dict(torch.load(TEST_CONFIG["weight_path"], map_location="cpu"))
    model.to(device)
    model.eval()

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    print("===== Start Testing =====")
    for imgs, masks in tqdm(test_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        out_seg, _, _ = model(imgs)

        if out_seg.shape[-1] != TEST_CONFIG["img_size"]:
            out_seg = F.interpolate(
                out_seg,
                size=(TEST_CONFIG["img_size"], TEST_CONFIG["img_size"]),
                mode="bilinear",
                align_corners=True
            )

        loss = structure_loss(out_seg, masks)
        dice, iou = dice_iou_from_logits(out_seg, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(test_loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
