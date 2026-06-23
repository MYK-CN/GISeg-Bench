import os
import glob
import numpy as np
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/hiformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from medpy import metric

from models.HiFormer import HiFormer
import configs.HiFormer_configs as hcfg

# ======================= 1. 测试数据集 ==========================
class WCEBleedTestDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        super().__init__()
        self.img_size = img_size

        self.image_dir = os.path.join(root_dir, 'test')
        self.mask_dir = os.path.join(root_dir, 'masktest')

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {self.image_dir} 中未找到测试图片")

        print(f"[Test] 已加载 {len(self.image_paths)} 张测试图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # img-xxx.png -> ann-xxx.png
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('img-', 'ann-', 1)
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到 Mask: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.asarray(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # 与训练一致

        mask = np.asarray(mask).astype(np.float32)
        mask = (mask > 128).astype(np.int64)

        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

# ======================= 2. Dice & IoU ==========================
def compute_dice_iou(preds, labels, num_classes=2):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    dice_list = []
    iou_list = []

    for c in range(1, num_classes):  # 只算出血类
        pred_c = preds == c
        label_c = labels == c

        if label_c.sum() == 0 and pred_c.sum() == 0:
            continue

        dice = metric.dc(pred_c, label_c)
        iou = metric.jc(pred_c, label_c)

        dice_list.append(dice)
        iou_list.append(iou)

    mean_dice = np.mean(dice_list) if len(dice_list) > 0 else 0.0
    mean_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0

    return mean_dice, mean_iou

# ======================= 3. 测试主函数 ==========================
def test_hiformer():
    # ---------- 路径 ----------
    data_root = r"./data/WCEBleedGen/bleeding"
    weight_path = r"./weights/hiformer_final.pth"

    img_size = 224
    num_classes = 2
    batch_size = 1  # 测试建议 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ---------- Dataset ----------
    test_dataset = WCEBleedTestDataset(data_root, img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ---------- Model ----------
    config = hcfg.get_hiformer_b_configs()
    model = HiFormer(
        config=config,
        img_size=img_size,
        in_chans=3,
        n_classes=num_classes
    ).to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    print("成功加载权重:", weight_path)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            dice, iou = compute_dice_iou(preds, labels, num_classes)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    print("\n========= 测试结果 =========")
    print(f"Average Loss : {avg_loss:.4f}")
    print(f"Average Dice : {avg_dice:.4f}")
    print(f"Average IoU  : {avg_iou:.4f}")
    print("============================")

if __name__ == "__main__":
    test_hiformer()
