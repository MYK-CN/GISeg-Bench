import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/daeformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.DAEFormer import DAEFormer
# 配置
DATA_ROOT = r"./data/Kvasir-SEG"

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "masktest")

MODEL_PATH = r"./weights/daeformer_kvasir_best.pth"
SAVE_PRED_DIR = r"./test_predictions"

IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 1
NUM_WORKERS = 0
# Dataset（测试专用，强一致性）
class KvasirSegTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 只加载合法图像文件
        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Resize(
            (img_size, img_size), interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到对应 mask：{mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask).long()  # [H, W]

        return img, mask, name
# 指标：Dice & IoU
def dice_score(pred, gt, smooth=1e-6):
    pred = pred.view(-1).float()
    gt = gt.view(-1).float()
    inter = (pred * gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)

def iou_score(pred, gt, smooth=1e-6):
    pred = pred.view(-1).float()
    gt = gt.view(-1).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)
# 测试主函数
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] device:", device)

    os.makedirs(SAVE_PRED_DIR, exist_ok=True)

    dataset = KvasirSegTestDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        IMG_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Test samples:", len(dataset))

    model = DAEFormer(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("[info] Model loaded:", MODEL_PATH)

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)               # [1, 2, H, W]
            preds = torch.argmax(logits, dim=1)  # [1, H, W]

            d = dice_score(preds[0], masks[0])
            i = iou_score(preds[0], masks[0])

            dice_list.append(d.item())
            iou_list.append(i.item())

            # 保存预测 mask
            pred_np = preds[0].cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(pred_np).save(
                os.path.join(SAVE_PRED_DIR, names[0])
            )

    print("=" * 60)
    print(f"[Test] Mean Dice : {np.mean(dice_list):.4f}")
    print(f"[Test] Mean IoU  : {np.mean(iou_list):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    test()
