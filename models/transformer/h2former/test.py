import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/h2former')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ======== 模型路径 ========
sys.path.append(r"./models")
from models.H2Former import Res34_Swin_MS, BasicBlock
# 配置
DATA_ROOT = r"./data/Kvasir-SEG"

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "masktest")

MODEL_PATH = r"./weights/h2former_kvasir_epoch_4.pth"   # 改成你的模型
SAVE_PRED_DIR = r"./test_predictions"

IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
# Dataset（测试专用）
class PolypTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        mask = (mask > 0).float()  # [1, H, W]

        return img, mask, name
# 指标
def dice_score(pred, gt, smooth=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    inter = (pred * gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)

def iou_score(pred, gt, smooth=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)
# 测试主函数
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] Device:", device)

    os.makedirs(SAVE_PRED_DIR, exist_ok=True)

    dataset = PolypTestDataset(TEST_IMG_DIR, TEST_MASK_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Test samples:", len(dataset))

    model = Res34_Swin_MS(
        image_size=IMAGE_SIZE,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=1
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # ===== 3 → 4 通道（和训练完全一致）=====
            extra_channel = imgs.mean(dim=1, keepdim=True)
            imgs = torch.cat([imgs, extra_channel], dim=1)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            d = dice_score(preds, masks)
            i = iou_score(preds, masks)

            dice_list.append(d.item())
            iou_list.append(i.item())

            # 保存预测结果（可选）
            pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(pred_np).save(
                os.path.join(SAVE_PRED_DIR, names[0])
            )

    print("=" * 60)
    print(f"[Test] Mean Dice: {np.mean(dice_list):.4f}")
    print(f"[Test] Mean IoU : {np.mean(iou_list):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    test()
