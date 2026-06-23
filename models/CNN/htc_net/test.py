import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/htc_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
# 路径配置
DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "masktest")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = r"./weights/HTC-Net-master-Kvasir-SEG-best.pth"
SAVE_PRED_DIR = os.path.join(ROOT_DIR, "test_predictions")
os.makedirs(SAVE_PRED_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset（与训练一致）
class SegmentationTestDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
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
        self.images = [p[0] for p in pairs]
        self.masks = [p[1] for p in pairs]

        if not self.images:
            raise RuntimeError("未找到任何 test image-mask 对，请检查路径")

        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask = (np.array(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

        name = os.path.basename(self.images[idx])
        return img, mask, name
# 指标
def dice_score(pred, gt, eps=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)

def iou_score(pred, gt, eps=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)
# 模型导入（与你训练一致）
from network.Net import model as SwinModelWrapper
import ml_collections
# 测试主函数
def test():
    print("[info] Device:", DEVICE)

    dataset = SegmentationTestDataset(
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

    # ---- 构建 config（必须与训练一致）----
    cfg = ml_collections.config_dict.ConfigDict()
    cfg.n_classes = 1
    cfg.decoder_channels = (128, 64, 32, 16)
    cfg.n_skip = 3

    model = SwinModelWrapper(
        config=cfg,
        img_size=IMG_SIZE,
        num_classes=1
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"模型未找到: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            d = dice_score(preds, masks)
            i = iou_score(preds, masks)

            dice_list.append(d.item())
            iou_list.append(i.item())

            # 保存预测
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
