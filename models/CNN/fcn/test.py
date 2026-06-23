import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/fcn')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================配置参数=================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = r"./data/Kvasir-SEG"
TEST_IMAGE_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./weights/fcn-Kvasir-SEG-best.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
# 1. 测试数据集类
class KvasirTestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, TEST_IMAGE_DIR)
        self.mask_dir = os.path.join(root_dir, TEST_MASK_DIR)

        exts = (".png", ".jpg", ".jpeg")

        # 建立 mask 索引
        mask_index = {}
        for f in os.listdir(self.mask_dir):
            if f.lower().endswith(exts):
                mask_index[os.path.splitext(f)[0]] = f

        self.samples = []
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            if stem in mask_index:
                self.samples.append(
                    (
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem])
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError("❌ test / masktest 中没有匹配的 image-mask 对")

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

        self.img_tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_tf(image)
        mask = self.mask_tf(mask)

        mask = (mask > 0).long().squeeze(0)
        return image, mask

# 2. Dice 系数和 IoU 计算函数
def dice_iou(pred_probs, targets, eps=1e-6):
    """
    pred_probs: softmax 概率 (B,2,H,W)
    targets: (B,H,W)
    """

    pred_mask = torch.argmax(pred_probs, dim=1)

    pred_flat = pred_mask.view(-1)
    target_flat = targets.view(-1)

    inter = (pred_flat * target_flat).sum()

    dice = (2 * inter + eps) / (pred_flat.sum() + target_flat.sum() + eps)
    iou = (inter + eps) / (pred_flat.sum() + target_flat.sum() - inter + eps)

    return dice.item(), iou.item()

# 3. 加载模型
def get_fcn_model():
    model = models.segmentation.fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # 输出 2 类
    model.aux_classifier = None
    return model

# 4. 测试主函数
@torch.no_grad()
def main():
    dataset = KvasirTestDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 加载模型
    model = get_fcn_model().to(DEVICE)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    print("===== Start Testing =====")
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # 模型前向传播 -> softmax 概率
        outputs = model(images)["out"]

        # 损失计算
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # 计算 Dice 和 IoU
        dice, iou = dice_iou(outputs, masks)
        total_dice += dice
        total_iou += iou

    n = len(loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

if __name__ == "__main__":
    main()
