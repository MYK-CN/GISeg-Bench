import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/densenet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"./data/Kvasir-SEG"
TEST_IMAGE_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./weights/DenseNet-Kvasir-SEG-best.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
# 测试数据集（stem 匹配，防止 _result）
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
                self.samples.append((
                    os.path.join(self.img_dir, f),
                    os.path.join(self.mask_dir, mask_index[stem])
                ))

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
# DenseNet 分割模型（与训练一致）
class DenseNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        self.backbone = backbone.features
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=32,
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        out = self.upsample(out)
        return {"out": out}
# 指标
def dice_iou(pred, target, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    inter = (pred * target).sum()
    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    iou = (inter + eps) / (pred.sum() + target.sum() - inter + eps)
    return dice.item(), iou.item()
# 测试主流程
@torch.no_grad()
def main():
    dataset = KvasirTestDataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0  # Windows 必须 0
    )

    model = DenseNetSeg(num_classes=2)
    print("[INFO] Loading weights:", WEIGHT_PATH)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = total_dice = total_iou = 0.0

    print("===== Start Testing =====")
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)["out"]

        loss = criterion(outputs, masks)
        dice, iou = dice_iou(outputs, masks)

        total_loss += loss.item()
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
