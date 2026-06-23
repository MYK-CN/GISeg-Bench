import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/hybrid/condseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from network.model import ConDSeg   # 你的模型

# ================== 设备 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== 预处理（必须和训练一致） ==================
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ================== 数据集 ==================
class KvasirSegTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None,
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, img_name)

        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
            return image, mask, img_name
        else:
            return image, img_name

# ================== 指标（可选） ==================
def dice_score(pred, gt, eps=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    intersection = (pred * gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)

def iou_score(pred, gt, eps=1e-6):
    pred = pred.view(-1)
    gt = gt.view(-1)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + eps) / (union + eps)

# ================== 路径 ==================
test_image_dir = r"./data/Kvasir-SEG/test"
test_mask_dir  = r"./data/Kvasir-SEG/masktest"  # 没有可设为 None

save_pred_dir = r"./pred_results"
os.makedirs(save_pred_dir, exist_ok=True)

# ================== DataLoader ==================
test_dataset = KvasirSegTestDataset(
    test_image_dir,
    test_mask_dir,
    image_transform=image_transform,
    mask_transform=mask_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,      # 测试建议 1，方便保存
    shuffle=False,
    num_workers=0
)

# ================== 加载模型 ==================
model = ConDSeg().to(device)
model_path = "con_dseg_epoch_20.pth"   # 换成你自己的
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded:", model_path)

# ================== 测试循环 ==================
dice_list = []
iou_list = []

with torch.no_grad():
    for data in tqdm(test_loader, desc="Testing"):
        if test_mask_dir is not None:
            images, masks, names = data
            masks = masks.to(device)
        else:
            images, names = data

        images = images.to(device)

        outputs, _, _, _ = model(images)
        preds = (outputs > 0.5).float()  # 二值化

        # ================== 保存预测结果 ==================
        pred_np = preds.squeeze().cpu().numpy() * 255
        pred_img = Image.fromarray(pred_np.astype(np.uint8))
        pred_img.save(os.path.join(save_pred_dir, names[0]))

        # ================== 计算指标 ==================
        if test_mask_dir is not None:
            d = dice_score(preds, masks)
            i = iou_score(preds, masks)
            dice_list.append(d.item())
            iou_list.append(i.item())

# ================== 输出结果 ==================
if test_mask_dir is not None:
    print(f"Mean Dice: {np.mean(dice_list):.4f}")
    print(f"Mean IoU : {np.mean(iou_list):.4f}")

print("Testing complete!")
