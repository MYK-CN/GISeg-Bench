import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from network.model import ConDSeg   # 你的模型

# ================== 基本配置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "ConDSeg"
MODEL_WEIGHT_PATH = "con_dseg_epoch_20.pth"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 图像预处理（需与训练一致） ==================
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================== 推理数据集 ==================
class XirouInferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name

# ================== DataLoader ==================
dataset = XirouInferenceDataset(
    image_dir=IMAGE_DIR,
    transform=image_transform
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ================== 加载模型 ==================
model = ConDSeg().to(device)
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device))
model.eval()

print(f"Model loaded: {MODEL_WEIGHT_PATH}")

# ================== 推理并保存结果 ==================
with torch.no_grad():
    for idx, (images, _) in enumerate(tqdm(loader, desc="Inferencing")):
        images = images.to(device)

        outputs, _, _, _ = model(images)

        # 二值化
        preds = (outputs > 0.5).float()

        # 转为 0 / 255
        pred_np = preds.squeeze().cpu().numpy()
        pred_np = (pred_np * 255).astype(np.uint8)

        pred_img = Image.fromarray(pred_np)

        save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
        save_path = os.path.join(SAVE_DIR, save_name)
        pred_img.save(save_path)

print("Inference finished!")
print(f"Results saved to: {SAVE_DIR}")
