import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from network.model import ConDSeg  # 你的模型文件

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/hybrid/condseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 一些加速相关设置 ==================
torch.backends.cudnn.benchmark = True  # 对固定尺寸输入加速卷积
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== 图像 & 掩膜预处理 ==================
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
    transforms.ToTensor()  # 掩膜只做 ToTensor，不做 normalize
])

# ================== 数据集定义 ==================
class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道掩膜

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 保证掩膜是 0/1（二值）
        mask = (mask > 0.5).float()  # [1, H, W]

        return image, mask

# ================== 数据加载器 ==================
image_dir = r"./data/Kvasir-SEG/images"
mask_dir = r"./data/Kvasir-SEG/masks"

dataset = KvasirSegDataset(
    image_dir,
    mask_dir,
    image_transform=image_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(
    dataset,
    batch_size=8,       # 显存不够可以改成 4 或 2
    shuffle=True,
    num_workers=4,      # 如果总是卡在这，可以改成 0 试试
    pin_memory=True
)

# ================== 模型 / 损失 / 优化器 ==================
model = ConDSeg().to(device)

# 模型里已经做了 Sigmoid，所以这里用 BCELoss
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 混合精度
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

num_epochs = 30

# ================== 训练循环 ==================
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)  # [B, 1, H, W]

            optimizer.zero_grad()

            # 只把“前向 + 反向的前向部分”放在 autocast 里
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs, mask_fg, mask_bg, mask_uc = model(images)  # outputs: [B,1,H,W]

            # ⚠️ 注意：损失在 autocast 外面算，强制用 float32，避免报错
            loss_main = criterion(outputs.float(), masks.float())

            scaler.scale(loss_main).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_main.item()
            pbar.set_postfix(loss=loss_main.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}")

        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"con_dseg_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print("模型已保存到:", save_path)

    print("Training complete!")
