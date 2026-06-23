import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/h2former')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 1. 路径配置 =================
sys.path.append(r"./models")

images_dir = r"./data/Kvasir-SEG/images"
masks_dir = r"./data/Kvasir-SEG/masks"
weight_path = r"./pretrained_ckpt/resnet34-333f7ec4.pth"

from models.H2Former import Res34_Swin_MS, BasicBlock

# ================= 2. Dice 指标 =================
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

# ================= 3. ResNet34 权重加载（3 → 4 通道） =================
def load_resnet34_weights(model, weight_path):
    print(f"Loading pretrained weights: {weight_path}")

    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location="cpu")

    model_dict = model.state_dict()
    new_dict = {}

    for k, v in checkpoint.items():
        if k in model_dict:
            if k == "conv1.weight":
                new_weight = torch.zeros_like(model_dict[k])
                new_weight[:, :3, :, :] = v
                new_weight[:, 3:, :, :] = torch.randn_like(new_weight[:, 3:, :, :]) * 0.01
                new_dict[k] = new_weight
            elif v.shape == model_dict[k].shape:
                new_dict[k] = v

    model.load_state_dict(new_dict, strict=False)
    print("Pretrained weights loaded (Conv1 adapted to 4 channels)")
    return model

# ================= 4. 数据集 =================
class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

# ================= 5. 主训练流程 =================
if __name__ == "__main__":

    # -------- 参数 --------
    image_size = 224
    batch_size = 16
    epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------- 数据 --------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = PolypDataset(images_dir, masks_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # -------- 模型 --------
    model = Res34_Swin_MS(
        image_size=image_size,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=1
    )

    model = load_resnet34_weights(model, weight_path)
    model.to(device)

    # -------- 损失 & 优化 --------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n===== Start Training =====\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # 3 → 4 通道
            extra_channel = imgs.mean(dim=1, keepdim=True)
            imgs = torch.cat([imgs, extra_channel], dim=1)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice.item():.4f}"
            })

        epoch_loss /= len(dataloader)
        epoch_dice /= len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} "
              f"Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}")

    # -------- 保存模型 --------
    save_path = "h2former_kvasir_final.pth"
    torch.save(model.state_dict(), save_path)
    print("\nTraining Finished!")
    print("Model saved to:", save_path)
