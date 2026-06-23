import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from VANet import VANet

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/cnn/viewpoint_aware_net')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ★★★ 导入 VANet 模型 ★★★
# 假设你将提供的模型代码保存为了 VANet.py
# 并且你的目录下有 lib 文件夹 (包含 config, models 等依赖)
# try:
#     from VANet import VANet
# except ImportError:
#     print("错误：无法导入 VANet。请确保 'VANet.py' 存在，且 'lib' 文件夹及依赖完整。")
#     exit()

# =================配置参数=================
DATA_DIR = r"./data/Kvasir-SEG"
# 预训练权重路径 (如果为 None 则随机初始化，不建议)
# 请确保下载了 CvT-13-224x224-IN-1k.pth 并放在对应位置
PRETRAINED_PATH = r"./weights/CvT-13-224x224-IN-1k.pth"
CONFIG_PATH = r"./experiments/imagenet/cvt/cvt-13-224x224.yaml"

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# VANet/CvT 通常使用 224x224
IMAGE_SIZE = (224, 224)
# 1. 自定义数据集
class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        self.images = [x for x in sorted(os.listdir(self.images_dir))
                       if x.endswith('.jpg') or x.endswith('.png')]

        self.img_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            # ImageNet 标准归一化，对应 CvT 预训练
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_trans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # 读取图片
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        image = self.img_trans(image)
        mask = self.mask_trans(mask)

        # 处理 Mask 为 FloatTensor (1, H, W)，用于 BCE Loss
        mask = (mask > 0.5).float()

        return image, mask

# 2. 获取模型
def get_vanet_model():
    print("正在加载 VANet 模型...")
    # 注意：需要确保 cfg 和 weights 路径正确，否则模型可能会报错或无法加载预训练参数
    # num_class=1 表示二分类
    model = VANet(num_class=1,
                  cfg=CONFIG_PATH if os.path.exists(CONFIG_PATH) else None,
                  weights=PRETRAINED_PATH if os.path.exists(PRETRAINED_PATH) else None)
    return model

# 3. 损失函数 (Deep Supervision)
def structure_loss(preds, target):
    """
    preds: tuple (out3, out2, out1, out0)
           out3 是最终输出 (最高分辨率)
    target: (B, 1, 224, 224)
    """

    # VANet 输出的是 Logits (没有经过 Sigmoid)，所以使用 BCEWithLogitsLoss
    bce = nn.BCEWithLogitsLoss()

    loss = 0.0
    # 遍历所有尺度的输出
    for pred in preds:
        # 将预测结果上采样到与 Target 一致的大小
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        loss += bce(pred, target)

    return loss

# 4. Dice 系数计算
def calculate_dice(pred_logits, target):
    # 先做 Sigmoid 得到概率
    pred_probs = torch.sigmoid(pred_logits)
    pred_mask = (pred_probs > 0.5).float()

    smooth = 1e-5
    intersection = (pred_mask * target).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + target.sum() + smooth)
    return dice.item()

# 5. 主训练
if __name__ == '__main__':

    # 检查依赖文件
    if not os.path.exists(CONFIG_PATH):
        print(f"警告：找不到配置文件 {CONFIG_PATH}，模型将使用默认配置。")
    if not os.path.exists(PRETRAINED_PATH):
        print(f"警告：找不到预训练权重 {PRETRAINED_PATH}，模型将从头开始训练（效果可能不佳）。")

    # 加载数据
    try:
        dataset = SimpleDataset(DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2, drop_last=True)
        print(f"成功加载数据集，共 {len(dataset)} 张图片。")
    except Exception as e:
        print("数据加载失败：", e)
        exit()

    # 初始化模型
    model = get_vanet_model().to(DEVICE)

    # 优化器
    # Transformer 类模型通常推荐使用 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率调整策略 (可选)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"开始训练，设备: {DEVICE}，批大小: {BATCH_SIZE}")

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            # VANet 的 forward 定义为 forward(self, x, cue=None)
            # 训练时我们只传入 images，cue 默认为 None
            outputs = model(images)

            # outputs 是一个 tuple: (out3, out2, out1, out0)
            # 计算深层监督 Loss
            loss = structure_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算 Dice，只使用最终输出 out3 (outputs[0])
            # 同样需要上采样对齐尺寸
            final_pred = outputs[0]
            if final_pred.shape[2:] != masks.shape[2:]:
                final_pred = F.interpolate(final_pred, size=masks.shape[2:], mode='bilinear', align_corners=True)

            running_dice += calculate_dice(final_pred, masks)

        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice: {epoch_dice:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存权重
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vanet_medical_model.pth"))
    print("训练完成！模型已保存为 vanet_medical_model.pth")
