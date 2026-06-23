import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from vit_seg_modeling import VisionTransformer  # 确保这个文件存在
from vit_seg_configs import get_r50_b16_config  # 确保这个文件存在

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transunet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================配置参数=================
# ★★★ 请务必确认你的数据集路径是正确的 ★★★
DATA_DIR = r'./data/Kvasir-SEG'  # 使用 r'...' 避免路径中的反斜杠问题
BATCH_SIZE = 4  # 如果显存不够(报OOM错误)，请改小，比如 2 或 1
LR = 1e-4  # 学习率
EPOCHS = 20  # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 定义图像和掩码的数据变换 (★★★ 关键修改点 ★★★)
# 图像变换：包含缩放、转Tensor和标准化
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 掩码变换：只包含缩放和转Tensor，绝不能有Normalize！
# 插值方式使用NEAREST，防止在边缘产生0和1之外的模糊值
mask_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# 2. 定义数据集加载器 (★★★ 关键修改点 ★★★)
class MedicalDataset(Dataset):
    def __init__(self, root_dir):
        # 不再需要外部传入 transform
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        # 读取文件名列表并排序，确保image和mask能对应上
        self.images = sorted(os.listdir(self.images_dir))

        # 将变换流程作为类的属性
        self.image_transform = image_transforms
        self.mask_transform = mask_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)  # 假设mask文件名和原图一致

        # 打开图片
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Mask转为灰度图 (单通道)

        # ★★★ 分别对 image 和 masks 应用不同的 transform ★★★
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # ★★★ 修复标签类型和维度问题 ★★★
        # CrossEntropyLoss 要求 target 是 (N, H, W) 且为 Long 类型
        # 1. 将 masks 从 [0, 1] 浮点数转为 0 或 1 整数
        # 2. .squeeze(0) 去掉通道维度，从 (1, H, W) -> (H, W)
        # 3. .long() 转换为 LongTensor
        mask = (mask > 0.5).long().squeeze(0)

        return image, mask

# 3. 加载 Vision Transformer 模型
def get_vit_model():
    print("正在加载 Vision Transformer 模型...")
    config = get_r50_b16_config()
    # num_classes=2 表示输出2个通道：(背景概率, 前景概率)
    model = VisionTransformer(config=config, img_size=256, num_classes=2)
    return model

# 4. Dice 系数计算函数 (评估指标)
def calculate_dice(pred, target):
    # pred: (B, 2, H, W) -> argmax -> (B, H, W)
    # 沿着通道维度(dim=1)取最大值的索引，得到预测的类别图 (0或1)
    pred_mask = torch.argmax(pred, dim=1)

    # 展平
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)  # 加一个极小值防止分母为0
    return dice.item()

# 5. 主训练流程
if __name__ == '__main__':
    # 加载数据
    try:
        # ★★★ 初始化Dataset时不再传入transform ★★★
        dataset = MedicalDataset(DATA_DIR)
        # 建议 num_workers 在 Windows 下设为 0 进行调试，成功后再改
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print(f"成功加载数据集，共 {len(dataset)} 张图片。")
    except Exception as e:
        print(f"数据加载失败，请检查路径: {e}")
        exit()

    # 初始化模型、损失函数、优化器
    model = get_vit_model().to(DEVICE)
    # CrossEntropyLoss 适用于多分类（这里是背景和前景2类）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始训练，设备: {DEVICE}，批次大小: {BATCH_SIZE}")

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        # 使用 for...in... 循环遍历 dataloader
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)  # masks 现在是 (B, H, W) 的 LongTensor

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)  # outputs 的形状是 (B, 2, H, W)

            # 计算损失 (outputs: (B, 2, H, W), masks: (B, H, W))
            loss = criterion(outputs, masks)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录数据
            running_loss += loss.item()
            running_dice += calculate_dice(outputs.detach(), masks.detach())  # detach() 避免影响计算图

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Dice Score: {epoch_dice:.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vit_medical_model.pth"))
    print("训练完成！模型已保存为 vit_medical_model.pth")
