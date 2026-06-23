import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
import torch.nn.functional as F

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medical_sam_adapter')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 引入混合精度训练模块
from torch.cuda.amp import autocast, GradScaler

# ================= 配置区域 =================

class Config:
    # 1. 数据集路径
    DATA_ROOT = r"./data/Kvasir-SEG"

    # 2. 基础 SAM 权重 (ViT-B)
    BASE_SAM_PATH = r"./pretrained_ckpt/sam_vit_b_01ec64.pth"

    # 3. Melanoma 适配器权重
    PRETRAINED_ADAPTER_PATH = r"./pretrained_ckpt/Melanoma_Photo_SAM_1024.pth"

    # 4. 训练参数
    IMAGE_SIZE = 256  # 如果还爆显存，请把这里改成 512
    BATCH_SIZE = 1  #  强制改为 1 以节省显存
    LR = 0.0001
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "outputs/foundation/medical_sam_adapter"
current_dir = os.getcwd()
sys.path.append(current_dir)

try:
    from models.sam.build_sam import build_sam_vit_b
except ImportError as e:
    print(f" 导入错误: {e}")
    sys.exit()

# === 1. 数据集加载器 ===
class PolypDataset(Dataset):
    def __init__(self, root_dir, image_size=1024):
        self.root_dir = root_dir
        self.image_size = image_size
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        self.image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f" 已加载数据集: {len(self.image_files)} 张图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        mask_candidates = [
            img_name.replace('.jpg', '.png').replace('.jpeg', '.png'),
            img_name,
            img_name.replace('.png', '.jpg')
        ]

        mask_path = None
        for cand in mask_candidates:
            p = os.path.join(self.masks_dir, cand)
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# === 2. Loss Function ===
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# === 3. 参数补全 ===
def get_args():
    parser = argparse.Namespace()
    parser.image_size = Config.IMAGE_SIZE
    parser.mod = 'sam_adpt'
    parser.sam_checkpoint = Config.BASE_SAM_PATH
    parser.type = 'map'
    parser.encoder_adapter = True
    parser.mid_dim = None
    parser.up_dim = None
    parser.multimask_output = 1
    parser.vit_out_dim = 256
    parser.thd = False
    parser.chunk = None
    parser.num_sample = 1
    parser.evl_chunk = None
    return parser

# === 4. 自定义前向传播 (绕过 no_grad) ===
def custom_forward(model, batched_input):
    # 1. 预处理
    input_images = torch.stack([model.preprocess(x["image"]) for x in batched_input], dim=0)

    # 2. Image Encoder
    image_embeddings = model.image_encoder(input_images)

    outputs = []
    for image_record, curr_embedding in zip(batched_input, image_embeddings):
        # 3. Prompt Encoder
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        # 4. Mask Decoder
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        outputs.append({"low_res_logits": low_res_masks})

    return outputs

# === 5. 主训练流程 ===
def train():
    if not os.path.exists(Config.SAVE_DIR):
        os.makedirs(Config.SAVE_DIR)

    dataset = PolypDataset(Config.DATA_ROOT, image_size=Config.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    print(" 正在构建 Med-SA (ViT-B) 模型...")

    my_args = get_args()
    try:
        model = build_sam_vit_b(checkpoint=Config.BASE_SAM_PATH, args=my_args)
    except Exception as e:
        print(f" 构建模型失败: {e}")
        return

    if os.path.exists(Config.PRETRAINED_ADAPTER_PATH):
        print(f" 加载适配器权重...")
        adapter_ckpt = torch.load(Config.PRETRAINED_ADAPTER_PATH, map_location='cpu')
        if 'model' in adapter_ckpt: adapter_ckpt = adapter_ckpt['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in adapter_ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f" 成功加载参数")

    model.to(Config.DEVICE)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f" 可训练参数量: {len(trainable_params)} 组 tensor")

    optimizer = optim.AdamW(trainable_params, lr=Config.LR)

    # === 关键：初始化 GradScaler 用于混合精度训练 ===
    scaler = GradScaler()

    criterion_dice = DiceLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    print(f" 开始训练 (Mixed Precision Mode)...")

    for epoch in range(Config.EPOCHS):
        epoch_loss = 0
        for i, (images, masks) in enumerate(dataloader):
            # 清理之前的梯度和显存缓存
            optimizer.zero_grad()

            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE).float()

            batched_input = []
            for j in range(len(images)):
                img_255 = images[j] * 255.0
                input_dict = {'image': img_255, 'original_size': (Config.IMAGE_SIZE, Config.IMAGE_SIZE)}
                batched_input.append(input_dict)

            # === 关键：使用 autocast 自动混合精度 ===
            with autocast():
                outputs = custom_forward(model, batched_input)

                pred_list = [out['low_res_logits'] for out in outputs]
                preds = torch.stack(pred_list, dim=0)

                if preds.dim() == 5: preds = preds.squeeze(1)

                preds = F.interpolate(preds, size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), mode='bilinear',
                                      align_corners=False)

                # 在计算 Loss 时，最好转回 float32 保证精度，或者由 autocast 处理
                loss = 0.5 * criterion_bce(preds, masks) + 0.5 * criterion_dice(preds, masks)

            # === 关键：使用 scaler 进行反向传播 ===
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # 手动清理缓存（如果显存极其紧张）
            if i % 5 == 0:
                torch.cuda.empty_cache()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "latest_model.pth"))
        print(f" Epoch {epoch + 1} 完成, Avg Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    # 设置显存分配策略，防止碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    try:
        train()
    except Exception as e:
        print(f" 错误: {e}")
        import traceback

        traceback.print_exc()
