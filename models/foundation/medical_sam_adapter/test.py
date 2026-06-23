import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medical_sam_adapter')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 配置区域 =================
class Config:
    # 测试集路径
    TEST_ROOT = r"./data/Kvasir-SEG"
    TEST_IMG_DIR = os.path.join(TEST_ROOT, "test")
    TEST_MASK_DIR = os.path.join(TEST_ROOT, "masktest")

    # 预训练模型路径（你给的）
    MODEL_PATH = r"./weights/Medical-SAM-Adapter-main-Kvasir-SEG-best.pth"

    # SAM 基础权重（仅用于构建模型结构）
    BASE_SAM_PATH = r"./pretrained_ckpt/sam_vit_b_01ec64.pth"

    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append(os.getcwd())

from models.sam.build_sam import build_sam_vit_b

# ================= 数据集（测试用，无增强） =================
class PolypTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)

        mask_path = None
        for cand in [
            name,
            name.replace(".jpg", ".png"),
            name.replace(".jpeg", ".png"),
        ]:
            p = os.path.join(self.mask_dir, cand)
            if os.path.exists(p):
                mask_path = p
                break

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0) if mask_path else np.zeros(image.shape[:2], np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1)
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return image, mask

# ================= Loss =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# ================= Metrics =================
def dice_metric(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2 * inter + eps) / (union + eps)).item()

def iou_metric(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()

# ================= SAM Adapter Forward =================
def custom_forward(model, batched_input):
    input_images = torch.stack([model.preprocess(x["image"]) for x in batched_input])
    image_embeddings = model.image_encoder(input_images)

    outputs = []
    for emb in image_embeddings:
        sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=emb.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        outputs.append({"low_res_logits": low_res_masks})
    return outputs

# ================= 测试主流程 =================
def test():
    print(f" Using device: {Config.DEVICE}")

    dataset = PolypTestDataset(
        Config.TEST_IMG_DIR,
        Config.TEST_MASK_DIR,
        Config.IMAGE_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # -------- 构建模型结构 --------
    sam_args = argparse.Namespace(
        image_size=Config.IMAGE_SIZE,
        mod="sam_adpt",
        sam_checkpoint=Config.BASE_SAM_PATH,
        type="map",
        encoder_adapter=True,
        mid_dim=None,
        up_dim=None,
        multimask_output=1,
        vit_out_dim=256,
        thd=False,
        chunk=None,
        num_sample=1,
        evl_chunk=None,
    )

    model = build_sam_vit_b(
        checkpoint=Config.BASE_SAM_PATH,
        args=sam_args
    )

    print(" Loading trained model weights...")
    state = torch.load(Config.MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(Config.DEVICE)
    model.eval()

    loss_bce = nn.BCEWithLogitsLoss()
    loss_dice = DiceLoss()

    total_loss = total_dice = total_iou = 0.0

    print(" Start testing...")

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            batched_input = [
                {"image": images[j] * 255., "original_size": (Config.IMAGE_SIZE, Config.IMAGE_SIZE)}
                for j in range(len(images))
            ]

            with autocast():
                outputs = custom_forward(model, batched_input)
                preds = torch.stack([o["low_res_logits"] for o in outputs]).squeeze(1)
                preds = F.interpolate(preds, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), mode="bilinear",
                                      align_corners=False)

                loss = 0.5 * loss_bce(preds, masks) + 0.5 * loss_dice(preds, masks)

            d = dice_metric(preds, masks)
            iou = iou_metric(preds, masks)

            total_loss += loss.item()
            total_dice += d
            total_iou += iou

            print(
                f"[{idx+1}/{len(dataloader)}] "
                f"Loss {loss.item():.4f} "
                f"Dice {d:.4f} "
                f"IoU {iou:.4f}"
            )

    n = len(dataloader)
    print("\n================ Test Summary ================")
    print(f"Avg Loss : {total_loss / n:.4f}")
    print(f"Avg Dice : {total_dice / n:.4f}")
    print(f"Avg IoU  : {total_iou / n:.4f}")
    print("=============================================")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    test()
