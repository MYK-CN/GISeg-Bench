import os
import sys
import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medical_sam_adapter')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 0. Universal Data Loader Utility =================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size, img_size):
    spec = importlib.util.spec_from_file_location("universal_data_loader", data_loader_path)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=img_size,
        num_workers=0,
    )

# ================= Configuration =================
class DefaultConfig:
    DATA_ROOT = r"./data/Kvasir-SEG"
    BASE_SAM_PATH = r"./pretrained_ckpt/sam_vit_b_01ec64.pth"
    PRETRAINED_ADAPTER_PATH = r"./pretrained_ckpt/Melanoma_Photo_SAM_1024.pth"
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "outputs/foundation/medical_sam_adapter"

sys.path.append(os.getcwd())

from models.sam.build_sam import build_sam_vit_b

# ================= Dataset =================
class PolypDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        mask_path = None
        for cand in [
            img_name.replace(".jpg", ".png"),
            img_name.replace(".jpeg", ".png"),
            img_name
        ]:
            p = os.path.join(self.masks_dir, cand)
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
        return 1 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()

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

# ================= Custom Forward =================
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

# ================= Training =================
def train(args):
    os.makedirs(DefaultConfig.SAVE_DIR, exist_ok=True)
    batch_size = args.batch_size or DefaultConfig.BATCH_SIZE
    img_size = DefaultConfig.IMAGE_SIZE

    if args.data_loader and args.image_folder and args.mask_folder:
        dataloader = load_external_dataloader(
            args.data_loader, args.image_folder, args.mask_folder,
            batch_size=batch_size, img_size=img_size
        )
    else:
        dataset = PolypDataset(DefaultConfig.DATA_ROOT, img_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ================= [FIX] 完整 SAM Adapter 参数 =================
    sam_args = argparse.Namespace(
        image_size=img_size,
        mod="sam_adpt",
        sam_checkpoint=DefaultConfig.BASE_SAM_PATH,
        type="map",
        encoder_adapter=True,
        mid_dim=None,          #  关键修复
        up_dim=None,           # 🔧关键修复
        multimask_output=1,
        vit_out_dim=256,
        thd=False,
        chunk=None,
        num_sample=1,
        evl_chunk=None,
    )
    model = build_sam_vit_b(checkpoint=DefaultConfig.BASE_SAM_PATH, args=sam_args)
    model.to(DefaultConfig.DEVICE)
    model.train()

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DefaultConfig.LR)
    scaler = GradScaler()

    loss_bce = nn.BCEWithLogitsLoss()
    loss_dice = DiceLoss()

    print(" Start training (Dice + IoU enabled)...", flush=True)

    for epoch in range(DefaultConfig.EPOCHS):
        epoch_loss = epoch_dice = epoch_iou = 0.0

        for i, (images, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            images = images.to(DefaultConfig.DEVICE)
            masks = masks.to(DefaultConfig.DEVICE)

            batched_input = [
                {"image": images[j] * 255., "original_size": (img_size, img_size)}
                for j in range(len(images))
            ]

            with autocast():
                outputs = custom_forward(model, batched_input)
                preds = torch.stack([o["low_res_logits"] for o in outputs]).squeeze(1)
                preds = F.interpolate(preds, (img_size, img_size), mode="bilinear", align_corners=False)
                loss = 0.5 * loss_bce(preds, masks) + 0.5 * loss_dice(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                d = dice_metric(preds, masks)
                iou = iou_metric(preds, masks)

            epoch_loss += loss.item()
            epoch_dice += d
            epoch_iou += iou

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] "
                    f"Loss {loss.item():.4f} Dice {d:.4f} IoU {iou:.4f}",
                    flush=True
                )

        n = len(dataloader)
        print(
            f" Epoch {epoch+1} Finished | "
            f"Loss {epoch_loss/n:.4f} "
            f"Dice {epoch_dice/n:.4f} "
            f"IoU {epoch_iou/n:.4f}",
            flush=True
        )

        torch.save(model.state_dict(), os.path.join(DefaultConfig.SAVE_DIR, f"sam_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loader", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--mask_folder", type=str)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    train(args)
