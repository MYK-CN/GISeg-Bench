import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medsam（wu-test）')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from segment_anything import sam_model_registry
# 1. 配置（直接写死）
DATA_ROOT = r"./data/Kvasir-SEG"
IMG_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")

CKPT_PRETRAIN = r"./pretrained_ckpt/medsam_point_prompt_flare22.pth"
SAVE_DIR = "outputs/foundation/medsam（wu-test）"
os.makedirs(SAVE_DIR, exist_ok=True)

SAM_TYPE = "vit_b"
IMG_SIZE = 1024
BATCH_SIZE = 1          # SAM 微调基本都是 1
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 2. 工具函数
def mask_to_box(mask):
    """从 GT mask 生成 bbox"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def resize_longest_side(image, size=1024):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    newh, neww = int(h * scale), int(w * scale)
    image = cv2.resize(image, (neww, newh))
    return image, scale
# 3. Dataset
class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        mask = (mask > 0).astype(np.uint8)

        # resize
        image, scale = resize_longest_side(image, IMG_SIZE)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        box = mask_to_box(mask)
        if box is None:
            box = np.array([0, 0, 1, 1], dtype=np.float32)

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        return image, mask, box, image.shape[-2:]
# 4. Loss
class DiceLoss(nn.Module):
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        return 1 - (2 * inter + 1e-5) / (union + 1e-5)
# 5. 训练主逻辑
def main():
    print("Loading SAM / MedSAM...")
    model = sam_model_registry[SAM_TYPE](checkpoint=CKPT_PRETRAIN)
    model.to(DEVICE)
    model.train()

    # 冻结 image encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(model.prompt_encoder.parameters()) +
        list(model.mask_decoder.parameters()),
        lr=LR
    )

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    dataset = KvasirDataset(IMG_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for image, mask, box, orig_size in pbar:
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            box = box.to(DEVICE)

            image = (image - model.pixel_mean) / model.pixel_std
            image = nn.functional.pad(
                image,
                (0, IMG_SIZE - image.shape[-1], 0, IMG_SIZE - image.shape[-2])
            )

            img_embed = model.image_encoder(image)

            box = box / torch.tensor(
                [orig_size[1], orig_size[0], orig_size[1], orig_size[0]],
                device=DEVICE
            ) * IMG_SIZE

            sparse, dense = model.prompt_encoder(
                points=None,
                boxes=box.unsqueeze(1),
                masks=None,
            )

            low_res_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            pred = nn.functional.interpolate(
                low_res_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

            loss = bce(pred, mask) + dice(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
        )

    print("Training finished.")

if __name__ == "__main__":
    main()
