import os
import cv2
import torch
import argparse
import importlib.util
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medsam（wu-test）')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from segment_anything import sam_model_registry
# 1. External Data Loader Support
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=1):
    spec = importlib.util.spec_from_file_location(
        "universal_data_loader", data_loader_path
    )
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        num_workers=0
    )
# 2. Built-in Dataset (Fallback)
IMG_SIZE = 1024

def mask_to_box(mask_np):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def resize_longest_side(image, size=1024):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    newh, neww = int(h * scale), int(w * scale)
    return cv2.resize(image, (neww, newh))

class DefaultSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask = (mask > 0).astype(np.uint8)

        img = resize_longest_side(img, IMG_SIZE)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        box = mask_to_box(mask)

        img = torch.from_numpy(img).float().permute(2, 0, 1)     # [3,H,W]
        mask = torch.from_numpy(mask).float()                    # [H,W]

        return img, mask, torch.from_numpy(box), img.shape[-2:]
# 3. Loss
class DiceLoss(nn.Module):
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        return 1 - (2 * inter + 1e-5) / (union + 1e-5)
# [NEW] Metrics
def dice_metric(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return ((2 * inter + eps) / (union + eps)).item()

def iou_metric(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return ((inter + eps) / (union + eps)).item()

# 4. Main Training Logic
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading MedSAM...")
    model = sam_model_registry["vit_b"](checkpoint=args.pretrain)
    model.to(device)
    model.train()

    # Freeze image encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(model.prompt_encoder.parameters()) +
        list(model.mask_decoder.parameters()),
        lr=1e-4
    )

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    # DataLoader
    if args.data_loader:
        print("[Training] Using external data loader")
        train_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            args.batch_size
        )
    else:
        print("[Training] Using built-in data loader")
        dataset = DefaultSegDataset(args.image_folder, args.mask_folder)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # -------- Unified batch structure --------
            if len(batch) == 2:
                image, mask = batch
                image = image.to(device)
                mask = mask.to(device)

                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)

                B, _, H, W = image.shape

                box_list = []
                for b in range(B):
                    mb = mask[b]
                    ys, xs = torch.where(mb > 0)
                    if len(xs) == 0:
                        box_list.append(torch.tensor([0, 0, 1, 1], device=device))
                    else:
                        box_list.append(
                            torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device)
                        )
                box = torch.stack(box_list, dim=0)
                orig_size = (H, W)

            else:
                image, mask, box, orig_size = batch
                image = image.to(device)
                mask = mask.to(device)
                box = box.to(device)

                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)

            # -------- SAM preprocess --------
            image = (image - model.pixel_mean) / model.pixel_std
            image = nn.functional.pad(
                image,
                (0, IMG_SIZE - image.shape[-1],
                 0, IMG_SIZE - image.shape[-2])
            )

            img_embed = model.image_encoder(image)

            box = box / torch.tensor(
                [orig_size[1], orig_size[0],
                 orig_size[1], orig_size[0]],
                device=device
            ) * IMG_SIZE

            sparse, dense = model.prompt_encoder(
                points=None,
                boxes=box.unsqueeze(1),
                masks=None
            )

            low_res_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )

            pred = nn.functional.interpolate(
                low_res_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

            # -------- loss --------
            loss = bce(pred, mask) + dice(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ================= [NEW] Metrics =================
            with torch.no_grad():
                d = dice_metric(pred, mask)
                iou = iou_metric(pred, mask)

            total_loss += loss.item()
            total_dice += d
            total_iou += iou

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{d:.4f}",
                iou=f"{iou:.4f}"
            )

        n = len(train_loader)
        print(
            f"[Epoch {epoch+1}] "
            f"Avg Loss: {total_loss/n:.4f} | "
            f"Dice: {total_dice/n:.4f} | "
            f"IoU: {total_iou/n:.4f}"
        )

    print("[INFO] Training finished.")
# 5. argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser("MedSAM Fine-tuning")

    parser.add_argument("--data_loader", type=str, default=None)
    parser.add_argument(
        "--image_folder",
        type=str,
        default=r"./data/Kvasir-SEG/images"
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=r"./data/Kvasir-SEG/masks"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=r"./pretrained_ckpt/medsam_point_prompt_flare22.pth"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()
    main(args)
