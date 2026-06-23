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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/foundation/medsam')
os.makedirs(OUTPUT_DIR, exist_ok=True)


from segment_anything import sam_model_registry

# ================= Configuration =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"./data/Kvasir-SEG"
TEST_IMAGE_DIR = "test"
TEST_MASK_DIR = "masktest"

WEIGHT_PATH = r"./pretrained_ckpt/medsam_point_prompt_flare22.pth"

IMG_SIZE = 1024
BATCH_SIZE = 1
# ========= External data loader =========
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
        num_workers=0,
        shuffle=False
    )

# ========= Built-in test dataset =========
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

class KvasirTestDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, TEST_IMAGE_DIR)
        self.mask_dir = os.path.join(root_dir, TEST_MASK_DIR)

        exts = (".png", ".jpg", ".jpeg")

        # build mask index by stem
        mask_index = {}
        for f in os.listdir(self.mask_dir):
            if f.lower().endswith(exts):
                mask_index[os.path.splitext(f)[0]] = f

        self.samples = []
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            if stem in mask_index:
                self.samples.append(
                    (
                        os.path.join(self.img_dir, f),
                        os.path.join(self.mask_dir, mask_index[stem]),
                        f  # keep original image name for reference
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError("No matched image-mask pairs found in test / masktest")

        print(f"[INFO] Test samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, name = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 0).astype(np.uint8)

        img = resize_longest_side(img, IMG_SIZE)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        box = mask_to_box(mask)

        img = torch.from_numpy(img).float().permute(2, 0, 1)     # [3,H,W]
        mask = torch.from_numpy(mask).float()                    # [H,W]

        return img, mask, torch.from_numpy(box), img.shape[-2:], name

# ========= Metrics =========
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

# ========= Main test =========
@torch.no_grad()
def main(args):
    device = DEVICE

    # -------- Data --------
    if args.data_loader:
        print("[Test] Using external data loader")
        test_loader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            args.batch_size
        )
    else:
        print("[Test] Using built-in KvasirTestDataset")
        dataset = KvasirTestDataset(DATA_DIR)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------- Model --------
    print("[INFO] Loading MedSAM...")
    model = sam_model_registry["vit_b"](checkpoint=args.pretrain)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    bce = nn.BCEWithLogitsLoss()

    print("===== Start Testing =====")
    for batch in tqdm(test_loader, desc="Testing"):
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
            image, mask, box, orig_size = batch[:4]
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

        # -------- Metrics --------
        loss = bce(pred, mask)
        d = dice_metric(pred, mask)
        iou = iou_metric(pred, mask)

        total_loss += loss.item()
        total_dice += d
        total_iou += iou

    n = len(test_loader)
    print("\n========== Test Results ==========")
    print(f"Loss : {total_loss / n:.4f}")
    print(f"Dice : {total_dice / n:.4f}")
    print(f"IoU  : {total_iou / n:.4f}")
    print("=================================")

# ========= Entry =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser("MedSAM Test")

    parser.add_argument("--data_loader", type=str, default=None)
    parser.add_argument(
        "--image_folder",
        type=str,
        default=r"./data/Kvasir-SEG/test"
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=r"./data/Kvasir-SEG/masktest"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=WEIGHT_PATH
    )
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    main(args)
