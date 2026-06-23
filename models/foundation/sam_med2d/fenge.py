import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 基本配置
# =========================
MODEL_NAME = "SAMMed2D"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

# 【SAM-Med2D 初始构架权重】—— 只用于 build
SAM_INIT_CKPT = r"./pretrained_ckpt/sam-med2d_b.pth"

# 🔴【你训练好的权重】—— 真正决定推理效果的
TRAINED_CKPT = r"./weights/sam_med2d_epoch_10.pth"

IMG_SIZE = 256
BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# SAM-Med2D 构建（与你训练一致）
# =========================
def build_sam_med2d():
    from segment_anything.build_sam import build_sam_vit_b

    class Args:
        def __init__(self):
            self.image_size = IMG_SIZE
            self.sam_checkpoint = SAM_INIT_CKPT
            self.encoder_adapter = True
            self.sam_type = "vit_b"

    model = build_sam_vit_b(Args())
    return model


# =========================
# 推理数据集（无 GT）
# =========================
class InferenceDataset(Dataset):
    def __init__(self, img_dir, image_size):
        self.image_paths = []
        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
            self.image_paths += glob.glob(os.path.join(img_dir, e))
        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise RuntimeError("❌ images 目录为空")

        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = np.asarray(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img


# =========================
# 推理主流程
# =========================
@torch.no_grad()
def inference():
    print("[info] Device:", DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = InferenceDataset(IMAGE_DIR, IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # -------- 1. 构建模型（加载 sam-med2d_b.pth）--------
    model = build_sam_med2d()

    # -------- 2. 加载你训练好的权重（关键步骤）--------
    ckpt = torch.load(TRAINED_CKPT, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        print("[info] Loaded trained weights from ckpt['model']")
    else:
        model.load_state_dict(ckpt, strict=False)
        print("[info] Loaded trained weights directly")

    model.to(DEVICE)
    model.eval()

    print("[info] Init backbone :", SAM_INIT_CKPT)
    print("[info] Trained ckpt  :", TRAINED_CKPT)

    # -------- 3. 推理 --------
    idx_global = 1

    for imgs in tqdm(loader, desc="Inferencing"):
        imgs = imgs.to(DEVICE)

        # preprocess（与你训练一致）
        imgs = (imgs - model.pixel_mean) / model.pixel_std
        imgs = F.pad(
            imgs,
            (0, IMG_SIZE - imgs.shape[-1],
             0, IMG_SIZE - imgs.shape[-2])
        )

        # encoder
        img_emb = model.image_encoder(imgs)

        # 无 prompt（与你训练一致）
        sparse, dense = model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )

        # decoder
        low_res, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )

        # resize + 二值化
        masks = F.interpolate(
            low_res,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        masks = (torch.sigmoid(masks) > 0.5).float()
        mask_np = masks[0, 0].cpu().numpy().astype(np.uint8) * 255

        save_path = os.path.join(
            SAVE_DIR,
            f"{idx_global}-{MODEL_NAME}-xirou.png"
        )
        Image.fromarray(mask_np).save(save_path)

        idx_global += 1

    print("[info] Inference finished")
    print("[info] Results saved to:", SAVE_DIR)


# =========================
# 直接运行
# =========================
if __name__ == "__main__":
    inference()
