import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry

# =====================================================
# 基本配置（直接改这里即可）
# =====================================================
MODEL_NAME = "MedSAM"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

# 训练好的 Adapter 权重
MODEL_PATH = r"./weights/Medical-SAM-Adapter-main-Kvasir-SEG-best.pth"

# SAM backbone 权重
BASE_SAM_PATH = r"./pretrained_ckpt/medsam_point_prompt_flare22.pth"

IMG_SIZE = 1024
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# 推理数据集（无 mask）
# =====================================================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir):
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = IMG_SIZE / max(h, w)
        newh, neww = int(h * scale), int(w * scale)
        img = cv2.resize(img, (neww, newh))

        img = torch.from_numpy(img).float().permute(2, 0, 1)

        return img, (h, w), os.path.basename(path)


# =====================================================
# 推理主流程
# =====================================================
def inference():
    print("[info] Device:", DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = XirouInferenceDataset(IMAGE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # ---------------- 构建 MedSAM ----------------
    model = sam_model_registry["vit_b"](checkpoint=BASE_SAM_PATH)
    model.to(DEVICE)
    model.eval()

    # 加载 Adapter 训练权重
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

    print("[info] Model loaded:", MODEL_PATH)

    # ---------------- 推理 ----------------
    with torch.no_grad():
        for idx, (imgs, orig_sizes, _) in enumerate(tqdm(loader, desc="Inferencing")):
            imgs = imgs.to(DEVICE)

            # SAM preprocess
            imgs = (imgs - model.pixel_mean) / model.pixel_std
            imgs = F.pad(
                imgs,
                (0, IMG_SIZE - imgs.shape[-1],
                 0, IMG_SIZE - imgs.shape[-2])
            )

            # image encoder
            image_embeddings = model.image_encoder(imgs)

            # 无 prompt
            sparse, dense = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )

            low_res_logits, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )

            preds = F.interpolate(
                low_res_logits,
                size=orig_sizes[0],
                mode="bilinear",
                align_corners=False
            )

            preds = (torch.sigmoid(preds) > 0.5).float()

            pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
            pred_img = Image.fromarray(pred_np)

            save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
            pred_img.save(os.path.join(SAVE_DIR, save_name))

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


# =====================================================
# 直接运行（无命令行）
# =====================================================
if __name__ == "__main__":
    inference()
