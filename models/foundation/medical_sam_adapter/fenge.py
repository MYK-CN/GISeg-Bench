import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 基本配置（直接改这里）
# =========================================================
MODEL_NAME = "MedicalSAM"

IMAGE_DIR = r"./data/Kvasir-SEG/xirou/images"
SAVE_DIR  = r"./results"

MODEL_PATH = r"./weights/Medical-SAM-Adapter-main-Kvasir-SEG-best.pth"
BASE_SAM_PATH = r"./pretrained_ckpt/sam_vit_b_01ec64.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 模型构建
# =========================================================
sys.path.append(os.getcwd())
from models.sam.build_sam import build_sam_vit_b


# =========================================================
# 推理数据集（无 mask）
# =========================================================
class XirouInferenceDataset(Dataset):
    def __init__(self, img_dir, image_size):
        self.image_size = image_size
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
        img_path = self.images[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        image = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        return image, os.path.basename(img_path)


# =========================================================
# SAM Adapter 前向（与你原测试代码一致）
# =========================================================
def custom_forward(model, batched_input):
    input_images = torch.stack(
        [model.preprocess(x["image"]) for x in batched_input]
    )

    image_embeddings = model.image_encoder(input_images)

    outputs = []
    for emb in image_embeddings:
        sparse, dense = model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=emb.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        outputs.append({"low_res_logits": low_res_masks})

    return outputs


# =========================================================
# 推理主流程
# =========================================================
def inference():
    print("[info] Device:", DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = XirouInferenceDataset(IMAGE_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[info] Total images:", len(dataset))

    # -------- 构建模型参数对象（不用 argparse） --------
    class Args:
        image_size = IMAGE_SIZE
        mod = "sam_adpt"
        sam_checkpoint = BASE_SAM_PATH
        type = "map"
        encoder_adapter = True
        mid_dim = None
        up_dim = None
        multimask_output = 1
        vit_out_dim = 256
        thd = False
        chunk = None
        num_sample = 1
        evl_chunk = None

    args = Args()

    model = build_sam_vit_b(
        checkpoint=BASE_SAM_PATH,
        args=args
    )

    # -------- 加载训练好的 Adapter 权重 --------
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()

    print("[info] Model loaded:", MODEL_PATH)

    # -------- 推理并保存 --------
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(loader, desc="Inferencing")):
            images = images.to(DEVICE)

            batched_input = [
                {
                    "image": images[j] * 255.0,
                    "original_size": (IMAGE_SIZE, IMAGE_SIZE)
                }
                for j in range(len(images))
            ]

            with autocast():
                outputs = custom_forward(model, batched_input)
                preds = torch.stack(
                    [o["low_res_logits"] for o in outputs]
                ).squeeze(1)

                preds = F.interpolate(
                    preds,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False
                )

                preds = (torch.sigmoid(preds) > 0.5).float()

            pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
            pred_img = Image.fromarray(pred_np)

            save_name = f"{idx + 1}-{MODEL_NAME}-xirou.png"
            save_path = os.path.join(SAVE_DIR, save_name)
            pred_img.save(save_path)

    print("[info] Inference finished!")
    print("[info] Results saved to:", SAVE_DIR)


# =========================================================
# 直接运行
# =========================================================
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    inference()
