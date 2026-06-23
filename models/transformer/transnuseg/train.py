import os
import glob
import cv2
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# Auto-generated: unified output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, r'outputs/transformer/transnuseg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Import model (ensure the models folder is in the current directory)
try:
    from models.transnuseg import TransNuSeg
except ImportError:
    print("[Error] Cannot find models.transnuseg. Please check the file structure.")

# ================= 0. [New] Dynamic Edge Generation Utility =================
def generate_edges_on_the_fly(masks):
    """
    Dynamically generate edges from masks using morphological operations (dilation - erosion)
    Args:
        masks: Tensor [B, 1, H, W], float, 0/1 (or 0-1)
    Returns:
        edges: Tensor [B, 1, H, W], float
    """
    # Ensure float type for max_pool operations
    if masks.dtype != torch.float32:
        masks = masks.float()

    # 1. Dilation -> simulated using MaxPool
    dilated = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)

    # 2. Erosion -> simulated using negative MaxPool
    eroded = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)

    # 3. Edge = Dilation - Erosion
    edges = dilated - eroded
    return edges

# ================= 1. [Modified] External DataLoader Support =================
def load_external_dataloader(data_loader_path, image_folder, mask_folder, batch_size=4, target_size=256):
    """
    Load external DataLoader and enforce Windows-compatible parameters
    """
    spec = importlib.util.spec_from_file_location(
        "universal_data_loader", data_loader_path
    )
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)

    print(f"[Info] External Loader: forcing img_size={target_size}, num_workers=0")

    return loader_module.get_data_loader(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        image_size=target_size,  # TransNuSeg usually requires 256
        num_workers=0,  # Must be 0 to fix Windows PicklingError
        shuffle=True
    )

# ================= Configuration =================
CONFIG = {
    "train_img_path": r"./data/fluorescence/data",
    "train_mask_path": r"./data/fluorescence/label",
    "img_size": 256,
    "batch_size": 4,
    "lr": 1e-4,
    "epochs": 15,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ================= 2. Built-in Dataset Class (Unchanged) =================
class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.img_list = glob.glob(os.path.join(img_dir, "*.*"))
        self.img_list = [x for x in self.img_list if x.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(self.img_list)} images.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_list)

    def generate_edge(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        mask_erosion = cv2.erode(mask, kernel, iterations=1)
        edge = mask - mask_erosion
        return edge

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        file_name = os.path.basename(img_path)

        mask_path = os.path.join(self.mask_dir, file_name)
        if not os.path.exists(mask_path):
            # Try finding a PNG with the same name
            mask_path = os.path.join(
                self.mask_dir, os.path.splitext(file_name)[0] + ".png"
            )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(
            mask.resize((self.img_size, self.img_size), Image.NEAREST)
        )
        _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        edge_np = self.generate_edge(mask_np)

        image = self.transform(image)

        # TransNuSeg internal CrossEntropy requires Long type
        mask_tensor = torch.from_numpy(mask_np // 255).long()
        edge_tensor = torch.from_numpy(edge_np // 255).long()

        return image, mask_tensor, edge_tensor

# ================= 3. Loss Function =================
def structure_loss(pred, mask):
    """
    pred: [B, 2, H, W] (logits)
    mask: [B, H, W] (LongTensor, values 0 or 1)
    """

    # CrossEntropyLoss expects mask to be Long and shape [B, H, W] (no channel dimension)
    ce_loss = F.cross_entropy(pred, mask)

    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1, :, :]  # foreground probability [B, H, W]

    # Dice computation requires float mask
    mask = mask.float()

    intersection = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2. * intersection + 1e-5) / (union + 1e-5)

    return ce_loss + dice_loss.mean()

# ================= 4. [Core Modification] Training Pipeline =================
def train(args):
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # ---------- DataLoader ----------
    if args.data_loader and args.image_folder and args.mask_folder:
        print("[INFO] Using external DataLoader (GUI Mode)")
        dataloader = load_external_dataloader(
            args.data_loader,
            args.image_folder,
            args.mask_folder,
            batch_size=args.batch_size,
            target_size=args.img_size  # pass specified image size
        )
    else:
        print("[INFO] Using built-in Dataset (Local Mode)")
        dataset = KvasirDataset(
            args.train_img_path,
            args.train_mask_path,
            args.img_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # set to 0 for safety
            drop_last=True
        )

    # ---------- Model ----------
    print(f"[INFO] Initializing TransNuSeg with img_size={args.img_size}")
    model = TransNuSeg(
        img_size=args.img_size,
        num_classes=CONFIG["num_classes"],
        depths=[2, 2, 2, 2],
        embed_dim=96
    ).to(device)

    if args.pretrain:
        print(f"[INFO] Loading pretrained weights: {args.pretrain}")
        try:
            state = torch.load(args.pretrain, map_location="cpu")
            # Handle possible state_dict key mismatch
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    print("Start Training...")

    steps_per_epoch = len(dataloader)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # [Core Modification] Compatible batch unpacking logic
        for i, batch_data in enumerate(dataloader):
            # 1. Dynamically determine the number of returned elements
            if len(batch_data) == 3:
                # Built-in loader returns (img, mask, edge)
                images, masks, edges = batch_data
            elif len(batch_data) == 2:
                # External loader returns (img, mask), edge must be generated
                images, masks = batch_data
                edges = None  # generate later
            else:
                raise ValueError(f"Unexpected batch length: {len(batch_data)}")

            # 2. Move to device
            images = images.to(device)
            masks = masks.to(device)  # External: [B, 1, H, W] Float

            # 3. Handle data format differences from external loader
            # If mask is [B, 1, H, W] Float, generate edge and convert to Long [B, H, W]

            # Generate edge (if not already provided)
            if edges is None:
                # masks should be float [B, 1, H, W]
                edges = generate_edges_on_the_fly(masks)
                edges = edges.to(device)
            else:
                edges = edges.to(device)

            # 4. Format normalization: TransNuSeg loss requires LongTensor [B, H, W]
            # If 4D [B, 1, H, W], squeeze channel dimension
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            if edges.dim() == 4 and edges.shape[1] == 1:
                edges = edges.squeeze(1)

            # Force conversion to Long (for float 0–1, apply threshold)
            if masks.dtype == torch.float32:
                masks = (masks > 0.5).long()
            if edges.dtype == torch.float32:
                edges = (edges > 0.5).long()

            # 5. Forward pass
            optimizer.zero_grad()

            out_seg, out_edge, out_cluster = model(images)

            # Align spatial size (prevent mismatch with GT)
            if out_seg.shape[-1] != args.img_size:
                out_seg = F.interpolate(out_seg, size=(args.img_size, args.img_size), mode="bilinear",
                                        align_corners=True)
                out_edge = F.interpolate(out_edge, size=(args.img_size, args.img_size), mode="bilinear",
                                         align_corners=True)
                out_cluster = F.interpolate(out_cluster, size=(args.img_size, args.img_size), mode="bilinear",
                                            align_corners=True)

            # 6. Compute loss
            loss_s = structure_loss(out_seg, masks)
            loss_e = structure_loss(out_edge, edges)
            loss_c = structure_loss(out_cluster, edges)

            loss = loss_s + loss_e + loss_c
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{i + 1}/{steps_per_epoch}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / max(len(dataloader), 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Avg Loss: {avg_loss:.4f}"
        )

        # Save model
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")

# ================= Argument Parsing =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_loader", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)

    parser.add_argument("--train_img_path", type=str, default=CONFIG["train_img_path"])
    parser.add_argument("--train_mask_path", type=str, default=CONFIG["train_mask_path"])

    parser.add_argument("--img_size", type=int, default=CONFIG["img_size"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="outputs/transformer/transnuseg")

    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
