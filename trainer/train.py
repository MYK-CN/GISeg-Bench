"""
GISeg-Bench  Unified Training Entry
===================================
Single CLI entry point for ALL models — no per-model train.py needed.

Usage::

    # Local dataset
    python -m trainer.train --model unet --dataset kvasir
          --data_root D:/data/Kvasir-SEG --epochs 100

    # External data loader (GUI mode)
    python -m trainer.train --model pranet --data_loader /path/to/loader.py
          --image_folder D:/data/Kvasir-SEG/images
          --mask_folder  D:/data/Kvasir-SEG/masks

    # With pretrained weights
    python -m trainer.train --model medsam --pretrain /path/to/sam.pth
          --epochs 20 --lr 1e-4

All model-specific behaviour is captured by the model builder registry below.
"""

import os
import sys
import argparse

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# SAM-family model names — outputs need logit normalisation for BCE/Dice loss
_SAM_MODELS = {"medsam", "sam_med2d", "medical_sam_adapter"}

# Models that require a specific input image_size (override CLI --image_size default)
_MODEL_IMAGE_SIZES = {
    "daeformer": 224,
    "hiformer": 224,
    "htc_net": 224,
    "swin_unet": 224,
    "mt_unet": 224,
    "transnuseg": 512,
}

# ===================================================================
#  SAM Training Wrapper — shared across all SAM-family models
# ===================================================================
class SAMTrainingWrapper(torch.nn.Module):
    """Training-friendly SAM forward that preserves gradient flow.

    The standard ``Sam.forward()`` is decorated with ``@torch.no_grad()``,
    which kills gradients.  This wrapper calls the sub-modules directly:
    ``image_encoder`` → ``prompt_encoder`` → ``mask_decoder``,
    bypassing the no-grad wrapper and returning upsampled logits.
    """

    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Preprocess (normalise + pad)
        input_images = torch.stack([self.sam.preprocess(x[i]) for i in range(B)])

        # 2. Image encoder — WITH gradients (supports adapter/LoRA fine-tuning)
        image_embeddings = self.sam.image_encoder(input_images)

        # 3. Prompt encoder — no prompts = unconditional segmentation
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        # Prompt encoder returns batch-size-1 when no prompts; expand to B
        sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)

        # 4. Mask decoder — produces low_res_logits [B, 1, 256, 256]
        low_res_masks, _iou = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 5. Upsample to input resolution
        logits = F.interpolate(low_res_masks, size=(H, W), mode="bilinear",
                               align_corners=False)
        return logits

from datasets.dataset_zoo import get_dataset
from datasets.transforms import SegTransform
from .trainer_core import Trainer
from .callbacks import (
    ConsoleReporter,
    BestModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
)


# ===================================================================
#  Model registry  (extensible — add new models here)
# ===================================================================
_MODEL_BUILDERS = {}


def _register(name):
    def dec(fn):
        _MODEL_BUILDERS[name] = fn
        return fn
    return dec


# ===================== CNN models =====================

@_register("unet")
def _build_unet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn", "unet"))
    from train import UNet
    return UNet(in_ch=3, out_ch=cfg.get("n_classes", 1))


@_register("pranet")
def _build_pranet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn", "pranet"))
    from PraNet_ResNet import CRANet
    return CRANet()


@_register("pranet_v2")
def _build_pranet_v2(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn", "pranet_v2"))
    from lib.pranet import PVT_PraNet_V2
    return PVT_PraNet_V2(channel=32, num_class=cfg.get("n_classes", 1),
                          sem_downsample=1, use_softmax=False)


@_register("fcn")
def _build_fcn(cfg):
    from torchvision import models
    import torch.nn as nn
    model = models.segmentation.fcn_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(512, cfg.get("n_classes", 2), kernel_size=1)
    model.aux_classifier = None
    return model


@_register("deeplabv3")
def _build_deeplabv3(cfg):
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    import torch.nn as nn
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(
        model.classifier[-1].in_channels, cfg.get("n_classes", 2), kernel_size=1
    )
    model.aux_classifier = None
    return model


@_register("densenet")
def _build_densenet(cfg):
    from torchvision import models
    import torch.nn as nn
    class DenseNetSeg(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.backbone = backbone.features
            self.classifier = nn.Conv2d(1024, cfg.get("n_classes", 2), kernel_size=1)
            self.upsample = nn.Upsample(
                scale_factor=32, mode="bilinear", align_corners=False)
        def forward(self, x):
            x = self.backbone(x)
            x = self.classifier(x)
            return {"out": self.upsample(x)}
    return DenseNetSeg()


@_register("resnet")
def _build_resnet(cfg):
    from torchvision import models
    import torch.nn as nn
    class ResNetUNet(nn.Module):
        def __init__(self, n_classes=1):
            super().__init__()
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, n_classes, kernel_size=1),
            )
        def forward(self, x):
            x = self.encoder(x)
            return {"out": self.decoder(x)}
    return ResNetUNet(n_classes=cfg.get("n_classes", 1))


@_register("ce_net")
def _build_ce_net(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn", "ce_net"))
    from cenet import CE_Net_
    return CE_Net_()


@_register("htc_net")
def _build_htc_net(cfg):
    htc_dir = os.path.join(PROJECT_ROOT, "models", "cnn", "htc_net")
    sys.path.insert(0, htc_dir)
    import ml_collections
    # Clear any cached 'network' module that might shadow the local package
    for key in list(sys.modules.keys()):
        if key == "network" or key.startswith("network."):
            if "htc_net" not in str(sys.modules[key].__file__ if hasattr(sys.modules[key], "__file__") else ""):
                del sys.modules[key]
    try:
        from network.Net import model as SwinModelWrapper
    except ImportError as e:
        raise ImportError(
            f"Failed to import HTCNet: {e}. "
            f"Ensure 'segmentation_models_pytorch' is installed "
            f"(pip install segmentation-models-pytorch)."
        )
    cfg_htc = ml_collections.ConfigDict()
    cfg_htc.n_classes = cfg.get("n_classes", 1)
    cfg_htc.decoder_channels = (128, 64, 32, 16)
    cfg_htc.n_skip = 3
    return SwinModelWrapper(config=cfg_htc, img_size=224,
                            num_classes=cfg.get("n_classes", 1))


@_register("viewpoint_aware_net")
def _build_viewpoint_aware_net(cfg):
    import torch.nn as nn
    vanet_dir = os.path.join(PROJECT_ROOT, "models", "cnn", "viewpoint_aware_net")
    sys.path.insert(0, vanet_dir)
    # Clear any pre-existing 'lib' module that shadows the local package
    for key in list(sys.modules.keys()):
        if key == "lib" or key.startswith("lib."):
            if "viewpoint_aware_net" not in str(sys.modules[key].__file__ if hasattr(sys.modules[key], "__file__") else ""):
                del sys.modules[key]
    from VANet import VANet
    yaml_path = os.path.join(vanet_dir, "experiments", "imagenet", "cvt",
                              "cvt-13-224x224.yaml")
    vanet = VANet(cfg=yaml_path, weights=None, num_class=cfg.get("n_classes", 1))
    class VANetWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return self.upsample(out)
    return VANetWrapper(vanet)


# ===================== Foundation models =====================

@_register("medsam")
def _build_medsam(cfg):
    medsam_dir = os.path.join(PROJECT_ROOT, "models", "foundation", "medsam")
    sys.path.insert(0, medsam_dir)
    # Clear cached segment_anything to avoid cross-SAM-variant conflicts
    for key in list(sys.modules.keys()):
        if key.startswith("segment_anything"):
            del sys.modules[key]
    from segment_anything import sam_model_registry
    sam_model = sam_model_registry["vit_b"](checkpoint=cfg.get("pretrain"))
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = False
    return SAMTrainingWrapper(sam_model)


@_register("sam_med2d")
def _build_sam_med2d(cfg):
    sam_dir = os.path.join(PROJECT_ROOT, "models", "foundation", "sam_med2d")
    sys.path.insert(0, sam_dir)
    # Clear cached segment_anything to avoid cross-SAM-variant conflicts
    for key in list(sys.modules.keys()):
        if key.startswith("segment_anything"):
            del sys.modules[key]
    from segment_anything import sam_model_registry
    import torch.nn as nn
    class _Args:
        image_size = cfg.get("image_size", 256)
        sam_checkpoint = cfg.get("pretrain")
        encoder_adapter = True
    sam_model = sam_model_registry["vit_b"](_Args())
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = False
    for p in sam_model.prompt_encoder.parameters():
        p.requires_grad = False
    return SAMTrainingWrapper(sam_model)


@_register("universeg")
def _build_universeg(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "foundation", "universeg"))
    from universeg import UniverSeg
    import torch.nn as nn
    useg = UniverSeg(encoder_blocks=[64, 64, 64, 64])
    class USegWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # UniverSeg expects: target=[B,1,H,W], support=[B,S,C,H,W], labels=[B,S,1,H,W]
            img = x[:, 0:1, :, :]                     # [B, 1, H, W]
            sup_img = img.unsqueeze(1)                  # [B, 1, 1, H, W] (S=1)
            sup_lbl = torch.zeros_like(sup_img)         # [B, 1, 1, H, W]
            return self.model(img, sup_img, sup_lbl)
    return USegWrapper(useg)


@_register("sam2_unet")
def _build_sam2_unet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "foundation", "sam2_unet"))
    from SAM2UNet import SAM2UNet
    return SAM2UNet()


@_register("scribbleprompt")
def _build_scribbleprompt(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "foundation", "scribbleprompt"))
    from scribbleprompt.models.unet import ScribblePromptUNet
    import torch.nn as nn
    sp = ScribblePromptUNet(pretrained=False)
    class SPWrapper(nn.Module):
        def __init__(self, sp_model):
            super().__init__()
            self.model = sp_model.model
        def forward(self, x):
            # ScribblePrompt expects 5-channel input; adapt for standard 3-channel training
            B, C, H, W = x.shape
            padding = torch.zeros(B, 2, H, W, device=x.device)
            x5 = torch.cat([x, padding], dim=1)
            return self.model(x5)
    return SPWrapper(sp)


@_register("medical_sam_adapter")
def _build_medical_sam_adapter(cfg):
    msa_dir = os.path.join(PROJECT_ROOT, "models", "foundation", "medical_sam_adapter")
    sys.path.insert(0, msa_dir)
    # Clear cached segment_anything to avoid cross-SAM-variant conflicts
    for key in list(sys.modules.keys()):
        if key.startswith("segment_anything"):
            del sys.modules[key]
    from models.sam.build_sam import build_sam_vit_b
    import torch.nn as nn
    class _Args:
        image_size = cfg.get("image_size", 256)
        multimask_output = 1
        mod = "sam_adpt"  # required by ImageEncoderViT for adapter mode
        mid_dim = None    # adapter hidden dim (None = default)
        thd = False       # whether to use threshold adapter
    sam_model = build_sam_vit_b(args=_Args(), checkpoint=cfg.get("pretrain"))
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = False
    return SAMTrainingWrapper(sam_model)


# ===================== Transformer models =====================

@_register("swin_unet")
def _build_swin_unet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "swin_unet"))
    import torch.nn as nn
    from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
    return SwinTransformerSys(
        img_size=cfg.get("image_size", 224), patch_size=4, in_chans=3,
        num_classes=cfg.get("n_classes", 1), embed_dim=96,
        depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
        qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
        patch_norm=True, final_upsample="expand_first",
    )


@_register("transunet")
def _build_transunet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "transunet"))
    from vit_seg_modeling import VisionTransformer
    from vit_seg_configs import get_r50_b16_config
    config = get_r50_b16_config()
    return VisionTransformer(config=config, img_size=256, num_classes=cfg.get("n_classes", 2))


@_register("hiformer")
def _build_hiformer(cfg):
    hiformer_dir = os.path.join(PROJECT_ROOT, "models", "transformer", "hiformer")
    sys.path.insert(0, hiformer_dir)
    import importlib.util
    # Load config
    spec_cfg = importlib.util.spec_from_file_location("HiFormer_configs",
        os.path.join(hiformer_dir, "configs", "HiFormer_configs.py"))
    hcfg = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(hcfg)
    config = hcfg.get_hiformer_b_configs()
    # Load utils from hiformer path (before models.HiFormer imports it)
    spec_utils = importlib.util.spec_from_file_location("utils",
        os.path.join(hiformer_dir, "utils.py"))
    hf_utils = importlib.util.module_from_spec(spec_utils)
    sys.modules["utils"] = hf_utils
    spec_utils.loader.exec_module(hf_utils)
    from models.HiFormer import HiFormer
    return HiFormer(config=config, img_size=224, in_chans=3, n_classes=cfg.get("n_classes", 2))


@_register("h2former")
def _build_h2former(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "h2former"))
    from models.H2Former import Res34_Swin_MS, BasicBlock
    import torch.nn as nn
    h2f = Res34_Swin_MS(image_size=cfg.get("image_size", 224), block=BasicBlock,
                         layers=[3, 4, 6, 3], num_classes=cfg.get("n_classes", 1))
    class H2FWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            extra = x.mean(dim=1, keepdim=True)
            x4 = torch.cat([x, extra], dim=1)
            return self.model(x4)
    return H2FWrapper(h2f)


@_register("daeformer")
def _build_daeformer(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "daeformer"))
    from networks.DAEFormer import DAEFormer
    return DAEFormer(num_classes=cfg.get("n_classes", 2))


@_register("transnuseg")
def _build_transnuseg(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "transnuseg"))
    from models.transnuseg import TransNuSeg
    return TransNuSeg(img_size=512, num_classes=cfg.get("n_classes", 2))


@_register("mt_unet")
def _build_mt_unet(cfg):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer", "mt_unet"))
    from model.MTUNet import MTUNet
    return MTUNet(out_ch=cfg.get("n_classes", 1))


# ===================== Hybrid models =====================

@_register("condseg")
def _build_condseg(cfg):
    condseg_dir = os.path.join(PROJECT_ROOT, "models", "hybrid", "condseg")
    sys.path.insert(0, condseg_dir)
    # Clear cached 'network' module to avoid conflict with htc_net
    for key in list(sys.modules.keys()):
        if key == "network" or key.startswith("network."):
            if "condseg" not in str(sys.modules[key].__file__ if hasattr(sys.modules[key], "__file__") else ""):
                del sys.modules[key]
    from network.model import ConDSeg
    return ConDSeg()


# ===================================================================
#  CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser("GISeg-Bench Unified Trainer")

    # ---- model ----
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name: {sorted(_MODEL_BUILDERS.keys())}")
    parser.add_argument("--pretrain", type=str, default=None)

    # ---- data ----
    parser.add_argument("--dataset", type=str, default="kvasir")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_loader", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)

    # ---- training ----
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--loss", type=str, default="auto")

    # ---- output ----
    parser.add_argument("--output_dir", type=str, default="outputs/trainer")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    # Override image_size for models with specific size requirements
    if args.model in _MODEL_IMAGE_SIZES:
        required = _MODEL_IMAGE_SIZES[args.model]
        if args.image_size != required:
            print(f"[Train] Overriding image_size {args.image_size} → {required} "
                  f"(required by model '{args.model}')")
            args.image_size = required

    # ---- build dataset ----
    if args.data_loader and args.image_folder and args.mask_folder:
        # External loader path
        import importlib.util
        spec = importlib.util.spec_from_file_location("ul", args.data_loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        train_loader = mod.get_data_loader(
            image_folder=args.image_folder,
            mask_folder=args.mask_folder,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=0,
        )
        val_loader = None
    elif args.data_root:
        tf = SegTransform(size=args.image_size, normalise="imagenet")
        train_ds = get_dataset(args.dataset, root=args.data_root,
                               split="train", split_file=args.split_file,
                               transform=tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        # try val split
        try:
            val_ds = get_dataset(args.dataset, root=args.data_root,
                                 split="val", split_file=args.split_file,
                                 transform=tf)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
        except Exception:
            val_loader = None
    else:
        parser.error("Provide --data_root or (--data_loader + --image_folder + --mask_folder)")

    # ---- build model ----
    if args.model not in _MODEL_BUILDERS:
        parser.error(f"Unknown model '{args.model}'.  Choices: {sorted(_MODEL_BUILDERS.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _MODEL_BUILDERS[args.model](vars(args)).to(device)

    # ---- pretrained weights ----
    if args.pretrain:
        print(f"[Train] Loading pretrained weights: {args.pretrain}")
        state = torch.load(args.pretrain, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    # ---- run ----
    trainer = Trainer(model, train_loader, args, val_loader)
    trainer.set_callbacks([
        ConsoleReporter(),
        BestModelCheckpoint(args.output_dir, monitor="train_dice"),
        EarlyStopping(monitor="train_dice", patience=args.patience),
    ])
    trainer.run()


if __name__ == "__main__":
    main()
