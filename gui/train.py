"""
GISeg-Bench  Interactive Training GUI
=====================================
PyQt5 GUI with 4-category model selection (CNN / Foundation / Hybrid /
Transformer) populated from the actual models/ directory structure.

Under the hood it calls ``trainer_core.Trainer`` on a QThread —
no subprocess.  Dataset config, pretrained weights, log viewer are
unchanged from the original design.
"""

import sys
import os
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox,
    QFileDialog, QTextEdit, QGroupBox, QLineEdit,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Project root  (trainer/  →  models/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================================================================
#  Worker thread — runs Trainer.run()
# ===================================================================
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model, train_loader, val_loader, cfg):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.trainer = None

    def run(self):
        try:
            from trainer.trainer_core import Trainer
            from trainer.callbacks import ConsoleReporter, BestModelCheckpoint, EarlyStopping

            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            self.trainer = Trainer(self.model, self.train_loader, self.cfg, self.val_loader)
            self.trainer.set_callbacks([
                ConsoleReporter(),
                BestModelCheckpoint(self.cfg.get("output_dir", "outputs/trainer")),
                EarlyStopping(patience=self.cfg.get("patience", 10)),
            ])

            def gui_log(msg):
                self.log_signal.emit(msg)
                try:
                    sys.stdout.write(msg + "\n")
                except Exception:
                    pass

            self.trainer.log = gui_log
            self.trainer.run()

        except Exception as e:
            self.log_signal.emit(f"[Error] {e}\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            self.finished_signal.emit()

    def stop(self):
        if self.trainer:
            self.trainer.stop()


# ===================================================================
#  Category → model list  (scanned from filesystem)
# ===================================================================
def _scan_models():
    """Walk models/ and return {category: [model_name, ...]}."""
    cats = {}
    models_root = os.path.join(PROJECT_ROOT, "models")
    for cat in ["cnn", "foundation", "hybrid", "transformer"]:
        cat_dir = os.path.join(models_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        names = sorted(
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
            and not d.startswith("_") and not d.startswith(".")
            and os.path.isfile(os.path.join(cat_dir, d, "train.py"))
        )
        if names:
            cats[cat] = names
    return cats


CATEGORY_MODELS = _scan_models()


# ===================================================================
#  Per-model defaults  (n_classes, image_size, loss)
# ===================================================================
MODEL_DEFAULTS = {
    # ---- CNN ----
    "unet":               {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},
    "pranet":             {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "pranet_v2":          {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "fcn":                {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "deeplabv3":          {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "densenet":           {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "resnet":             {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "ce_net":             {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},
    "htc_net":            {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "viewpoint_aware_net":{"n_classes": 2, "image_size": 256, "loss": "ce"},

    # ---- Foundation ----
    "medsam":             {"n_classes": 1, "image_size": 1024, "loss": "bce_dice"},
    "sam_med2d":          {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},
    "universeg":          {"n_classes": 1, "image_size": 128, "loss": "bce_dice"},
    "sam2_unet":          {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},
    "medical_sam_adapter":{"n_classes": 1, "image_size": 1024, "loss": "bce_dice"},
    "scribbleprompt":     {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},

    # ---- Hybrid ----
    "condseg":            {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},

    # ---- Transformer ----
    "swin_unet":          {"n_classes": 1, "image_size": 224, "loss": "bce_dice"},
    "transunet":          {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "hiformer":           {"n_classes": 2, "image_size": 224, "loss": "ce"},
    "h2former":           {"n_classes": 1, "image_size": 224, "loss": "bce_dice"},
    "daeformer":          {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "transnuseg":         {"n_classes": 2, "image_size": 256, "loss": "ce"},
    "mt_unet":            {"n_classes": 1, "image_size": 256, "loss": "bce_dice"},
}


# ===================================================================
#  GUI
# ===================================================================
class MedicalSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_root = PROJECT_ROOT
        self.default_dataset = os.path.join(PROJECT_ROOT, "data", "Kvasir-SEG")
        self.worker = None
        self.init_ui()

    # ------------------------------------------------------------------
    #  UI layout
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Medical Image Segmentation Training System")
        self.setGeometry(100, 100, 1000, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---- title ----
        title = QLabel("Medical Image Segmentation Training System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title)

        # ==================================================================
        #  1. Model Selection  (4-category dropdown)
        # ==================================================================
        model_group = QGroupBox("1. Model Selection")
        model_layout = QVBoxLayout()

        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(CATEGORY_MODELS.keys()))
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.category_combo)

        cat_layout.addSpacing(20)
        cat_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        cat_layout.addWidget(self.model_combo)
        cat_layout.addStretch()
        model_layout.addLayout(cat_layout)

        # info line
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self.model_info_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # populate the first category
        self._on_category_changed(self.category_combo.currentText())

        # ==================================================================
        #  2. Dataset Configuration  (unchanged)
        # ==================================================================
        data_group = QGroupBox("2. Dataset Configuration")
        data_layout = QVBoxLayout()

        ds_sel_layout = QHBoxLayout()
        ds_sel_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_combo = QComboBox()
        self.dataset_name_combo.addItems(["kvasir", "cvc", "wce", "edd"])
        ds_sel_layout.addWidget(self.dataset_name_combo)
        ds_sel_layout.addStretch()
        data_layout.addLayout(ds_sel_layout)

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset Root Directory:"))
        self.dataset_path = QLineEdit(self.default_dataset)
        dataset_layout.addWidget(self.dataset_path)
        dataset_btn = QPushButton("Browse")
        dataset_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(dataset_btn)
        data_layout.addLayout(dataset_layout)

        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Image Folder:"))
        self.image_folder = QLineEdit("images")
        image_layout.addWidget(self.image_folder)
        image_btn = QPushButton("Browse")
        image_btn.clicked.connect(self.browse_image_folder)
        image_layout.addWidget(image_btn)
        data_layout.addLayout(image_layout)

        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Mask Folder:"))
        self.mask_folder = QLineEdit("masks")
        mask_layout.addWidget(self.mask_folder)
        mask_btn = QPushButton("Browse")
        mask_btn.clicked.connect(self.browse_mask_folder)
        mask_layout.addWidget(mask_btn)
        data_layout.addLayout(mask_layout)

        pretrain_layout = QHBoxLayout()
        pretrain_layout.addWidget(QLabel("Pretrained Weights:"))
        self.pretrain_path = QLineEdit()
        pretrain_layout.addWidget(self.pretrain_path)
        pretrain_btn = QPushButton("Browse")
        pretrain_btn.clicked.connect(self.browse_pretrain)
        pretrain_layout.addWidget(pretrain_btn)
        data_layout.addLayout(pretrain_layout)

        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # ==================================================================
        #  3. Controls
        # ==================================================================
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet(
            "QPushButton {background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;}")
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton {background-color: #f44336; color: white; padding: 10px; font-size: 14px;}")
        control_layout.addWidget(self.stop_btn)

        main_layout.addLayout(control_layout)

        # ==================================================================
        #  4. Log
        # ==================================================================
        log_group = QGroupBox("3. Training Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

    # ------------------------------------------------------------------
    #  Category → Model list sync
    # ------------------------------------------------------------------
    def _on_category_changed(self, cat):
        self.model_combo.clear()
        names = CATEGORY_MODELS.get(cat, [])
        self.model_combo.addItems(names)
        if names:
            self._update_model_info(names[0])
        self.model_combo.currentTextChanged.connect(self._update_model_info)

    def _update_model_info(self, name):
        d = MODEL_DEFAULTS.get(name, {})
        nc = d.get("n_classes", "?")
        sz = d.get("image_size", "?")
        lo = d.get("loss", "?")
        self.model_info_label.setText(
            f"n_classes={nc}  |  image_size={sz}  |  loss={lo}"
        )

    # ------------------------------------------------------------------
    #  File / folder browsing
    # ------------------------------------------------------------------
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Root Directory", self.project_root)
        if folder:
            self.dataset_path.setText(folder)

    def browse_image_folder(self):
        dataset_root = self.dataset_path.text()
        if not os.path.exists(dataset_root):
            QMessageBox.warning(self, "Warning", "Please select a valid dataset root directory")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", dataset_root)
        if folder:
            self.image_folder.setText(os.path.basename(folder))

    def browse_mask_folder(self):
        dataset_root = self.dataset_path.text()
        if not os.path.exists(dataset_root):
            QMessageBox.warning(self, "Warning", "Please select a valid dataset root directory")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder", dataset_root)
        if folder:
            self.mask_folder.setText(os.path.basename(folder))

    def browse_pretrain(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Pretrained Weights", self.project_root,
            "Model files (*.pth *.pt *.h5 *.ckpt);;All files (*.*)")
        if file:
            self.pretrain_path.setText(file)

    # ------------------------------------------------------------------
    #  Training control
    # ------------------------------------------------------------------
    def start_training(self):
        from datasets.transforms import SegTransform
        from datasets.dataset_zoo import get_dataset
        import torch
        from torch.utils.data import DataLoader

        model_name = self.model_combo.currentText().strip()
        cat = self.category_combo.currentText().strip()
        defaults = MODEL_DEFAULTS.get(model_name, {})

        dataset_root = self.dataset_path.text()
        if not os.path.exists(dataset_root):
            QMessageBox.warning(self, "Warning", "Dataset path does not exist")
            return

        image_folder = os.path.join(dataset_root, self.image_folder.text())
        mask_folder = os.path.join(dataset_root, self.mask_folder.text())

        if not os.path.exists(image_folder):
            QMessageBox.warning(self, "Warning", f"Image folder does not exist: {image_folder}")
            return
        if not os.path.exists(mask_folder):
            QMessageBox.warning(self, "Warning", f"Mask folder does not exist: {mask_folder}")
            return

        self.log_text.clear()
        self.log_text.append("[System] Training started...")
        self.log_text.append(f"[System] Category: {cat}  |  Model: {model_name}")
        self.log_text.append(f"[System] Image folder: {image_folder}")
        self.log_text.append(f"[System] Mask folder: {mask_folder}")
        self.log_text.append(f"[System] Pretrained weights: {self.pretrain_path.text() or 'None'}")
        self.log_text.append("-" * 80)

        dataset_name = self.dataset_name_combo.currentText().strip()
        self.log_text.append(f"[System] Dataset: {dataset_name}")

        # ---- build dataset ----
        img_size = defaults.get("image_size", 256)
        tf = SegTransform(size=img_size, normalise="imagenet")
        try:
            train_ds = get_dataset(dataset_name, root=dataset_root, split="train", transform=tf)
        except Exception:
            try:
                from datasets.base_dataset import BaseSegDataset
                class _AdHoc(BaseSegDataset):
                    def _build_index(self):
                        import os as _os
                        msk_stems = {}
                        for f in _os.listdir(mask_folder):
                            s, e = _os.path.splitext(f)
                            if e.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}:
                                msk_stems[s] = _os.path.join(mask_folder, f)
                        valid = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                        for f in sorted(_os.listdir(image_folder)):
                            s, e = _os.path.splitext(f)
                            if e.lower() in valid and s in msk_stems:
                                self.samples.append((_os.path.join(image_folder, f), msk_stems[s], f))
                train_ds = _AdHoc(root="", split="all")
            except Exception as e2:
                QMessageBox.critical(self, "Error", f"Failed to create dataset:\n{e2}")
                return

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

        # ---- build model ----
        n_classes = defaults.get("n_classes", 1)
        model = self._build_model(cat, model_name, n_classes, img_size)

        if model is None:
            QMessageBox.critical(self, "Error",
                f"Cannot build model '{model_name}'.\n\n"
                f"This model may have complex dependencies that require "
                f"running its original train.py directly.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # ---- pretrained weights ----
        pretrain = self.pretrain_path.text()
        if pretrain and os.path.exists(pretrain):
            state = torch.load(pretrain, map_location=device)
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)

        # ---- config ----
        cfg = {
            "epochs": 20,
            "batch_size": 4,
            "lr": 1e-4,
            "n_classes": n_classes,
            "image_size": img_size,
            "loss": defaults.get("loss", "bce_dice"),
            "output_dir": os.path.join(
                os.path.dirname(__file__), "..", "outputs", "trainer"
            ),
            "patience": 10,
        }

        # ---- launch ----
        self.worker = TrainingWorker(model, train_loader, None, cfg)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    # ------------------------------------------------------------------
    #  Model builder  (category + name → nn.Module)
    # ------------------------------------------------------------------
    def _build_model(self, cat, name, n_classes, img_size):
        import torch.nn as nn
        cat = cat.lower()

        # ---- helpers ----
        def _add_model_dir(subpath):
            d = os.path.join(PROJECT_ROOT, subpath)
            if d not in sys.path:
                sys.path.insert(0, d)
            return d

        # ===================== CNN =====================
        if cat == "cnn":
            if name == "unet":
                _add_model_dir("models/cnn/unet")
                from train import UNet
                return UNet(in_ch=3, out_ch=n_classes)

            elif name in ("pranet", "pranet_v2"):
                if name == "pranet":
                    _add_model_dir("models/cnn/pranet")
                else:
                    _add_model_dir("models/cnn/pranet_v2")
                from PraNet_ResNet import CRANet
                return CRANet()

            elif name == "fcn":
                from torchvision import models
                model = models.segmentation.fcn_resnet50(weights="DEFAULT")
                model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)
                model.aux_classifier = None
                return model

            elif name == "deeplabv3":
                from torchvision.models.segmentation import (
                    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
                )
                model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
                model.classifier[-1] = nn.Conv2d(
                    model.classifier[-1].in_channels, n_classes, kernel_size=1)
                model.aux_classifier = None
                return model

            elif name == "densenet":
                from torchvision import models
                class DenseNetSeg(nn.Module):
                    def __init__(self):
                        super().__init__()
                        backbone = models.densenet121(
                            weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                        self.backbone = backbone.features
                        self.classifier = nn.Conv2d(1024, n_classes, kernel_size=1)
                        self.upsample = nn.Upsample(
                            scale_factor=32, mode="bilinear", align_corners=False)
                    def forward(self, x):
                        x = self.backbone(x)
                        x = self.classifier(x)
                        return {"out": self.upsample(x)}
                return DenseNetSeg()

            elif name == "resnet":
                # ResNet-based UNet-style from cnn/resnet
                _add_model_dir("models/cnn/resnet")
                from model import ResNetUNet
                return ResNetUNet(n_classes=n_classes)

            elif name == "ce_net":
                _add_model_dir("models/cnn/ce_net")
                from cenet import CE_Net_
                return CE_Net_()

            elif name == "htc_net":
                _add_model_dir("models/cnn/htc_net")
                # HTC-Net uses specific config; try dynamic import
                from network.Net import model as SwinModelWrapper
                import ml_collections
                cfg = ml_collections.ConfigDict()
                cfg.n_classes = n_classes
                cfg.image_size = img_size
                return SwinModelWrapper(cfg)

            elif name == "viewpoint_aware_net":
                _add_model_dir("models/cnn/viewpoint_aware_net")
                from VANet import VANet
                return VANet(n_classes=n_classes)

        # ===================== Foundation =====================
        elif cat == "foundation":
            if name == "medsam":
                _add_model_dir("models/foundation/medsam")
                from segment_anything import sam_model_registry
                ckpt = self.pretrain_path.text() or None
                model = sam_model_registry["vit_b"](checkpoint=ckpt)
                for p in model.image_encoder.parameters():
                    p.requires_grad = False
                return model

            elif name == "sam_med2d":
                _add_model_dir("models/foundation/sam_med2d")
                from segment_anything import sam_model_registry
                class _Args:
                    image_size = img_size
                    sam_checkpoint = self.pretrain_path.text() or None
                    encoder_adapter = True
                    sam_type = "vit_b"
                model = sam_model_registry["vit_b"](_Args())
                for p in model.image_encoder.parameters():
                    p.requires_grad = False
                for p in model.prompt_encoder.parameters():
                    p.requires_grad = False
                return model

            elif name == "universeg":
                from universeg import universeg
                return universeg(version="v1", pretrained=False)

            elif name == "sam2_unet":
                _add_model_dir("models/foundation/sam2_unet")
                from SAM2UNet import SAM2UNet
                return SAM2UNet()

            elif name == "scribbleprompt":
                _add_model_dir("models/foundation/scribbleprompt")
                from scribbleprompt.models.unet import ScribblePromptUNet
                return ScribblePromptUNet()

            elif name == "medical_sam_adapter":
                _add_model_dir("models/foundation/medical_sam_adapter")
                from models.sam.build_sam import build_sam_vit_b
                return build_sam_vit_b(checkpoint=self.pretrain_path.text() or None)

        # ===================== Hybrid =====================
        elif cat == "hybrid":
            if name == "condseg":
                _add_model_dir("models/hybrid/condseg")
                from network.model import ConDSeg
                return ConDSeg()

        # ===================== Transformer =====================
        elif cat == "transformer":
            if name == "swin_unet":
                _add_model_dir("models/transformer/swin_unet")
                from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
                return SwinTransformerSys(
                    img_size=img_size, patch_size=4, in_chans=3,
                    num_classes=n_classes, embed_dim=96,
                    depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
                    num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                    qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                    patch_norm=True, final_upsample="expand_first",
                )

            elif name == "transunet":
                _add_model_dir("models/transformer/transunet")
                from vit_seg_modeling import VisionTransformer
                from vit_seg_configs import get_r50_b16_config
                config = get_r50_b16_config()
                return VisionTransformer(config=config, img_size=256,
                                         num_classes=n_classes)

            elif name == "hiformer":
                _add_model_dir("models/transformer/hiformer")
                from models.HiFormer import HiFormer
                import configs.HiFormer_configs as hcfg
                config = hcfg.get_hiformer_b_configs()
                return HiFormer(config=config, img_size=224, in_chans=3,
                                n_classes=n_classes)

            elif name == "h2former":
                _add_model_dir("models/transformer/h2former")
                from models.H2Former import Res34_Swin_MS, BasicBlock
                return Res34_Swin_MS(image_size=img_size, block=BasicBlock,
                                     layers=[3, 4, 6, 3], num_classes=n_classes)

            elif name == "daeformer":
                _add_model_dir("models/transformer/daeformer")
                from networks.DAEFormer import DAEFormer
                return DAEFormer(n_classes=n_classes)

            elif name == "transnuseg":
                _add_model_dir("models/transformer/transnuseg")
                from models.transnuseg import TransNuSeg
                return TransNuSeg(n_classes=n_classes)

            elif name == "mt_unet":
                _add_model_dir("models/transformer/mt_unet")
                from model.MTUNet import MTUNet
                return MTUNet(n_classes=n_classes)

        # fallback — try generic dynamic import
        return self._try_dynamic_build(cat, name, n_classes, img_size)

    def _try_dynamic_build(self, cat, name, n_classes, img_size):
        """Last-resort: add model dir to path, import its train module,
        and look for a model class."""
        model_dir = os.path.join(PROJECT_ROOT, "models", cat, name)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        try:
            import importlib
            mod = importlib.import_module("train")
            # look for common class names
            candidates = [
                "UNet", "ResUNet", "AttentionUNet", "SegNet", "Model",
                "Net", "Network", "SegModel",
            ]
            for cname in candidates:
                if hasattr(mod, cname):
                    cls = getattr(mod, cname)
                    try:
                        return cls(n_classes=n_classes)
                    except TypeError:
                        try:
                            return cls()
                        except Exception:
                            continue
        except Exception as e:
            print(f"[Warn] Dynamic build failed: {e}")
        return None

    # ------------------------------------------------------------------
    #  Stop / log / finish
    # ------------------------------------------------------------------
    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_text.append("[System] Stopping training...")

    def update_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def training_finished(self):
        self.log_text.append("-" * 80)
        self.log_text.append("[System] Training finished")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ===================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MedicalSegmentationGUI()
    gui.show()
    sys.exit(app.exec_())
