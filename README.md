# GISeg-Bench — Unified Interactive Medical Image Segmentation Benchmark

## Overview

GISeg-Bench is a unified benchmark and experimentation platform for interactive
medical image segmentation. It integrates 24 state-of-the-art segmentation
models across 4 medical imaging datasets, delivering a complete workflow from
training to inference to evaluation to visualization, with both command-line
and graphical user interface (GUI) entry points.

Author: myk
Version: 1.0

## Project Structure

```
 gui/                              GUI entry points
     fenge.py                      Unified interactive segmentation GUI (all 24 models)
     train.py                      Training GUI (model selection + dataset config + log viewer)

 main.py                           CLI orchestrator — config → train → inference → metrics → report

 models/                           24 segmentation models (grouped by family)
     cnn/                          CNN family (10 models)
         unet/                     U-Net (Ronneberger 2015)
         pranet/                   PraNet — Parallel Reverse Attention Network
         pranet_v2/                PraNet v2 improved variant
         fcn/                      FCN-ResNet50 (torchvision)
         deeplabv3/                DeepLabV3-ResNet50 (torchvision)
         densenet/                 DenseNet121-FCN
         resnet/                   ResNet-UNet
         ce_net/                   CE-Net — Context Encoder Network
         htc_net/                  HTC-Net — Hybrid Transformer-CNN
         viewpoint_aware_net/      Viewpoint-Aware Net
     transformer/                  Transformer family (7 models)
         swin_unet/                Swin-UNet (Cao 2021)
         transunet/                TransUNet (Chen 2021)
         hiformer/                 HiFormer — Hierarchical Multi-scale Transformer
         h2former/                 H2Former — Hybrid High-resolution Hierarchical
         daeformer/                DAE-Former — Dual Attention Enhanced
         transnuseg/               TransNuSeg — Nucleus Segmentation Transformer
         mt_unet/                  MT-UNet — Mixed Transformer UNet
     foundation/                   Foundation model family (6 models)
         medsam/                   MedSAM — Medical Segment Anything Model
         sam_med2d/                SAM-Med2D
         universeg/                UniverSeg — Universal Segmentation
         sam2_unet/                SAM2-UNet
         scribbleprompt/           ScribblePrompt — Weakly-supervised
         medical_sam_adapter/      Medical SAM Adapter
     hybrid/                       Hybrid family (1 model)
         condseg/                  ConDSeg — Conditional Diffusion Segmentation

 configs/                          Configuration system
     model_config.py               24-model registry (name / family / input size / classes)
     dataset_config.py             4-dataset registry (path / modality / task / splits)
     config_loader.py              Unified config loader (train/infer mode auto-merge)
     base_config.py                Base Config class
     train_config.py               Training hyperparameter defaults
     infer_config.py               Inference parameter defaults

 datasets/                         Data loading layer
     dataset_zoo.py                Unified get_dataset() entry point
     base_dataset.py               Abstract BaseSegDataset class
     kvasir.py / cvc.py            Per-dataset implementations
     wce.py / edd.py
     transforms.py                 Image/mask transforms (ImageNet normalization)
     split_utils.py                Train/val/test split generation and loading
     splits/                       Pre-generated split files (kvasir / cvc / wce / edd .json)

 trainer/                          Training engine
     train.py                      Model builder registry + CLI training entry point
     trainer_core.py               Trainer core class (epoch loop)
     engine.py                     Training/validation step implementation
     callbacks.py                  Callbacks: checkpointing / early stopping / TensorBoard / logging
     loss_builder.py               Loss function builder
     optimizer_builder.py          Optimizer builder
     scheduler_builder.py          Learning rate scheduler builder
     validator.py                  Validation logic

 inference/                        Inference module
     loader.py                     Model builder (build_model) + weight loader (load_checkpoint)
     predictor.py                  Predictor class — batched inference + output normalization
     postprocess.py                Post-processing: sigmoid/softmax → threshold/argmax → resize
     run_inference.py              Inference script entry point
     utils.py                      Device management / tensor utilities

 metrics/                          Evaluation metrics
     evaluator.py                  Unified evaluate() entry point
     aggregator.py                 Cross-sample / cross-model aggregation
     dice.py / iou.py              Per-metric implementations
     hd95.py / precision_recall.py
     utils.py                      Metric helper functions

 utils/                            Global utility layer
     logger.py                     Unified logging (console + file)
     seed.py                       Deterministic random seeding
     file_utils.py                 Path management / output dir creation / checkpoint search
     image_utils.py                Image resizing / normalization
     tensor_utils.py               Tensor conversion / sigmoid / softmax
     metrics_utils.py              Dice/IoU helpers / confusion matrix
     visual_utils.py               Mask overlay / GT-vs-prediction comparison

 data/                             Dataset storage
     Kvasir-SEG/
     cvc-clinicdb-DatasetNinja/
     WCEBleedGen (updated)/
     EDD2020/

 weights/                          Pretrained weights
     swin_tiny_patch4_window7_224.pth

 outputs/                          Output directory (checkpoints / predictions / metrics)
```

## Supported Datasets

| Name         | Modality                    | Task                    | Classes        |
|------------- |---------------------------- |------------------------ |--------------- |
| Kvasir-SEG   | Gastrointestinal endoscopy  | Polyp segmentation      | 2 (bg / polyp) |
| CVC-ClinicDB | Colonoscopy                 | Polyp segmentation      | 2 (bg / polyp) |
| WCEBleedGen  | Wireless capsule endoscopy  | Bleeding segmentation   | 2 (bg / bleed) |
| EDD2020      | Multi-site endoscopy        | Disease (multi-class)   | 5+             |

Download links:

- CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/
- Kvasir-SEG:   https://datasets.simula.no/kvasir-seg/
- WCEBleedGen:  https://zenodo.org/records/10156571
- EDD2020:      https://edd2020.grand-challenge.org/

Data directory format (each dataset):

```
<dataset_root>/
├── images/        # RGB images (.png / .jpg / .jpeg)
│   ├── img001.png
│   └── ...
└── masks/         # Grayscale masks (filename-matched to images)
    ├── img001.png
    └── ...
```

## Supported Models (24 total)

By family:

- **CNN (10):** unet, pranet, pranet_v2, fcn, deeplabv3, densenet, resnet, ce_net, htc_net, viewpoint_aware_net
- **Transformer (7):** swin_unet, transunet, hiformer, h2former, daeformer, transnuseg, mt_unet
- **Foundation (6):** medsam, sam_med2d, universeg, sam2_unet, scribbleprompt, medical_sam_adapter
- **Hybrid (1):** condseg

## Dependencies

Core:

- Python >= 3.8
- PyTorch >= 1.12
- torchvision
- numpy
- scikit-image
- scipy
- Pillow
- matplotlib
- tqdm

GUI (optional):

- PyQt5

Model-specific (install as needed):

- segmentation-models-pytorch (U-Net and other SMP architectures)
- timm (some Transformer models)
- nibabel (MedSAM — NIfTI support)
- ml_collections (HTC-Net)

## Usage

### 1. CLI Benchmark Orchestrator (main.py)

`main.py` is the unified orchestrator. A single command drives the full
pipeline: config → train → inference → metrics → report.

```bash
# Full benchmark — all 24 models × all 4 datasets
python main.py

# Quick smoke test — 1 model × 1 dataset × 2 epochs
python main.py --quick

# Specific models and datasets
python main.py --models unet,pranet,swin_unet --datasets kvasir,cvc

# Skip training — inference + metrics only (requires existing checkpoints)
python main.py --skip_train

# Skip training and inference — metrics only (requires existing predictions)
python main.py --skip_train --skip_infer

# Override number of epochs
python main.py --epochs 50 --models unet --datasets kvasir

# Show all options
python main.py --help
```

Output directory structure:

```
outputs/
└── <model_name>/
    └── <dataset_name>/
        ├── best.pth              # Best model checkpoint
        ├── final.pth             # Final model checkpoint
        ├── metrics.json          # Evaluation metrics
        └── inference_summary.json
```

After completion the terminal prints:

- Model ranking (by average Dice score)
- Best model per dataset
- Full results table

### 2. Interactive Segmentation GUI (gui/fenge.py)

`fenge.py` is the unified interactive segmentation GUI. It supports all 24 models
from a single interface — no need to navigate into individual model directories.

```bash
python gui/fenge.py
```

Workflow:

1. Select a segmentation model from the dropdown (all 24 models auto-detected)
2. Click "Select Checkpoint" to choose a trained `.pth` weight file
3. Click "Load Image" to open a medical image
4. Drag the mouse on the image to draw an ROI bounding box
5. Click "Run Segmentation" to execute inference

Features:

- Auto model discovery — scans MODEL_REGISTRY for all 24 models at startup
- Interactive ROI drawing — green rectangle with real-time preview
- Result overlay — prediction region highlighted in green on the image
- Three-panel in-GUI display:
  - Panel 1: original image + ROI box + segmentation overlay
  - Panel 2: prediction mask (binary)
  - Panel 3: ground truth mask (if found) or blank
- Auto-save results:
  - `outputs/infer_results/result_<timestamp>.png` — 3-panel composite
  - `outputs/infer_results/mask_<timestamp>.png` — prediction mask
  - `outputs/infer_results/overlay_<timestamp>.png` — overlay image
  - `outputs/infer_results/metrics_<timestamp>.json` — Dice / IoU / HD95 / Recall / Precision
- Automatic ground-truth discovery — searches `../masks/`, `../masktest/`,
  and same-directory `*_mask.*` patterns
- Live metrics display — Dice, IoU shown in status bar and result panel

Use cases:

- Quick visual inspection of model segmentation quality
- Interactive single-image analysis
- Model comparison (load different checkpoints on the same image)

Each model subdirectory also contains an independent `gui.py`
(e.g. `models/cnn/unet/gui.py`), but those require manual model architecture
configuration and weight paths. `gui/fenge.py` unifies them under one entry point.

### 3. Training GUI (gui/train.py)

`train.py` is a dedicated training GUI with 4-category model selection,
dataset configuration, and real-time log viewing.

```bash
python gui/train.py
```

Workflow:

1. Select model category (CNN / Transformer / Foundation / Hybrid)
2. Select a specific model
3. Configure dataset root directory, image folder, and mask folder
4. Optionally select pretrained weights
5. Click "Start Training"

Features:

- Linked dropdowns — changing category refreshes the model list automatically
- Auto-display of model parameters (number of classes, input size, loss function)
- Flexible dataset configuration with custom image/mask folders
- Real-time training log in the text viewer
- Start/stop training control
- Automatic model building via dynamic import by category + name
- Pretrained weight loading — supports `.pth`, `.pt`, `.h5`, `.ckpt`

### 4. CLI Single-Model Training

```bash
python -m trainer.train --model unet --dataset kvasir \
    --data_root D:/data/Kvasir-SEG --epochs 100

python -m trainer.train --model medsam --pretrain /path/to/sam.pth \
    --epochs 20 --lr 1e-4
```

### 5. CLI Inference

```bash
python -m inference.run_inference --model unet --dataset kvasir \
    --checkpoint outputs/unet/kvasir/best.pth
```

### 6. Dataset Split Generation

```python
from datasets.split_utils import generate_split

generate_split(
    image_dir="data/Kvasir-SEG/images",
    mask_dir="data/Kvasir-SEG/masks",
    save_path="datasets/splits/kvasir.json",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
```

## Evaluation Metrics

Built-in segmentation metrics:

- Dice coefficient (F1-score)
- IoU (Jaccard index)
- Recall (sensitivity)
- Precision
- HD95 (95th-percentile Hausdorff distance)

The `metrics/` module supports:

- Single-model, single-dataset evaluation
- Cross-model and cross-dataset aggregation
- Model ranking generation

## Design Principles

- **Non-invasive:** `main.py` and `gui/fenge.py` only import public APIs from
  existing modules. They never modify code inside `models/`, `trainer/`, or
  `inference/`.
- **Unified entry points:** All 24 models share the same train / inference /
  evaluation pipeline.
- **Dual-mode operation:** CLI for batch automation, GUI for interactive
  visualization.
- **Reproducible:** Deterministic random seed + pre-generated dataset split files.
- **Extensible:** Adding a new model only requires registration in
  `MODEL_REGISTRY` and `_MODEL_BUILDERS`.

## Quick Start

1. Install dependencies:

   ```bash
   pip install torch torchvision numpy scikit-image scipy pillow matplotlib tqdm PyQt5
   ```

2. Prepare a dataset (example: Kvasir-SEG):

   - Download from https://datasets.simula.no/kvasir-seg/
   - Extract to `data/Kvasir-SEG/`
   - Ensure the directory structure has `images/` and `masks/` subdirectories

3. Generate dataset splits:

   ```bash
   python -c "
   from datasets.split_utils import generate_split
   generate_split('data/Kvasir-SEG/images', 'data/Kvasir-SEG/masks',
                   'datasets/splits/kvasir.json')
   "
   ```

4. Train a model:

   ```bash
   python main.py --quick    # Quick test: U-Net on Kvasir-SEG
   ```

5. Run interactive segmentation:

   ```bash
   python gui/fenge.py
   # Select unet → select outputs/unet/kvasir/best.pth → load image → draw ROI → run
   ```

6. Check results:

   - Terminal output: training loss, evaluation metrics
   - `outputs/` directory: model weights, metrics JSON, inference summaries
   - GUI: three-panel segmentation result displayed directly in the interface

## FAQ

**Q: `ImportError` when launching `gui/fenge.py`?**

A: Make sure you run from the project root (`gui-github/`), or add the
project root to `PYTHONPATH`:

```bash
# Windows
set PYTHONPATH=D:\gui-github

# Linux / macOS
export PYTHONPATH=/path/to/gui-github
```

**Q: A specific model fails to load?**

A: Some models (e.g. SAM, ConDSeg) have extra dependencies. Check the
`import` statements in the model's original `train.py` and install any
missing packages.

**Q: No ground truth mask found?**

A: `fenge.py` auto-searches these locations:

- `../masks/<basename>.png` or `.jpg`
- `../masktest/<basename>.png` or `.jpg`
- `./<basename>_mask.png` or `.jpg`

If none are found, panel 3 (GT) will be blank and metrics will be zero.

## Changelog

v1.0 (2026-06):

- Initial release
- 24 segmentation models across 4 families
- 4 medical imaging datasets
- `main.py` CLI benchmark orchestrator
- `gui/fenge.py` unified interactive segmentation GUI with in-interface
  three-panel result display
- `gui/train.py` training GUI
- Complete train / inference / evaluation pipeline
