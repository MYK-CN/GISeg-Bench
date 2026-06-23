# GISeg-Bench Datasets Module

Unified data-loading layer for the GISeg-Bench medical image segmentation benchmark.
**No raw data is stored in this repository.**

---

## Supported Datasets

| Name        | Modality               | Task              | Classes |
|-------------|------------------------|-------------------|---------|
| CVC-ClinicDB | Colonoscopy           | Polyp segmentation | 2 (bg / polyp) |
| Kvasir-SEG  | Gastrointestinal endoscopy | Polyp segmentation | 2 (bg / polyp) |
| WCEBleedGen | Wireless Capsule Endoscopy | Bleeding segmentation | 2 (bg / bleeding) |
| EDD2020     | Multi-site endoscopy   | Disease segmentation | 5+ (multi-class) |

---

## Download Links

| Dataset      | Official Source |
|--------------|----------------|
| CVC-ClinicDB | https://polyp.grand-challenge.org/CVCClinicDB/ |
| Kvasir-SEG   | https://datasets.simula.no/kvasir-seg/ |
| WCEBleedGen  | https://zenodo.org/records/10156571 |
| EDD2020      | https://edd2020.grand-challenge.org/ |

---

## Data Directory Format

After downloading, each dataset should be organised as:

```
<dataset_root>/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001.png
    ├── img002.png
    └── ...
```

**Rules:**
- Image and mask filenames must be identical (stem-matched).
- Images: RGB, any common format (`.png`, `.jpg`, `.jpeg`).
- Masks: grayscale, same format or `.gif` (Kvasir-SEG legacy).
- For Kvasir-SEG, `test/` and `masktest/` subdirectories are used for the held-out test set when present.

---

## Generating Splits

Run the built-in split utility **once** after downloading the data:

```python
from datasets.split_utils import generate_split

# Example: Kvasir-SEG
generate_split(
    image_dir="D:/data/Kvasir-SEG/images",
    mask_dir="D:/data/Kvasir-SEG/masks",
    save_path="datasets/splits/kvasir.json",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
```

This writes a `.json` file with deterministic train/val/test filename lists
(seed = 42).  Commit the generated split files to version control for full
reproducibility.

---

## Loading a Dataset

```python
from datasets.dataset_zoo import get_dataset
from datasets.transforms import SegTransform

# Build a transform (optional)
tf = SegTransform(size=256, normalise="imagenet")

# Load training split
train_ds = get_dataset(
    name="kvasir",
    root="/data/Kvasir-SEG",
    split="train",
    split_file="datasets/splits/kvasir.json",
    transform=tf,
)

# Cross-dataset validation
test_ds = get_dataset(
    name="cvc",
    root="/data/CVC-ClinicDB",
    split="test",
    split_file="datasets/splits/cvc.json",
    transform=tf,
)
```

---

## Module Structure

```
datasets/
├── README.md            # ← this file
├── dataset_zoo.py       # unified get_dataset() entry point
├── base_dataset.py      # abstract BaseSegDataset class
├── transforms.py        # standard image/mask transforms
├── split_utils.py       # deterministic split generation & loading
├── splits/
│   ├── cvc.json         # CVC-ClinicDB split (stem list)
│   ├── kvasir.json      # Kvasir-SEG split
│   ├── wce.json         # WCEBleedGen split
│   └── edd.json         # EDD2020 split
├── cvc.py               # CVCClinicDB dataset class
├── kvasir.py            # KvasirSEG dataset class
├── wce.py               # WCEBleedGen dataset class
└── edd.py               # EDD2020 dataset class (multi-class)
```

---

## Cross-Dataset Experiment Example

```python
from datasets.dataset_zoo import get_dataset
from torch.utils.data import DataLoader

# Train on Kvasir-SEG, test on CVC-ClinicDB
train_ds = get_dataset("kvasir", root="/data/Kvasir-SEG", split="train",
                       split_file="datasets/splits/kvasir.json")
test_ds  = get_dataset("cvc", root="/data/CVC-ClinicDB", split="test",
                       split_file="datasets/splits/cvc.json")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)
```
