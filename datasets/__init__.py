"""
GISeg-Bench Datasets Module
===========================
Unified data-loading layer for medical image segmentation.
"""

from .dataset_zoo import get_dataset, list_datasets
from .base_dataset import BaseSegDataset
from .transforms import SegTransform, imagenet_normalize, per_image_normalize
