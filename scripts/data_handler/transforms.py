"""
Data Transformations Module

This module defines the image augmentation pipelines for training and 
the standard normalization for validation/testing. It also includes 
utilities for deterministic worker initialization.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import random
from typing import Tuple, Final

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
from torchvision import transforms

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Config

# Constants for standard ImageNet normalization
IMG_SIZE: Final[int] = 28
NORM_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
NORM_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)


def get_augmentations_transforms(cfg: Config) -> str:
    """
    Generates a descriptive string of the augmentations using values from Config.
    """ 
    return (
        f"RandomHorizontalFlip({cfg.hflip}), "
        f"RandomRotation({cfg.rotation_angle}), "
        f"ColorJitter ({cfg.jitter_val}), "
        f"RandomResizedCrop(28, scale=(0.9, 1.0)), "
        f"MixUp(alpha={cfg.mixup_alpha})"
    )


def worker_init_fn(worker_id: int):
    """
    Initializes random number generators (PRNGs) for each DataLoader worker.
    """
    initial_seed = Config().seed
    worker_seed = initial_seed + worker_id 
    
    np.random.seed(worker_seed)
    random.seed(worker_seed) 
    torch.manual_seed(worker_seed)


def get_pipeline_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Defines the transformation pipelines for training and evaluation.
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=cfg.hflip),
        transforms.RandomRotation(cfg.rotation_angle),
        transforms.ColorJitter(
            brightness=cfg.jitter_val,
            contrast=cfg.jitter_val,
            saturation=cfg.jitter_val
        ),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    
    return train_transform, val_transform