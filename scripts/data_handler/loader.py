"""
DataLoader Generation Module

This module orchestrates the creation of PyTorch DataLoaders by combining 
the fetched data, the dataset structure, and the transformation pipelines.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
from typing import Tuple, Final

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Config, Logger
from .fetcher import BloodMNISTData
from .dataset import BloodMNISTDataset
from .transforms import get_pipeline_transforms, worker_init_fn

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


def get_dataloaders(
        data: BloodMNISTData,
        cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    """
    # 1. Get Transformation Pipelines
    train_transform, val_transform = get_pipeline_transforms(cfg)

    # 2. Create Datasets
    train_ds = BloodMNISTDataset(data.X_train, data.y_train, transform=train_transform)
    val_ds   = BloodMNISTDataset(data.X_val,   data.y_val,   transform=val_transform)
    test_ds  = BloodMNISTDataset(data.X_test,  data.y_test,  transform=val_transform)

    # 3. Setup DataLoader Parameters
    init_fn = worker_init_fn if cfg.num_workers > 0 else None
    pin_memory = torch.cuda.is_available()

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    
    logger.info(f"Dataset loaded → Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")
    
    return train_loader, val_loader, test_loader