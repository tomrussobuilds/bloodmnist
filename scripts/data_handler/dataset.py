"""
PyTorch Dataset Definition Module

This module contains the custom Dataset class for BloodMNIST, handling
the conversion from NumPy arrays to PyTorch tensors and applying 
image transformations for training and inference.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Tuple

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# =========================================================================== #
#                                DATASET CLASS
# =========================================================================== #

class BloodMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for BloodMNIST data. Handles NumPy to Tensor conversion
    and optional image transformations.
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 transform: transforms.Compose | None = None):
        """
        Args:
            images (np.ndarray): Image data (H, W, C).
            labels (np.ndarray): Label data.
            transform (transforms.Compose | None): Torchvision transformations.
        """
        # Normalize pixel values to [0.0, 1.0]
        self.images = images.astype(np.float32) / 255.0
        # Convert labels to int64 (long) for loss function
        self.labels = labels.astype(np.int64) 
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Apply transformation (e.g., augmentation for training)
            img = self.transform(img)
        else:
             # Convert NumPy array (H, W, C) to PyTorch Tensor (C, H, W)
             img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.tensor(label, dtype=torch.long)