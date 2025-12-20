"""
Data Visualization Module

This module provides utilities for inspecting the dataset visually, 
specifically by generating grids of sample images from the raw NumPy arrays.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
from pathlib import Path
from typing import Final

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Config, FIGURES_DIR, Logger, BLOODMNIST_CLASSES
# We import the dataclass from the local fetcher (that we will create next)
from .fetcher import BloodMNISTData

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


def show_sample_images(
        data: BloodMNISTData,
        save_path: Path | None = None,
        cfg: Config | None = None
    ) -> None:  
    """
    Generates and saves a figure showing 9 random samples from the training set.

    Args:
        data (BloodMNISTData): The structured dataset (to access training images).
        save_path (Path | None, optional): Path to save the figure.
                                           Defaults to FIGURES_DIR/bloodmnist_samples.png.
        cfg (Config | None): Configuration object for title and naming.
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"{cfg.model_name}_samples.png"

    if save_path.exists():
        logger.info(f"Sample images figure already exists → {save_path}")
        return

    indices = np.random.choice(len(data.X_train), size=9, replace=False)

    plt.figure(figsize=(9, 9))
    for i, idx in enumerate(indices):
        img = data.X_train[idx]
        label = int(data.y_train[idx])

        plt.subplot(3, 3, i + 1)

        # Handle grayscale (1 channel) or color (3 channels) images
        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imshow(img)
        else:
            plt.imshow(img.squeeze(), cmap='gray')

        plt.title(f"{label} — {BLOODMNIST_CLASSES[label]}", fontsize=11)
        plt.axis("off")

    plt.suptitle(f"{cfg.model_name} — 9 Random Samples from Training Set", fontsize=16)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")