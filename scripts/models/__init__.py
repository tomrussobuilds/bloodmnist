"""
Models Factory Package

This package implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It routes requests to specific architecture 
definitions based on the configuration provided.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch
import torch.nn as nn

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Logger, Config, BLOODMNIST_CLASSES
# Temporary import: we keep the original name for now to avoid breaking changes
from .resnet_18_adapted import build_resnet18_adapted

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


def get_model(
    device: torch.device,
    cfg: Config
) -> nn.Module:
    """
    Factory function to instantiate and prepare the model.

    Based on the 'model_name' defined in the Config object, this function 
    calls the appropriate builder and moves the resulting model to the 
    specified hardware device.

    Args:
        device (torch.device): The hardware device (CPU/CUDA) to host the model.
        cfg (Config): The global configuration object containing model metadata.

    Returns:
        nn.Module: The instantiated and hardware-assigned PyTorch model.

    Raises:
        ValueError: If the requested model_name is not registered in the factory.
    """
    
    # Normalize model name for robust matching
    model_name_lower = cfg.model_name.lower()
    num_classes = len(BLOODMNIST_CLASSES)

    # Routing logic (Factory Pattern)
    if "resnet-18 adapted" in model_name_lower:
        # Currently routes to the adapted ResNet-18 implementation
        model = build_resnet18_adapted(
            device=device, 
            num_classes=num_classes, 
            cfg=cfg
        )
    else:
        error_msg = f"Model architecture '{cfg.model_name}' is not recognized by the Factory."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return model