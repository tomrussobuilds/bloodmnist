"""
Configuration and Command-Line Interface Module

This module defines the training hyperparameters using Pydantic for validation
and type safety. It also provides the argument parsing logic for the 
command-line interface (CLI).
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
import argparse

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                                HELPER FUNCTIONS
# =========================================================================== #

def _get_num_workers_config() -> int:
    """
    Calculates the default value for num_workers based on the environment.

    If DOCKER_REPRODUCIBILITY_MODE is set to '1' or 'TRUE', it returns 0
    to force single-thread execution for bit-per-bit determinism.
    
    Returns:
        int: The determined number of data loader workers (0 or 4).
    """
    is_docker_reproducible = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE")
    return 0 if is_docker_reproducible else 4

# =========================================================================== #
#                                CONFIGURATION
# =========================================================================== #

class Config(BaseModel):
    """Configuration class for training hyperparameters using Pydantic validation."""
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Core Hyperparameters
    seed: int = 42
    batch_size: int = Field(default=128, gt=0)
    num_workers: int = Field(default_factory=_get_num_workers_config)
    epochs: int = Field(default=60, gt=0)
    patience: int = Field(default=15, ge=0)
    learning_rate: float = Field(default=0.008, gt=0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    mixup_alpha: float = Field(default=0.002, ge=0.0)
    use_tta: bool = True
    
    # Metadata for Reporting
    model_name: str = "ResNet-18 Adapted"
    dataset_name: str = "BloodMNIST"
    normalization_info: str = "ImageNet Mean/Std"
    
    # Data Augmentation Parameters
    hflip: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_angle: int = Field(default=10, ge=0, le=180)
    jitter_val: float = Field(default=0.2, ge=0.0)

# =========================================================================== #
#                                ARGUMENT PARSING
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """
    Configure and analyze command line arguments for the training script.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="BloodMNIST training pipeline based on adapted ResNet-18."
    )
    
    # Use an instance of Config to get default values dynamically
    default_cfg = Config()

    parser.add_argument(
        '--epochs', type=int, default=default_cfg.epochs,
        help=f"Number of training epochs. Default: {default_cfg.epochs}"
    )
    parser.add_argument(
        '--batch_size', type=int, default=default_cfg.batch_size,
        help=f"Batch size for data loaders. Default: {default_cfg.batch_size}"
    )
    parser.add_argument(
        '--lr', '--learning_rate', type=float, default=default_cfg.learning_rate,
        help=f"Initial learning rate. Default: {default_cfg.learning_rate}"
    )
    parser.add_argument(
        '--seed', type=int, default=default_cfg.seed,
        help=f"Random seed for reproducibility. Default: {default_cfg.seed}"
    )
    parser.add_argument(
        '--mixup_alpha', type=float, default=default_cfg.mixup_alpha,
        help=f"Alpha for MixUp. Default: {default_cfg.mixup_alpha}"
    )
    parser.add_argument(
        '--patience', type=int, default=default_cfg.patience,
        help=f"Early stopping patience. Default: {default_cfg.patience}"
    )
    parser.add_argument(
        '--no_tta', action='store_true',
        help="Disable Test-Time Augmentation (TTA) during final evaluation."
    )
    parser.add_argument(
        '--momentum', type=float, default=default_cfg.momentum,
        help=f"Momentum for SGD. Default: {default_cfg.momentum}"
    )
    parser.add_argument(
        '--weight_decay', type=float, default=default_cfg.weight_decay,
        help=f"Weight decay for SGD. Default: {default_cfg.weight_decay}"
    )
    parser.add_argument(
        '--hflip', type=float, default=default_cfg.hflip,
        help=f"Probability of Horizontal Flip. Default: {default_cfg.hflip}"
    )
    parser.add_argument(
        '--rotation_angle', type=int, default=default_cfg.rotation_angle,
        help=f"Max rotation angle. Default: {default_cfg.rotation_angle}"
    )
    parser.add_argument(
        '--jitter_val', type=float, default=default_cfg.jitter_val,
        help=f"Color jitter value. Default: {default_cfg.jitter_val}"
    )
    
    return parser.parse_args()