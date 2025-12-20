"""
Dataset Fetching and Loading Module

This module handles the physical retrieval of the dataset, including robust 
download logic with retries, MD5 checksum verification, and loading the 
compressed NumPy (.npz) files into structured data containers.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Final

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import requests
import numpy as np

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import (
    Logger, Config, md5_checksum, validate_npz_keys,
    NPZ_PATH, EXPECTED_MD5, URL
)

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


# =========================================================================== #
#                                DATA CONTAINERS
# =========================================================================== #

@dataclass(frozen=True)
class BloodMNISTData:
    """A container for the BloodMNIST dataset splits stored as NumPy arrays."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


# =========================================================================== #
#                                FETCHING LOGIC
# =========================================================================== #

def ensure_mnist_npz(
        target_npz: Path,
        retries: int = 5,
        delay: float = 5.0,
        cfg: Config | None = None
    ) -> Path:
    """
    Downloads the BloodMNIST dataset NPZ file robustly, with retries and MD5 validation.

    Args:
        target_npz (Path): The expected path for the dataset NPZ file.
        retries (int): Max number of download attempts.
        delay (float): Delay (seconds) between retries.
        cfg (Config | None): Configuration object for logging context.

    Returns:
        Path: Path to the successfully validated .npz file.

    Raises:
        RuntimeError: If the dataset cannot be downloaded and verified.
    """
    def _is_valid(path: Path) -> bool:
        """Checks file existence, size, and MD5 checksum."""
        if not path.exists() or path.stat().st_size < 50_000:
            return False
        # Check for ZIP header (NPZ files are ZIP archives)
        if path.read_bytes()[:2] != b"PK":
            return False
        return md5_checksum(path) == EXPECTED_MD5

    if _is_valid(target_npz):
        logger.info(f"Valid dataset found: {target_npz}")
        return target_npz

    if target_npz.exists():
        logger.warning(f"Corrupted or incomplete dataset found, deleting: {target_npz}")
        target_npz.unlink()

    logger.info(f"Downloading {cfg.model_name if cfg else 'dataset'} from {URL}")
    tmp_path = target_npz.with_suffix(".tmp")
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status() 
            tmp_path.write_bytes(response.content)

            if not _is_valid(tmp_path):
                raise ValueError("Downloaded file failed validation (wrong size/header/MD5)")

            tmp_path.replace(target_npz) # Atomic move
            logger.info(f"Successfully downloaded and verified: {target_npz}")
            return target_npz

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                model_info = cfg.model_name if cfg else "dataset"
                logger.error(f"Failed to download dataset after {retries} attempts")
                raise RuntimeError(f"Could not download {model_info} dataset") from e

            logger.warning(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unexpected error in dataset download logic.")


def load_bloodmnist(npz_path: Path = NPZ_PATH,
                    cfg: Config | None = None
) -> BloodMNISTData:
    """
    Loads the dataset from the NPZ file, validates its keys, and returns
    the structured data splits.
    
    Args:
        npz_path (Path, optional): Path to the NPZ file.
        cfg (Config | None): Configuration object for download context.

    Returns:
        BloodMNISTData: The structured dataset splits.
    """
    path = ensure_mnist_npz(npz_path, cfg=cfg)

    logger.info(f"Loading dataset from {path}")

    # Use mmap_mode="r" for memory efficiency
    with np.load(npz_path, mmap_mode="r") as data:
        validate_npz_keys(data)
        logger.info(f"Keys in NPZ file: {data.files}")

        return BloodMNISTData(
            X_train=data["train_images"],
            X_val=data["val_images"],
            X_test=data["test_images"],
            y_train=data["train_labels"].ravel(), # Flatten labels
            y_val=data["val_labels"].ravel(),
            y_test=data["test_labels"].ravel(),
        )