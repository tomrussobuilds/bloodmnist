"""
System and Hardware Utilities Module

This module provides low-level utilities for hardware abstraction (device selection),
reproducibility (seeding), file integrity (checksums), and process management.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
import random
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import psutil

# =========================================================================== #
#                               SYSTEM UTILITIES
# =========================================================================== #

def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across NumPy, Python, and PyTorch.

    Args:
        seed (int): The integer seed value to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(logger: logging.Logger) -> torch.device:
    """
    Determines the appropriate device (CUDA or CPU) for computation.
    
    Args:
        logger (logging.Logger): Logger instance to report the selected device.
    
    Returns:
        torch.device: The selected device object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device 


def md5_checksum(path: Path) -> str:
    """
    Calculates the MD5 checksum of a file in chunks for efficiency.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        # Read the file in 8192-byte chunks to avoid memory issues
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all expected keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        ValueError: If any required key is missing from the NPZ file.
    """
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"NPZ file is missing required keys: {missing}")
    

def kill_duplicate_processes(logger: logging.Logger, script_name: Optional[str] = None) -> None:
    """
    Kills all Python processes executing the same script, excluding the current one.

    Args:
        logger (logging.Logger): Logger instance to report actions.
        script_name (str, optional): The filename of the script to check.
                                     Defaults to the current script's filename.
    """
    if script_name is None:
        script_name = os.path.basename(__file__)
    
    current_pid = os.getpid()
    killed = 0
    python_executables = ('python', 'python3', 'python.exe')

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Filter for Python processes
            if proc.info['name'] not in python_executables:
                continue
            
            cmdline = proc.cmdline()
            if proc.pid == current_pid:
                continue           
            
            # Match script name in command line arguments
            is_match = any(script_name in arg for arg in cmdline)
            
            if is_match:
                proc.terminate()
                killed += 1
                logger.info(f"Killed duplicate process PID {proc.pid}")
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        logger.info(f"Cleaned up {killed} duplicate process(es). Waiting for resources...")
        time.sleep(1)