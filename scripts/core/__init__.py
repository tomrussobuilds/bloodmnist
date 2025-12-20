"""
Core Utilities Package

This package exposes the essential components for configuration, logging, 
system management, and project constants.
"""

from .config import Config, parse_args
from .constants import (
    PROJECT_ROOT, 
    DATASET_DIR, 
    FIGURES_DIR, 
    MODELS_DIR, 
    LOG_DIR, 
    REPORTS_DIR,
    ALL_DIRS,
    NPZ_PATH,
    EXPECTED_MD5,
    URL,
    BLOODMNIST_CLASSES,
    setup_directories
)
from .logger import Logger, logger, log_file
from .system import (
    set_seed, 
    get_device, 
    md5_checksum, 
    validate_npz_keys, 
    kill_duplicate_processes
)