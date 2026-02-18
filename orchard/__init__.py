"""
Orchard ML: Type-Safe Deep Learning for Reproducible Research.

Top-level convenience API re-exporting the most commonly used components
from subpackages, so users and entry points (forge.py) can write:

    from orchard import Config, RootOrchestrator, get_model
"""

from orchard.core import (
    Config,
    LogStyle,
    RootOrchestrator,
    log_pipeline_summary,
    parse_args,
)
from orchard.core.paths import MLRUNS_DB
from orchard.models import get_model
from orchard.pipeline import run_export_phase, run_optimization_phase, run_training_phase
from orchard.tracking import create_tracker

__all__ = [
    # Core
    "Config",
    "LogStyle",
    "RootOrchestrator",
    "log_pipeline_summary",
    "parse_args",
    # Paths
    "MLRUNS_DB",
    # Models
    "get_model",
    # Pipeline
    "run_export_phase",
    "run_optimization_phase",
    "run_training_phase",
    # Tracking
    "create_tracker",
]
