"""
Distributed Environment Detection Utilities.

Provides lightweight rank and world-size detection from environment variables
set by distributed launchers (torchrun, torch.distributed.launch). All functions
return safe single-process defaults when no distributed environment is active.

These utilities are used by guards, orchestrator, and infrastructure management
to gate rank-specific behavior (e.g., locking, filesystem provisioning, logging)
without requiring torch.distributed to be initialized.

Environment Variables (set automatically by torchrun):
    RANK: Global rank of the current process (0-indexed).
    LOCAL_RANK: Rank within the current node (0-indexed).
    WORLD_SIZE: Total number of processes across all nodes.

Typical Usage:
    >>> from orchard.core.environment import is_main_process, get_rank
    >>> if is_main_process():
    ...     # Only rank 0 writes checkpoints, logs, etc.
    ...     save_checkpoint(model)
"""

import os


def get_rank() -> int:
    """
    Return the global rank of the current process.

    Reads from the ``RANK`` environment variable set by torchrun or
    torch.distributed.launch.  Returns 0 when running outside a
    distributed context (single-process default).
    """
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """
    Return the node-local rank of the current process.

    Reads from the ``LOCAL_RANK`` environment variable.  Used primarily
    for per-rank GPU assignment (``torch.device(f"cuda:{local_rank}")``).
    Returns 0 in single-process mode.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """
    Return the total number of distributed processes.

    Reads from the ``WORLD_SIZE`` environment variable.  Returns 1
    when running outside a distributed context.
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    """
    Detect whether the current process was launched in a distributed context.

    Returns True when either ``RANK`` or ``LOCAL_RANK`` is present in
    the environment, indicating a torchrun or equivalent launcher.
    """
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


def is_main_process() -> bool:
    """
    Check whether the current process is the main (rank 0) process.

    Always returns True in single-process mode.  In distributed mode,
    only the process with ``RANK=0`` returns True.
    """
    return get_rank() == 0
