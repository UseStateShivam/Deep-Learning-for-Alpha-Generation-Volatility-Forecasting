"""
Model Checkpointing Module
Provides atomic serialization and deserialization of neural model states and experiment metadata.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state_dict: Dict[str, Any],
    file_path: str | Path,
) -> Path:
    """
    Saves checkpoint dictionary atomically using temporary file swap.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        torch.save(state_dict, tmp_path)

    # Atomic rename
    tmp_path.replace(target)
    return target


def load_checkpoint(file_path: str | Path, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """
    Loads checkpoint dictionary.
    """
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found at {target}")
    return torch.load(target, map_location=map_location)
