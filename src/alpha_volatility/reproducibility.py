"""
Reproducibility Utilities
Ensures deterministic random seeds and execution configurations across NumPy, PyTorch, and Python.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """
    Safely resolves the PyTorch compute device.

    Args:
        preference: "auto", "cuda", or "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preference == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
