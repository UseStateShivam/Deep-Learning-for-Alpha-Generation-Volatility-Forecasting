"""
Validation module for purged cross-validation, walk-forward splits, and embargo periods.
"""

from .diagnostics import export_fold_manifest
from .embargo import apply_embargo
from .purging import purge_overlapping_samples
from .splits import PurgedWalkForwardCV, SplitFold

__all__ = [
    "purge_overlapping_samples",
    "apply_embargo",
    "SplitFold",
    "PurgedWalkForwardCV",
    "export_fold_manifest",
]
