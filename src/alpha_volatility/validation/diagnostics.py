"""
Validation Diagnostics Module
Generates and serializes fold manifests tracking purging and embargo counts.
"""

import json
from pathlib import Path
from typing import List

from .splits import SplitFold


def export_fold_manifest(
    folds: List[SplitFold],
    output_path: str | Path = "artifacts/manifests/fold_manifest.json",
    seed: int = 42,
) -> Path:
    """
    Exports a structured fold manifest documenting sample counts, purge rates, and time boundaries.
    """
    manifest_records = []
    for f in folds:
        record = {
            "fold_id": f.fold_id,
            "train_start": str(f.train_start),
            "train_end": str(f.train_end),
            "val_start": str(f.val_start),
            "val_end": str(f.val_end),
            "initial_train_samples": f.initial_train_count,
            "purged_samples": f.purged_count,
            "embargoed_samples": f.embargoed_count,
            "final_train_samples": f.final_train_count,
            "val_samples": f.val_count,
            "seed": seed,
        }
        manifest_records.append(record)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump({"folds": manifest_records}, fp, indent=2)

    return out_file
