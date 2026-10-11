"""
Data Storage Module
Provides idempotent Parquet storage and loading with schema enforcement and integrity hashing.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .schemas import CanonicalBarSchema


class ParquetStorage:
    """Handles persistent Parquet storage for raw, processed, and synthetic data."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_dataframe_hash(df: pd.DataFrame) -> str:
        """Computes a SHA-256 hash of DataFrame values for integrity tracking."""
        sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(sample_bytes).hexdigest()

    def save(
        self,
        df: pd.DataFrame,
        filename: str,
        partition_cols: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Saves DataFrame to Parquet idempotently.
        """
        target_path = self.base_dir / filename
        df_enforced = CanonicalBarSchema.enforce_schema(df)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        df_enforced.to_parquet(
            target_path,
            index=False,
            partition_cols=partition_cols,
            engine="pyarrow",
        )

        # Write sidecar metadata
        meta_path = target_path.with_suffix(".meta.json")
        meta_dict = {
            "num_rows": len(df_enforced),
            "num_assets": df_enforced["asset_id"].nunique(),
            "sha256": self.compute_dataframe_hash(df_enforced),
            "min_timestamp": df_enforced["timestamp"].min().isoformat(),
            "max_timestamp": df_enforced["timestamp"].max().isoformat(),
            "custom_metadata": metadata or {},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return target_path

    def load(self, filename: str) -> pd.DataFrame:
        """
        Loads Parquet file and enforces schema.
        """
        target_path = self.base_dir / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Parquet file not found at {target_path}")

        df = pd.read_parquet(target_path, engine="pyarrow")
        return CanonicalBarSchema.enforce_schema(df)
