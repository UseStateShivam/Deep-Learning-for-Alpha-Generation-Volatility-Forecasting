"""
Pytest Fixtures for Quantitative Deep Learning Suite
Provides small deterministic market bar datasets, features, and model fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_volatility.data.synthetic import generate_synthetic_market_data
from alpha_volatility.features.pipeline import FeaturePipeline
from alpha_volatility.labels.alignment import generate_all_labels


@pytest.fixture(scope="session")
def sample_market_bars() -> pd.DataFrame:
    """Generates a small deterministic 10-asset, 5-session market bar dataset."""
    df = generate_synthetic_market_data(
        num_assets=10,
        num_sessions=5,
        bars_per_session=78,
        seed=123,
    )
    return df


@pytest.fixture(scope="session")
def sample_features_and_labels(sample_market_bars: pd.DataFrame) -> pd.DataFrame:
    """Generates features and aligned labels on sample market bars."""
    pipeline = FeaturePipeline()
    feat_df = pipeline.generate_features(sample_market_bars)
    labeled_df = generate_all_labels(feat_df, horizon_bars=3, drop_unobservable=True)
    return labeled_df
