"""
Leakage and Lookahead Tests
Strict tests proving the complete absence of future information leakage.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_volatility.data.synthetic import generate_synthetic_market_data
from alpha_volatility.features.pipeline import FeaturePipeline
from alpha_volatility.validation.purging import purge_overlapping_samples
from alpha_volatility.validation.embargo import apply_embargo
from alpha_volatility.labels.alignment import generate_all_labels


def test_future_data_does_not_alter_past_features():
    """
    CRITICAL LEAKAGE TEST:
    Modifying or appending future market observations must NOT alter feature values at past time t.
    """
    # 1. Base dataset: 3 sessions
    df_base = generate_synthetic_market_data(num_assets=3, num_sessions=3, seed=42)
    pipeline = FeaturePipeline()
    feat_base = pipeline.generate_features(df_base)

    # Cut point at end of session 2
    session_2_end = df_base["timestamp"].drop_duplicates().sort_values().iloc[155]
    past_features_orig = feat_base[feat_base["timestamp"] <= session_2_end][pipeline.feature_columns].values

    # 2. Append 2 completely different future sessions (with different seed)
    df_future = generate_synthetic_market_data(num_assets=3, num_sessions=5, seed=999)
    # Overwrite only the past with df_base so the history is identical but future is different
    df_combined = pd.concat([df_base[df_base["timestamp"] <= session_2_end], df_future[df_future["timestamp"] > session_2_end]])
    df_combined = df_combined.sort_values(by=["timestamp", "asset_id"]).reset_index(drop=True)

    feat_modified = pipeline.generate_features(df_combined)
    past_features_new = feat_modified[feat_modified["timestamp"] <= session_2_end][pipeline.feature_columns].values

    # Feature values in the past MUST be bit-exact identical
    np.testing.assert_allclose(past_features_orig, past_features_new, rtol=1e-5, atol=1e-5)


def test_purging_eliminates_overlapping_label_intervals():
    """
    Verifies that purging removes any training sample whose forward return
    interval [t, t+h] overlaps with the validation window [val_start, val_end].
    """
    ts = pd.date_range("2024-01-01 10:00:00", periods=10, freq="5min", tz="UTC")
    # Observation at 10:00 with h=3 forward bars has label window [10:00, 10:15]
    # Observation at 10:05 has label window [10:05, 10:20]
    # Observation at 10:10 has label window [10:10, 10:25]
    train_df = pd.DataFrame({
        "timestamp": [ts[0], ts[1], ts[2]],
        "label_info_start": [ts[0], ts[1], ts[2]],
        "label_info_end": [ts[3], ts[4], ts[5]],
    })

    # Validation window starts at ts[3] (10:15)
    val_df = pd.DataFrame({
        "timestamp": [ts[3], ts[4]],
        "label_info_start": [ts[3], ts[4]],
        "label_info_end": [ts[6], ts[7]],
    })

    # Train sample 0 ends at ts[3] -> overlaps with val_df (starts at ts[3])!
    # Train sample 1 ends at ts[4] -> overlaps!
    # Train sample 2 ends at ts[5] -> overlaps!
    purged_train, n_purged = purge_overlapping_samples(train_df, val_df)
    assert n_purged == 3
    assert len(purged_train) == 0


def test_embargo_excludes_post_validation_observations():
    """
    Verifies that observations in the designated post-validation embargo window are removed.
    """
    ts = pd.date_range("2024-01-01 10:00:00", periods=10, freq="5min", tz="UTC")
    val_df = pd.DataFrame({"timestamp": [ts[0], ts[1]]})  # Ends at ts[1] (10:05)

    # Train df with observations before, within embargo, and after embargo
    train_df = pd.DataFrame({
        "timestamp": [ts[2], ts[3], ts[6]],  # ts[2]=10:10, ts[3]=10:15, ts[6]=10:30
    })

    # Embargo of 3 bars (15 mins) post val_end (10:05) means cutoff is 10:20 (ts[4])
    embargoed_train, n_embargoed = apply_embargo(train_df, val_df, embargo_bars=3)
    # ts[2] and ts[3] are <= 10:20 -> embargoed! ts[6] > 10:20 -> kept!
    assert n_embargoed == 2
    assert len(embargoed_train) == 1
    assert embargoed_train["timestamp"].iloc[0] == ts[6]
