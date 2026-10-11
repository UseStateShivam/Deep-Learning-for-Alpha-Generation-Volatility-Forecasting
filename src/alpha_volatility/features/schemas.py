"""
Feature Schemas and Registry
Defines the metadata registry for all engineered alpha and risk features.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata contract describing an engineered feature."""
    name: str
    category: str              # "price", "volatility", "volume", "momentum", "temporal", "cross_sectional"
    formula: str
    lookback_bars: int
    required_warmup: int
    input_columns: List[str]
    is_cross_sectional: bool
    description: str


# Feature Registry containing formal mathematical definitions and lookbacks
FEATURE_REGISTRY: Dict[str, FeatureMetadata] = {
    # Returns & Price
    "log_return_1": FeatureMetadata(
        name="log_return_1",
        category="price",
        formula="ln(close_t / close_{t-1})",
        lookback_bars=1,
        required_warmup=2,
        input_columns=["close"],
        is_cross_sectional=False,
        description="One-bar (5-minute) backward log return.",
    ),
    "log_return_3": FeatureMetadata(
        name="log_return_3",
        category="price",
        formula="ln(close_t / close_{t-3})",
        lookback_bars=3,
        required_warmup=4,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Three-bar (15-minute) backward cumulative log return.",
    ),
    "log_return_6": FeatureMetadata(
        name="log_return_6",
        category="price",
        formula="ln(close_t / close_{t-6})",
        lookback_bars=6,
        required_warmup=7,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Six-bar (30-minute) backward cumulative log return.",
    ),
    "log_return_12": FeatureMetadata(
        name="log_return_12",
        category="price",
        formula="ln(close_t / close_{t-12})",
        lookback_bars=12,
        required_warmup=13,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Twelve-bar (60-minute) backward cumulative log return.",
    ),
    "high_low_range": FeatureMetadata(
        name="high_low_range",
        category="price",
        formula="ln(high_t / low_t)",
        lookback_bars=1,
        required_warmup=1,
        input_columns=["high", "low"],
        is_cross_sectional=False,
        description="Single-bar log high-to-low price dispersion.",
    ),
    "open_close_return": FeatureMetadata(
        name="open_close_return",
        category="price",
        formula="ln(close_t / open_t)",
        lookback_bars=1,
        required_warmup=1,
        input_columns=["open", "close"],
        is_cross_sectional=False,
        description="Intra-bar log return from open to close.",
    ),
    "price_dev_12": FeatureMetadata(
        name="price_dev_12",
        category="price",
        formula="(close_t - mean(close_{t-11..t})) / std(close_{t-11..t})",
        lookback_bars=12,
        required_warmup=12,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Z-score deviation of close price from 12-bar rolling average.",
    ),
    # Volatility
    "realized_vol_6": FeatureMetadata(
        name="realized_vol_6",
        category="volatility",
        formula="sqrt(sum(r_{t-i}^2, i=0..5))",
        lookback_bars=6,
        required_warmup=7,
        input_columns=["close"],
        is_cross_sectional=False,
        description="6-bar backward realized return volatility.",
    ),
    "realized_vol_12": FeatureMetadata(
        name="realized_vol_12",
        category="volatility",
        formula="sqrt(sum(r_{t-i}^2, i=0..11))",
        lookback_bars=12,
        required_warmup=13,
        input_columns=["close"],
        is_cross_sectional=False,
        description="12-bar backward realized return volatility.",
    ),
    "realized_vol_36": FeatureMetadata(
        name="realized_vol_36",
        category="volatility",
        formula="sqrt(sum(r_{t-i}^2, i=0..35))",
        lookback_bars=36,
        required_warmup=37,
        input_columns=["close"],
        is_cross_sectional=False,
        description="36-bar (3-hour) backward realized return volatility.",
    ),
    "parkinson_vol_12": FeatureMetadata(
        name="parkinson_vol_12",
        category="volatility",
        formula="sqrt(1/(4*ln(2)*12) * sum((ln(H/L))^2))",
        lookback_bars=12,
        required_warmup=12,
        input_columns=["high", "low"],
        is_cross_sectional=False,
        description="12-bar Parkinson extreme-value range volatility proxy.",
    ),
    "vol_ratio_6_36": FeatureMetadata(
        name="vol_ratio_6_36",
        category="volatility",
        formula="realized_vol_6 / (realized_vol_36 + 1e-8)",
        lookback_bars=36,
        required_warmup=37,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Short-to-long volatility term structure ratio.",
    ),
    # Volume & Liquidity Proxies
    "log_volume": FeatureMetadata(
        name="log_volume",
        category="volume",
        formula="ln(volume_t + 1)",
        lookback_bars=1,
        required_warmup=1,
        input_columns=["volume"],
        is_cross_sectional=False,
        description="Log-transformed traded volume.",
    ),
    "volume_zscore_12": FeatureMetadata(
        name="volume_zscore_12",
        category="volume",
        formula="(vol_t - mean(vol_{t-11..t})) / std(vol_{t-11..t})",
        lookback_bars=12,
        required_warmup=12,
        input_columns=["volume"],
        is_cross_sectional=False,
        description="12-bar rolling volume z-score.",
    ),
    "relative_volume_12": FeatureMetadata(
        name="relative_volume_12",
        category="volume",
        formula="vol_t / (mean(vol_{t-11..t}) + 1e-6)",
        lookback_bars=12,
        required_warmup=12,
        input_columns=["volume"],
        is_cross_sectional=False,
        description="Current volume relative to 12-bar rolling baseline.",
    ),
    "pv_interaction": FeatureMetadata(
        name="pv_interaction",
        category="volume",
        formula="ln(volume_t + 1) * abs(log_return_1)",
        lookback_bars=1,
        required_warmup=2,
        input_columns=["volume", "close"],
        is_cross_sectional=False,
        description="Interaction between volume magnitude and absolute return.",
    ),
    # Momentum & Mean Reversion
    "momentum_6": FeatureMetadata(
        name="momentum_6",
        category="momentum",
        formula="log_return_6",
        lookback_bars=6,
        required_warmup=7,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Short 30-minute momentum signal.",
    ),
    "momentum_24": FeatureMetadata(
        name="momentum_24",
        category="momentum",
        formula="ln(close_t / close_{t-24})",
        lookback_bars=24,
        required_warmup=25,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Medium 2-hour momentum signal.",
    ),
    "trend_consistency_12": FeatureMetadata(
        name="trend_consistency_12",
        category="momentum",
        formula="mean(sign(r_{t-i}) == sign(r_t), i=0..11)",
        lookback_bars=12,
        required_warmup=13,
        input_columns=["close"],
        is_cross_sectional=False,
        description="Proportion of bars in past 12 aligned with current direction.",
    ),
    # Temporal
    "sin_time_of_day": FeatureMetadata(
        name="sin_time_of_day",
        category="temporal",
        formula="sin(2*pi*bar_idx / bars_per_session)",
        lookback_bars=0,
        required_warmup=0,
        input_columns=["timestamp"],
        is_cross_sectional=False,
        description="Cyclical intraday sine encoding.",
    ),
    "cos_time_of_day": FeatureMetadata(
        name="cos_time_of_day",
        category="temporal",
        formula="cos(2*pi*bar_idx / bars_per_session)",
        lookback_bars=0,
        required_warmup=0,
        input_columns=["timestamp"],
        is_cross_sectional=False,
        description="Cyclical intraday cosine encoding.",
    ),
    "session_progress": FeatureMetadata(
        name="session_progress",
        category="temporal",
        formula="bar_idx / bars_per_session",
        lookback_bars=0,
        required_warmup=0,
        input_columns=["timestamp"],
        is_cross_sectional=False,
        description="Linear fraction of trading session elapsed.",
    ),
    # Cross-Sectional
    "cs_rank_return_1": FeatureMetadata(
        name="cs_rank_return_1",
        category="cross_sectional",
        formula="rank_t(log_return_1) / N_t - 0.5",
        lookback_bars=1,
        required_warmup=2,
        input_columns=["close"],
        is_cross_sectional=True,
        description="Cross-sectional percentile rank of 1-bar return at timestamp t.",
    ),
    "cs_rank_momentum_12": FeatureMetadata(
        name="cs_rank_momentum_12",
        category="cross_sectional",
        formula="rank_t(log_return_12) / N_t - 0.5",
        lookback_bars=12,
        required_warmup=13,
        input_columns=["close"],
        is_cross_sectional=True,
        description="Cross-sectional percentile rank of 1-hour momentum at timestamp t.",
    ),
    "cs_rank_vol_12": FeatureMetadata(
        name="cs_rank_vol_12",
        category="cross_sectional",
        formula="rank_t(realized_vol_12) / N_t - 0.5",
        lookback_bars=12,
        required_warmup=13,
        input_columns=["close"],
        is_cross_sectional=True,
        description="Cross-sectional percentile rank of 12-bar realized volatility.",
    ),
}
