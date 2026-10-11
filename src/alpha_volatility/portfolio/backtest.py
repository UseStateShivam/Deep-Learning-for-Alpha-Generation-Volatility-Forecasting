"""
Backtesting and Decile Portfolio Simulation Engine
Executes chronological event simulation, tracks gross and net returns, computes turnover,
verifies dollar-neutrality, and tests decile monotonicity.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from ..evaluation.portfolio_metrics import compute_portfolio_performance_metrics
from .construction import assign_decile_quantiles, construct_dollar_neutral_weights
from .costs import TransactionCostModel
from .signals import process_alpha_signal


@dataclass
class BacktestResult:
    """Encapsulates time-series returns and summary metrics of portfolio backtest."""
    equity_curve: pd.DataFrame
    decile_returns: pd.DataFrame
    summary_metrics: Dict[str, float]
    monotonicity_rank_corr: float
    is_dollar_neutral: bool


class DecilePortfolioBacktester:
    """Simulates a dollar-neutral decile rebalancing strategy."""

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        num_quantiles: int = 10,
        min_universe_size: int = 10,
        rebalance_interval_bars: int = 3,
        execution_delay_bars: int = 1,
        bars_per_year: int = 19656,
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.num_quantiles = num_quantiles
        self.min_universe_size = min_universe_size
        self.rebalance_interval_bars = rebalance_interval_bars
        self.execution_delay_bars = execution_delay_bars
        self.bars_per_year = bars_per_year

    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        signal_col: str = "pred_return_mean",
        return_col: str = "target_return",
    ) -> BacktestResult:
        """
        Executes backtest over out-of-sample prediction history.

        Args:
            predictions_df: DataFrame containing ['timestamp', 'asset_id', signal_col, return_col].
            signal_col: Name of prediction alpha signal.
            return_col: Realized return column over the horizon.

        Returns:
            BacktestResult object.
        """
        # 1. Process signals & assign deciles
        proc_df = process_alpha_signal(predictions_df, signal_col=signal_col)
        decile_df = assign_decile_quantiles(
            proc_df,
            signal_col="alpha_signal",
            num_quantiles=self.num_quantiles,
            min_universe_size=self.min_universe_size,
        )

        # 2. Compute decile average forward returns for monotonicity analysis
        decile_summary = []
        for q in range(1, self.num_quantiles + 1):
            q_subset = decile_df[decile_df["quantile"] == q]
            mean_ret = float(q_subset[return_col].mean()) if len(q_subset) > 0 else 0.0
            decile_summary.append({
                "decile": q,
                "mean_return": mean_ret,
                "count": len(q_subset),
            })
        decile_table = pd.DataFrame(decile_summary)

        # Calculate Spearman monotonicity correlation between decile (1..10) and mean_return
        deciles_idx = decile_table["decile"].values
        deciles_ret = decile_table["mean_return"].values
        monotonicity_corr, _ = stats.spearmanr(deciles_idx, deciles_ret)
        if np.isnan(monotonicity_corr):
            monotonicity_corr = 0.0

        # 3. Simulate rebalancing timeline
        unique_times = decile_df["timestamp"].drop_duplicates().sort_values().tolist()
        all_assets = sorted(decile_df["asset_id"].unique().tolist())
        num_assets = len(all_assets)
        asset_to_idx = {a: i for i, a in enumerate(all_assets)}

        # Filter rebalance times according to rebalance_interval_bars
        rebalance_times = set(unique_times[:: self.rebalance_interval_bars])

        current_weights = np.zeros(num_assets)
        portfolio_records = []

        is_dollar_neutral = True

        for _t_idx, ts in enumerate(unique_times):
            ts_slice = decile_df[decile_df["timestamp"] == ts]

            # Rebalance decision occurs at rebalance times
            if ts in rebalance_times:
                w_dict = construct_dollar_neutral_weights(
                    ts_slice,
                    quantile_col="quantile",
                    num_quantiles=self.num_quantiles,
                )
                target_w = np.zeros(num_assets)
                for a, w in w_dict.items():
                    if a in asset_to_idx:
                        target_w[asset_to_idx[a]] = w

                # Check dollar neutrality
                net_exp = float(np.sum(target_w))
                if abs(net_exp) > 1e-4:
                    is_dollar_neutral = False

                # Transaction costs
                turnover = float(0.5 * np.sum(np.abs(target_w - current_weights)))
                cost = self.cost_model.calculate_cost(target_w, current_weights)
                current_weights = target_w
            else:
                turnover = 0.0
                cost = 0.0

            # Realized asset returns for this bar
            # Asset forward returns in this bar:
            asset_ret_map = dict(zip(ts_slice["asset_id"], ts_slice[return_col], strict=False))
            bar_rets = np.array([asset_ret_map.get(a, 0.0) for a in all_assets])

            # Separate long and short returns
            long_mask = current_weights > 0
            short_mask = current_weights < 0

            long_ret = float(np.sum(current_weights[long_mask] * bar_rets[long_mask]))
            short_ret = float(np.sum(current_weights[short_mask] * bar_rets[short_mask]))
            gross_ret = long_ret + short_ret
            net_ret = gross_ret - cost

            portfolio_records.append({
                "timestamp": ts,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "long_leg_return": long_ret,
                "short_leg_return": short_ret,
                "turnover": turnover,
                "cost": cost,
                "gross_exposure": float(np.sum(np.abs(current_weights))),
                "net_exposure": float(np.sum(current_weights)),
            })

        eq_df = pd.DataFrame(portfolio_records)
        eq_df["cum_gross_wealth"] = np.cumprod(1.0 + eq_df["gross_return"])
        eq_df["cum_net_wealth"] = np.cumprod(1.0 + eq_df["net_return"])

        # Summary performance metrics
        perf_metrics = compute_portfolio_performance_metrics(
            returns=eq_df["net_return"].values,
            bars_per_year=self.bars_per_year,
            turnover_series=eq_df["turnover"].values,
            cost_series=eq_df["cost"].values,
        )
        perf_metrics_gross = compute_portfolio_performance_metrics(
            returns=eq_df["gross_return"].values,
            bars_per_year=self.bars_per_year,
        )
        perf_metrics["sharpe_ratio_gross"] = perf_metrics_gross.get("sharpe_ratio", 0.0)
        perf_metrics["cumulative_return_gross"] = perf_metrics_gross.get("cumulative_return", 0.0)
        perf_metrics["monotonicity_score"] = round(float(monotonicity_corr), 4)

        return BacktestResult(
            equity_curve=eq_df,
            decile_returns=decile_table,
            summary_metrics=perf_metrics,
            monotonicity_rank_corr=float(monotonicity_corr),
            is_dollar_neutral=is_dollar_neutral,
        )
