"""
Transaction Cost Model
Computes realistic friction costs from portfolio rebalancing turnover and short borrow fees.
"""

import numpy as np


class TransactionCostModel:
    """
    Linear friction model accounting for half-spread, broker commission, and execution slippage.
    """

    def __init__(
        self,
        spread_bps: float = 2.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        total_one_way_cost_bps: float = 5.0,
    ):
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        # One-way cost in decimals (e.g. 5 bps = 0.0005)
        self.cost_rate = total_one_way_cost_bps * 1e-4

    def calculate_cost(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> float:
        """
        Computes trading friction incurred moving from current_weights to target_weights:
          cost = sum(|w_target - w_current|) * cost_rate
        """
        turnover_traded = float(np.sum(np.abs(target_weights - current_weights)))
        cost = turnover_traded * self.cost_rate
        return cost
