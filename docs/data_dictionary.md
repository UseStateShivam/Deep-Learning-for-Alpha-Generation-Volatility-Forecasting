# Data Dictionary & Schema Specification

## 1. Canonical Raw Bar Schema

| Column Name | Data Type | Unit / Format | Description | Availability Timing |
|---|---|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | ISO-8601 UTC | Bar close timestamp (end of 5m interval) | At bar close |
| `asset_id` | `string` | Ticker symbol | Unique asset identifier | At bar close |
| `open` | `float64` | Base currency | First traded price in 5m interval | At bar close |
| `high` | `float64` | Base currency | Highest traded price in 5m interval | At bar close |
| `low` | `float64` | Base currency | Lowest traded price in 5m interval | At bar close |
| `close` | `float64` | Base currency | Final traded price in 5m interval | At bar close |
| `volume` | `float64` | Shares / Contracts | Total traded volume in 5m interval | At bar close |

## 2. Engineered Feature Registry

| Feature Name | Category | Formula / Definition | Lookback | Availability Time |
|---|---|---|---|---|
| `log_return_1` | Price | $\ln(P_t / P_{t-1})$ | 1 bar | $t$ |
| `log_return_3` | Price | $\ln(P_t / P_{t-3})$ | 3 bars | $t$ |
| `log_return_6` | Price | $\ln(P_t / P_{t-6})$ | 6 bars | $t$ |
| `log_return_12` | Price | $\ln(P_t / P_{t-12})$ | 12 bars | $t$ |
| `high_low_range` | Price | $\ln(H_t / L_t)$ | 1 bar | $t$ |
| `open_close_return` | Price | $\ln(C_t / O_t)$ | 1 bar | $t$ |
| `price_dev_12` | Price | $(P_t - \mu_{12}) / \sigma_{12}$ | 12 bars | $t$ |
| `realized_vol_6` | Volatility | $\sqrt{\sum_{i=0}^5 r_{t-i}^2}$ | 6 bars | $t$ |
| `realized_vol_12` | Volatility | $\sqrt{\sum_{i=0}^{11} r_{t-i}^2}$ | 12 bars | $t$ |
| `realized_vol_36` | Volatility | $\sqrt{\sum_{i=0}^{35} r_{t-i}^2}$ | 36 bars | $t$ |
| `parkinson_vol_12` | Volatility | $\sqrt{\frac{1}{4\ln 2 \cdot 12}\sum (\ln(H/L))^2}$ | 12 bars | $t$ |
| `vol_ratio_6_36` | Volatility | $\text{realized\_vol\_6} / \text{realized\_vol\_36}$ | 36 bars | $t$ |
| `log_volume` | Volume | $\ln(V_t + 1)$ | 1 bar | $t$ |
| `volume_zscore_12` | Volume | $(V_t - \mu_{V,12}) / \sigma_{V,12}$ | 12 bars | $t$ |
| `relative_volume_12`| Volume | $V_t / \mu_{V,12}$ | 12 bars | $t$ |
| `pv_interaction` | Volume | $\ln(V_t + 1) \cdot \|r_{t,1}\|$ | 1 bar | $t$ |
| `momentum_6` | Momentum | Backward 6-bar return | 6 bars | $t$ |
| `momentum_24` | Momentum | Backward 24-bar return | 24 bars | $t$ |
| `trend_consistency_12`| Momentum | Rolling % of matching signs | 12 bars | $t$ |
| `sin_time_of_day` | Temporal | $\sin(2\pi \cdot \text{bar} / 78)$ | 0 bars | $t$ |
| `cos_time_of_day` | Temporal | $\cos(2\pi \cdot \text{bar} / 78)$ | 0 bars | $t$ |
| `session_progress` | Temporal | $\text{bar} / 78$ | 0 bars | $t$ |
| `cs_rank_log_return_1`| Cross-Sectional | Centered percentile rank of $r_{t,1}$ | 1 bar | $t$ |
| `cs_rank_momentum_6` | Cross-Sectional | Centered percentile rank of mom_6 | 6 bars | $t$ |
| `cs_rank_realized_vol_12`| Cross-Sectional| Centered percentile rank of vol_12 | 12 bars | $t$ |

## 3. Label & Interval Alignment Schema

| Target Column | Type | Formula | Information Interval |
|---|---|---|---|
| `target_return` | `float64` | $\ln(P_{t+h} / P_t)$ | $[t, t+h]$ |
| `target_realized_vol` | `float64` | $\sqrt{\sum_{i=1}^h r_{t+i}^2}$ | $[t, t+h]$ |
| `target_log_realized_vol` | `float64` | $\ln(RV(t, h) + \epsilon)$ | $[t, t+h]$ |
| `label_info_start` | `datetime64[ns, UTC]` | $t$ | Information interval start |
| `label_info_end` | `datetime64[ns, UTC]` | $t+h$ | Information interval end (purging anchor) |

## 4. Prediction Schema

| Prediction Column | Description |
|---|---|
| `pred_return_mean` | Estimated expected return $\hat{\mu}_t$ (primary alpha score) |
| `pred_return_scale` | Dispersion / scale parameter $\hat{\sigma}_t$ of conditional return distribution |
| `pred_return_df` | Estimated Student-t degrees of freedom $\hat{\nu}_t$ |
| `pred_volatility` | Forward realized volatility estimate $\hat{\sigma}_{vol, t}$ |

## 5. Backtest Equity Schema

| Column | Description |
|---|---|
| `timestamp` | Rebalance timestamp |
| `gross_return` | Portfolio gross return: $\sum w_{i, t} r_{i, t+1}$ |
| `net_return` | Portfolio net return after turnover costs: $R_{gross} - C$ |
| `long_leg_return` | Return contribution from long decile |
| `short_leg_return` | Return contribution from short decile |
| `turnover` | Rebalance turnover: $\frac{1}{2}\sum \|w_{i, t} - w_{i, t-1}\|$ |
| `cost` | Incurred friction: $\sum \|w_{i, t} - w_{i, t-1}\| \times c$ |
| `gross_exposure` | Total leverage: $\sum \|w_i\| = 1.0$ |
| `net_exposure` | Net dollar exposure: $\sum w_i = 0.0$ |
| `cum_gross_wealth` | Cumulative gross equity path |
| `cum_net_wealth` | Cumulative net equity path |
