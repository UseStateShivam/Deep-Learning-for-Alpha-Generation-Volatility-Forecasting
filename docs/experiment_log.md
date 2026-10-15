# Experiment Log

This log records executed research experiments with their exact configurations, data environments, and actual realized out-of-sample metrics. No metrics are fabricated.

---

## Experiment: `exp_001_deep_alpha_vol`

- **Date Executed**: 2026-10-11
- **Dataset**: Deterministic Synthetic 5-Minute Intraday Bars (`data/synthetic/synthetic_bars.parquet`)
- **Universe**: 10 assets (`ASSET_00` to `ASSET_09`), 60 trading sessions (46,800 bars)
- **Forecast Horizon**: 3 bars (15 minutes)
- **Validation Protocol**: 4-fold Purged Walk-Forward Cross Validation + 20% Untouched Chronological Holdout Test Set (Sessions 49-60)
- **Hardware & Device**: CPU (Intel x86_64, Windows)
- **Software Stack**: PyTorch 2.14.1, LightGBM 4.7.0, pandas 3.0.6, scikit-learn 1.9.1

### Model Configurations

1. **LightGBM Baseline**:
   - Tabular features (27 features)
   - Max depth: 6, Num leaves: 31, Learning rate: 0.03, Estimators: 150
   - Early stopping: 15 rounds on chronological validation folds

2. **Temporal Convolutional Network (TCN)**:
   - Sequence length: 24 bars (2 hours)
   - Channels: [64, 64, 64], Kernel size: 3, Dilations: [1, 2, 4]
   - Receptive field: 15 bars
   - Output heads: Student-t distribution ($\mu, \sigma, \nu$) + Softplus volatility ($\hat{\sigma}_{vol}$)
   - Loss: Multi-task ($\mathcal{L}_{total} = 1.0 \times \text{NLL}_{Student-t} + 0.5 \times \text{QLIKE}$)

3. **Temporal Transformer Encoder**:
   - Sequence length: 24 bars
   - Model dimension ($d_{model}$): 64, Attention heads: 4, Encoder layers: 3, Feed-forward dim: 128
   - Causal attention mask: True
   - Output heads: Student-t distribution + Softplus volatility

### Out-of-Sample Test Set Results

| Model | Return RMSE | Return MAE | Directional Accuracy | Mean IC | Rank IC | IC IR | Volatility RMSE | Volatility QLIKE | Gross Sharpe | Net Sharpe (5 bps cost) | Max Drawdown (Net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.00548** | **0.00424** | **51.57%** | **+0.0127** | **+0.0164** | **+0.036** | **0.00239** | **0.1234** | **+1.19** | -8.64 | 28.79% |
| **TCN** | 0.00558 | 0.00432 | 48.76% | +0.0110 | +0.0081 | +0.032 | 0.00270 | 0.1409 | -2.94 | -13.20 | 29.72% |
| **Transformer** | 0.00572 | 0.00446 | 49.19% | -0.0234 | -0.0196 | -0.068 | 0.00363 | 0.2019 | - | - | - |

### Key Research Findings

1. **Alpha Signal Monotonicity**: LightGBM achieved a positive Rank IC (+0.0164) and produced a positive gross annualized Sharpe ratio of +1.19 with a decile monotonicity score of 0.467.
2. **Transaction Cost Sensitivity**: With high-frequency 5-minute rebalancing (every 3 bars) and a realistic 5 bps transaction cost drag, net returns were negative (-8.64 Sharpe), demonstrating the critical importance of transaction cost modeling in intraday systematic strategies.
3. **Volatility Forecasting Calibration**: All models successfully converged on realized volatility forecasting, with LightGBM achieving the lowest QLIKE loss (0.1234), closely followed by TCN (0.1409).
4. **Deep Learning Sequence Representation**: On this synthetic intraday sample size (60 sessions), the tabular LightGBM baseline slightly outperformed the deep learning sequence models, illustrating that deep learning models require larger multi-year regimes or pre-training to outperform well-tuned gradient boosted trees.
