# Quantitative Research Methodology

## Research Design

The objective of this research repository is twofold:
1. **Task A: Short-Horizon Alpha Generation**: Forecast the forward return distribution from 5-minute OHLCV bars over a 3-bar (15-minute) forward horizon.
2. **Task B: Realized Volatility Forecasting**: Forecast forward realized volatility over the same 15-minute interval for risk scaling and position sizing.

## Mathematical Formulations

### 1. Log Returns and Targets
For price series $P_t$ (bar close at timestamp $t$):
- Backward 1-bar log return:
  $$r_{t, 1} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
- Forward return target over horizon $h$:
  $$r_{t, h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$

### 2. Forward Realized Volatility Target
Over forward horizon $h$ bars:
$$RV(t, h) = \sqrt{\sum_{i=1}^h r_{t+i, 1}^2}$$
$$\ln RV(t, h) = \ln(RV(t, h) + \epsilon)$$

### 3. Probabilistic Return Modeling: Conditional Student-t Distribution
To capture financial fat tails, the neural networks parameterize a conditional Student-t distribution with location $\mu_t$, scale $\sigma_t > 0$, and degrees of freedom $\nu_t > 2$:
$$p(y; \mu, \sigma, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\pi \nu}\sigma} \left(1 + \frac{(y - \mu)^2}{\nu \sigma^2}\right)^{-\frac{\nu+1}{2}}$$

Trained via Negative Log-Likelihood (NLL):
$$\mathcal{L}_{ret} = -\ln p(y; \mu, \sigma, \nu) = -\ln\Gamma\left(\frac{\nu+1}{2}\right) + \ln\Gamma\left(\frac{\nu}{2}\right) + \frac{1}{2}\ln(\pi \nu) + \ln\sigma + \frac{\nu+1}{2}\ln\left(1 + \frac{(y - \mu)^2}{\nu\sigma^2}\right)$$

### 4. Volatility Forecast Loss: Quasi-Likelihood (QLIKE)
For realized volatility $y$ and model forecast $\hat{y} > 0$:
$$QLIKE(y, \hat{y}) = \frac{y}{\hat{y}} - \ln\left(\frac{y}{\hat{y}}\right) - 1$$
Properties:
- Non-negative, zero if and only if $y = \hat{y}$.
- Scale-free and asymmetric: penalizes underforecasting of volatility more heavily than overforecasting.

### 5. Multi-Task Objective
$$\mathcal{L}_{total} = w_{ret}\mathcal{L}_{ret} + w_{vol}\mathcal{L}_{vol}$$
Default weights: $w_{ret} = 1.0, w_{vol} = 0.5$.

## Model Architectures

### 1. Temporal Convolutional Network (TCN)
- Dilated causal 1D convolutions with strict left padding $P = (K - 1) \times d$.
- Residual connections with $1 \times 1$ channel projections.
- Receptive field:
  $$RF = 1 + \sum_{l=0}^{L-1} 2(K - 1)2^l$$
- Multi-task heads:
  - $\mu = W_\mu z_t + b_\mu$
  - $\sigma = \text{Softplus}(W_\sigma z_t + b_\sigma) + \epsilon$
  - $\nu = 2.1 + \text{Softplus}(W_\nu z_t + b_\nu)$
  - $\hat{\sigma}_{vol} = \text{Softplus}(W_v z_t + b_v) + \epsilon$

### 2. Temporal Transformer Encoder
- Linear feature projection to $d_{model} = 64$.
- Sinusoidal positional encoding.
- Causal upper-triangular attention mask preventing attendance to future sequence positions.
- Multi-head self-attention with pre-LayerNorm residual blocks.
- Multi-task linear heads on final sequence step representation.

### 3. LightGBM Tabular Baseline
- Gradient-boosted decision trees trained on point-in-time engineered features.
- Early stopping based on validation fold loss.
- Separate regressors for return and volatility tasks.

## Validation & Leakage Controls

1. **Chronological Holdout**: Final 20% of data is strictly isolated as an untouched out-of-sample test set.
2. **Purged Walk-Forward CV**: Expanding development window divided into 4 chronological folds.
3. **Purging**: Eliminates training observations whose forward return interval $[t, t+h]$ overlaps the validation window $[V_{start}, V_{end}]$.
4. **Embargo**: Eliminates training samples in a 12-bar (60-minute) window post validation to prevent autoregressive contamination.
5. **Fold-Specific Scaling**: Feature scalers fitted strictly on training fold data.

## Portfolio Construction and Simulation

- Cross-sectional ranking of assets by predicted return $\hat{\mu}_t$ at each rebalance timestamp.
- Partition into 10 quantiles (Deciles 1 to 10).
- Long top decile (D10) with equal weights summing to $+0.5$.
- Short bottom decile (D1) with equal weights summing to $-0.5$.
- Net dollar exposure = $0.0$; Gross exposure = $1.0$.
- Linear transaction cost model: half-spread + commission + slippage (default 5 bps per 100% turnover).
- Monotonicity evaluated via Spearman rank correlation between decile numbers (1..10) and realized forward return.
