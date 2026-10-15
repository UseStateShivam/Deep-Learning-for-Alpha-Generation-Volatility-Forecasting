# Validation Protocol & Leakage Controls

## Executive Summary

Standard k-fold cross-validation and random train/test splits are invalid for financial time-series modeling. Randomly sampling across time causes:
1. **Lookahead Bias**: Training on future observations to predict earlier periods.
2. **Serial Correlation Contamination**: Adjacent intraday bars share overlapping information.
3. **Overlapping Label Leakage**: When the forecast horizon is $h > 1$, forward labels overlap in time.

The `alpha-volatility` validation protocol implements strict point-in-time walk-forward evaluation with purged intervals and post-validation embargoes.

## Validation Design

### 1. Chronological Holdout Set
- The final 20% of the dataset is permanently quarantined as an untouched holdout test set.
- It is never accessed for feature selection, hyperparameter tuning, model architecture choices, or calibration.
- Any development observation whose forward label information interval extends into the holdout start timestamp is purged.

### 2. Purged Walk-Forward Cross-Validation
Within the initial 80% development period, an expanding window cross-validation protocol executes across 4 chronological folds:

| Fold | Training Start | Training End | Validation Start | Validation End | Purged Samples | Embargoed Samples |
|---|---|---|---|---|---|---|
| Fold 1 | Session 01 | Session 24 | Session 25 | Session 30 | 30 | 120 |
| Fold 2 | Session 01 | Session 30 | Session 31 | Session 36 | 30 | 120 |
| Fold 3 | Session 01 | Session 36 | Session 37 | Session 42 | 30 | 120 |
| Fold 4 | Session 01 | Session 42 | Session 43 | Session 48 | 30 | 120 |

### 3. Purging Implementation
Each sample $i$ at timestamp $t$ has an information interval $[T_{start, i}, T_{end, i}] = [t, t+h]$.
When evaluating a validation period $[V_{start}, V_{end}]$, any training observation $i$ is purged if:
$$\max(T_{start, i}, V_{start}) \le \min(T_{end, i}, V_{end})$$
This guarantees that training labels do not contain returns realized during the validation interval.

### 4. Embargo Implementation
When expanding the training set after a validation period, autoregressive memory can leak information from the validation interval into subsequent bars. An embargo buffer of $E = 12$ bars (60 minutes) is placed immediately following validation:
$$\text{Excluded Training Timestamps: } t \in (V_{end}, V_{end} + \Delta_{embargo}]$$

### 5. Preprocessing Isolation
- Feature scalers (e.g., `StandardScaler`, `RobustScaler`) are fitted strictly on the purged training set of each fold.
- Scaler parameters ($\mu_{train}, \sigma_{train}$) are applied out-of-sample to validation and test data without refitting.

## Automated Leakage Verification Tests

The test suite contains dedicated leakage tests (`tests/leakage/test_leakage.py`):
1. `test_future_data_does_not_alter_past_features`: Mutates future market bars and proves past features remain bit-exact identical.
2. `test_purging_eliminates_overlapping_label_intervals`: Verifies that samples with forward-looking labels spanning validation boundaries are removed.
3. `test_embargo_excludes_post_validation_observations`: Proves post-validation buffer exclusion.
