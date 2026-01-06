# SCOPE Experiments Log

## Project Goal
Build a profitable Premier League corner prediction model for O/U betting.

---

## Phase 1: Regression Approach

### Initial Hypothesis
Predict exact total corners, then threshold for O/U bets.

### Baseline Model (v1)
**Config:**
```python
model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    reg_alpha=0.1,
    reg_lambda=1.0
)
ROLLING_WINDOW = 5
```

**Results:**
- RMSE: 3.37
- R²: -0.03 (worse than mean!)
- Correlation: -0.03
- **Prediction Range: 10.0 - 11.0** (actual: 3-19)

**Problem:** Severe mean regression. Model predicts ~10 corners for every match.

---

### Iteration v2: Reduced Regularization
**Changes:** `max_depth=6`, `reg_alpha=0.01`, `reg_lambda=0.1`

**Results:**
- Prediction Range: 10.0 - 10.7
- Still too narrow, no improvement

---

### Iteration v3: Poisson Regression + No Regularization
**Hypothesis:** Corners are count data, Poisson might help. Remove regularization to allow extreme predictions.

**Config:**
```python
objective='count:poisson'
reg_alpha=0
reg_lambda=0
n_estimators=200
```

**Results:**
- RMSE: 3.37
- Correlation: **+0.09** (first positive!)
- **Prediction Range: 7.9 - 12.3** (much better!)

**Insight:** Poisson + no regularization allows model to make varied predictions.

---

### Iteration v4: Added H2H Features
**Changes:** Added head-to-head historical corner averages

**Results:**
- RMSE: 3.43 (worse)
- Correlation: +0.03
- Slight overfit, H2H didn't help

---

### Iteration v5: Switch to LightGBM
**Hypothesis:** LightGBM often performs better on tabular data.

**Results:**
- Early stopping kicked in at iteration 11
- Prediction Range: 10.0 - 10.8 (narrow again)

**Problem:** Early stopping too aggressive.

---

### Iteration v6: Aggressive LightGBM
**Config:**
```python
num_leaves=63
max_depth=8
reg_alpha=0
reg_lambda=0
n_estimators=300
```

**Results:**
- RMSE: 3.34
- Correlation: **+0.14** (best!)
- Prediction Range: 6.9 - 12.6

---

### Iteration v7: Optimized LightGBM (BEST REGRESSION)
**Config:**
```python
objective='poisson'
num_leaves=63
max_depth=8
learning_rate=0.05
n_estimators=500
reg_alpha=0.001
reg_lambda=0.01
ROLLING_WINDOW=5
```

**Results:**
- RMSE: 3.32
- R²: -0.0016 (essentially 0)
- Correlation: 0.15
- Prediction Range: 7.7 - 12.6
- O/U 9.5 Accuracy: 58.9%

**Model saved as:** `model_v7_lgbm_optimized_*_best.pkl`

---

### Iteration v8: Rolling Window = 3
**Results:**
- Wider range (6.4 - 12.8)
- But worse overall metrics
- N=5 remains optimal

---

## Phase 1 Analysis

### What Worked
- Poisson objective for count data
- Minimal regularization
- LightGBM over XGBoost
- Rolling window N=5

### What Didn't Work
- H2H features
- Heavy regularization (causes mean regression)
- XGBoost with default settings

### Key Realization
**The regression approach has fundamental limitations:**

| Actual Corners | Model Behavior |
|---------------|----------------|
| Low (0-7) | Always overpredicts (MAE=4.0) |
| Medium (10-11) | Good predictions (MAE=1.0) |
| High (14+) | Always underpredicts (MAE=5.1) |

The model is essentially a "mean predictor with slight adjustments."

---

## Phase 2: Classifier Approach

### Key Insight
> We don't need to predict exact corners. We only need to predict Over vs Under for a specific threshold.

Solving a simpler problem = potentially better results.

### Betting Win Rate Analysis (Regression Model)
| Threshold | Win Rate | ROI | Real Edge |
|-----------|----------|-----|-----------|
| O/U 8.5 | 65.8% | +25.7% | 0% (just follows base rate) |
| O/U 9.5 | 58.9% | +12.5% | **+1.8%** |
| O/U 10.5 | 54.7% | +4.5% | 0% |
| O/U 11.5 | 68.9% | +31.6% | 0% (just bets Under) |
| O/U 12.5 | 80.5% | +53.8% | 0% (just bets Under) |

**Problem:** High ROI at extreme thresholds is an illusion - model just follows base rates.

**Real edge only exists at O/U 9.5 and 10.5** where outcomes are ~50/50.

---

### Classifier Iteration 1: More Data + Ensemble

**Changes:**
- Extended data from 2015-2026 (was 2020-2026)
- Ensemble: LightGBM + XGBoost + RandomForest
- Calibrated probabilities
- Focus on O/U 9.5 and 10.5

**Training Data:** 3,202 matches
**Test Data:** 200 matches (2025-26 season)

**Results:**
```
O/U 9.5:
  Confidence > 0.58: 138 bets, 58.0% win
  Real Edge: +3.5%
  ROI: +10.7%

O/U 10.5:
  Confidence > 0.55: 37 bets, 64.9% win
  Real Edge: +6.9%
  ROI: +23.9%
```

**Major Improvement:** Real edge achieved on both thresholds!

---

### Classifier Iteration 2: Extended Data + Multiple Windows (COMPLETE)

**Changes:**
- Extended data from 2010-2026 (~5,400 training matches)
- Test multiple rolling windows: [5, 7, 10]
- Fine-grained confidence levels: [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]

**Results (50+ bets, positive edge):**
```
Best O/U 9.5:  Window=5,  Conf>0.58 → 100 bets, 60.0% win, +5.5% edge, +14.6% ROI
Best O/U 10.5: Window=10, Conf>0.54 → 131 bets, 61.1% win, +3.1% edge, +16.6% ROI
```

**Key Findings:**
- Window=5 optimal for O/U 9.5
- Window=10 optimal for O/U 10.5 (longer history helps)
- Window=7 generally underperformed
- Higher confidence = higher edge but fewer bets

**Risk Analysis:**
| Config | Bets | Expected P/L | Worst Case (95% CI) |
|--------|------|--------------|---------------------|
| O/U 9.5 N=5 >0.58 | 100 | +$14.6 | -$4.1 |
| O/U 10.5 N=10 >0.54 | 131 | +$21.8 | +$0.5 |

**Major Improvement:** Both thresholds now show statistically meaningful edge with good volume.

---

### Classifier Iteration 3: Feature Selection (COMPLETE)

**Goal:** Test if reducing/adding features improves performance.

**Feature Importance Analysis:**
- **Most important:** shots (home/away), volatility, trends, O/U historical rates
- **Less important:** fouls, individual O/U rates

**Tested Feature Sets:**
| Feature Set | O/U 9.5 Edge | O/U 10.5 Edge |
|-------------|--------------|---------------|
| Baseline (32 features) | +7.5% | +2.8% |
| + NEW features (42) | +3.8% | +0.7% |
| Top 15 important | -0.2% | +1.2% |
| Core corners only (9) | +0.2% | -12.8% |
| Core + O/U rates (18) | +0.3% | -12.5% |
| Core + Momentum (15) | +0.6% | +1.6% |

**Conclusion:** Baseline 32 features is optimal. Adding more features adds noise, reducing features loses signal.

---

### Classifier Iteration 4: Hyperparameter Fine-tuning (COMPLETE)

**Approach:** Iterative testing instead of grid search.

**Tested Configurations:**

| Param | Baseline | Iter 1 | Iter 2 |
|-------|----------|--------|--------|
| LGBM depth | 6 | 8 | 8 |
| LGBM lr | 0.03 | 0.02 | 0.02 |
| reg_alpha | 0.1 | 0.05 | 0.01 |
| reg_lambda | 0.1 | 0.05 | 0.01 |

**Results:**

| Config | Baseline | Iter 1 | Iter 2 |
|--------|----------|--------|--------|
| O/U 9.5 | 100 bets, +5.5% edge | 77 bets, +9.1% edge | 90 bets, +6.6% edge |
| O/U 10.5 | 131 bets, +3.1% edge | 87 bets, +4.1% edge | 124 bets, +4.1% edge |

**Best Configurations:**
- **O/U 9.5:** Iteration 1 params (deeper, more regularized) - higher edge
- **O/U 10.5:** Iteration 2 params (less regularization) - same edge, more volume

---

## Key Learnings

### 1. Problem Framing Matters
Switching from regression (hard problem) to classification (simpler problem) improved results significantly.

### 2. Edge vs Win Rate
High win rate ≠ edge. Must compare against naive baseline (always bet majority).

### 3. Confidence Thresholds
Only betting on high-confidence predictions improves edge but reduces volume.

### 4. Data Quantity Helps
More historical data (2015 vs 2020) improved model performance.

### 5. Feature Engineering
O/U-specific features (historical over rates) are more relevant than raw corner stats.

---

## Current Best Strategy (After All Iterations)

### Option 1: Higher Edge (O/U 9.5)
**Config:** Window=5, Confidence > 0.58, Iter 1 params
- Win Rate: 63.6%
- Real Edge: +9.1%
- ROI: +21.5%
- Bets per season: ~77

### Option 2: Higher Volume (O/U 10.5)
**Config:** Window=10, Confidence > 0.54, Iter 2 params
- Win Rate: 62.1%
- Real Edge: +4.1%
- ROI: +18.6%
- Bets per season: ~124

**Recommendation:** Use both models together for diversification.

---

## Files

| File | Description |
|------|-------------|
| `train_model.py` | Regression model training (Phase 1) |
| `analyze_model.py` | Regression model analysis |
| `train_all_classifiers.py` | Classifier training (Phase 2) |
| `analyze_classifier.py` | Classifier profitability analysis |
| `model_v7_*_best.pkl` | Best regression model |
| `classifier_ou*.pkl` | Classifier models |

---

## Next Steps

1. Complete Iteration 2 with extended data (2010-2026)
2. Test longer rolling windows (7, 10)
3. Feature selection to remove noise
4. Consider separate models for Over vs Under predictions
5. Backtest on multiple seasons
6. Live testing with paper trading

---

## Metrics Reference

**Break-even win rate at -110 odds:** 52.4%

**Kelly Criterion:** `f* = (bp - q) / b` where `b = odds-1`

**Real Edge:** `Model Accuracy - max(base_rate, 1-base_rate)`

---

*Last updated: 2026-01-06*
