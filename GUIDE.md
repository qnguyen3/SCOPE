# SCOPE - Statistical Corner Outcome Prediction Engine
# Premier League Corner Prediction Guide

A complete guide to building a corner prediction model using Football-Data.co.uk data and XGBoost.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Data Source and Available Columns](#2-data-source-and-available-columns)
3. [Target Variable](#3-target-variable)
4. [Feature Engineering](#4-feature-engineering)
5. [Data Preparation Pipeline](#5-data-preparation-pipeline)
6. [Model: XGBoost](#6-model-xgboost)
7. [Training and Validation Protocol](#7-training-and-validation-protocol)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Making Predictions](#9-making-predictions)
10. [Implementation Checklist](#10-implementation-checklist)
11. [Common Pitfalls](#11-common-pitfalls)
12. [Appendix: Formula Reference](#appendix-formula-reference)

---

## 1. Objective

Build a model that predicts the total number of corners in a Premier League match at full time.

**Primary prediction target:**
```
Total Corners = HC + AC
```
Where:
- `HC` = Home team corners
- `AC` = Away team corners

**Use cases:**
- Predict whether a match will finish Over or Under X total corners (e.g., Over/Under 9.5)
- Estimate the expected total corners for any given match
- Compare predictions against betting lines to identify value (separate from this guide)

**Constraints:**
- Use only football-related statistics (no betting odds)
- Use only pre-match information (no in-game data for the match being predicted)
- Use only data available from Football-Data.co.uk

---

## 2. Data Source and Available Columns

### 2.1 Data Source

Download Premier League match data from:
- Website: https://www.football-data.co.uk/englandm.php
- Download CSV files for each season you want to use
- Recommended: Use at least 3-5 seasons for sufficient training data

### 2.2 Columns Used in This Guide

The following columns from Football-Data.co.uk are used:

| Column | Description | Used For |
|--------|-------------|----------|
| `Date` | Match date | Sorting, time-based splits |
| `HomeTeam` | Home team name | Team identification |
| `AwayTeam` | Away team name | Team identification |
| `FTHG` | Full-time home goals | Context features (optional) |
| `FTAG` | Full-time away goals | Context features (optional) |
| `HS` | Home team shots | Shot-based features |
| `AS` | Away team shots | Shot-based features |
| `HST` | Home team shots on target | Shot accuracy features, blocked shots proxy |
| `AST` | Away team shots on target | Shot accuracy features, blocked shots proxy |
| `HC` | Home team corners | **Target variable** + historical features |
| `AC` | Away team corners | **Target variable** + historical features |

### 2.3 Columns NOT Used

The following columns are available but explicitly excluded:
- All odds columns (B365H, PSH, etc.) — excluded by design choice
- Referee column — excluded due to data availability concerns
- `HF`, `AF` (Fouls) — excluded after correlation analysis showed no predictive value
- `HY`, `AY`, `HR`, `AR` (Cards) — excluded after correlation analysis showed no predictive value

---

## 3. Target Variable

### 3.1 Primary Target: Total Corners (Regression)

```
target = HC + AC
```

This is a count variable representing the total corners in a match.

**Typical range:** 5 to 18 corners per match
**Average:** Approximately 10-11 corners per Premier League match

### 3.2 Secondary Target: Over/Under Classification (Optional)

For a specific threshold X (e.g., 9.5):

```
target_over_X = 1 if (HC + AC) > X else 0
```

You can create multiple binary targets for different thresholds:
- `over_8.5` = 1 if total corners > 8.5
- `over_9.5` = 1 if total corners > 9.5
- `over_10.5` = 1 if total corners > 10.5
- etc.

### 3.3 Recommendation

Train a **regression model** to predict total corners. From the predicted value, you can derive probabilities for any Over/Under threshold.

---

## 4. Feature Engineering

All features must be computed using **only historical data** — information available before the match takes place. This section defines every feature in detail.

### 4.1 Core Principle: Venue-Aware Rolling Statistics

Teams perform differently at home versus away. All rolling statistics must be computed separately for home and away games.

**Rolling window:** Last N games (recommended: N = 5 for recent form, N = 10 for stability)

**General formula for rolling features:**
```
For a team T playing at home on date D:
  rolling_stat = mean of [stat from T's last N home games before date D]

For a team T playing away on date D:
  rolling_stat = mean of [stat from T's last N away games before date D]
```

---

### 4.2 Feature Category 1: Rolling Corner Statistics

These are the most direct predictors of future corners.

#### Features for Home Team (computed from their previous HOME games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_corners_for_N` | mean(HC) over last N home games | Corners won by home team at home |
| `home_corners_against_N` | mean(AC) over last N home games | Corners conceded by home team at home |
| `home_corners_total_N` | mean(HC + AC) over last N home games | Total corners in home team's home games |

#### Features for Away Team (computed from their previous AWAY games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `away_corners_for_N` | mean(AC) over last N away games | Corners won by away team when away |
| `away_corners_against_N` | mean(HC) over last N away games | Corners conceded by away team when away |
| `away_corners_total_N` | mean(HC + AC) over last N away games | Total corners in away team's away games |

#### Match-Level Composite Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `expected_corners_for` | home_corners_for_N + away_corners_for_N | Combined attacking corner threat |
| `expected_corners_against` | home_corners_against_N + away_corners_against_N | Combined defensive corner vulnerability |
| `expected_corners_total` | (home_corners_total_N + away_corners_total_N) / 2 | Average total corners from both teams' recent games |
| `corner_differential` | home_corners_for_N - away_corners_for_N | Relative corner-winning strength |

---

### 4.3 Feature Category 2: Rolling Shot Statistics

Shots indicate attacking intent and are a leading indicator of corners (shots that miss or are blocked often result in corners).

#### Features for Home Team (from previous HOME games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_shots_for_N` | mean(HS) over last N home games | Shots by home team at home |
| `home_shots_against_N` | mean(AS) over last N home games | Shots conceded at home |
| `home_sot_for_N` | mean(HST) over last N home games | Shots on target by home team at home |
| `home_sot_against_N` | mean(AST) over last N home games | Shots on target conceded at home |

#### Features for Away Team (from previous AWAY games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `away_shots_for_N` | mean(AS) over last N away games | Shots by away team when away |
| `away_shots_against_N` | mean(HS) over last N away games | Shots conceded when away |
| `away_sot_for_N` | mean(AST) over last N away games | Shots on target by away team when away |
| `away_sot_against_N` | mean(HST) over last N away games | Shots on target conceded when away |

#### Match-Level Composite Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `combined_shots_for` | home_shots_for_N + away_shots_for_N | Total expected shots in match |
| `combined_sot_for` | home_sot_for_N + away_sot_for_N | Total expected shots on target |
| `shot_differential` | home_shots_for_N - away_shots_for_N | Shot dominance indicator |

---

### 4.4 Feature Category 3: Shot Accuracy and Efficiency

Ratios that capture playing style independent of volume.

#### Team-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_shot_accuracy_N` | home_sot_for_N / max(home_shots_for_N, 1) | Home team's shooting accuracy |
| `away_shot_accuracy_N` | away_sot_for_N / max(away_shots_for_N, 1) | Away team's shooting accuracy |
| `home_corners_per_shot_N` | home_corners_for_N / max(home_shots_for_N, 1) | Corners won per shot (home) |
| `away_corners_per_shot_N` | away_corners_for_N / max(away_shots_for_N, 1) | Corners won per shot (away) |

#### Match-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `avg_shot_accuracy` | (home_shot_accuracy_N + away_shot_accuracy_N) / 2 | Combined shooting accuracy |
| `combined_corners_per_shot` | home_corners_per_shot_N + away_corners_per_shot_N | Style indicator for corner generation |

**Why corners per shot matters:** Teams that play wide, cross frequently, or shoot from angles tend to generate more corners per shot. This captures tactical style.

---

### 4.5 Feature Category 4: Pressure Index (Share-Based)

Share-based metrics reduce scale effects and indicate dominance patterns.

#### Team-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_shot_share_N` | home_shots_for_N / (home_shots_for_N + home_shots_against_N) | Home team's share of shots in their home games |
| `away_shot_share_N` | away_shots_for_N / (away_shots_for_N + away_shots_against_N) | Away team's share of shots in their away games |
| `home_corner_share_N` | home_corners_for_N / (home_corners_for_N + home_corners_against_N) | Home team's share of corners at home |
| `away_corner_share_N` | away_corners_for_N / (away_corners_for_N + away_corners_against_N) | Away team's share of corners away |

#### Match-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `pressure_sum` | home_shot_share_N + away_shot_share_N | Combined attacking pressure (higher = more open game) |
| `pressure_gap` | home_shot_share_N - away_shot_share_N | Pressure imbalance |

**Interpretation:**
- `pressure_sum` near 1.0: Balanced game
- `pressure_sum` > 1.0: Both teams attack-minded → likely more corners
- `pressure_sum` < 1.0: Both teams defensive-minded → likely fewer corners

---

### 4.6 Feature Category 5: Blocked Shots Proxy

Shots that miss the target or are blocked often result in corners. The difference between total shots and shots on target provides a proxy for blocked/deflected shots.

**Correlation with TotalCorners: r = 0.298 (validated)**

#### Features for Home Team (from previous HOME games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_blocked_shots_N` | mean(HS - HST) over last N home games | Blocked/missed shots by home team at home |
| `home_blocked_against_N` | mean(AS - AST) over last N home games | Blocked/missed shots conceded at home |

#### Features for Away Team (from previous AWAY games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `away_blocked_shots_N` | mean(AS - AST) over last N away games | Blocked/missed shots by away team when away |
| `away_blocked_against_N` | mean(HS - HST) over last N away games | Blocked/missed shots conceded when away |

#### Match-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `combined_blocked_shots` | home_blocked_shots_N + away_blocked_shots_N | Expected blocked shots in match |
| `blocked_shot_ratio` | combined_blocked_shots / max(combined_shots_for, 1) | Proportion of shots likely to be blocked |

**Why blocked shots matter for corners:** When a shot is blocked or deflected, it often goes out for a corner. Teams with high blocked shot rates generate more corner opportunities.

---

### 4.7 Feature Category 6: Shot Imbalance

One-sided games where one team dominates possession/shots tend to generate more corners. Shot imbalance captures this dynamic.

**Correlation with TotalCorners: r = 0.157 (validated)**

#### Features for Home Team (from previous HOME games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_shot_dominance_N` | mean(HS - AS) over last N home games | Shot differential when playing at home |

#### Features for Away Team (from previous AWAY games):

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `away_shot_dominance_N` | mean(AS - HS) over last N away games | Shot differential when playing away |

#### Match-Level Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `expected_shot_imbalance` | abs(home_shot_dominance_N - away_shot_dominance_N) | Expected shot differential in match |
| `dominance_mismatch` | home_shot_dominance_N + away_shot_dominance_N | Combined dominance (if both positive = open game) |

**Why shot imbalance matters:** When one team heavily outshots another, the dominant team creates more attacking opportunities, leading to more corners from their attacks and more defensive clearances from the defending team.

---

### 4.8 Feature Category 7: Volatility Features

Volatility measures consistency and helps quantify prediction uncertainty.

#### Features:

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `home_corner_std_N` | std(HC) over last N home games | Volatility in home team's corners at home |
| `away_corner_std_N` | std(AC) over last N away games | Volatility in away team's corners when away |
| `home_corner_cv_N` | home_corner_std_N / max(home_corners_for_N, 1) | Coefficient of variation (home) |
| `away_corner_cv_N` | away_corner_std_N / max(away_corners_for_N, 1) | Coefficient of variation (away) |
| `combined_corner_volatility` | home_corner_std_N + away_corner_std_N | Match-level volatility |

**Why volatility matters:** A team averaging 5 corners with std=1 is more predictable than one averaging 5 with std=3. High volatility → wider prediction intervals.

---

### 4.9 Complete Feature List Summary

**Category 1: Rolling Corners (12 features)** — Core predictors (r = 0.53-0.63)
- home_corners_for_N, home_corners_against_N, home_corners_total_N
- away_corners_for_N, away_corners_against_N, away_corners_total_N
- expected_corners_for, expected_corners_against, expected_corners_total
- corner_differential
- home_corner_std_N, away_corner_std_N

**Category 2: Rolling Shots (12 features)** — Strong predictors (r = 0.31)
- home_shots_for_N, home_shots_against_N, home_sot_for_N, home_sot_against_N
- away_shots_for_N, away_shots_against_N, away_sot_for_N, away_sot_against_N
- combined_shots_for, combined_sot_for, shot_differential, avg_shot_accuracy

**Category 3: Efficiency Ratios (4 features)** — Best predictors (r = 0.55)
- home_shot_accuracy_N, away_shot_accuracy_N
- home_corners_per_shot_N, away_corners_per_shot_N ← **Highest correlation**

**Category 4: Pressure Index (4 features)**
- home_shot_share_N, away_shot_share_N
- pressure_sum, pressure_gap

**Category 5: Blocked Shots Proxy (6 features)** — Validated (r = 0.30)
- home_blocked_shots_N, home_blocked_against_N
- away_blocked_shots_N, away_blocked_against_N
- combined_blocked_shots, blocked_shot_ratio

**Category 6: Shot Imbalance (4 features)** — Validated (r = 0.16)
- home_shot_dominance_N, away_shot_dominance_N
- expected_shot_imbalance, dominance_mismatch

**Category 7: Volatility (3 features)**
- home_corner_cv_N, away_corner_cv_N, combined_corner_volatility

**Total: ~45 features** (exact count depends on whether you use N=5, N=10, or both)

### 4.10 Feature Correlation Summary (Validated)

Based on correlation analysis of 1,900 Premier League matches (2020-2025):

| Feature | Correlation | Priority |
|---------|-------------|----------|
| AvgCornersPerShot | 0.551 | **HIGH** |
| TotalShots | 0.312 | **HIGH** |
| TotalBlockedShots | 0.298 | **HIGH** |
| ShotImbalance | 0.157 | MEDIUM |
| TotalShotsOnTarget | 0.142 | MEDIUM |

---

## 5. Data Preparation Pipeline

### 5.1 Step 1: Load and Combine Data

```
1. Download CSV files for each season from Football-Data.co.uk
2. Load each CSV into a dataframe
3. Add a 'season' column to each (e.g., "2023-24")
4. Concatenate all seasons into one dataframe
5. Sort by Date (ascending)
```

### 5.2 Step 2: Parse Dates

```
1. Convert 'Date' column to datetime format
2. Handle different date formats across seasons (DD/MM/YY vs DD/MM/YYYY)
3. Verify chronological ordering
```

### 5.3 Step 3: Standardize Team Names

```
1. Check for team name variations across seasons
2. Create a mapping dictionary for inconsistent names
3. Apply mapping to HomeTeam and AwayTeam columns
```

Example issues:
- "Man United" vs "Manchester United"
- "Spurs" vs "Tottenham"

### 5.4 Step 4: Create Target Variable

```
df['total_corners'] = df['HC'] + df['AC']
```

### 5.5 Step 5: Compute Rolling Features

**Critical: Avoid data leakage**

For each match at time t, features must be computed using only matches before time t.

**Algorithm for computing rolling features:**

```
For each team T:
    For each match M where T plays:
        Get all previous matches where T played at the same venue (home/away)
        Take the last N of these matches
        Compute mean (and std where needed) of relevant statistics
        Assign to match M
```

**Important considerations:**
- First few matches of a season will have missing features (not enough history)
- Options: Use expanding window for early season, or exclude early matches from training
- Recommended: Require at least 3 previous venue-specific matches; otherwise, use overall average or mark as missing

### 5.6 Step 6: Handle Missing Values

**Sources of missing values:**
- Early season matches (insufficient rolling history)
- New promoted teams (no previous Premier League data)
- Missing data in source files

**Strategies:**
1. For early season: Use expanding mean instead of rolling mean
2. For promoted teams: Use league average as fallback
3. For missing source data: Drop rows or impute with league average

### 5.7 Step 7: Create Train/Test Split

**Do NOT use random splitting.** Use time-based splits only.

**Option A: Season-based split**
```
Training: Seasons 2019-20, 2020-21, 2021-22, 2022-23
Testing: Season 2023-24
```

**Option B: Date-based split**
```
Training: All matches before 2023-08-01
Testing: All matches from 2023-08-01 onward
```

---

## 6. Model: XGBoost

### 6.1 Why XGBoost

- Handles non-linear relationships automatically
- Robust to outliers and missing values
- Provides feature importance rankings
- Fast training and prediction
- Works well with count data when configured properly

### 6.2 Model Configuration for Corner Prediction

**Objective function:** Use `reg:squarederror` (standard regression) or `count:poisson` (for count data)

**Recommended starting parameters:**

```python
params = {
    'objective': 'reg:squarederror',  # or 'count:poisson'
    'eval_metric': 'rmse',
    'max_depth': 4,                    # shallow trees to prevent overfitting
    'learning_rate': 0.05,             # conservative learning rate
    'n_estimators': 500,               # use early stopping
    'min_child_weight': 10,            # requires substantial samples per leaf
    'subsample': 0.8,                  # row sampling
    'colsample_bytree': 0.8,           # feature sampling
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.0,                 # L2 regularization
    'random_state': 42
}
```

### 6.3 Training with Early Stopping

```
1. Split training data into train and validation sets (time-based)
2. Train with early stopping:
   - Monitor validation RMSE
   - Stop if no improvement for 50 rounds
3. Use the best iteration for final model
```

### 6.4 Hyperparameter Tuning

**Parameters to tune (in order of importance):**

1. `max_depth`: Try [3, 4, 5, 6]
2. `learning_rate`: Try [0.01, 0.03, 0.05, 0.1]
3. `min_child_weight`: Try [5, 10, 20]
4. `subsample` and `colsample_bytree`: Try [0.7, 0.8, 0.9]

**Tuning method:** Time-series cross-validation (see Section 7)

---

## 7. Training and Validation Protocol

### 7.1 Time-Series Cross-Validation

Standard k-fold cross-validation is invalid for time-series data because it leaks future information.

**Correct approach: Rolling origin evaluation**

```
Fold 1: Train on Season 1-2, Validate on Season 3
Fold 2: Train on Season 1-3, Validate on Season 4
Fold 3: Train on Season 1-4, Validate on Season 5
...
```

Or with monthly rolling:
```
Fold 1: Train up to Month 6, Validate on Month 7
Fold 2: Train up to Month 7, Validate on Month 8
...
```

### 7.2 Validation Metrics to Track

For each fold, compute:
- RMSE (primary metric)
- MAE
- Correlation between predicted and actual
- Calibration (does predicting 10 corners mean ~10 corners on average?)

### 7.3 Final Model Training

After tuning hyperparameters:
```
1. Train on all available historical data
2. Save the model for production use
3. Retrain periodically (e.g., weekly or monthly) as new data becomes available
```

---

## 8. Evaluation Metrics

### 8.1 Regression Metrics (for corner count prediction)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | sqrt(mean((predicted - actual)²)) | Lower is better; penalizes large errors |
| MAE | mean(abs(predicted - actual)) | Lower is better; average error magnitude |
| R² | 1 - (SS_res / SS_tot) | Higher is better; variance explained |

**Expected ranges for corner prediction:**
- RMSE: 2.5 - 3.5 corners (good models)
- MAE: 2.0 - 3.0 corners (good models)
- R²: 0.05 - 0.15 (corners are noisy; low R² is normal)

### 8.2 Classification Metrics (for Over/Under prediction)

If evaluating Over/Under X.5 predictions:

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | % of correct Over/Under calls | > 52-53% is good |
| Log Loss | Probabilistic accuracy | Lower is better |
| Brier Score | Calibration + accuracy combined | Lower is better |
| AUC-ROC | Ranking ability | > 0.55 is good |

### 8.3 Calibration Analysis

**What is calibration?**
When the model predicts 60% probability of Over 9.5, it should happen ~60% of the time.

**How to check:**
1. Bin predictions into groups (e.g., 0-10%, 10-20%, ..., 90-100%)
2. For each bin, compute actual Over rate
3. Plot predicted probability vs actual frequency
4. Perfect calibration = diagonal line

**Why it matters:**
Even if RMSE is low, poor calibration means predicted probabilities are unreliable for betting decisions.

---

## 9. Making Predictions

### 9.1 For a New Match

**Input required:**
- HomeTeam name
- AwayTeam name
- Match date

**Process:**
```
1. Look up both teams' recent matches (home games for home team, away games for away team)
2. Compute all rolling features using historical data only
3. Create feature vector
4. Pass to trained XGBoost model
5. Model outputs: predicted total corners
```

### 9.2 Converting to Over/Under Probabilities

**Method 1: Simple threshold**
```
If predicted_corners > 9.5: lean Over
If predicted_corners < 9.5: lean Under
```

This is crude and doesn't give probabilities.

**Method 2: Empirical distribution from residuals**
```
1. On validation set, compute residuals: actual - predicted
2. Fit a distribution to residuals (e.g., normal with mean 0)
3. For new prediction:
   P(Over X) = P(predicted + residual > X)
             = 1 - CDF(X - predicted)
```

**Method 3: Train separate classification models**
```
For each threshold X (8.5, 9.5, 10.5, 11.5):
    Train XGBoost classifier with target: 1 if total > X else 0
    Output: P(Over X)
```

### 9.3 Example Prediction Output

```
Match: Arsenal (H) vs Chelsea (A)
Date: 2024-03-15

Predicted Total Corners: 10.8

Over/Under Probabilities:
  Over 8.5:  78%
  Over 9.5:  62%
  Over 10.5: 48%
  Over 11.5: 31%
  Over 12.5: 18%

Confidence: Medium (based on feature completeness and recent form stability)
```

---

## 10. Implementation Checklist

### Phase 1: Data Collection
- [ ] Download Premier League CSV files from Football-Data.co.uk (at least 3 seasons)
- [ ] Verify all required columns are present (HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR)
- [ ] Document any missing columns or seasons

### Phase 2: Data Preparation
- [ ] Load and concatenate all season files
- [ ] Parse dates correctly
- [ ] Standardize team names
- [ ] Create target variable (total_corners = HC + AC)
- [ ] Verify data integrity (no duplicate matches, reasonable value ranges)

### Phase 3: Feature Engineering
- [ ] Implement rolling window function (venue-aware)
- [ ] Compute Category 1 features (rolling corners)
- [ ] Compute Category 2 features (rolling shots)
- [ ] Compute Category 3 features (efficiency ratios)
- [ ] Compute Category 4 features (pressure index)
- [ ] Compute Category 5 features (blocked shots proxy)
- [ ] Compute Category 6 features (shot imbalance)
- [ ] Compute Category 7 features (volatility)
- [ ] Handle missing values (early season, promoted teams)
- [ ] Verify no data leakage (features only use past data)

### Phase 4: Model Development
- [ ] Create time-based train/validation/test splits
- [ ] Train baseline XGBoost model
- [ ] Evaluate on validation set
- [ ] Tune hyperparameters using time-series CV
- [ ] Evaluate final model on test set
- [ ] Analyze feature importance
- [ ] Check calibration

### Phase 5: Production
- [ ] Create prediction pipeline for new matches
- [ ] Implement Over/Under probability calculation
- [ ] Set up model retraining schedule
- [ ] Document model version and performance

---

## 11. Common Pitfalls

### Pitfall 1: Data Leakage
**Problem:** Using current match statistics as features
**Solution:** Always verify features are computed from past matches only. Add assertion checks.

### Pitfall 2: Random Train/Test Splits
**Problem:** Random splitting leaks future information and inflates performance metrics
**Solution:** Always use time-based splits (by date or season)

### Pitfall 3: Ignoring Early Season Matches
**Problem:** Rolling features are undefined or unreliable for first few matches
**Solution:** Either exclude early matches from training, or use expanding windows with minimum match requirements

### Pitfall 4: Team Name Inconsistency
**Problem:** Same team has different names across seasons
**Solution:** Create and maintain a team name mapping dictionary

### Pitfall 5: Overfitting to Noise
**Problem:** Corners are inherently noisy; model learns patterns that don't generalize
**Solution:** Use regularization, limit tree depth, require minimum samples per leaf, validate on out-of-time data

### Pitfall 6: Expecting High Accuracy
**Problem:** Disappointment when R² is low
**Reality:** Corners depend heavily on in-game events (shots hitting posts, keeper saves, deflections). Expect R² of 0.05-0.15. Focus on calibration and edge over baseline.

### Pitfall 7: Not Updating Features After Model Training
**Problem:** Using stale rolling statistics for predictions
**Solution:** Always recompute features using the most recent available data before prediction

---

## Appendix: Formula Reference

### A.1 Rolling Mean Formula

For team T's home games, computing rolling mean of statistic S over last N games:

```
rolling_mean_S_N = (S[t-1] + S[t-2] + ... + S[t-N]) / N

where:
  t = current match index for team T at home
  S[t-k] = value of S in T's kth most recent home game
```

### A.2 Rolling Standard Deviation Formula

```
rolling_std_S_N = sqrt( sum((S[t-k] - rolling_mean_S_N)²) / (N-1) )

for k = 1 to N
```

### A.3 Coefficient of Variation

```
CV = rolling_std / max(rolling_mean, 1)
```

Division by max(rolling_mean, 1) prevents division by zero.

### A.4 Share-Based Metrics

```
shot_share = shots_for / (shots_for + shots_against)
corner_share = corners_for / (corners_for + corners_against)
```

Denominator uses team's own for and against values, not opponent's.

### A.5 Efficiency Ratios

```
shot_accuracy = shots_on_target / max(total_shots, 1)
corners_per_shot = corners / max(shots, 1)
```

### A.6 Blocked Shots Proxy

```
blocked_shots = total_shots - shots_on_target
blocked_shot_ratio = blocked_shots / max(total_shots, 1)
```

### A.7 Shot Imbalance

```
shot_imbalance = abs(home_shots - away_shots)
shot_dominance = home_shots - away_shots  (positive = home dominant)
```

---

## Document Version

- Version: 3.0
- Last Updated: January 2025
- Changes from v2:
  - Performed correlation analysis on 1,900 matches (2020-2025)
  - Added Category 5: Blocked Shots Proxy (r = 0.30)
  - Added Category 6: Shot Imbalance (r = 0.16)
  - Added Feature Correlation Summary (Section 4.10)
  - Updated Appendix with new formulas (A.6, A.7)
  - Streamlined feature set based on validated correlations

- Version: 2.0
- Changes from v1:
  - Removed referee features
  - Removed all odds-based features
  - Changed primary model from Negative Binomial to XGBoost
  - Added efficiency ratio features
  - Added volatility features
  - Expanded feature engineering documentation
  - Added complete implementation checklist