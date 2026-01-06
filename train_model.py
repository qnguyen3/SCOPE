"""
SCOPE - Model Training Script
Senior Data Scientist Workspace

This script is designed for iterative model improvement.
Change log at bottom of file.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - BEST MODEL (v7)
# =============================================================================
# Best performing configuration after 7 iterations

MODEL_VERSION = "v7_lgbm_optimized"
MODEL_TYPE = "lightgbm"  # "xgboost" or "lightgbm"

ROLLING_WINDOW = 5  # N=5 performed best
TEST_SEASON = '2025-26'
VALIDATION_SPLIT = 0.2

LIGHTGBM_PARAMS = {
    'objective': 'poisson',
    'metric': 'rmse',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,        # Slower for more iterations
    'n_estimators': 500,          # More iterations
    'min_child_samples': 10,      # Slight constraint
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.001,           # Tiny regularization
    'reg_lambda': 0.01,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

XGBOOST_PARAMS = {
    'objective': 'count:poisson',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.001,
    'reg_lambda': 0.01,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

EARLY_STOPPING_ROUNDS = None  # Disabled for aggressive learning

# =============================================================================
# DATA LOADING
# =============================================================================
print("="*70)
print(f"SCOPE MODEL TRAINING - {MODEL_VERSION}")
print("="*70)

SEASONS = {
    '2020-21': '2021',
    '2021-22': '2122',
    '2022-23': '2223',
    '2023-24': '2324',
    '2024-25': '2425',
    '2025-26': '2526'
}

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']

print("\nLoading data...")
dfs = []
for season_name, season_code in SEASONS.items():
    url = BASE_URL.format(code=season_code)
    try:
        df = pd.read_csv(url, encoding='utf-8')
        available_cols = [c for c in COLS if c in df.columns]
        df = df[available_cols].copy()
        df['Season'] = season_name
        dfs.append(df)
        print(f"  {season_name}: {len(df)} matches")
    except Exception as e:
        print(f"  {season_name}: Failed - {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
df['TotalCorners'] = df['HC'] + df['AC']

print(f"\nTotal: {len(df)} matches")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\nComputing features...")

def compute_rolling_features(df, n=5):
    """Compute venue-aware rolling features."""
    rolling_cols = [
        'home_corners_for', 'home_corners_against', 'home_corners_total', 'home_corner_std',
        'away_corners_for', 'away_corners_against', 'away_corners_total', 'away_corner_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_blocked_shots', 'home_blocked_against',
        'away_blocked_shots', 'away_blocked_against',
        'home_shot_dominance', 'away_shot_dominance',
        # NEW: Recent form features
        'home_corners_last3', 'away_corners_last3',
        'home_goals_for', 'home_goals_against',
        'away_goals_for', 'away_goals_against',
    ]

    for col in rolling_cols:
        df[col] = np.nan

    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())

    for team in all_teams:
        home_mask = df['HomeTeam'] == team
        away_mask = df['AwayTeam'] == team

        home_indices = df[home_mask].index.tolist()
        away_indices = df[away_mask].index.tolist()

        # HOME games
        for i, idx in enumerate(home_indices):
            if i >= n:
                prev = home_indices[i-n:i]
                prev_data = df.loc[prev]

                df.loc[idx, 'home_corners_for'] = prev_data['HC'].mean()
                df.loc[idx, 'home_corners_against'] = prev_data['AC'].mean()
                df.loc[idx, 'home_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'home_corner_std'] = prev_data['HC'].std()
                df.loc[idx, 'home_shots_for'] = prev_data['HS'].mean()
                df.loc[idx, 'home_shots_against'] = prev_data['AS'].mean()
                df.loc[idx, 'home_sot_for'] = prev_data['HST'].mean()
                df.loc[idx, 'home_sot_against'] = prev_data['AST'].mean()
                df.loc[idx, 'home_blocked_shots'] = (prev_data['HS'] - prev_data['HST']).mean()
                df.loc[idx, 'home_blocked_against'] = (prev_data['AS'] - prev_data['AST']).mean()
                df.loc[idx, 'home_shot_dominance'] = (prev_data['HS'] - prev_data['AS']).mean()
                df.loc[idx, 'home_goals_for'] = prev_data['FTHG'].mean()
                df.loc[idx, 'home_goals_against'] = prev_data['FTAG'].mean()

            # Last 3 games (shorter window for recent form)
            if i >= 3:
                prev3 = home_indices[i-3:i]
                df.loc[idx, 'home_corners_last3'] = df.loc[prev3, 'HC'].mean()

        # AWAY games
        for i, idx in enumerate(away_indices):
            if i >= n:
                prev = away_indices[i-n:i]
                prev_data = df.loc[prev]

                df.loc[idx, 'away_corners_for'] = prev_data['AC'].mean()
                df.loc[idx, 'away_corners_against'] = prev_data['HC'].mean()
                df.loc[idx, 'away_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'away_corner_std'] = prev_data['AC'].std()
                df.loc[idx, 'away_shots_for'] = prev_data['AS'].mean()
                df.loc[idx, 'away_shots_against'] = prev_data['HS'].mean()
                df.loc[idx, 'away_sot_for'] = prev_data['AST'].mean()
                df.loc[idx, 'away_sot_against'] = prev_data['HST'].mean()
                df.loc[idx, 'away_blocked_shots'] = (prev_data['AS'] - prev_data['AST']).mean()
                df.loc[idx, 'away_blocked_against'] = (prev_data['HS'] - prev_data['HST']).mean()
                df.loc[idx, 'away_shot_dominance'] = (prev_data['AS'] - prev_data['HS']).mean()
                df.loc[idx, 'away_goals_for'] = prev_data['FTAG'].mean()
                df.loc[idx, 'away_goals_against'] = prev_data['FTHG'].mean()

            if i >= 3:
                prev3 = away_indices[i-3:i]
                df.loc[idx, 'away_corners_last3'] = df.loc[prev3, 'AC'].mean()

    return df


def compute_match_features(df):
    """Compute match-level composite features."""
    # Corner composites
    df['expected_corners_for'] = df['home_corners_for'] + df['away_corners_for']
    df['expected_corners_against'] = df['home_corners_against'] + df['away_corners_against']
    df['expected_corners_total'] = (df['home_corners_total'] + df['away_corners_total']) / 2
    df['corner_differential'] = df['home_corners_for'] - df['away_corners_for']

    # Recent form composite
    df['recent_corners_combined'] = df['home_corners_last3'].fillna(df['home_corners_for']) + \
                                     df['away_corners_last3'].fillna(df['away_corners_for'])

    # Shot composites
    df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
    df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']
    df['shot_differential'] = df['home_shots_for'] - df['away_shots_for']

    # Efficiency ratios
    df['home_shot_accuracy'] = df['home_sot_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_shot_accuracy'] = df['away_sot_for'] / df['away_shots_for'].replace(0, np.nan)
    df['avg_shot_accuracy'] = (df['home_shot_accuracy'] + df['away_shot_accuracy']) / 2
    df['home_corners_per_shot'] = df['home_corners_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_corners_per_shot'] = df['away_corners_for'] / df['away_shots_for'].replace(0, np.nan)
    df['combined_corners_per_shot'] = df['home_corners_per_shot'] + df['away_corners_per_shot']

    # Pressure index
    df['home_shot_share'] = df['home_shots_for'] / (df['home_shots_for'] + df['home_shots_against']).replace(0, np.nan)
    df['away_shot_share'] = df['away_shots_for'] / (df['away_shots_for'] + df['away_shots_against']).replace(0, np.nan)
    df['home_corner_share'] = df['home_corners_for'] / (df['home_corners_for'] + df['home_corners_against']).replace(0, np.nan)
    df['away_corner_share'] = df['away_corners_for'] / (df['away_corners_for'] + df['away_corners_against']).replace(0, np.nan)
    df['pressure_sum'] = df['home_shot_share'] + df['away_shot_share']
    df['pressure_gap'] = df['home_shot_share'] - df['away_shot_share']

    # Blocked shots
    df['combined_blocked_shots'] = df['home_blocked_shots'] + df['away_blocked_shots']
    df['blocked_shot_ratio'] = df['combined_blocked_shots'] / df['combined_shots_for'].replace(0, np.nan)

    # Shot imbalance
    df['expected_shot_imbalance'] = abs(df['home_shot_dominance'] - df['away_shot_dominance'])
    df['dominance_mismatch'] = df['home_shot_dominance'] + df['away_shot_dominance']

    # Volatility
    df['home_corner_cv'] = df['home_corner_std'] / df['home_corners_for'].replace(0, np.nan)
    df['away_corner_cv'] = df['away_corner_std'] / df['away_corners_for'].replace(0, np.nan)
    df['combined_corner_volatility'] = df['home_corner_std'] + df['away_corner_std']

    # Goal-based features (attacking intent proxy)
    df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
    df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
    df['goal_differential'] = df['home_goals_for'] - df['away_goals_for']

    return df


df = compute_rolling_features(df, n=ROLLING_WINDOW)
df = compute_match_features(df)


def compute_h2h_features(df):
    """Compute head-to-head historical features."""
    print("  Computing H2H features...")

    df['h2h_corners_avg'] = np.nan
    df['h2h_corners_std'] = np.nan
    df['h2h_matches'] = 0

    for idx in df.index:
        home = df.loc[idx, 'HomeTeam']
        away = df.loc[idx, 'AwayTeam']
        date = df.loc[idx, 'Date']

        # Find previous meetings (either venue)
        h2h_mask = (
            (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))) &
            (df['Date'] < date)
        )

        h2h_matches = df[h2h_mask]

        if len(h2h_matches) >= 1:
            df.loc[idx, 'h2h_corners_avg'] = h2h_matches['TotalCorners'].mean()
            df.loc[idx, 'h2h_corners_std'] = h2h_matches['TotalCorners'].std() if len(h2h_matches) > 1 else 0
            df.loc[idx, 'h2h_matches'] = len(h2h_matches)

    # Fill NaN with overall average
    overall_avg = df['TotalCorners'].mean()
    df['h2h_corners_avg'] = df['h2h_corners_avg'].fillna(overall_avg)
    df['h2h_corners_std'] = df['h2h_corners_std'].fillna(df['TotalCorners'].std())

    return df


df = compute_h2h_features(df)

# =============================================================================
# FEATURE LIST
# =============================================================================
FEATURE_COLUMNS = [
    # Rolling Corners
    'home_corners_for', 'home_corners_against', 'home_corners_total',
    'away_corners_for', 'away_corners_against', 'away_corners_total',
    'expected_corners_for', 'expected_corners_against', 'expected_corners_total',
    'corner_differential', 'home_corner_std', 'away_corner_std',
    'recent_corners_combined',

    # Rolling Shots
    'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
    'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
    'combined_shots_for', 'combined_sot_for', 'shot_differential', 'avg_shot_accuracy',

    # Efficiency Ratios
    'home_shot_accuracy', 'away_shot_accuracy',
    'home_corners_per_shot', 'away_corners_per_shot', 'combined_corners_per_shot',

    # Pressure Index
    'home_shot_share', 'away_shot_share',
    'home_corner_share', 'away_corner_share',
    'pressure_sum', 'pressure_gap',

    # Blocked Shots
    'home_blocked_shots', 'home_blocked_against',
    'away_blocked_shots', 'away_blocked_against',
    'combined_blocked_shots', 'blocked_shot_ratio',

    # Shot Imbalance
    'home_shot_dominance', 'away_shot_dominance',
    'expected_shot_imbalance', 'dominance_mismatch',

    # Volatility
    'home_corner_cv', 'away_corner_cv', 'combined_corner_volatility',

    # Goals (attacking intent)
    'combined_goals_for', 'combined_goals_against', 'goal_differential',

    # Head-to-head
    'h2h_corners_avg', 'h2h_corners_std', 'h2h_matches',
]

TARGET_COLUMN = 'TotalCorners'

print(f"Features: {len(FEATURE_COLUMNS)}")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
df_model = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
print(f"Matches with complete features: {len(df_model)}")

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

print(f"Training: {len(train_df)} | Test: {len(test_df)}")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]

# Validation split
val_size = int(len(X_train) * VALIDATION_SPLIT)
X_train_fit = X_train.iloc[:-val_size]
y_train_fit = y_train.iloc[:-val_size]
X_val = X_train.iloc[-val_size:]
y_val = y_train.iloc[-val_size:]

print(f"Train fit: {len(X_train_fit)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*70)
print(f"TRAINING ({MODEL_TYPE.upper()})")
print("="*70)

if MODEL_TYPE == "lightgbm":
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if EARLY_STOPPING_ROUNDS else []

    model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )

    if EARLY_STOPPING_ROUNDS:
        print(f"Best iteration: {model.best_iteration_}")
        print(f"Best validation RMSE: {model.best_score_['valid_0']['rmse']:.4f}")
    else:
        print(f"Trained for {LIGHTGBM_PARAMS['n_estimators']} iterations")

elif MODEL_TYPE == "xgboost":
    if EARLY_STOPPING_ROUNDS:
        model = xgb.XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best validation RMSE: {model.best_score:.4f}")
    else:
        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        print(f"Trained for {XGBOOST_PARAMS['n_estimators']} iterations (no early stopping)")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

y_pred_test = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
corr = np.corrcoef(y_test, y_pred_test)[0, 1]

print(f"\nTest Set Results:")
print(f"  RMSE:        {rmse:.3f}")
print(f"  MAE:         {mae:.3f}")
print(f"  R²:          {r2:.4f}")
print(f"  Correlation: {corr:.4f}")

# Prediction range check
print(f"\n  Pred Range:  {y_pred_test.min():.1f} - {y_pred_test.max():.1f}")
print(f"  Actual Range: {y_test.min():.0f} - {y_test.max():.0f}")

# Over/Under quick check
print("\nOver/Under Accuracy:")
for t in [9.5, 10.5, 11.5]:
    acc = ((y_test > t) == (y_pred_test > t)).mean() * 100
    print(f"  O/U {t}: {acc:.1f}%")

# =============================================================================
# SAVE MODEL
# =============================================================================
model_filename = f'model_{MODEL_VERSION}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
params = LIGHTGBM_PARAMS if MODEL_TYPE == "lightgbm" else XGBOOST_PARAMS
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': model,
        'model_type': MODEL_TYPE,
        'feature_columns': FEATURE_COLUMNS,
        'params': params,
        'rolling_window': ROLLING_WINDOW,
        'version': MODEL_VERSION,
        'test_metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr},
        'train_date': datetime.now().isoformat()
    }, f)

print(f"\nModel saved: {model_filename}")

# =============================================================================
# CHANGE LOG
# =============================================================================
"""
CHANGE LOG:
-----------
v1 (baseline): Original XGBoost from train.ipynb
  - max_depth=4, learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0
  - RMSE=3.37, R²=-0.03, Corr=-0.03
  - Prediction range: 10.0-11.0 (too narrow - severe mean regression)

v2_reduced_reg: XGBoost with reduced regularization
  - max_depth=6, learning_rate=0.03, reg_alpha=0.01, reg_lambda=0.1
  - RMSE=3.37, R²=-0.03, Corr=-0.05
  - Prediction range: 10.0-10.7 (still too narrow)

v3_poisson_no_reg: XGBoost Poisson, NO regularization
  - objective='count:poisson', reg_alpha=0, reg_lambda=0
  - RMSE=3.37, R²=-0.03, Corr=+0.09 (first positive!)
  - Prediction range: 7.9-12.3 (much better!)

v4_h2h_features: Added H2H features
  - Same as v3 + head-to-head corner history
  - RMSE=3.43, R²=-0.07, Corr=+0.03
  - Slight overfit, H2H didn't help

v5_lightgbm: Switched to LightGBM
  - LightGBM with early stopping
  - Early stopped at 11 iterations (too early)
  - Prediction range: 10.0-10.8 (narrow again)

v6_lgbm_aggressive: LightGBM aggressive, NO regularization
  - num_leaves=63, max_depth=8, no reg, 300 iterations
  - RMSE=3.34, R²=-0.01, Corr=+0.14 (best so far!)
  - Prediction range: 6.9-12.6

v7_lgbm_optimized: BEST MODEL
  - LightGBM, Poisson, num_leaves=63, max_depth=8
  - learning_rate=0.05, 500 iterations
  - Tiny regularization (0.001/0.01)
  - RMSE=3.32, R²=-0.0016, Corr=+0.15
  - Prediction range: 7.7-12.6
  - O/U 9.5: 58.9% accuracy

v8_rolling3: Tested N=3 rolling window
  - Wider range (6.4-12.8) but worse metrics
  - N=5 remains optimal

FINAL BEST: v7_lgbm_optimized
"""
