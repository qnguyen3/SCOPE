"""
SCOPE - Fine-tuning Best Models (Iterative Approach)
Test one config at a time, observe, improve
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - EDIT THESE TO ITERATE
# =============================================================================
TEST_SEASON = '2025-26'
ODDS_DECIMAL = 1.91

# ITERATION 2: Less regularization for more confident predictions
LGBM_PARAMS = {
    'n_estimators': 700,
    'max_depth': 8,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
}

XGB_PARAMS = {
    'n_estimators': 700,
    'max_depth': 6,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
}

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 10,
    'min_samples_leaf': 10,
}

# Configs to test
CONFIGS = [
    {'threshold': 9.5, 'window': 5, 'conf': 0.58},
    {'threshold': 10.5, 'window': 10, 'conf': 0.54},
]

print("="*70)
print("SCOPE FINE-TUNING - ITERATIVE")
print("="*70)
print(f"\nLGBM: depth={LGBM_PARAMS['max_depth']}, lr={LGBM_PARAMS['learning_rate']}, est={LGBM_PARAMS['n_estimators']}")
print(f"XGB:  depth={XGB_PARAMS['max_depth']}, lr={XGB_PARAMS['learning_rate']}, est={XGB_PARAMS['n_estimators']}")
print(f"RF:   depth={RF_PARAMS['max_depth']}, est={RF_PARAMS['n_estimators']}")

# =============================================================================
# DATA LOADING
# =============================================================================
SEASONS = {
    '2010-11': '1011', '2011-12': '1112', '2012-13': '1213',
    '2013-14': '1314', '2014-15': '1415', '2015-16': '1516',
    '2016-17': '1617', '2017-18': '1718', '2018-19': '1819',
    '2019-20': '1920', '2020-21': '2021', '2021-22': '2122',
    '2022-23': '2223', '2023-24': '2324', '2024-25': '2425',
    '2025-26': '2526'
}

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']

print("\nLoading data...")
dfs = []
for season_name, season_code in SEASONS.items():
    try:
        df = pd.read_csv(BASE_URL.format(code=season_code), encoding='utf-8', on_bad_lines='skip')
        df = df[[c for c in COLS if c in df.columns]].copy()
        df['Season'] = season_name
        dfs.append(df)
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'HC', 'AC'])
df = df.sort_values('Date').reset_index(drop=True)
df['TotalCorners'] = df['HC'] + df['AC']
print(f"Total matches: {len(df)}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_features(df, n=5):
    feature_cols = [
        'home_corners_for', 'home_corners_against', 'home_corners_total', 'home_corner_std',
        'away_corners_for', 'away_corners_against', 'away_corners_total', 'away_corner_std',
        'home_shots', 'away_shots', 'home_sot', 'away_sot',
        'home_goals', 'away_goals', 'home_fouls', 'away_fouls',
        'home_over_9', 'home_over_10', 'home_over_11',
        'away_over_9', 'away_over_10', 'away_over_11',
        'home_corner_trend', 'away_corner_trend',
    ]
    for col in feature_cols:
        df[col] = np.nan

    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_idx = df[df['HomeTeam'] == team].index.tolist()
        away_idx = df[df['AwayTeam'] == team].index.tolist()

        for i, idx in enumerate(home_idx):
            if i >= n:
                prev = df.loc[home_idx[i-n:i]]
                df.loc[idx, 'home_corners_for'] = prev['HC'].mean()
                df.loc[idx, 'home_corners_against'] = prev['AC'].mean()
                df.loc[idx, 'home_corners_total'] = (prev['HC'] + prev['AC']).mean()
                df.loc[idx, 'home_corner_std'] = prev['HC'].std()
                if 'HS' in df.columns: df.loc[idx, 'home_shots'] = prev['HS'].mean()
                if 'HST' in df.columns: df.loc[idx, 'home_sot'] = prev['HST'].mean()
                if 'FTHG' in df.columns: df.loc[idx, 'home_goals'] = prev['FTHG'].mean()
                if 'HF' in df.columns: df.loc[idx, 'home_fouls'] = prev['HF'].mean()
                total = prev['HC'] + prev['AC']
                df.loc[idx, 'home_over_9'] = (total > 9.5).mean()
                df.loc[idx, 'home_over_10'] = (total > 10.5).mean()
                df.loc[idx, 'home_over_11'] = (total > 11.5).mean()
                if i >= 3:
                    df.loc[idx, 'home_corner_trend'] = df.loc[home_idx[i-3:i], 'HC'].mean() - df.loc[home_idx[i-n:i-3], 'HC'].mean()

        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = df.loc[away_idx[i-n:i]]
                df.loc[idx, 'away_corners_for'] = prev['AC'].mean()
                df.loc[idx, 'away_corners_against'] = prev['HC'].mean()
                df.loc[idx, 'away_corners_total'] = (prev['HC'] + prev['AC']).mean()
                df.loc[idx, 'away_corner_std'] = prev['AC'].std()
                if 'AS' in df.columns: df.loc[idx, 'away_shots'] = prev['AS'].mean()
                if 'AST' in df.columns: df.loc[idx, 'away_sot'] = prev['AST'].mean()
                if 'FTAG' in df.columns: df.loc[idx, 'away_goals'] = prev['FTAG'].mean()
                if 'AF' in df.columns: df.loc[idx, 'away_fouls'] = prev['AF'].mean()
                total = prev['HC'] + prev['AC']
                df.loc[idx, 'away_over_9'] = (total > 9.5).mean()
                df.loc[idx, 'away_over_10'] = (total > 10.5).mean()
                df.loc[idx, 'away_over_11'] = (total > 11.5).mean()
                if i >= 3:
                    df.loc[idx, 'away_corner_trend'] = df.loc[away_idx[i-3:i], 'AC'].mean() - df.loc[away_idx[i-n:i-3], 'AC'].mean()

    df['expected_corners'] = (df['home_corners_total'] + df['away_corners_total']) / 2
    df['combined_corners_for'] = df['home_corners_for'] + df['away_corners_for']
    df['corner_diff'] = df['home_corners_for'] - df['away_corners_for']
    df['combined_volatility'] = df['home_corner_std'] + df['away_corner_std']
    df['combined_shots'] = df['home_shots'].fillna(0) + df['away_shots'].fillna(0)
    df['combined_goals'] = df['home_goals'].fillna(0) + df['away_goals'].fillna(0)
    df['combined_trend'] = df['home_corner_trend'].fillna(0) + df['away_corner_trend'].fillna(0)
    df['combined_over_9'] = (df['home_over_9'].fillna(0.5) + df['away_over_9'].fillna(0.5)) / 2
    df['combined_over_10'] = (df['home_over_10'].fillna(0.5) + df['away_over_10'].fillna(0.5)) / 2
    df['combined_over_11'] = (df['home_over_11'].fillna(0.5) + df['away_over_11'].fillna(0.5)) / 2
    return df

FEATURES = [
    'home_corners_for', 'home_corners_against', 'home_corners_total',
    'away_corners_for', 'away_corners_against', 'away_corners_total',
    'expected_corners', 'combined_corners_for', 'corner_diff',
    'home_corner_std', 'away_corner_std', 'combined_volatility',
    'home_shots', 'away_shots', 'combined_shots',
    'home_sot', 'away_sot',
    'home_goals', 'away_goals', 'combined_goals',
    'home_corner_trend', 'away_corner_trend', 'combined_trend',
    'home_over_9', 'away_over_9', 'combined_over_9',
    'home_over_10', 'away_over_10', 'combined_over_10',
    'home_over_11', 'away_over_11', 'combined_over_11',
]

# =============================================================================
# TRAIN AND EVALUATE
# =============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)

for config in CONFIGS:
    threshold = config['threshold']
    window = config['window']
    conf = config['conf']

    print(f"\n--- O/U {threshold} (Window={window}, Conf>{conf}) ---")

    df_w = compute_features(df.copy(), n=window)
    valid_feat = [c for c in FEATURES if c in df_w.columns]
    df_m = df_w.dropna(subset=valid_feat).copy()

    train = df_m[df_m['Season'] != TEST_SEASON].copy()
    test = df_m[df_m['Season'] == TEST_SEASON].copy()

    train['Target'] = (train['TotalCorners'] > threshold).astype(int)
    test['Target'] = (test['TotalCorners'] > threshold).astype(int)

    X_train, y_train = train[valid_feat], train['Target']
    X_test, y_test = test[valid_feat], test['Target']

    naive = max(test['Target'].mean(), 1 - test['Target'].mean())

    # Train
    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS, verbose=-1, random_state=42, class_weight='balanced')
    xgbm = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0, random_state=42)
    rf = RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1, class_weight='balanced')

    ensemble = VotingClassifier([('lgbm', lgbm), ('xgb', xgbm), ('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]

    # Evaluate at multiple confidence levels
    print(f"{'Conf':<8} {'Bets':<8} {'Win%':<8} {'Edge%':<10} {'ROI%':<10}")
    print("-" * 50)

    for c in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62]:
        over = probs > c
        under = (1 - probs) > c
        bets = over.sum() + under.sum()
        wins = ((over) & (y_test == 1)).sum() + ((under) & (y_test == 0)).sum()

        if bets > 5:
            wr = wins / bets * 100
            roi = (wins * (ODDS_DECIMAL - 1) - (bets - wins)) / bets * 100
            edge = wr - naive * 100
            status = "✅" if edge > 3 else "⚠️" if edge > 0 else "❌"
            marker = " <-- CURRENT" if c == conf else ""
            print(f">{c:<7} {bets:<8} {wr:<8.1f} {edge:>+8.1f}% {roi:>+8.1f}% {status}{marker}")

print("\n" + "="*70)
print("To iterate: edit LGBM_PARAMS, XGB_PARAMS, RF_PARAMS and re-run")
print("="*70)
