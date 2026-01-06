"""
SCOPE - Train Production Classifiers
Best params from fine-tuning iterations

O/U 9.5: Window=5, Iter 1 params (deeper, more regularized)
O/U 10.5: Window=10, Iter 2 params (less regularization)
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_SEASON = '2025-26'
ODDS_DECIMAL = 1.91
MODEL_DIR = 'models/classifier'

# Best configurations - tuned params applied to all thresholds
# Lower thresholds (8.5, 9.5): Window=5, more regularization
# Higher thresholds (10.5, 11.5, 12.5): Window=10, less regularization

PARAMS_LOW = {
    'lgbm': {
        'n_estimators': 700,
        'max_depth': 8,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
    },
    'xgb': {
        'n_estimators': 700,
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
    },
    'rf': {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_leaf': 15,
    },
}

PARAMS_HIGH = {
    'lgbm': {
        'n_estimators': 700,
        'max_depth': 8,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'min_child_samples': 20,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
    },
    'xgb': {
        'n_estimators': 700,
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
    },
    'rf': {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_leaf': 10,
    },
}

CONFIGS = [
    {'threshold': 8.5, 'window': 5, 'confidence': 0.58, **PARAMS_LOW},
    {'threshold': 9.5, 'window': 5, 'confidence': 0.58, **PARAMS_LOW},
    {'threshold': 10.5, 'window': 10, 'confidence': 0.54, **PARAMS_HIGH},
    {'threshold': 11.5, 'window': 10, 'confidence': 0.54, **PARAMS_HIGH},
    {'threshold': 12.5, 'window': 10, 'confidence': 0.54, **PARAMS_HIGH},
]

print("="*70)
print("SCOPE - PRODUCTION CLASSIFIER TRAINING")
print("="*70)

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

print("\nLoading data (2010-2026)...")
dfs = []
for season_name, season_code in SEASONS.items():
    try:
        df = pd.read_csv(BASE_URL.format(code=season_code), encoding='utf-8', on_bad_lines='skip')
        df = df[[c for c in COLS if c in df.columns]].copy()
        df['Season'] = season_name
        dfs.append(df)
        print(f"  {season_name}: {len(df)} matches")
    except Exception as e:
        print(f"  {season_name}: Failed - {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'HC', 'AC'])
df = df.sort_values('Date').reset_index(drop=True)
df['TotalCorners'] = df['HC'] + df['AC']

print(f"\nTotal: {len(df)} matches")

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
# TRAIN MODELS
# =============================================================================
print("\n" + "="*70)
print("TRAINING PRODUCTION MODELS")
print("="*70)

os.makedirs(MODEL_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for config in CONFIGS:
    threshold = config['threshold']
    window = config['window']
    conf = config['confidence']

    print(f"\n{'#'*60}")
    print(f"O/U {threshold} | Window={window} | Conf>{conf}")
    print(f"{'#'*60}")

    # Compute features
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

    print(f"Training: {len(train)} | Test: {len(test)}")
    print(f"Naive accuracy: {naive*100:.1f}%")

    # Build models with config-specific params
    lgbm = lgb.LGBMClassifier(**config['lgbm'], verbose=-1, random_state=42, class_weight='balanced')
    xgbm = xgb.XGBClassifier(**config['xgb'], verbosity=0, random_state=42)
    rf = RandomForestClassifier(**config['rf'], random_state=42, n_jobs=-1, class_weight='balanced')

    ensemble = VotingClassifier([('lgbm', lgbm), ('xgb', xgbm), ('rf', rf)], voting='soft')
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate
    probs = calibrated.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    print(f"\nResults at Conf > {conf}:")
    over = probs > conf
    under = (1 - probs) > conf
    bets = over.sum() + under.sum()
    wins = ((over) & (y_test == 1)).sum() + ((under) & (y_test == 0)).sum()

    if bets > 0:
        wr = wins / bets * 100
        roi = (wins * (ODDS_DECIMAL - 1) - (bets - wins)) / bets * 100
        edge = wr - naive * 100
        print(f"  Bets: {bets}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Real Edge: {edge:+.1f}%")
        print(f"  ROI: {roi:+.1f}%")

    # Save model
    model_data = {
        'model': calibrated,
        'ensemble': ensemble,
        'features': valid_feat,
        'threshold': threshold,
        'window': window,
        'confidence': conf,
        'params': {
            'lgbm': config['lgbm'],
            'xgb': config['xgb'],
            'rf': config['rf'],
        },
        'metrics': {
            'bets': bets,
            'win_rate': wr,
            'edge': edge,
            'roi': roi,
        }
    }

    filename = os.path.join(MODEL_DIR, f'ou{threshold}_w{window}_{timestamp}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nSaved: {filename}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nModels saved to: {MODEL_DIR}/")
