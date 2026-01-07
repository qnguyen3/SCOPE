"""
SCOPE - Train Production Classifiers
LightGBM with SMOTE resampling for class balance
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_SEASON = '2025-26'
ODDS_DECIMAL = 1.91
MODEL_DIR = 'models/classifier'

# LightGBM params
LGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# Optimized confidence thresholds based on probability distribution analysis
CONFIGS = [
    {'threshold': 8.5, 'window': 5, 'confidence': 0.70},
    {'threshold': 9.5, 'window': 5, 'confidence': 0.60},
    {'threshold': 10.5, 'window': 5, 'confidence': 0.65},
    {'threshold': 11.5, 'window': 5, 'confidence': 0.70},
    {'threshold': 12.5, 'window': 5, 'confidence': 0.70},
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
COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY']

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
# FEATURE ENGINEERING V2
# New features: shot accuracy, corners per shot, yellow cards, goal difference
# =============================================================================
def compute_features(df, n=5):
    feature_cols = [
        # Core corner stats
        'home_corners_avg', 'away_corners_avg',
        'home_corners_conceded', 'away_corners_conceded',
        # Shot stats
        'home_shots_avg', 'away_shots_avg',
        'home_sot_avg', 'away_sot_avg',
        # Shot efficiency (SOT/Shots)
        'home_shot_accuracy', 'away_shot_accuracy',
        # Corners per shot (corner generation efficiency)
        'home_corners_per_shot', 'away_corners_per_shot',
        # Yellow cards (aggression indicator)
        'home_yellows_avg', 'away_yellows_avg',
        # Goal difference (form indicator)
        'home_goal_diff', 'away_goal_diff',
        # Fouls
        'home_fouls_avg', 'away_fouls_avg',
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
                df.loc[idx, 'home_corners_avg'] = prev['HC'].mean()
                df.loc[idx, 'home_corners_conceded'] = prev['AC'].mean()
                df.loc[idx, 'home_shots_avg'] = prev['HS'].mean()
                df.loc[idx, 'home_sot_avg'] = prev['HST'].mean()

                # Shot accuracy
                shots = prev['HS'].sum()
                if shots > 0:
                    df.loc[idx, 'home_shot_accuracy'] = prev['HST'].sum() / shots
                    df.loc[idx, 'home_corners_per_shot'] = prev['HC'].sum() / shots

                # Yellow cards
                if 'HY' in prev.columns:
                    df.loc[idx, 'home_yellows_avg'] = prev['HY'].mean()

                # Goal difference
                df.loc[idx, 'home_goal_diff'] = (prev['FTHG'] - prev['FTAG']).mean()

                # Fouls
                if 'HF' in prev.columns:
                    df.loc[idx, 'home_fouls_avg'] = prev['HF'].mean()

        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = df.loc[away_idx[i-n:i]]
                df.loc[idx, 'away_corners_avg'] = prev['AC'].mean()
                df.loc[idx, 'away_corners_conceded'] = prev['HC'].mean()
                df.loc[idx, 'away_shots_avg'] = prev['AS'].mean()
                df.loc[idx, 'away_sot_avg'] = prev['AST'].mean()

                # Shot accuracy
                shots = prev['AS'].sum()
                if shots > 0:
                    df.loc[idx, 'away_shot_accuracy'] = prev['AST'].sum() / shots
                    df.loc[idx, 'away_corners_per_shot'] = prev['AC'].sum() / shots

                # Yellow cards
                if 'AY' in prev.columns:
                    df.loc[idx, 'away_yellows_avg'] = prev['AY'].mean()

                # Goal difference
                df.loc[idx, 'away_goal_diff'] = (prev['FTAG'] - prev['FTHG']).mean()

                # Fouls
                if 'AF' in prev.columns:
                    df.loc[idx, 'away_fouls_avg'] = prev['AF'].mean()

    # Combined features (non-redundant)
    df['total_corners_expected'] = df['home_corners_avg'] + df['away_corners_avg']
    df['total_shots_expected'] = df['home_shots_avg'] + df['away_shots_avg']
    df['corner_efficiency_combined'] = df['home_corners_per_shot'].fillna(0) + df['away_corners_per_shot'].fillna(0)
    df['aggression_combined'] = df['home_yellows_avg'].fillna(0) + df['away_yellows_avg'].fillna(0)
    df['form_diff'] = df['home_goal_diff'].fillna(0) - df['away_goal_diff'].fillna(0)
    return df

FEATURES = [
    'home_corners_avg', 'away_corners_avg',
    'home_corners_conceded', 'away_corners_conceded',
    'home_shots_avg', 'away_shots_avg',
    'home_sot_avg', 'away_sot_avg',
    'home_shot_accuracy', 'away_shot_accuracy',
    'home_corners_per_shot', 'away_corners_per_shot',
    'home_yellows_avg', 'away_yellows_avg',
    'home_goal_diff', 'away_goal_diff',
    'home_fouls_avg', 'away_fouls_avg',
    'total_corners_expected', 'total_shots_expected',
    'corner_efficiency_combined', 'aggression_combined', 'form_diff',
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

    # Class balance before SMOTE
    pos_rate = (y_train == 1).mean()
    print(f"Class balance before SMOTE: {pos_rate*100:.1f}% positive / {(1-pos_rate)*100:.1f}% negative")

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal)} samples ({(y_train_bal == 1).mean()*100:.1f}% positive)")

    # Train LightGBM (no class_weight needed after SMOTE)
    model = lgb.LGBMClassifier(**LGBM_PARAMS, verbose=-1, random_state=42)
    print("Training LightGBM...")
    model.fit(X_train_bal, y_train_bal)

    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    print(f"\nResults at Conf > {conf}:")
    over = probs > conf
    under = (1 - probs) > conf
    bets = over.sum() + under.sum()
    wins = ((over) & (y_test == 1)).sum() + ((under) & (y_test == 0)).sum()

    wr, roi, edge = 0, 0, 0  # defaults
    if bets > 0:
        wr = wins / bets * 100
        roi = (wins * (ODDS_DECIMAL - 1) - (bets - wins)) / bets * 100
        edge = wr - naive * 100
        print(f"  Bets: {bets}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Real Edge: {edge:+.1f}%")
        print(f"  ROI: {roi:+.1f}%")

    # Probability distribution analysis
    print(f"\n--- Probability Distribution ---")
    print(f"  Min: {probs.min():.3f} | Max: {probs.max():.3f} | Mean: {probs.mean():.3f} | Std: {probs.std():.3f}")
    print(f"  P < 0.3 (confident Under): {(probs < 0.3).sum()} ({(probs < 0.3).mean()*100:.1f}%)")
    print(f"  P 0.3-0.7 (uncertain): {((probs >= 0.3) & (probs <= 0.7)).sum()} ({((probs >= 0.3) & (probs <= 0.7)).mean()*100:.1f}%)")
    print(f"  P > 0.7 (confident Over): {(probs > 0.7).sum()} ({(probs > 0.7).mean()*100:.1f}%)")

    # Confusion Matrix Analysis
    print(f"\n--- Confusion Matrix Analysis ---")
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    total_test = len(y_test)
    actual_over = (y_test == 1).sum()
    actual_under = (y_test == 0).sum()
    pred_over = (preds == 1).sum()
    pred_under = (preds == 0).sum()

    print(f"  Actual:    Over={actual_over} ({actual_over/total_test*100:.1f}%) | Under={actual_under} ({actual_under/total_test*100:.1f}%)")
    print(f"  Predicted: Over={pred_over} ({pred_over/total_test*100:.1f}%) | Under={pred_under} ({pred_under/total_test*100:.1f}%)")
    print(f"  Confusion Matrix:")
    print(f"                 Pred Under   Pred Over")
    print(f"    Actual Under    {tn:4d}        {fp:4d}")
    print(f"    Actual Over     {fn:4d}        {tp:4d}")

    # Check if model is just following base rate
    base_rate_over = actual_over / total_test
    pred_rate_over = pred_over / total_test
    base_rate_diff = abs(pred_rate_over - base_rate_over) * 100

    if pred_over == 0 or pred_under == 0:
        print(f"  ⚠️  WARNING: Model predicting ALL {'Under' if pred_over == 0 else 'Over'}!")
    elif base_rate_diff < 5:
        print(f"  ⚠️  Model prediction rate similar to base rate (diff={base_rate_diff:.1f}%)")
    else:
        print(f"  ✅ Model deviating from base rate by {base_rate_diff:.1f}%")

    # Save model
    model_data = {
        'model': model,
        'features': valid_feat,
        'threshold': threshold,
        'window': window,
        'confidence': conf,
        'params': LGBM_PARAMS,
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
