"""
SCOPE - Feature Selection Analysis
Identifies important features and tests reduced feature sets
"""

import pandas as pd
import numpy as np
import pickle
import glob
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_SEASON = '2025-26'
ODDS_DECIMAL = 1.91

print("="*70)
print("SCOPE FEATURE SELECTION ANALYSIS")
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

print("\nLoading data...")
dfs = []
for season_name, season_code in SEASONS.items():
    url = BASE_URL.format(code=season_code)
    try:
        df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
        available_cols = [c for c in COLS if c in df.columns]
        df = df[available_cols].copy()
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
# FEATURE ENGINEERING (Same as train_all_classifiers.py)
# =============================================================================
def compute_features(df, n=5):
    feature_cols = [
        'home_corners_for', 'home_corners_against', 'home_corners_total', 'home_corner_std',
        'away_corners_for', 'away_corners_against', 'away_corners_total', 'away_corner_std',
        'home_shots', 'away_shots', 'home_sot', 'away_sot',
        'home_goals', 'away_goals',
        'home_fouls', 'away_fouls',
        'home_over_9', 'home_over_10', 'home_over_11',
        'away_over_9', 'away_over_10', 'away_over_11',
        'home_corner_trend', 'away_corner_trend',
    ]

    for col in feature_cols:
        df[col] = np.nan

    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())

    for team in all_teams:
        home_mask = df['HomeTeam'] == team
        away_mask = df['AwayTeam'] == team
        home_indices = df[home_mask].index.tolist()
        away_indices = df[away_mask].index.tolist()

        for i, idx in enumerate(home_indices):
            if i >= n:
                prev = home_indices[i-n:i]
                prev_data = df.loc[prev]

                df.loc[idx, 'home_corners_for'] = prev_data['HC'].mean()
                df.loc[idx, 'home_corners_against'] = prev_data['AC'].mean()
                df.loc[idx, 'home_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'home_corner_std'] = prev_data['HC'].std()

                if 'HS' in df.columns:
                    df.loc[idx, 'home_shots'] = prev_data['HS'].mean()
                if 'HST' in df.columns:
                    df.loc[idx, 'home_sot'] = prev_data['HST'].mean()
                if 'FTHG' in df.columns:
                    df.loc[idx, 'home_goals'] = prev_data['FTHG'].mean()
                if 'HF' in df.columns:
                    df.loc[idx, 'home_fouls'] = prev_data['HF'].mean()

                total_corners = prev_data['HC'] + prev_data['AC']
                df.loc[idx, 'home_over_9'] = (total_corners > 9.5).mean()
                df.loc[idx, 'home_over_10'] = (total_corners > 10.5).mean()
                df.loc[idx, 'home_over_11'] = (total_corners > 11.5).mean()

                if i >= n:
                    recent = df.loc[home_indices[i-3:i], 'HC'].mean() if i >= 3 else prev_data['HC'].mean()
                    older = df.loc[home_indices[i-n:i-3], 'HC'].mean() if i >= n else prev_data['HC'].mean()
                    df.loc[idx, 'home_corner_trend'] = recent - older

        for i, idx in enumerate(away_indices):
            if i >= n:
                prev = away_indices[i-n:i]
                prev_data = df.loc[prev]

                df.loc[idx, 'away_corners_for'] = prev_data['AC'].mean()
                df.loc[idx, 'away_corners_against'] = prev_data['HC'].mean()
                df.loc[idx, 'away_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'away_corner_std'] = prev_data['AC'].std()

                if 'AS' in df.columns:
                    df.loc[idx, 'away_shots'] = prev_data['AS'].mean()
                if 'AST' in df.columns:
                    df.loc[idx, 'away_sot'] = prev_data['AST'].mean()
                if 'FTAG' in df.columns:
                    df.loc[idx, 'away_goals'] = prev_data['FTAG'].mean()
                if 'AF' in df.columns:
                    df.loc[idx, 'away_fouls'] = prev_data['AF'].mean()

                total_corners = prev_data['HC'] + prev_data['AC']
                df.loc[idx, 'away_over_9'] = (total_corners > 9.5).mean()
                df.loc[idx, 'away_over_10'] = (total_corners > 10.5).mean()
                df.loc[idx, 'away_over_11'] = (total_corners > 11.5).mean()

                if i >= n:
                    recent = df.loc[away_indices[i-3:i], 'AC'].mean() if i >= 3 else prev_data['AC'].mean()
                    older = df.loc[away_indices[i-n:i-3], 'AC'].mean() if i >= n else prev_data['AC'].mean()
                    df.loc[idx, 'away_corner_trend'] = recent - older

    # Composite features
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

    # === NEW THEORETICALLY-JUSTIFIED FEATURES ===

    # 1. Shot-to-corner efficiency: teams that convert shots to corners well
    # Theory: Some teams attack in ways that generate more corners per shot
    df['home_shots_per_corner'] = df['home_shots'] / (df['home_corners_for'] + 0.1)
    df['away_shots_per_corner'] = df['away_shots'] / (df['away_corners_for'] + 0.1)
    df['combined_shot_efficiency'] = (df['home_shots_per_corner'] + df['away_shots_per_corner']) / 2

    # 2. Goal differential context: teams behind tend to push for more corners
    # Theory: Trailing teams attack more aggressively, generating corners
    df['home_goal_diff'] = df['home_goals'] - df['away_goals'].fillna(0)
    df['away_goal_diff'] = -df['home_goal_diff']

    # 3. Attack-defense asymmetry: mismatches create corner opportunities
    # Theory: Strong attack vs weak defense = more corners
    df['attack_defense_mismatch'] = df['home_corners_for'] - df['away_corners_against'].fillna(0)
    df['reverse_mismatch'] = df['away_corners_for'] - df['home_corners_against'].fillna(0)

    # 4. Historical variance interaction: two volatile teams = unpredictable
    df['volatility_product'] = df['home_corner_std'] * df['away_corner_std']

    # 5. Combined shot-on-target (more precise attacking metric)
    df['combined_sot'] = df['home_sot'].fillna(0) + df['away_sot'].fillna(0)
    df['sot_ratio'] = df['combined_sot'] / (df['combined_shots'] + 0.1)

    return df

# Compute features for best windows
print("\nComputing features...")
df_n5 = compute_features(df.copy(), n=5)
df_n10 = compute_features(df.copy(), n=10)

FEATURE_COLUMNS = [
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

# New features added for testing
NEW_FEATURES = [
    'home_shots_per_corner', 'away_shots_per_corner', 'combined_shot_efficiency',
    'home_goal_diff', 'away_goal_diff',
    'attack_defense_mismatch', 'reverse_mismatch',
    'volatility_product',
    'combined_sot', 'sot_ratio',
]

ALL_FEATURES = FEATURE_COLUMNS + NEW_FEATURES

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

def analyze_features(df_window, threshold, n):
    """Analyze feature importance for a specific configuration."""

    valid_features = [c for c in FEATURE_COLUMNS if c in df_window.columns]
    df_model = df_window.dropna(subset=valid_features).copy()
    train_df = df_model[df_model['Season'] != TEST_SEASON].copy()

    train_df['Target'] = (train_df['TotalCorners'] > threshold).astype(int)
    X = train_df[valid_features]
    y = train_df['Target']

    # 1. LightGBM Feature Importance
    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, verbose=-1, random_state=42)
    lgbm.fit(X, y)
    lgbm_importance = pd.DataFrame({
        'feature': valid_features,
        'lgbm_importance': lgbm.feature_importances_
    })

    # 2. XGBoost Feature Importance
    xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=5, verbosity=0, random_state=42)
    xgbm.fit(X, y)
    xgb_importance = pd.DataFrame({
        'feature': valid_features,
        'xgb_importance': xgbm.feature_importances_
    })

    # 3. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': valid_features,
        'mi_score': mi_scores
    })

    # Combine
    importance = lgbm_importance.merge(xgb_importance).merge(mi_importance)

    # Normalize
    importance['lgbm_norm'] = importance['lgbm_importance'] / importance['lgbm_importance'].max()
    importance['xgb_norm'] = importance['xgb_importance'] / importance['xgb_importance'].max()
    importance['mi_norm'] = importance['mi_score'] / importance['mi_score'].max() if importance['mi_score'].max() > 0 else 0

    # Combined score
    importance['combined_score'] = (importance['lgbm_norm'] + importance['xgb_norm'] + importance['mi_norm']) / 3
    importance = importance.sort_values('combined_score', ascending=False)

    return importance

# Analyze for best configs
print("\n--- O/U 9.5 (N=5) Feature Importance ---")
imp_9_5 = analyze_features(df_n5, 9.5, 5)
print(f"{'Feature':<30} {'LGBM':<10} {'XGB':<10} {'MI':<10} {'Combined':<10}")
print("-" * 70)
for _, row in imp_9_5.head(15).iterrows():
    print(f"{row['feature']:<30} {row['lgbm_norm']:<10.3f} {row['xgb_norm']:<10.3f} {row['mi_norm']:<10.3f} {row['combined_score']:<10.3f}")

print("\n--- O/U 10.5 (N=10) Feature Importance ---")
imp_10_5 = analyze_features(df_n10, 10.5, 10)
print(f"{'Feature':<30} {'LGBM':<10} {'XGB':<10} {'MI':<10} {'Combined':<10}")
print("-" * 70)
for _, row in imp_10_5.head(15).iterrows():
    print(f"{row['feature']:<30} {row['lgbm_norm']:<10.3f} {row['xgb_norm']:<10.3f} {row['mi_norm']:<10.3f} {row['combined_score']:<10.3f}")

# =============================================================================
# TEST REDUCED FEATURE SETS
# =============================================================================
print("\n" + "="*70)
print("TESTING REDUCED FEATURE SETS")
print("="*70)

def train_and_evaluate(df_window, threshold, features, conf_threshold):
    """Train model with specific features and evaluate."""

    valid_features = [c for c in features if c in df_window.columns]
    df_model = df_window.dropna(subset=valid_features).copy()
    train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
    test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

    train_df['Target'] = (train_df['TotalCorners'] > threshold).astype(int)
    test_df['Target'] = (test_df['TotalCorners'] > threshold).astype(int)

    X_train = train_df[valid_features]
    y_train = train_df['Target']
    X_test = test_df[valid_features]
    y_test = test_df['Target']

    base_rate = test_df['Target'].mean()
    naive_acc = max(base_rate, 1 - base_rate)

    # Ensemble
    lgbm = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, verbose=-1, random_state=42)
    xgbm = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, verbosity=0, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

    ensemble = VotingClassifier([('lgbm', lgbm), ('xgb', xgbm), ('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]

    # Betting simulation
    bet_over = probs > conf_threshold
    bet_under = (1 - probs) > conf_threshold

    over_wins = ((bet_over) & (y_test == 1)).sum()
    under_wins = ((bet_under) & (y_test == 0)).sum()
    total_bets = bet_over.sum() + bet_under.sum()
    total_wins = over_wins + under_wins

    if total_bets > 0:
        win_rate = total_wins / total_bets
        roi = (total_wins * (ODDS_DECIMAL - 1) - (total_bets - total_wins)) / total_bets * 100
        edge = (win_rate - naive_acc) * 100
    else:
        win_rate, roi, edge = 0, 0, 0

    return {
        'bets': total_bets,
        'win_rate': win_rate * 100,
        'edge': edge,
        'roi': roi,
        'features': len(valid_features)
    }

# Define feature sets to test
print("\nDefining feature sets...")

# Top features from importance analysis
top_9_5 = imp_9_5.head(15)['feature'].tolist()
top_10_5 = imp_10_5.head(15)['feature'].tolist()

# Core corner features only
CORE_FEATURES = [
    'home_corners_for', 'home_corners_against', 'home_corners_total',
    'away_corners_for', 'away_corners_against', 'away_corners_total',
    'expected_corners', 'combined_corners_for', 'corner_diff',
]

# O/U specific features
OU_FEATURES = [
    'home_over_9', 'away_over_9', 'combined_over_9',
    'home_over_10', 'away_over_10', 'combined_over_10',
    'home_over_11', 'away_over_11', 'combined_over_11',
]

# Momentum features
MOMENTUM_FEATURES = [
    'home_corner_trend', 'away_corner_trend', 'combined_trend',
    'home_corner_std', 'away_corner_std', 'combined_volatility',
]

# Test different feature sets for O/U 9.5
print("\n--- O/U 9.5 (N=5) Feature Set Comparison ---")
print(f"{'Feature Set':<30} {'#Feat':<8} {'Bets':<8} {'Win%':<8} {'Edge%':<10} {'ROI%':<10}")
print("-" * 75)

feature_sets_9 = {
    'Baseline (32 features)': FEATURE_COLUMNS,
    'Baseline + NEW features': ALL_FEATURES,
    'Top 15 important': top_9_5,
    'Core corners only': CORE_FEATURES,
    'Core + O/U rates': CORE_FEATURES + OU_FEATURES,
    'Core + Momentum': CORE_FEATURES + MOMENTUM_FEATURES,
}

results_9 = {}
for name, features in feature_sets_9.items():
    result = train_and_evaluate(df_n5, 9.5, features, 0.58)
    results_9[name] = result
    status = "✅" if result['edge'] > 5 else "⚠️" if result['edge'] > 0 else "❌"
    print(f"{name:<30} {result['features']:<8} {result['bets']:<8} {result['win_rate']:<8.1f} {result['edge']:>+8.1f}% {result['roi']:>+8.1f}% {status}")

# Test different feature sets for O/U 10.5
print("\n--- O/U 10.5 (N=10) Feature Set Comparison ---")
print(f"{'Feature Set':<30} {'#Feat':<8} {'Bets':<8} {'Win%':<8} {'Edge%':<10} {'ROI%':<10}")
print("-" * 75)

feature_sets_10 = {
    'Baseline (32 features)': FEATURE_COLUMNS,
    'Baseline + NEW features': ALL_FEATURES,
    'Top 15 important': top_10_5,
    'Core corners only': CORE_FEATURES,
    'Core + O/U rates': CORE_FEATURES + OU_FEATURES,
    'Core + Momentum': CORE_FEATURES + MOMENTUM_FEATURES,
}

results_10 = {}
for name, features in feature_sets_10.items():
    result = train_and_evaluate(df_n10, 10.5, features, 0.54)
    results_10[name] = result
    status = "✅" if result['edge'] > 3 else "⚠️" if result['edge'] > 0 else "❌"
    print(f"{name:<30} {result['features']:<8} {result['bets']:<8} {result['win_rate']:<8.1f} {result['edge']:>+8.1f}% {result['roi']:>+8.1f}% {status}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FEATURE SELECTION SUMMARY")
print("="*70)

print("\nO/U 9.5 Best Feature Set:")
best_9 = max(results_9.items(), key=lambda x: x[1]['edge'] if x[1]['bets'] >= 50 else -999)
print(f"  {best_9[0]}: {best_9[1]['features']} features, {best_9[1]['bets']} bets, {best_9[1]['edge']:+.1f}% edge")

print("\nO/U 10.5 Best Feature Set:")
best_10 = max(results_10.items(), key=lambda x: x[1]['edge'] if x[1]['bets'] >= 50 else -999)
print(f"  {best_10[0]}: {best_10[1]['features']} features, {best_10[1]['bets']} bets, {best_10[1]['edge']:+.1f}% edge")

print("\n" + "="*70)
print("FEATURE SELECTION COMPLETE")
print("="*70)
