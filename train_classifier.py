"""
SCOPE - Binary Classifier for O/U Betting
Optimized for profitability, not prediction accuracy

Key insight: We don't need to predict exact corners.
We only need to predict Over vs Under for a specific threshold.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_VERSION = "classifier_v1"
TARGET_THRESHOLD = 9.5  # <-- CHANGE THIS TO TARGET DIFFERENT O/U
ROLLING_WINDOW = 5
TEST_SEASON = '2025-26'

# Classifier params - optimized for precision (fewer but better bets)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',  # Handle imbalanced classes
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Betting parameters
MIN_CONFIDENCE = 0.55  # Only bet if probability > this
ODDS_DECIMAL = 1.91    # Standard -110 odds

print("="*70)
print(f"SCOPE CLASSIFIER - O/U {TARGET_THRESHOLD}")
print("="*70)

# =============================================================================
# DATA LOADING
# =============================================================================
SEASONS = {
    '2020-21': '2021', '2021-22': '2122', '2022-23': '2223',
    '2023-24': '2324', '2024-25': '2425', '2025-26': '2526'
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
    except Exception as e:
        print(f"  {season_name}: Failed - {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
df['TotalCorners'] = df['HC'] + df['AC']

# Binary target: 1 = Over, 0 = Under
df['Target'] = (df['TotalCorners'] > TARGET_THRESHOLD).astype(int)

print(f"Total matches: {len(df)}")
print(f"Over {TARGET_THRESHOLD} rate: {df['Target'].mean()*100:.1f}%")

# =============================================================================
# FEATURE ENGINEERING (same as before)
# =============================================================================
print("\nComputing features...")

def compute_rolling_features(df, n=5):
    rolling_cols = [
        'home_corners_for', 'home_corners_against', 'home_corners_total', 'home_corner_std',
        'away_corners_for', 'away_corners_against', 'away_corners_total', 'away_corner_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_blocked_shots', 'home_blocked_against',
        'away_blocked_shots', 'away_blocked_against',
        'home_shot_dominance', 'away_shot_dominance',
        'home_corners_last3', 'away_corners_last3',
        'home_goals_for', 'home_goals_against',
        'away_goals_for', 'away_goals_against',
        # NEW: O/U specific features
        'home_over_rate', 'away_over_rate',
    ]

    for col in rolling_cols:
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
                df.loc[idx, 'home_shots_for'] = prev_data['HS'].mean()
                df.loc[idx, 'home_shots_against'] = prev_data['AS'].mean()
                df.loc[idx, 'home_sot_for'] = prev_data['HST'].mean()
                df.loc[idx, 'home_sot_against'] = prev_data['AST'].mean()
                df.loc[idx, 'home_blocked_shots'] = (prev_data['HS'] - prev_data['HST']).mean()
                df.loc[idx, 'home_blocked_against'] = (prev_data['AS'] - prev_data['AST']).mean()
                df.loc[idx, 'home_shot_dominance'] = (prev_data['HS'] - prev_data['AS']).mean()
                df.loc[idx, 'home_goals_for'] = prev_data['FTHG'].mean()
                df.loc[idx, 'home_goals_against'] = prev_data['FTAG'].mean()
                # O/U rate in recent home games
                df.loc[idx, 'home_over_rate'] = (prev_data['TotalCorners'] > TARGET_THRESHOLD).mean()

            if i >= 3:
                prev3 = home_indices[i-3:i]
                df.loc[idx, 'home_corners_last3'] = df.loc[prev3, 'HC'].mean()

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
                # O/U rate in recent away games
                df.loc[idx, 'away_over_rate'] = (prev_data['TotalCorners'] > TARGET_THRESHOLD).mean()

            if i >= 3:
                prev3 = away_indices[i-3:i]
                df.loc[idx, 'away_corners_last3'] = df.loc[prev3, 'AC'].mean()

    return df


def compute_match_features(df):
    df['expected_corners_for'] = df['home_corners_for'] + df['away_corners_for']
    df['expected_corners_against'] = df['home_corners_against'] + df['away_corners_against']
    df['expected_corners_total'] = (df['home_corners_total'] + df['away_corners_total']) / 2
    df['corner_differential'] = df['home_corners_for'] - df['away_corners_for']
    df['recent_corners_combined'] = df['home_corners_last3'].fillna(df['home_corners_for']) + \
                                     df['away_corners_last3'].fillna(df['away_corners_for'])
    df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
    df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']
    df['shot_differential'] = df['home_shots_for'] - df['away_shots_for']
    df['home_shot_accuracy'] = df['home_sot_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_shot_accuracy'] = df['away_sot_for'] / df['away_shots_for'].replace(0, np.nan)
    df['avg_shot_accuracy'] = (df['home_shot_accuracy'] + df['away_shot_accuracy']) / 2
    df['home_corners_per_shot'] = df['home_corners_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_corners_per_shot'] = df['away_corners_for'] / df['away_shots_for'].replace(0, np.nan)
    df['combined_corners_per_shot'] = df['home_corners_per_shot'] + df['away_corners_per_shot']
    df['home_shot_share'] = df['home_shots_for'] / (df['home_shots_for'] + df['home_shots_against']).replace(0, np.nan)
    df['away_shot_share'] = df['away_shots_for'] / (df['away_shots_for'] + df['away_shots_against']).replace(0, np.nan)
    df['home_corner_share'] = df['home_corners_for'] / (df['home_corners_for'] + df['home_corners_against']).replace(0, np.nan)
    df['away_corner_share'] = df['away_corners_for'] / (df['away_corners_for'] + df['away_corners_against']).replace(0, np.nan)
    df['pressure_sum'] = df['home_shot_share'] + df['away_shot_share']
    df['pressure_gap'] = df['home_shot_share'] - df['away_shot_share']
    df['combined_blocked_shots'] = df['home_blocked_shots'] + df['away_blocked_shots']
    df['blocked_shot_ratio'] = df['combined_blocked_shots'] / df['combined_shots_for'].replace(0, np.nan)
    df['expected_shot_imbalance'] = abs(df['home_shot_dominance'] - df['away_shot_dominance'])
    df['dominance_mismatch'] = df['home_shot_dominance'] + df['away_shot_dominance']
    df['home_corner_cv'] = df['home_corner_std'] / df['home_corners_for'].replace(0, np.nan)
    df['away_corner_cv'] = df['away_corner_std'] / df['away_corners_for'].replace(0, np.nan)
    df['combined_corner_volatility'] = df['home_corner_std'] + df['away_corner_std']
    df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
    df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
    df['goal_differential'] = df['home_goals_for'] - df['away_goals_for']

    # NEW: O/U specific composite features
    df['combined_over_rate'] = (df['home_over_rate'].fillna(0.5) + df['away_over_rate'].fillna(0.5)) / 2
    df['over_rate_diff'] = df['home_over_rate'].fillna(0.5) - df['away_over_rate'].fillna(0.5)

    return df


df = compute_rolling_features(df, n=ROLLING_WINDOW)
df = compute_match_features(df)

# =============================================================================
# FEATURE LIST
# =============================================================================
FEATURE_COLUMNS = [
    'home_corners_for', 'home_corners_against', 'home_corners_total',
    'away_corners_for', 'away_corners_against', 'away_corners_total',
    'expected_corners_for', 'expected_corners_against', 'expected_corners_total',
    'corner_differential', 'home_corner_std', 'away_corner_std',
    'recent_corners_combined',
    'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
    'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
    'combined_shots_for', 'combined_sot_for', 'shot_differential', 'avg_shot_accuracy',
    'home_shot_accuracy', 'away_shot_accuracy',
    'home_corners_per_shot', 'away_corners_per_shot', 'combined_corners_per_shot',
    'home_shot_share', 'away_shot_share',
    'home_corner_share', 'away_corner_share',
    'pressure_sum', 'pressure_gap',
    'home_blocked_shots', 'home_blocked_against',
    'away_blocked_shots', 'away_blocked_against',
    'combined_blocked_shots', 'blocked_shot_ratio',
    'home_shot_dominance', 'away_shot_dominance',
    'expected_shot_imbalance', 'dominance_mismatch',
    'home_corner_cv', 'away_corner_cv', 'combined_corner_volatility',
    'combined_goals_for', 'combined_goals_against', 'goal_differential',
    # O/U specific
    'home_over_rate', 'away_over_rate', 'combined_over_rate', 'over_rate_diff',
]

print(f"Features: {len(FEATURE_COLUMNS)}")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
df_model = df.dropna(subset=FEATURE_COLUMNS + ['Target']).copy()
print(f"Matches with complete features: {len(df_model)}")

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df['Target']
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df['Target']

print(f"Training: {len(train_df)} | Test: {len(test_df)}")
print(f"Train Over rate: {y_train.mean()*100:.1f}% | Test Over rate: {y_test.mean()*100:.1f}%")

# =============================================================================
# TRAINING WITH CALIBRATION
# =============================================================================
print("\n" + "="*70)
print("TRAINING CLASSIFIER")
print("="*70)

# Train base model
base_model = lgb.LGBMClassifier(**LGBM_PARAMS)
base_model.fit(X_train, y_train)

# Calibrate probabilities using isotonic regression
print("Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

print("Training complete!")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Get probabilities
y_prob = calibrated_model.predict_proba(X_test)[:, 1]  # Probability of Over
y_pred = (y_prob > 0.5).astype(int)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nBasic Metrics (threshold=0.5):")
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall:    {recall*100:.1f}%")
print(f"  F1 Score:  {f1*100:.1f}%")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

# =============================================================================
# BETTING SIMULATION
# =============================================================================
print("\n" + "="*70)
print("BETTING SIMULATION")
print("="*70)

test_df['prob_over'] = y_prob
test_df['prob_under'] = 1 - y_prob

# Break-even probability at -110 odds
break_even = 1 / ODDS_DECIMAL
print(f"\nOdds: {ODDS_DECIMAL} (break-even: {break_even*100:.1f}%)")

# Simulate different confidence thresholds
print("\n--- Confidence-Based Betting ---")
print(f"{'Conf':<8} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Profit':<10} {'ROI':<8}")
print("-" * 50)

for conf in [0.50, 0.55, 0.60, 0.65, 0.70]:
    # Bet Over when prob_over > conf
    bet_over = test_df['prob_over'] > conf
    # Bet Under when prob_under > conf
    bet_under = test_df['prob_under'] > conf

    over_wins = ((bet_over) & (test_df['Target'] == 1)).sum()
    over_losses = ((bet_over) & (test_df['Target'] == 0)).sum()
    under_wins = ((bet_under) & (test_df['Target'] == 0)).sum()
    under_losses = ((bet_under) & (test_df['Target'] == 1)).sum()

    total_bets = bet_over.sum() + bet_under.sum()
    total_wins = over_wins + under_wins
    total_losses = over_losses + under_losses

    if total_bets > 0:
        win_rate = total_wins / total_bets * 100
        # Profit calculation: win pays (odds-1), loss costs 1
        profit = total_wins * (ODDS_DECIMAL - 1) - total_losses * 1
        roi = profit / total_bets * 100
        print(f"{conf:<8.2f} {total_bets:<6} {total_wins:<6} {win_rate:<8.1f} ${profit:<+9.0f} {roi:<+7.1f}%")

# =============================================================================
# VALUE BETTING SIMULATION
# =============================================================================
print("\n--- Value Betting (only bet when our prob > implied prob) ---")
print("Assuming market odds imply 52.4% for both Over and Under")

market_implied = 0.524  # What -110 odds imply

# Only bet when we have edge
value_over = (test_df['prob_over'] > market_implied) & (test_df['prob_over'] > 0.5)
value_under = (test_df['prob_under'] > market_implied) & (test_df['prob_under'] > 0.5)

over_wins = ((value_over) & (test_df['Target'] == 1)).sum()
over_losses = ((value_over) & (test_df['Target'] == 0)).sum()
under_wins = ((value_under) & (test_df['Target'] == 0)).sum()
under_losses = ((value_under) & (test_df['Target'] == 1)).sum()

total_bets = value_over.sum() + value_under.sum()
total_wins = over_wins + under_wins

if total_bets > 0:
    win_rate = total_wins / total_bets * 100
    profit = total_wins * (ODDS_DECIMAL - 1) - (total_bets - total_wins) * 1
    roi = profit / total_bets * 100
    print(f"\nValue bets: {total_bets}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Profit: ${profit:+.0f}")
    print(f"ROI: {roi:+.1f}%")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*70)
print("TOP 10 FEATURES")
print("="*70)

importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': base_model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"  {row['Feature']:35s} {row['Importance']:4.0f}")

# =============================================================================
# SAVE MODEL
# =============================================================================
model_filename = f'classifier_ou{TARGET_THRESHOLD}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': calibrated_model,
        'base_model': base_model,
        'feature_columns': FEATURE_COLUMNS,
        'target_threshold': TARGET_THRESHOLD,
        'params': LGBM_PARAMS,
        'rolling_window': ROLLING_WINDOW,
        'version': MODEL_VERSION,
        'train_date': datetime.now().isoformat()
    }, f)

print(f"\nModel saved: {model_filename}")
