"""
SCOPE Model Analysis Script
Analyzes model performance on 2025-26 season and provides improvement recommendations.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = 'model_v7_lgbm_optimized_20260106_203923_best.pkl'
SEASON_CODE = '2526'
SEASON_NAME = '2025-26'

# =============================================================================
# LOAD MODEL
# =============================================================================
print("="*70)
print("SCOPE MODEL ANALYSIS")
print("="*70)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
FEATURE_COLUMNS = model_data['feature_columns']
ROLLING_WINDOW = model_data['rolling_window']
test_metrics = model_data['test_metrics']

print(f"\nModel: {MODEL_PATH}")
print(f"Rolling Window: {ROLLING_WINDOW}")
print(f"Features: {len(FEATURE_COLUMNS)}")

# =============================================================================
# LOAD ALL DATA (need historical for rolling features)
# =============================================================================
print("\n" + "="*70)
print("LOADING DATA")
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

dfs = []
for season_name, season_code in SEASONS.items():
    url = BASE_URL.format(code=season_code)
    try:
        df = pd.read_csv(url, encoding='utf-8')
        available_cols = [c for c in COLS if c in df.columns]
        df = df[available_cols].copy()
        df['Season'] = season_name
        print(f"  {season_name}: {len(df)} matches")
        dfs.append(df)
    except Exception as e:
        print(f"  {season_name}: Failed - {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
df['TotalCorners'] = df['HC'] + df['AC']

# =============================================================================
# COMPUTE ROLLING FEATURES
# =============================================================================
print("\n" + "="*70)
print("COMPUTING FEATURES")
print("="*70)

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
    return df


def compute_h2h_features(df):
    """Compute head-to-head historical features."""
    df['h2h_corners_avg'] = np.nan
    df['h2h_corners_std'] = np.nan
    df['h2h_matches'] = 0

    for idx in df.index:
        home = df.loc[idx, 'HomeTeam']
        away = df.loc[idx, 'AwayTeam']
        date = df.loc[idx, 'Date']

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

    overall_avg = df['TotalCorners'].mean()
    df['h2h_corners_avg'] = df['h2h_corners_avg'].fillna(overall_avg)
    df['h2h_corners_std'] = df['h2h_corners_std'].fillna(df['TotalCorners'].std())

    return df


df = compute_rolling_features(df, n=ROLLING_WINDOW)
df = compute_match_features(df)
print("  Computing H2H features...")
df = compute_h2h_features(df)
print("Features computed.")

# =============================================================================
# FILTER TO TEST SEASON
# =============================================================================
test_df = df[df['Season'] == SEASON_NAME].dropna(subset=FEATURE_COLUMNS + ['TotalCorners']).copy()
print(f"\nTest matches with complete features: {len(test_df)}")

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df['TotalCorners']

# =============================================================================
# MAKE PREDICTIONS
# =============================================================================
print("\n" + "="*70)
print("PREDICTIONS")
print("="*70)

test_df['Predicted'] = model.predict(X_test)
test_df['Residual'] = test_df['TotalCorners'] - test_df['Predicted']
test_df['AbsError'] = test_df['Residual'].abs()

# =============================================================================
# OVERALL METRICS
# =============================================================================
print("\n" + "-"*50)
print("OVERALL METRICS")
print("-"*50)

rmse = np.sqrt((test_df['Residual'] ** 2).mean())
mae = test_df['AbsError'].mean()
r2 = 1 - (test_df['Residual'] ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
corr = np.corrcoef(y_test, test_df['Predicted'])[0, 1]

print(f"RMSE:        {rmse:.3f}")
print(f"MAE:         {mae:.3f}")
print(f"R²:          {r2:.4f}")
print(f"Correlation: {corr:.4f}")

# =============================================================================
# OVER/UNDER ANALYSIS
# =============================================================================
print("\n" + "-"*50)
print("OVER/UNDER ANALYSIS")
print("-"*50)

thresholds = [8.5, 9.5, 10.5, 11.5, 12.5]
ou_results = []

for t in thresholds:
    actual_over = (y_test > t)
    pred_over = (test_df['Predicted'] > t)

    overall_acc = (actual_over == pred_over).mean() * 100

    # Over precision
    over_correct = ((pred_over) & (actual_over)).sum()
    over_total = pred_over.sum()
    over_prec = (over_correct / over_total * 100) if over_total > 0 else 0

    # Under precision
    under_correct = ((~pred_over) & (~actual_over)).sum()
    under_total = (~pred_over).sum()
    under_prec = (under_correct / under_total * 100) if under_total > 0 else 0

    actual_rate = actual_over.mean() * 100
    pred_rate = pred_over.mean() * 100

    ou_results.append({
        'threshold': t,
        'overall_acc': overall_acc,
        'over_prec': over_prec,
        'under_prec': under_prec,
        'actual_rate': actual_rate,
        'pred_rate': pred_rate,
        'over_bets': over_total,
        'under_bets': under_total
    })

    print(f"\nOver/Under {t}:")
    print(f"  Overall Accuracy: {overall_acc:.1f}%")
    print(f"  Over Precision:   {over_prec:.1f}% ({over_correct}/{over_total})")
    print(f"  Under Precision:  {under_prec:.1f}% ({under_correct}/{under_total})")
    print(f"  Actual Over Rate: {actual_rate:.1f}% | Predicted: {pred_rate:.1f}%")

# =============================================================================
# ANALYSIS BY ERROR SIZE
# =============================================================================
print("\n" + "-"*50)
print("PREDICTION ERROR DISTRIBUTION")
print("-"*50)

error_bins = [0, 1, 2, 3, 4, 5, 100]
error_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']
test_df['ErrorBin'] = pd.cut(test_df['AbsError'], bins=error_bins, labels=error_labels)

error_dist = test_df['ErrorBin'].value_counts().sort_index()
print("\nAbsolute Error Distribution:")
for label in error_labels:
    count = error_dist.get(label, 0)
    pct = count / len(test_df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {label} corners: {count:3d} ({pct:5.1f}%) {bar}")

# =============================================================================
# WORST PREDICTIONS
# =============================================================================
print("\n" + "-"*50)
print("WORST PREDICTIONS (Top 10)")
print("-"*50)

worst = test_df.nlargest(10, 'AbsError')[['Date', 'HomeTeam', 'AwayTeam', 'TotalCorners', 'Predicted', 'Residual']]
worst['Predicted'] = worst['Predicted'].round(1)
worst['Residual'] = worst['Residual'].round(1)
print(worst.to_string(index=False))

# =============================================================================
# BEST PREDICTIONS
# =============================================================================
print("\n" + "-"*50)
print("BEST PREDICTIONS (Top 10)")
print("-"*50)

best = test_df.nsmallest(10, 'AbsError')[['Date', 'HomeTeam', 'AwayTeam', 'TotalCorners', 'Predicted', 'Residual']]
best['Predicted'] = best['Predicted'].round(1)
best['Residual'] = best['Residual'].round(1)
print(best.to_string(index=False))

# =============================================================================
# ANALYSIS BY ACTUAL CORNER COUNT
# =============================================================================
print("\n" + "-"*50)
print("PERFORMANCE BY ACTUAL CORNER COUNT")
print("-"*50)

corner_bins = [0, 7, 9, 11, 13, 100]
corner_labels = ['Low (0-7)', 'Med-Low (8-9)', 'Medium (10-11)', 'Med-High (12-13)', 'High (14+)']
test_df['CornerBin'] = pd.cut(test_df['TotalCorners'], bins=corner_bins, labels=corner_labels)

for label in corner_labels:
    subset = test_df[test_df['CornerBin'] == label]
    if len(subset) > 0:
        mae_bin = subset['AbsError'].mean()
        bias = subset['Residual'].mean()
        count = len(subset)
        print(f"  {label:20s}: MAE={mae_bin:.2f}, Bias={bias:+.2f}, N={count}")

# =============================================================================
# ANALYSIS BY TEAM
# =============================================================================
print("\n" + "-"*50)
print("PERFORMANCE BY TEAM (Home)")
print("-"*50)

team_perf = test_df.groupby('HomeTeam').agg({
    'AbsError': 'mean',
    'Residual': 'mean',
    'TotalCorners': ['mean', 'count']
}).round(2)
team_perf.columns = ['MAE', 'Bias', 'AvgCorners', 'Matches']
team_perf = team_perf.sort_values('MAE')

print("\nBest predicted teams (lowest MAE):")
print(team_perf.head(5).to_string())

print("\nWorst predicted teams (highest MAE):")
print(team_perf.tail(5).to_string())

# =============================================================================
# BIAS ANALYSIS
# =============================================================================
print("\n" + "-"*50)
print("BIAS ANALYSIS")
print("-"*50)

mean_pred = test_df['Predicted'].mean()
mean_actual = test_df['TotalCorners'].mean()
overall_bias = mean_pred - mean_actual

print(f"Mean Predicted:  {mean_pred:.2f}")
print(f"Mean Actual:     {mean_actual:.2f}")
print(f"Overall Bias:    {overall_bias:+.2f} ({'overpredicts' if overall_bias > 0 else 'underpredicts'})")

# Check prediction range
print(f"\nPrediction Range: {test_df['Predicted'].min():.1f} - {test_df['Predicted'].max():.1f}")
print(f"Actual Range:     {test_df['TotalCorners'].min():.0f} - {test_df['TotalCorners'].max():.0f}")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "-"*50)
print("TOP 15 FEATURE IMPORTANCES")
print("-"*50)

importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importance_df.head(15).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {row['Feature']:30s} {row['Importance']:.4f} {bar}")

# =============================================================================
# CONCLUSIONS AND RECOMMENDATIONS
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY & RECOMMENDATIONS")
print("="*70)

print("\n📊 WHAT'S WORKING:")
print("-" * 40)

# Check what's good
good_points = []

if rmse < 3.5:
    good_points.append(f"✓ Reasonable RMSE ({rmse:.2f}) for corner prediction")
if corr > 0.15:
    good_points.append(f"✓ Positive correlation ({corr:.3f}) between predictions and actuals")

# Find best O/U threshold
best_ou = max(ou_results, key=lambda x: x['overall_acc'])
if best_ou['overall_acc'] > 55:
    good_points.append(f"✓ Best O/U accuracy at {best_ou['threshold']}: {best_ou['overall_acc']:.1f}%")

# Check error distribution
low_error_pct = (test_df['AbsError'] <= 2).mean() * 100
if low_error_pct > 40:
    good_points.append(f"✓ {low_error_pct:.0f}% of predictions within 2 corners of actual")

for p in good_points:
    print(f"  {p}")

print("\n⚠️  WHAT NEEDS IMPROVEMENT:")
print("-" * 40)

issues = []

# Check for bias
if abs(overall_bias) > 0.5:
    direction = "high" if overall_bias > 0 else "low"
    issues.append(f"• Systematic bias: model predicts {abs(overall_bias):.1f} corners too {direction}")

# Check prediction range compression
pred_range = test_df['Predicted'].max() - test_df['Predicted'].min()
actual_range = test_df['TotalCorners'].max() - test_df['TotalCorners'].min()
if pred_range < actual_range * 0.5:
    issues.append(f"• Prediction range too narrow ({pred_range:.1f} vs actual {actual_range:.0f})")
    issues.append("  → Model regresses too heavily to the mean")

# Check extreme corners performance
high_corner_games = test_df[test_df['TotalCorners'] >= 14]
if len(high_corner_games) > 0:
    high_mae = high_corner_games['AbsError'].mean()
    if high_mae > mae * 1.5:
        issues.append(f"• Poor prediction for high-corner games (14+): MAE={high_mae:.1f}")

low_corner_games = test_df[test_df['TotalCorners'] <= 6]
if len(low_corner_games) > 0:
    low_mae = low_corner_games['AbsError'].mean()
    if low_mae > mae * 1.5:
        issues.append(f"• Poor prediction for low-corner games (≤6): MAE={low_mae:.1f}")

# Check if R² is very low
if r2 < 0.05:
    issues.append(f"• Very low R² ({r2:.4f}) - model explains little variance")

for issue in issues:
    print(f"  {issue}")

print("\n🔧 RECOMMENDATIONS:")
print("-" * 40)

recommendations = []

# Based on issues found
if abs(overall_bias) > 0.5:
    recommendations.append("1. CALIBRATION: Add post-hoc calibration or adjust predictions by bias offset")

if pred_range < actual_range * 0.5:
    recommendations.append("2. REDUCE REGULARIZATION: Lower reg_alpha/reg_lambda to allow more extreme predictions")
    recommendations.append("   Try: reg_alpha=0.01, reg_lambda=0.1")

if r2 < 0.1:
    recommendations.append("3. FEATURE ENGINEERING:")
    recommendations.append("   - Add head-to-head historical corners")
    recommendations.append("   - Add referee corner statistics")
    recommendations.append("   - Add weather/pitch conditions if available")
    recommendations.append("   - Add match importance (league position, relegation battle)")

# Check feature importance concentration
top3_importance = importance_df.head(3)['Importance'].sum()
if top3_importance > 0.5:
    recommendations.append("4. FEATURE DEPENDENCY: Top 3 features dominate - consider:")
    recommendations.append(f"   - Current top: {', '.join(importance_df.head(3)['Feature'].tolist())}")
    recommendations.append("   - Add interaction terms or polynomial features")

# Model suggestions
recommendations.append("5. MODEL ALTERNATIVES TO TRY:")
recommendations.append("   - Poisson regression (corners are count data)")
recommendations.append("   - Quantile regression for O/U predictions")
recommendations.append("   - Ensemble with LightGBM or CatBoost")

recommendations.append("6. HYPERPARAMETER TUNING:")
recommendations.append("   - Try different rolling windows: N=3, N=10")
recommendations.append("   - Increase max_depth to 5 or 6")
recommendations.append("   - Lower learning_rate to 0.01 with more estimators")

for rec in recommendations:
    print(f"  {rec}")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
output_file = f'analysis_results_{SEASON_NAME.replace("-", "")}.csv'
test_df[['Date', 'HomeTeam', 'AwayTeam', 'TotalCorners', 'Predicted', 'Residual', 'AbsError']].to_csv(output_file, index=False)
print(f"\n\nDetailed results saved to: {output_file}")
print("="*70)
