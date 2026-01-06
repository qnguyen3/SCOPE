"""
SCOPE - Comprehensive Feature Analysis
=======================================
This script calculates ALL features defined in GUIDE.md using venue-aware
rolling statistics and validates their correlation with TotalCorners.

Features are computed using only historical data (no data leakage).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# =============================================================================
# Configuration
# =============================================================================

SEASONS = {
    '2020-21': '2021',
    '2021-22': '2122',
    '2022-23': '2223',
    '2023-24': '2324',
    '2024-25': '2425',
}

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'

COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC']

# Rolling window size
N = 5

# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load all seasons data from Football-Data.co.uk"""
    print("Loading data from Football-Data.co.uk...\n")
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

    # Create target variable
    df['TotalCorners'] = df['HC'] + df['AC']

    return df

# =============================================================================
# Rolling Feature Calculations (Venue-Aware)
# =============================================================================

def compute_team_rolling_stats(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Compute venue-aware rolling statistics for each team.

    For each match, computes rolling stats using only previous matches
    at the same venue (home stats from home games, away stats from away games).
    """
    print(f"\nComputing rolling statistics (N={n})...")

    # Initialize feature columns
    feature_cols = []

    # ----- Category 1: Rolling Corner Statistics -----
    # Home team features (from their previous HOME games)
    df['home_corners_for'] = np.nan
    df['home_corners_against'] = np.nan
    df['home_corners_total'] = np.nan
    df['home_corner_std'] = np.nan

    # Away team features (from their previous AWAY games)
    df['away_corners_for'] = np.nan
    df['away_corners_against'] = np.nan
    df['away_corners_total'] = np.nan
    df['away_corner_std'] = np.nan

    # ----- Category 2: Rolling Shot Statistics -----
    df['home_shots_for'] = np.nan
    df['home_shots_against'] = np.nan
    df['home_sot_for'] = np.nan
    df['home_sot_against'] = np.nan

    df['away_shots_for'] = np.nan
    df['away_shots_against'] = np.nan
    df['away_sot_for'] = np.nan
    df['away_sot_against'] = np.nan

    # ----- Category 5: Blocked Shots -----
    df['home_blocked_shots'] = np.nan
    df['home_blocked_against'] = np.nan
    df['away_blocked_shots'] = np.nan
    df['away_blocked_against'] = np.nan

    # ----- Category 6: Shot Imbalance -----
    df['home_shot_dominance'] = np.nan
    df['away_shot_dominance'] = np.nan

    # Get all teams
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())

    # Process each team
    for team in all_teams:
        # Get indices where team plays home/away
        home_mask = df['HomeTeam'] == team
        away_mask = df['AwayTeam'] == team

        home_indices = df[home_mask].index.tolist()
        away_indices = df[away_mask].index.tolist()

        # Compute rolling stats for HOME games
        for i, idx in enumerate(home_indices):
            if i >= n:
                prev_home = home_indices[i-n:i]
                prev_data = df.loc[prev_home]

                # Category 1: Corners
                df.loc[idx, 'home_corners_for'] = prev_data['HC'].mean()
                df.loc[idx, 'home_corners_against'] = prev_data['AC'].mean()
                df.loc[idx, 'home_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'home_corner_std'] = prev_data['HC'].std()

                # Category 2: Shots
                df.loc[idx, 'home_shots_for'] = prev_data['HS'].mean()
                df.loc[idx, 'home_shots_against'] = prev_data['AS'].mean()
                df.loc[idx, 'home_sot_for'] = prev_data['HST'].mean()
                df.loc[idx, 'home_sot_against'] = prev_data['AST'].mean()

                # Category 5: Blocked shots
                df.loc[idx, 'home_blocked_shots'] = (prev_data['HS'] - prev_data['HST']).mean()
                df.loc[idx, 'home_blocked_against'] = (prev_data['AS'] - prev_data['AST']).mean()

                # Category 6: Shot dominance
                df.loc[idx, 'home_shot_dominance'] = (prev_data['HS'] - prev_data['AS']).mean()

        # Compute rolling stats for AWAY games
        for i, idx in enumerate(away_indices):
            if i >= n:
                prev_away = away_indices[i-n:i]
                prev_data = df.loc[prev_away]

                # Category 1: Corners (note: away team's corners are AC)
                df.loc[idx, 'away_corners_for'] = prev_data['AC'].mean()
                df.loc[idx, 'away_corners_against'] = prev_data['HC'].mean()
                df.loc[idx, 'away_corners_total'] = (prev_data['HC'] + prev_data['AC']).mean()
                df.loc[idx, 'away_corner_std'] = prev_data['AC'].std()

                # Category 2: Shots (away team's shots are AS)
                df.loc[idx, 'away_shots_for'] = prev_data['AS'].mean()
                df.loc[idx, 'away_shots_against'] = prev_data['HS'].mean()
                df.loc[idx, 'away_sot_for'] = prev_data['AST'].mean()
                df.loc[idx, 'away_sot_against'] = prev_data['HST'].mean()

                # Category 5: Blocked shots
                df.loc[idx, 'away_blocked_shots'] = (prev_data['AS'] - prev_data['AST']).mean()
                df.loc[idx, 'away_blocked_against'] = (prev_data['HS'] - prev_data['HST']).mean()

                # Category 6: Shot dominance (from away perspective)
                df.loc[idx, 'away_shot_dominance'] = (prev_data['AS'] - prev_data['HS']).mean()

    print("  Rolling statistics computed.")
    return df


def compute_match_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute match-level composite features from team rolling stats.
    """
    print("Computing match-level features...")

    # ----- Category 1: Corner Composites -----
    df['expected_corners_for'] = df['home_corners_for'] + df['away_corners_for']
    df['expected_corners_against'] = df['home_corners_against'] + df['away_corners_against']
    df['expected_corners_total'] = (df['home_corners_total'] + df['away_corners_total']) / 2
    df['corner_differential'] = df['home_corners_for'] - df['away_corners_for']

    # ----- Category 2: Shot Composites -----
    df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
    df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']
    df['shot_differential'] = df['home_shots_for'] - df['away_shots_for']

    # ----- Category 3: Efficiency Ratios -----
    df['home_shot_accuracy'] = df['home_sot_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_shot_accuracy'] = df['away_sot_for'] / df['away_shots_for'].replace(0, np.nan)
    df['avg_shot_accuracy'] = (df['home_shot_accuracy'] + df['away_shot_accuracy']) / 2

    df['home_corners_per_shot'] = df['home_corners_for'] / df['home_shots_for'].replace(0, np.nan)
    df['away_corners_per_shot'] = df['away_corners_for'] / df['away_shots_for'].replace(0, np.nan)
    df['combined_corners_per_shot'] = df['home_corners_per_shot'] + df['away_corners_per_shot']

    # ----- Category 4: Pressure Index -----
    df['home_shot_share'] = df['home_shots_for'] / (df['home_shots_for'] + df['home_shots_against']).replace(0, np.nan)
    df['away_shot_share'] = df['away_shots_for'] / (df['away_shots_for'] + df['away_shots_against']).replace(0, np.nan)
    df['home_corner_share'] = df['home_corners_for'] / (df['home_corners_for'] + df['home_corners_against']).replace(0, np.nan)
    df['away_corner_share'] = df['away_corners_for'] / (df['away_corners_for'] + df['away_corners_against']).replace(0, np.nan)
    df['pressure_sum'] = df['home_shot_share'] + df['away_shot_share']
    df['pressure_gap'] = df['home_shot_share'] - df['away_shot_share']

    # ----- Category 5: Blocked Shots Composites -----
    df['combined_blocked_shots'] = df['home_blocked_shots'] + df['away_blocked_shots']
    df['blocked_shot_ratio'] = df['combined_blocked_shots'] / df['combined_shots_for'].replace(0, np.nan)

    # ----- Category 6: Shot Imbalance Composites -----
    df['expected_shot_imbalance'] = abs(df['home_shot_dominance'] - df['away_shot_dominance'])
    df['dominance_mismatch'] = df['home_shot_dominance'] + df['away_shot_dominance']

    # ----- Category 7: Volatility -----
    df['home_corner_cv'] = df['home_corner_std'] / df['home_corners_for'].replace(0, np.nan)
    df['away_corner_cv'] = df['away_corner_std'] / df['away_corners_for'].replace(0, np.nan)
    df['combined_corner_volatility'] = df['home_corner_std'] + df['away_corner_std']

    print("  Match-level features computed.")
    return df


# =============================================================================
# Correlation Analysis
# =============================================================================

def get_all_feature_columns() -> Dict[str, List[str]]:
    """Return all feature columns organized by category"""
    return {
        'Category 1: Rolling Corners': [
            'home_corners_for', 'home_corners_against', 'home_corners_total',
            'away_corners_for', 'away_corners_against', 'away_corners_total',
            'expected_corners_for', 'expected_corners_against', 'expected_corners_total',
            'corner_differential', 'home_corner_std', 'away_corner_std'
        ],
        'Category 2: Rolling Shots': [
            'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
            'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
            'combined_shots_for', 'combined_sot_for', 'shot_differential', 'avg_shot_accuracy'
        ],
        'Category 3: Efficiency Ratios': [
            'home_shot_accuracy', 'away_shot_accuracy',
            'home_corners_per_shot', 'away_corners_per_shot', 'combined_corners_per_shot'
        ],
        'Category 4: Pressure Index': [
            'home_shot_share', 'away_shot_share',
            'home_corner_share', 'away_corner_share',
            'pressure_sum', 'pressure_gap'
        ],
        'Category 5: Blocked Shots': [
            'home_blocked_shots', 'home_blocked_against',
            'away_blocked_shots', 'away_blocked_against',
            'combined_blocked_shots', 'blocked_shot_ratio'
        ],
        'Category 6: Shot Imbalance': [
            'home_shot_dominance', 'away_shot_dominance',
            'expected_shot_imbalance', 'dominance_mismatch'
        ],
        'Category 7: Volatility': [
            'home_corner_cv', 'away_corner_cv', 'combined_corner_volatility'
        ]
    }


def analyze_correlations(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Analyze correlations of all features with TotalCorners"""

    feature_categories = get_all_feature_columns()
    all_features = []
    for features in feature_categories.values():
        all_features.extend(features)

    # Calculate correlations
    correlations = {}
    for feat in all_features:
        if feat in df.columns:
            # Use only rows with valid data
            valid_mask = df[feat].notna() & df['TotalCorners'].notna()
            if valid_mask.sum() > 100:  # Need sufficient data
                corr = df.loc[valid_mask, 'TotalCorners'].corr(df.loc[valid_mask, feat])
                correlations[feat] = corr

    # Create correlation dataframe
    corr_df = pd.DataFrame([
        {'Feature': k, 'Correlation': v, 'AbsCorr': abs(v)}
        for k, v in correlations.items()
    ]).sort_values('AbsCorr', ascending=False)

    return correlations, corr_df


def print_results_by_category(correlations: Dict[str, float]):
    """Print correlation results organized by category"""

    feature_categories = get_all_feature_columns()

    print("\n" + "="*70)
    print("FEATURE CORRELATIONS WITH TOTAL CORNERS (by Category)")
    print("="*70)

    category_summaries = []

    for category, features in feature_categories.items():
        print(f"\n{category}")
        print("-"*70)
        print(f"{'Feature':<30} {'Correlation':>12} {'Strength':<12}")
        print("-"*70)

        cat_corrs = []
        for feat in features:
            if feat in correlations:
                corr = correlations[feat]
                cat_corrs.append(abs(corr))

                if abs(corr) >= 0.15:
                    strength = "GOOD"
                elif abs(corr) >= 0.08:
                    strength = "MODERATE"
                elif abs(corr) >= 0.03:
                    strength = "WEAK"
                else:
                    strength = "NEGLIGIBLE"

                print(f"{feat:<30} {corr:>12.4f} {strength:<12}")

        if cat_corrs:
            avg_corr = np.mean(cat_corrs)
            max_corr = np.max(cat_corrs)
            category_summaries.append((category, avg_corr, max_corr))

    # Category summary
    print("\n" + "="*70)
    print("CATEGORY SUMMARY")
    print("="*70)
    print(f"{'Category':<35} {'Avg |r|':>10} {'Max |r|':>10}")
    print("-"*70)
    for cat, avg, mx in sorted(category_summaries, key=lambda x: -x[2]):
        print(f"{cat:<35} {avg:>10.4f} {mx:>10.4f}")


def plot_results(df: pd.DataFrame, correlations: Dict[str, float]):
    """Create visualization of correlations"""

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Top 20 features bar chart
    top_20 = sorted_corr[:20]
    features = [f[0] for f in top_20]
    values = [f[1] for f in top_20]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(features)), values, color=colors, alpha=0.8)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Correlation with TotalCorners')
    plt.title('Top 20 Features by Correlation with Total Corners')
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.axvline(x=0.15, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-0.15, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('all_features_correlation.png', dpi=150)
    print("\nSaved: all_features_correlation.png")

    # Correlation matrix for top features
    top_features = [f[0] for f in sorted_corr[:15]]
    plot_cols = ['TotalCorners'] + top_features
    available_cols = [c for c in plot_cols if c in df.columns]

    # Filter to valid rows
    valid_df = df[available_cols].dropna()

    if len(valid_df) > 100:
        corr_matrix = valid_df.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            annot_kws={'size': 8}
        )
        plt.title('Correlation Matrix: Top Features', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_correlation_matrix.png', dpi=150)
        print("Saved: feature_correlation_matrix.png")


def print_final_summary(correlations: Dict[str, float]):
    """Print final summary and recommendations"""

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Categorize features by correlation strength
    strong = [(f, c) for f, c in correlations.items() if abs(c) >= 0.15]
    moderate = [(f, c) for f, c in correlations.items() if 0.08 <= abs(c) < 0.15]
    weak = [(f, c) for f, c in correlations.items() if abs(c) < 0.08]

    print(f"\nSTRONG FEATURES (|r| >= 0.15): {len(strong)}")
    for f, c in sorted(strong, key=lambda x: -abs(x[1])):
        print(f"  {f:<35} {c:>8.4f}")

    print(f"\nMODERATE FEATURES (0.08 <= |r| < 0.15): {len(moderate)}")
    for f, c in sorted(moderate, key=lambda x: -abs(x[1])):
        print(f"  {f:<35} {c:>8.4f}")

    print(f"\nWEAK FEATURES (|r| < 0.08): {len(weak)}")
    for f, c in sorted(weak, key=lambda x: -abs(x[1]))[:10]:
        print(f"  {f:<35} {c:>8.4f}")
    if len(weak) > 10:
        print(f"  ... and {len(weak) - 10} more")

    print("\n" + "="*70)
    print(f"Total features analyzed: {len(correlations)}")
    print(f"Features with |r| >= 0.10: {len([c for c in correlations.values() if abs(c) >= 0.10])}")
    print("="*70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data()
    print(f"\nTotal matches loaded: {len(df)}")

    # Compute rolling statistics
    df = compute_team_rolling_stats(df, n=N)

    # Compute match-level features
    df = compute_match_level_features(df)

    # Count valid rows (with all features computed)
    feature_cols = []
    for features in get_all_feature_columns().values():
        feature_cols.extend(features)

    valid_mask = df[feature_cols].notna().all(axis=1)
    print(f"\nMatches with complete features: {valid_mask.sum()} / {len(df)}")

    # Analyze correlations
    correlations, corr_df = analyze_correlations(df)

    # Print results
    print_results_by_category(correlations)

    # Create visualizations
    plot_results(df, correlations)

    # Final summary
    print_final_summary(correlations)

    # Save correlation data
    corr_df.to_csv('feature_correlations.csv', index=False)
    print("\nSaved: feature_correlations.csv")
