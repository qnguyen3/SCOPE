"""
SCOPE - Feature Engineering Module
Extracted from train_all_classifiers.py
"""
import pandas as pd
import numpy as np

# Feature list used by the models
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


def compute_features(df, n=5):
    """
    Compute rolling window features for corner prediction.

    Args:
        df: DataFrame with columns Date, HomeTeam, AwayTeam, HC, AC, HS, AS, HST, AST, FTHG, FTAG, HF, AF
        n: Rolling window size (number of previous home/away games)

    Returns:
        DataFrame with computed features
    """
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


def compute_features_for_match(df, home_team, away_team, window=5):
    """
    Compute features for a specific upcoming match.

    Args:
        df: Full DataFrame with historical data
        home_team: Home team name
        away_team: Away team name
        window: Rolling window size

    Returns:
        dict: Feature values for the match
    """
    # Get last N home games for home team
    home_games = df[df['HomeTeam'] == home_team].tail(window)
    # Get last N away games for away team
    away_games = df[df['AwayTeam'] == away_team].tail(window)

    features = {}

    if len(home_games) >= window:
        features['home_corners_for'] = home_games['HC'].mean()
        features['home_corners_against'] = home_games['AC'].mean()
        features['home_corners_total'] = (home_games['HC'] + home_games['AC']).mean()
        features['home_corner_std'] = home_games['HC'].std()
        features['home_shots'] = home_games['HS'].mean() if 'HS' in df.columns else 0
        features['home_sot'] = home_games['HST'].mean() if 'HST' in df.columns else 0
        features['home_goals'] = home_games['FTHG'].mean() if 'FTHG' in df.columns else 0
        total = home_games['HC'] + home_games['AC']
        features['home_over_9'] = (total > 9.5).mean()
        features['home_over_10'] = (total > 10.5).mean()
        features['home_over_11'] = (total > 11.5).mean()
        if len(home_games) >= 3:
            features['home_corner_trend'] = home_games.tail(3)['HC'].mean() - home_games.head(window-3)['HC'].mean()
        else:
            features['home_corner_trend'] = 0
    else:
        # Default values if not enough data
        features['home_corners_for'] = 5.0
        features['home_corners_against'] = 5.0
        features['home_corners_total'] = 10.0
        features['home_corner_std'] = 1.5
        features['home_shots'] = 12.0
        features['home_sot'] = 4.0
        features['home_goals'] = 1.5
        features['home_over_9'] = 0.5
        features['home_over_10'] = 0.4
        features['home_over_11'] = 0.3
        features['home_corner_trend'] = 0

    if len(away_games) >= window:
        features['away_corners_for'] = away_games['AC'].mean()
        features['away_corners_against'] = away_games['HC'].mean()
        features['away_corners_total'] = (away_games['HC'] + away_games['AC']).mean()
        features['away_corner_std'] = away_games['AC'].std()
        features['away_shots'] = away_games['AS'].mean() if 'AS' in df.columns else 0
        features['away_sot'] = away_games['AST'].mean() if 'AST' in df.columns else 0
        features['away_goals'] = away_games['FTAG'].mean() if 'FTAG' in df.columns else 0
        total = away_games['HC'] + away_games['AC']
        features['away_over_9'] = (total > 9.5).mean()
        features['away_over_10'] = (total > 10.5).mean()
        features['away_over_11'] = (total > 11.5).mean()
        if len(away_games) >= 3:
            features['away_corner_trend'] = away_games.tail(3)['AC'].mean() - away_games.head(window-3)['AC'].mean()
        else:
            features['away_corner_trend'] = 0
    else:
        # Default values if not enough data
        features['away_corners_for'] = 4.0
        features['away_corners_against'] = 5.0
        features['away_corners_total'] = 10.0
        features['away_corner_std'] = 1.5
        features['away_shots'] = 10.0
        features['away_sot'] = 3.0
        features['away_goals'] = 1.2
        features['away_over_9'] = 0.5
        features['away_over_10'] = 0.4
        features['away_over_11'] = 0.3
        features['away_corner_trend'] = 0

    # Combined features
    features['expected_corners'] = (features['home_corners_total'] + features['away_corners_total']) / 2
    features['combined_corners_for'] = features['home_corners_for'] + features['away_corners_for']
    features['corner_diff'] = features['home_corners_for'] - features['away_corners_for']
    features['combined_volatility'] = features['home_corner_std'] + features['away_corner_std']
    features['combined_shots'] = features['home_shots'] + features['away_shots']
    features['combined_goals'] = features['home_goals'] + features['away_goals']
    features['combined_trend'] = features['home_corner_trend'] + features['away_corner_trend']
    features['combined_over_9'] = (features['home_over_9'] + features['away_over_9']) / 2
    features['combined_over_10'] = (features['home_over_10'] + features['away_over_10']) / 2
    features['combined_over_11'] = (features['home_over_11'] + features['away_over_11']) / 2

    return features


def get_team_statistics(df, team, n=5):
    """
    Get recent statistics for a team.

    Args:
        df: DataFrame with match data
        team: Team name
        n: Number of recent matches

    Returns:
        dict: Team statistics
    """
    # Home matches
    home = df[df['HomeTeam'] == team].tail(n)
    # Away matches
    away = df[df['AwayTeam'] == team].tail(n)

    # All recent matches (sorted by date)
    all_matches = pd.concat([
        home[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HS', 'AS']].assign(venue='Home'),
        away[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HS', 'AS']].assign(venue='Away')
    ]).sort_values('Date').tail(n)

    # Calculate corners for/against
    corners_for = []
    corners_against = []
    for _, row in all_matches.iterrows():
        if row['venue'] == 'Home':
            corners_for.append(row['HC'])
            corners_against.append(row['AC'])
        else:
            corners_for.append(row['AC'])
            corners_against.append(row['HC'])

    # Recent form - last 5 matches
    recent_matches = []
    for _, row in all_matches.iterrows():
        if row['venue'] == 'Home':
            gf, ga = row['FTHG'], row['FTAG']
            opponent = row['AwayTeam']
            cf, ca = row['HC'], row['AC']
        else:
            gf, ga = row['FTAG'], row['FTHG']
            opponent = row['HomeTeam']
            cf, ca = row['AC'], row['HC']

        result = 'W' if gf > ga else ('D' if gf == ga else 'L')
        recent_matches.append({
            'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': opponent,
            'venue': row['venue'],
            'result': result,
            'score': f"{int(gf)}-{int(ga)}",
            'corners_for': int(cf),
            'corners_against': int(ca),
            'total_corners': int(cf + ca)
        })

    # Calculate over rates
    total_corners = [m['total_corners'] for m in recent_matches]

    return {
        'recent_matches': recent_matches,
        'avg_corners_for': np.mean(corners_for) if corners_for else 0,
        'avg_corners_against': np.mean(corners_against) if corners_against else 0,
        'avg_total_corners': np.mean(total_corners) if total_corners else 0,
        'over_rates': {
            '8.5': sum(1 for c in total_corners if c > 8.5) / len(total_corners) if total_corners else 0,
            '9.5': sum(1 for c in total_corners if c > 9.5) / len(total_corners) if total_corners else 0,
            '10.5': sum(1 for c in total_corners if c > 10.5) / len(total_corners) if total_corners else 0,
            '11.5': sum(1 for c in total_corners if c > 11.5) / len(total_corners) if total_corners else 0,
            '12.5': sum(1 for c in total_corners if c > 12.5) / len(total_corners) if total_corners else 0,
        }
    }


def get_head_to_head(df, home_team, away_team, n=5):
    """
    Get head-to-head history between two teams.

    Args:
        df: DataFrame with match data
        home_team: Home team name
        away_team: Away team name
        n: Number of recent H2H matches

    Returns:
        dict: H2H statistics
    """
    # Get matches between these two teams
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].tail(n)

    if len(h2h) == 0:
        return {
            'matches': [],
            'avg_total_corners': 0,
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0
        }

    matches = []
    home_wins = 0
    away_wins = 0
    draws = 0
    total_corners = []

    for _, row in h2h.iterrows():
        total = row['HC'] + row['AC']
        total_corners.append(total)

        if row['FTHG'] > row['FTAG']:
            result = row['HomeTeam']
            if row['HomeTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        elif row['FTHG'] < row['FTAG']:
            result = row['AwayTeam']
            if row['AwayTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        else:
            result = 'Draw'
            draws += 1

        matches.append({
            'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'home': row['HomeTeam'],
            'away': row['AwayTeam'],
            'score': f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            'corners': f"{int(row['HC'])}-{int(row['AC'])}",
            'total_corners': int(total),
            'winner': result
        })

    return {
        'matches': matches,
        'avg_total_corners': np.mean(total_corners),
        'home_team_wins': home_wins,
        'away_team_wins': away_wins,
        'draws': draws
    }
