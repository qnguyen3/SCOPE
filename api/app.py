"""
EPL SCOPE - Flask REST API
Premier League Corner Prediction API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import glob
import requests
import os
from dotenv import load_dotenv
from features import FEATURES, compute_features_for_match, get_team_statistics, get_head_to_head

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": ["https://corner.qnguyen3.dev", "http://localhost:3000", "http://localhost:5173"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

# Configuration
DATA_URL = 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
HISTORICAL_SEASONS = ['2324', '2425', '2526']  # Last 3 seasons for H2H
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Model configurations (threshold -> window, confidence)
MODEL_CONFIGS = {
    8.5: {'window': 5, 'confidence': 0.58, 'has_real_edge': False},
    9.5: {'window': 5, 'confidence': 0.58, 'has_real_edge': True},
    10.5: {'window': 10, 'confidence': 0.54, 'has_real_edge': True},
    11.5: {'window': 10, 'confidence': 0.54, 'has_real_edge': False},
    12.5: {'window': 10, 'confidence': 0.54, 'has_real_edge': False},
}

# Load models at startup
models = {}

# URL for downloading models zip (set via environment variable or use default)
MODELS_ZIP_URL = os.environ.get(
    'MODELS_ZIP_URL',
    'https://huggingface.co/qnguyen3/epl-corners-2526-v1/resolve/main/models.zip'
)


def download_and_extract_models():
    """Download and extract models zip from external storage if not present locally."""
    import zipfile
    from io import BytesIO

    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Check if models already exist
    if os.path.exists(model_dir) and any(f.endswith('.pkl') for f in os.listdir(model_dir)):
        print("Models already present, skipping download")
        return

    if not MODELS_ZIP_URL:
        print("MODELS_ZIP_URL not set, skipping model download")
        return

    print(f"Downloading models from {MODELS_ZIP_URL}...")
    try:
        response = requests.get(MODELS_ZIP_URL, timeout=300)
        response.raise_for_status()
        print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

        # Extract zip
        os.makedirs(model_dir, exist_ok=True)
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(model_dir)
        print(f"Extracted models to {model_dir}")
    except Exception as e:
        print(f"Failed to download/extract models: {e}")


def load_models():
    """Load all classifier models from the models directory."""
    global models
    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Try to download models if not present
    download_and_extract_models()

    if not os.path.exists(model_dir):
        print(f"Warning: Models directory not found at {model_dir}")
        return

    for f in glob.glob(os.path.join(model_dir, '*.pkl')):
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                threshold = data['threshold']
                models[threshold] = data
                print(f"Loaded model for O/U {threshold}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

# Load models on startup
load_models()


def fetch_data():
    """Fetch fresh data from football-data.co.uk (current season only)."""
    try:
        df = pd.read_csv(DATA_URL, encoding='utf-8', on_bad_lines='skip')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HC', 'AC'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['TotalCorners'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def fetch_historical_data():
    """Fetch multiple seasons for H2H analysis."""
    try:
        dfs = []
        for season in HISTORICAL_SEASONS:
            url = f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
            try:
                df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
                df['Season'] = season
                dfs.append(df)
            except:
                pass
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HC', 'AC'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['TotalCorners'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def run_predictions(df, home_team, away_team):
    """Run predictions for all thresholds."""
    predictions = {}

    for threshold, config in MODEL_CONFIGS.items():
        window = config['window']
        confidence = config['confidence']
        has_real_edge = config['has_real_edge']

        # Get features for this match
        features = compute_features_for_match(df, home_team, away_team, window=window)

        # Check if model is loaded
        if threshold not in models:
            predictions[str(threshold)] = {
                'prob_over': 0.5,
                'prob_under': 0.5,
                'recommendation': 'NO BET',
                'has_edge': False,
                'confidence_threshold': confidence,
                'error': 'Model not loaded'
            }
            continue

        model_data = models[threshold]
        model = model_data['model']
        feature_list = model_data['features']

        # Prepare feature vector
        X = pd.DataFrame([features])[feature_list]

        # Make prediction
        try:
            prob_over = model.predict_proba(X)[0][1]
            prob_under = 1 - prob_over

            # Determine recommendation
            if prob_over > confidence:
                recommendation = 'OVER'
                has_edge = has_real_edge
            elif prob_under > confidence:
                recommendation = 'UNDER'
                has_edge = has_real_edge
            else:
                recommendation = 'NO BET'
                has_edge = False

            predictions[str(threshold)] = {
                'prob_over': round(float(prob_over), 4),
                'prob_under': round(float(prob_under), 4),
                'recommendation': recommendation,
                'has_edge': has_edge,
                'confidence_threshold': confidence,
                'has_real_edge': has_real_edge
            }
        except Exception as e:
            predictions[str(threshold)] = {
                'prob_over': 0.5,
                'prob_under': 0.5,
                'recommendation': 'NO BET',
                'has_edge': False,
                'confidence_threshold': confidence,
                'error': str(e)
            }

    return predictions


def get_llm_assessment(home_team, away_team, predictions, statistics):
    """Get LLM assessment from OpenRouter API."""
    if not OPENROUTER_API_KEY:
        return "LLM assessment not available (API key not configured)"

    # Build prompt
    prompt = f"""You are a sports betting analyst. Analyze this Premier League match and provide betting recommendations.

## Match: {home_team} vs {away_team}

## Model Predictions (Corner O/U)

| Threshold | P(Over) | P(Under) | Recommendation | Has Real Edge |
|-----------|---------|----------|----------------|---------------|
"""
    for threshold in ['8.5', '9.5', '10.5', '11.5', '12.5']:
        p = predictions.get(threshold, {})
        prompt += f"| O/U {threshold} | {p.get('prob_over', 0):.1%} | {p.get('prob_under', 0):.1%} | {p.get('recommendation', 'N/A')} | {'Yes' if p.get('has_edge') else 'No'} |\n"

    # Add team statistics
    home_stats = statistics.get('home_team', {})
    away_stats = statistics.get('away_team', {})

    prompt += f"""

## {home_team} Recent Form (Home Games)
- Avg Corners For: {home_stats.get('avg_corners_for', 0):.1f}
- Avg Corners Against: {home_stats.get('avg_corners_against', 0):.1f}
- Avg Total Corners: {home_stats.get('avg_total_corners', 0):.1f}
- Over 9.5 Rate: {home_stats.get('over_rates', {}).get('9.5', 0):.0%}
- Over 10.5 Rate: {home_stats.get('over_rates', {}).get('10.5', 0):.0%}

## {away_team} Recent Form (Away Games)
- Avg Corners For: {away_stats.get('avg_corners_for', 0):.1f}
- Avg Corners Against: {away_stats.get('avg_corners_against', 0):.1f}
- Avg Total Corners: {away_stats.get('avg_total_corners', 0):.1f}
- Over 9.5 Rate: {away_stats.get('over_rates', {}).get('9.5', 0):.0%}
- Over 10.5 Rate: {away_stats.get('over_rates', {}).get('10.5', 0):.0%}

## Head-to-Head
"""
    h2h = statistics.get('head_to_head', {})
    if h2h.get('matches'):
        prompt += f"- Avg Total Corners in H2H: {h2h.get('avg_total_corners', 0):.1f}\n"
        prompt += f"- {home_team} Wins: {h2h.get('home_team_wins', 0)}, {away_team} Wins: {h2h.get('away_team_wins', 0)}, Draws: {h2h.get('draws', 0)}\n"
    else:
        prompt += "- No recent head-to-head data available\n"

    prompt += """

## Important Notes
- Only O/U 9.5 and O/U 10.5 have statistically significant predictive edge
- Other thresholds (8.5, 11.5, 12.5) mostly follow base rates
- Standard odds are around 1.91 (-110), requiring 52.4% win rate to break even

---

## INSTRUCTIONS

You MUST follow this EXACT output format. Do not deviate from this structure.

---

## OUTPUT FORMAT (Follow Exactly)

### 🎯 PRIMARY RECOMMENDATION

**Bet:** [OVER/UNDER] [threshold] corners
**Confidence:** [HIGH/MEDIUM/LOW]
**Edge:** [Yes - model shows edge / No - no statistical edge]

### 📊 MATCH ANALYSIS

[2-3 sentences analyzing key factors: team form, playing styles, tactical tendencies that affect corner count]

### 📈 MODEL INSIGHTS

| Threshold | Signal | Strength |
|-----------|--------|----------|
| O/U 8.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 9.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 10.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 11.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 12.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |

### ⚠️ RISK FACTORS

- [Risk factor 1]
- [Risk factor 2]
- [Risk factor 3 if applicable]

### 🔄 ALTERNATIVE BETS

1. **[Bet 1]:** [Brief reasoning]
2. **[Bet 2]:** [Brief reasoning]

### 💡 VERDICT

[1-2 sentence final summary with clear action recommendation]

---

IMPORTANT: Follow the exact format above. Use the exact headers with emojis. Keep each section concise. Do not add extra sections."""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://epl-scope.vercel.app',
                'X-Title': 'EPL SCOPE'
            },
            json={
                'model': 'google/gemini-3-flash-preview',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM assessment failed: {str(e)}"


def prepare_charts_data(df, home_team, away_team, predictions):
    """Prepare data for frontend charts."""
    # Home team recent corner history
    home_matches = df[df['HomeTeam'] == home_team].tail(10)
    home_corner_history = []
    for _, row in home_matches.iterrows():
        home_corner_history.append({
            'date': row['Date'].strftime('%m/%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': row['AwayTeam'],
            'corners_for': int(row['HC']),
            'corners_against': int(row['AC']),
            'total': int(row['HC'] + row['AC'])
        })

    # Away team recent corner history
    away_matches = df[df['AwayTeam'] == away_team].tail(10)
    away_corner_history = []
    for _, row in away_matches.iterrows():
        away_corner_history.append({
            'date': row['Date'].strftime('%m/%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': row['HomeTeam'],
            'corners_for': int(row['AC']),
            'corners_against': int(row['HC']),
            'total': int(row['HC'] + row['AC'])
        })

    # Probability by threshold for line chart
    prob_by_threshold = []
    for threshold in ['8.5', '9.5', '10.5', '11.5', '12.5']:
        p = predictions.get(threshold, {})
        prob_by_threshold.append({
            'threshold': threshold,
            'prob_over': p.get('prob_over', 0.5),
            'prob_under': p.get('prob_under', 0.5),
            'has_edge': p.get('has_edge', False)
        })

    return {
        'home_corner_history': home_corner_history,
        'away_corner_history': away_corner_history,
        'probability_by_threshold': prob_by_threshold
    }


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'thresholds': list(models.keys())
    })


@app.route('/api/teams')
def get_teams():
    """Get list of current season teams."""
    try:
        df = fetch_data()
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 500

        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        return jsonify({'teams': teams})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run predictions for a match."""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')

        if not home_team or not away_team:
            return jsonify({'error': 'home_team and away_team are required'}), 400

        if home_team == away_team:
            return jsonify({'error': 'Home and away teams must be different'}), 400

        # Fetch fresh data
        df = fetch_data()
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 500

        # Validate teams
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        if home_team not in all_teams:
            return jsonify({'error': f'Unknown team: {home_team}'}), 400
        if away_team not in all_teams:
            return jsonify({'error': f'Unknown team: {away_team}'}), 400

        # Run predictions
        predictions = run_predictions(df, home_team, away_team)

        # Fetch historical data for H2H (last 3 seasons)
        df_historical = fetch_historical_data()

        # Get statistics (use current season for team stats, historical for H2H)
        statistics = {
            'home_team': get_team_statistics(df, home_team),
            'away_team': get_team_statistics(df, away_team),
            'head_to_head': get_head_to_head(df_historical if df_historical is not None else df, home_team, away_team)
        }

        # Get LLM assessment
        llm_assessment = get_llm_assessment(home_team, away_team, predictions, statistics)

        # Prepare charts data
        charts_data = prepare_charts_data(df, home_team, away_team, predictions)

        return jsonify({
            'match': {
                'home_team': home_team,
                'away_team': away_team
            },
            'predictions': predictions,
            'statistics': statistics,
            'llm_assessment': llm_assessment,
            'charts_data': charts_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
