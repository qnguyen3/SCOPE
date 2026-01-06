"""
SCOPE - Classifier Analysis for Profitability
Analyzes trained classifiers to determine real betting edge
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Find most recent classifier files
classifier_files = sorted(glob.glob('classifier_ou*.pkl'))
summary_files = sorted(glob.glob('classifier_summary_*.csv'))

if not classifier_files:
    print("ERROR: No classifier files found. Run train_all_classifiers.py first.")
    exit()

print("="*70)
print("SCOPE CLASSIFIER PROFITABILITY ANALYSIS")
print("="*70)

# =============================================================================
# LOAD LATEST SUMMARY
# =============================================================================
if summary_files:
    latest_summary = summary_files[-1]
    results_df = pd.read_csv(latest_summary)
    print(f"\nLoaded: {latest_summary}")
else:
    print("No summary file found")
    results_df = None

# =============================================================================
# PROFITABILITY ANALYSIS - ALL CONFIGURATIONS
# =============================================================================
print("\n" + "="*70)
print("ALL CONFIGURATIONS WITH POSITIVE EDGE (Min 20 bets)")
print("="*70)

MIN_BETS_THRESHOLD = 20

if results_df is not None:
    # Show all positive edge configurations with sufficient bets
    positive_edge = results_df[(results_df['RealEdge'] > 0) & (results_df['Bets'] >= MIN_BETS_THRESHOLD)]
    positive_edge = positive_edge.sort_values(['Threshold', 'ROI'], ascending=[True, False])

    print(f"\n{'Thresh':<8} {'Window':<8} {'Conf':<6} {'Bets':<6} {'Win%':<8} {'Edge%':<8} {'ROI%':<8}")
    print("-" * 60)

    for _, row in positive_edge.iterrows():
        status = "✅" if row['RealEdge'] > 2 else "⚠️"
        print(f"O/U {row['Threshold']:<4} N={int(row['Window']):<5} >{row['Confidence']:<5} {int(row['Bets']):<6} {row['WinRate']:<8.1f} {row['RealEdge']:>+6.1f}% {row['ROI']:>+6.1f}% {status}")

    if len(positive_edge) == 0:
        print("No configurations with positive edge and sufficient bets found.")

# =============================================================================
# BEST BY BET VOLUME
# =============================================================================
print("\n" + "="*70)
print("RECOMMENDED CONFIGURATIONS (Balance of edge + volume)")
print("="*70)

if results_df is not None:
    print("\n--- Best with 50+ bets ---")
    high_vol = results_df[(results_df['RealEdge'] > 0) & (results_df['Bets'] >= 50)]
    if len(high_vol) > 0:
        for threshold in sorted(high_vol['Threshold'].unique()):
            thresh_data = high_vol[high_vol['Threshold'] == threshold]
            best = thresh_data.loc[thresh_data['ROI'].idxmax()]
            print(f"O/U {threshold}: Window={int(best['Window'])}, Conf>{best['Confidence']}")
            print(f"  {int(best['Bets'])} bets, {best['WinRate']:.1f}% win, {best['RealEdge']:+.1f}% edge, {best['ROI']:+.1f}% ROI")
    else:
        print("No configurations with 50+ bets and positive edge")

    print("\n--- Best with 100+ bets ---")
    very_high_vol = results_df[(results_df['RealEdge'] > 0) & (results_df['Bets'] >= 100)]
    if len(very_high_vol) > 0:
        for threshold in sorted(very_high_vol['Threshold'].unique()):
            thresh_data = very_high_vol[very_high_vol['Threshold'] == threshold]
            best = thresh_data.loc[thresh_data['ROI'].idxmax()]
            print(f"O/U {threshold}: Window={int(best['Window'])}, Conf>{best['Confidence']}")
            print(f"  {int(best['Bets'])} bets, {best['WinRate']:.1f}% win, {best['RealEdge']:+.1f}% edge, {best['ROI']:+.1f}% ROI")
    else:
        print("No configurations with 100+ bets and positive edge")

# Standard betting odds scenarios
print("\n" + "="*70)
print("ROI AT DIFFERENT ODDS (Best configs)")
print("="*70)

ODDS_SCENARIOS = {
    'Best (-105)': 1.95,
    'Standard (-110)': 1.91,
    'Worse (-115)': 1.87,
}

if results_df is not None:
    print(f"\n{'Config':<30} ", end="")
    for name in ODDS_SCENARIOS.keys():
        print(f"{name:<15}", end="")
    print()
    print("-" * 75)

    # Show best high-volume configs
    for threshold in sorted(results_df['Threshold'].unique()):
        high_vol = results_df[(results_df['Threshold'] == threshold) & (results_df['Bets'] >= 50) & (results_df['RealEdge'] > 0)]
        if len(high_vol) > 0:
            best = high_vol.loc[high_vol['ROI'].idxmax()]
            config = f"O/U {threshold} N={int(best['Window'])} >{best['Confidence']}"
            print(f"{config:<30} ", end="")

            for name, odds in ODDS_SCENARIOS.items():
                wins = best['Wins']
                total = best['Bets']
                losses = total - wins
                profit = wins * (odds - 1) - losses
                roi = profit / total * 100
                status = "✅" if roi > 0 else "❌"
                print(f"{roi:>+6.1f}% {status}    ", end="")
            print()

# =============================================================================
# EDGE ANALYSIS (High-volume configs only)
# =============================================================================
print("\n" + "="*70)
print("EDGE ANALYSIS - Model vs Base Rate (50+ bets)")
print("="*70)

print("""
Real edge = Model accuracy - max(base_rate, 1-base_rate)
Only showing configs with 50+ bets for statistical reliability.
""")

if results_df is not None:
    base_rates = {
        8.5: 0.658,   # 65.8% Over
        9.5: 0.553,   # 55.3% Over (from test set: 54.5%)
        10.5: 0.426,  # 42.6% Over (from test set: 42.0%)
        11.5: 0.279,  # 27.9% Over
        12.5: 0.195,  # 19.5% Over
    }

    print(f"{'Config':<25} {'Win%':<8} {'Naive%':<8} {'Edge':<10} {'Bets':<8}")
    print("-" * 65)

    for threshold in sorted(results_df['Threshold'].unique()):
        thresh_data = results_df[(results_df['Threshold'] == threshold) & (results_df['Bets'] >= 50) & (results_df['RealEdge'] > 0)]
        if len(thresh_data) > 0:
            best = thresh_data.loc[thresh_data['ROI'].idxmax()]
            base = base_rates.get(threshold, 0.5)
            naive_win = max(base, 1-base) * 100

            config = f"O/U {threshold} N={int(best['Window'])} >{best['Confidence']}"
            status = "✅" if best['RealEdge'] > 2 else "⚠️"
            print(f"{config:<25} {best['WinRate']:<8.1f} {naive_win:<8.1f} {best['RealEdge']:>+6.1f}%   {int(best['Bets']):<8} {status}")

# =============================================================================
# KELLY CRITERION (High-volume configs)
# =============================================================================
print("\n" + "="*70)
print("KELLY CRITERION - Optimal Bet Sizing (50+ bets)")
print("="*70)

print("""
Kelly formula: f* = (bp - q) / b
Where: b = odds-1, p = win prob, q = 1-p

Use fractional Kelly (1/4 to 1/2) to reduce variance.
""")

if results_df is not None:
    odds = 1.91  # Standard -110
    b = odds - 1

    print(f"{'Config':<25} {'Win%':<8} {'Kelly':<10} {'1/4 Kelly':<12} {'Rec':<20}")
    print("-" * 80)

    for threshold in sorted(results_df['Threshold'].unique()):
        thresh_data = results_df[(results_df['Threshold'] == threshold) & (results_df['Bets'] >= 50) & (results_df['RealEdge'] > 0)]
        if len(thresh_data) > 0:
            best = thresh_data.loc[thresh_data['ROI'].idxmax()]

            p = best['WinRate'] / 100
            q = 1 - p
            kelly = (b * p - q) / b * 100
            quarter_kelly = kelly / 4

            if kelly <= 0:
                rec = "❌ Don't bet"
            elif kelly < 5:
                rec = "⚠️ Bet 1%"
            elif kelly < 10:
                rec = "✅ Bet 1-2%"
            else:
                rec = "✅ Bet 2-3%"

            config = f"O/U {threshold} N={int(best['Window'])} >{best['Confidence']}"
            print(f"{config:<25} {best['WinRate']:<8.1f} {kelly:<10.1f} {quarter_kelly:<12.1f} {rec}")

# =============================================================================
# RISK ANALYSIS (High-volume configs)
# =============================================================================
print("\n" + "="*70)
print("RISK ANALYSIS (50+ bets)")
print("="*70)

if results_df is not None:
    print(f"\n{'Config':<25} {'Bets':<6} {'E[P/L]':<10} {'Worst*':<10} {'Sharpe':<8}")
    print("-" * 65)
    print("*Worst case = 2 std deviations below mean (95% CI)\n")

    for threshold in sorted(results_df['Threshold'].unique()):
        thresh_data = results_df[(results_df['Threshold'] == threshold) & (results_df['Bets'] >= 50) & (results_df['RealEdge'] > 0)]
        if len(thresh_data) > 0:
            best = thresh_data.loc[thresh_data['ROI'].idxmax()]

            n = best['Bets']
            p = best['WinRate'] / 100

            # Expected value per bet (1 unit stake)
            ev = p * 0.91 - (1-p) * 1  # Win 0.91, lose 1

            # Variance per bet
            var = p * (0.91 - ev)**2 + (1-p) * (-1 - ev)**2
            std = np.sqrt(var)

            # For n bets
            total_ev = ev * n
            total_std = std * np.sqrt(n)
            worst_case = total_ev - 2 * total_std

            # Sharpe-like ratio
            sharpe = (ev / std) if std > 0 else 0

            config = f"O/U {threshold} N={int(best['Window'])} >{best['Confidence']}"
            print(f"{config:<25} {int(n):<6} ${total_ev:<+9.1f} ${worst_case:<+9.1f} {sharpe:<8.2f}")

# =============================================================================
# FINAL RECOMMENDATION
# =============================================================================
print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)

if results_df is not None:
    # Only consider high-volume configs with positive edge
    viable = results_df[(results_df['Bets'] >= 50) & (results_df['RealEdge'] > 0)].copy()

    if len(viable) > 0:
        # Score: edge * 2 + roi * 0.3 + log(bets) * 3
        viable['Score'] = viable['RealEdge'] * 2 + viable['ROI'] * 0.3 + np.log1p(viable['Bets']) * 3
        viable = viable.sort_values('Score', ascending=False)

        print("\nTop configurations (by score = edge + ROI + volume):\n")
        for i, (_, row) in enumerate(viable.head(5).iterrows(), 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            config = f"O/U {row['Threshold']} N={int(row['Window'])} >{row['Confidence']}"
            print(f"{status} #{i} {config}")
            print(f"      {int(row['Bets'])} bets | {row['WinRate']:.1f}% win | {row['RealEdge']:+.1f}% edge | {row['ROI']:+.1f}% ROI")

        best = viable.iloc[0]
        print(f"\n{'='*70}")
        print(f"RECOMMENDED: O/U {best['Threshold']} with N={int(best['Window'])}, Confidence > {best['Confidence']}")
        print(f"Expected: {int(best['Bets'])} bets/season, {best['WinRate']:.0f}% win rate, {best['ROI']:+.0f}% ROI")
        print(f"{'='*70}")
    else:
        print("\nNo configurations with positive edge and 50+ bets found.")
        print("Model needs more tuning.")
