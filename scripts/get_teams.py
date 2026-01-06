"""
Get current 25-26 Premier League team list from football-data.co.uk
"""
import pandas as pd

url = 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
df = pd.read_csv(url)
teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

print("2025-26 Premier League Teams:")
print("-" * 30)
for team in teams:
    print(f"  {team}")
print(f"\nTotal: {len(teams)} teams")
print(f"\nAs Python list:")
print(teams)
