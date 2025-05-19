import os
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


# Add a User-Agent header to avoid 403 errors
HEADERS = {
    'User-Agent': 'ChessDataAnalysisBot/1.0 (+https://example.com)'
}

# 1. Fetch first 1000 British players
COUNTRY_CODE = 'GB'
PLAYERS_URL = f'https://api.chess.com/pub/country/{COUNTRY_CODE}/players'
players = []
page = 1

while len(players) < 1000:
    try:
        resp = requests.get(PLAYERS_URL, params={'page': page}, headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch page {page}: {e}")
        break
    data = resp.json()
    batch = data.get('players', [])
    if not batch:
        break
    players.extend(batch)
    page += 1
    time.sleep(0.5)

players = players[:1000]
print(f"Fetched {len(players)} players")

# 2. Prepare container for Elo history and accuracy data
records = []

for username in tqdm(players, desc="Processing players"):
    archives_url = f'https://api.chess.com/pub/player/{username}/games/archives'
    try:
        resp = requests.get(archives_url, headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        continue
    archives = resp.json().get('archives', [])
    for archive_url in archives:
        try:
            resp2 = requests.get(archive_url, headers=HEADERS)
            resp2.raise_for_status()
        except requests.exceptions.HTTPError:
            continue
        games = resp2.json().get('games', [])
        for game in games:
            timestamp = game.get('end_time', 0)
            date = datetime.utcfromtimestamp(timestamp)
            for side in ['white', 'black']:
                player = game.get(side, {})
                if player.get('username', '').lower() != username.lower():
                    continue
                rating = player.get('rating')
                accuracy = game.get('accuracies', {}).get(side)
                records.append({
                    'username': username,
                    'date': date,
                    'rating': rating,
                    'accuracy': accuracy
                })
        time.sleep(0.2)
    time.sleep(0.5)

# 3. Build DataFrame

df = pd.DataFrame(records)



#save all records to a CSV file
csv_file_path = "~/Library/CloudStorage/OneDrive-Nexus365/Documents/Github/Github_new/chess_performance/Data/random_1k_GB.csv"

# Use the to_csv method to write the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)  # Set index=False to avoid writing row numbers as a column

# Filter out players with no ratings
df.dropna(subset=['rating'], inplace=True)
df.sort_values(['username', 'date'], inplace=True)

# 4. Compute rolling Elo per player (10-game window)
df['rolling_elo'] = df.groupby('username')['rating'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

# 5. Monthly grouping for averages
df['month'] = df['date'].dt.to_period('M')

monthly_elo = df.groupby('month')['rolling_elo'].mean()
monthly_accuracy = df.groupby('month')['accuracy'].mean()

# Directories
FIGURES_PATH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Nexus365/Documents/Github/Github_new/chess_performance/Figures/"
)
os.makedirs(FIGURES_PATH, exist_ok=True)
