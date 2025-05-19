import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# 1. Fetch first 1000 British players
# Chess.com Public API country listing endpoint
COUNTRY_CODE = 'GB'
PLAYERS_URL = f'https://api.chess.com/pub/country/{COUNTRY_CODE}/players'

players = []
page = 1
while len(players) < 1000:
    resp = requests.get(PLAYERS_URL, params={'page': page})
    resp.raise_for_status()
    data = resp.json()
    if 'players' not in data or not data['players']:
        break
    players.extend(data['players'])
    page += 1
    time.sleep(0.5)
players = players[:1000]

# 2. Prepare containers for Elo history and accuracy
all_histories = []

for username in tqdm(players, desc="Processing players"):
    # get archive list
    url_archives = f'https://api.chess.com/pub/player/{username}/games/archives'
    resp = requests.get(url_archives)
    if resp.status_code != 200:
        continue
    archives = resp.json().get('archives', [])
    for archive_url in archives:
        resp2 = requests.get(archive_url)
        if resp2.status_code != 200:
            continue
        games = resp2.json().get('games', [])
        for game in games:
            # parse date
            end_time = datetime.utcfromtimestamp(game.get('end_time', 0))
            # ratings
            white = game.get('white', {})
            black = game.get('black', {})
            for side in ['white', 'black']:
                player = game.get(side, {})
                if player.get('username', '').lower() != username.lower():
                    continue
                rating = player.get('rating')
                accuracy = game.get('accuracies', {}).get(side)
                all_histories.append({
                    'username': username,
                    'date': end_time,
                    'rating': rating,
                    'accuracy': accuracy
                })
        time.sleep(0.2)
    time.sleep(0.5)

# 3. Build DataFrame
df = pd.DataFrame(all_histories)
# sort
df = df.sort_values(['username', 'date'])

# 4. Compute rolling Elo: apply per-player rolling mean (window=10 games)
df['elo_rolling'] = df.groupby('username')['rating'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
# 5. Compute average accuracy over time: group by month

df['month'] = df['date'].dt.to_period('M')
avg_acc = df.groupby('month')['accuracy'].mean()

# 6. Plotting
plt.figure()
avg_rating = df.groupby('month')['elo_rolling'].mean()
avg_rating.plot()
plt.title('Average Rolling Elo over Time (GB Players)')
plt.xlabel('Month')
plt.ylabel('Elo (10-game rolling average)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
avg_acc.plot()
plt.title('Average Accuracy over Time (GB Players)')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
