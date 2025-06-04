import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from config import *
from utils import setup_logging, calculate_statistics, save_checkpoint, load_checkpoint

# Set up logging
logger = setup_logging(LOG_FILE)

def fetch_all_players_from_country(country_code):
    """Fetch ALL players from a country (not limited to 1000)"""
    players_url = f'https://api.chess.com/pub/country/{country_code}/players'
    all_players = []
    page = 1
    
    logger.info(f"Starting to fetch all players from {COUNTRIES[country_code]} ({country_code})")
    
    while True:
        try:
            resp = requests.get(players_url, params={'page': page}, headers=HEADERS)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to fetch page {page} for {country_code}: {e}")
            break
        
        data = resp.json()
        batch = data.get('players', [])
        
        if not batch:
            logger.info(f"No more players found at page {page} for {country_code}")
            break
            
        all_players.extend(batch)
        logger.info(f"Fetched page {page} for {country_code}: {len(batch)} players (total: {len(all_players)})")
        
        page += 1
        time.sleep(RATE_LIMIT_PAGES)
        
        # Save checkpoint every 10 pages
        if page % 10 == 0:
            checkpoint_data = [{'username': p, 'country': country_code} for p in all_players]
            save_checkpoint(checkpoint_data, f"{country_code}_players_checkpoint.csv")
        
        # Prevent infinite loops
        if page > MAX_PAGES_PER_COUNTRY:
            logger.warning(f"Reached page limit for {country_code}")
            break
            
        # Optional limit for testing
        if MAX_PLAYERS_PER_COUNTRY and len(all_players) >= MAX_PLAYERS_PER_COUNTRY:
            logger.info(f"Reached player limit for {country_code}: {MAX_PLAYERS_PER_COUNTRY}")
            all_players = all_players[:MAX_PLAYERS_PER_COUNTRY]
            break
    
    logger.info(f"Total players fetched from {country_code}: {len(all_players)}")
    return all_players

def process_player_games(username, country_code):
    """Process all games for a single player"""
    archives_url = f'https://api.chess.com/pub/player/{username}/games/archives'
    player_records = []
    
    try:
        resp = requests.get(archives_url, headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Failed to fetch archives for {username}: {e}")
        return player_records
    
    archives = resp.json().get('archives', [])
    
    for archive_url in archives:
        try:
            resp2 = requests.get(archive_url, headers=HEADERS)
            resp2.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Failed to fetch archive {archive_url} for {username}: {e}")
            continue
            
        games = resp2.json().get('games', [])
        
        for game in games:
            timestamp = game.get('end_time', 0)
            if timestamp == 0:
                continue
                
            date = datetime.utcfromtimestamp(timestamp)
            
            for side in ['white', 'black']:
                player = game.get(side, {})
                if player.get('username', '').lower() != username.lower():
                    continue
                    
                rating = player.get('rating')
                accuracy = game.get('accuracies', {}).get(side) if game.get('accuracies') else None
                
                player_records.append({
                    'username': username,
                    'country': country_code,
                    'date': date,
                    'rating': rating,
                    'accuracy': accuracy
                })
        
        time.sleep(RATE_LIMIT_ARCHIVES)
    
    return player_records

def main():
    logger.info("Starting chess performance analysis")
    logger.info(f"Output directories: {DATA_DIR}")
    
    # Try to load existing checkpoint
    all_records = load_checkpoint(RAW_DATA_FILE) or []
    processed_countries = set()
    
    if all_records:
        processed_countries = set([record['country'] for record in all_records])
        logger.info(f"Loaded {len(all_records)} records from checkpoint")
        logger.info(f"Already processed countries: {processed_countries}")
    
    country_summary = {}
    
    # Process each country
    for country_code, country_name in COUNTRIES.items():
        if country_code in processed_countries:
            logger.info(f"Skipping {country_name} ({country_code}) - already processed")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {country_name} ({country_code})")
        logger.info(f"{'='*50}")
        
        # Fetch all players from this country
        players = fetch_all_players_from_country(country_code)
        
        if not players:
            logger.warning(f"No players found for {country_code}")
            continue
        
        country_records = []
        games_per_player = []
        
        # Process each player
        for i, username in enumerate(tqdm(players, desc=f"Processing {country_code} players")):
            player_games = process_player_games(username, country_code)
            country_records.extend(player_games)
            
            # Track games per player
            if player_games:
                games_per_player.append(len(player_games))
            
            # Save checkpoint every 100 players
            if (i + 1) % 100 == 0:
                temp_records = all_records + country_records
                save_checkpoint(temp_records, RAW_DATA_FILE)
                logger.info(f"Checkpoint saved after processing {i + 1} players")
            
            time.sleep(RATE_LIMIT_PLAYERS)
        
        # Store country-level summary
        country_summary[country_code] = {
            'total_players': len(players),
            'players_with_games': len([x for x in games_per_player if x > 0]),
            'total_games': len(country_records),
            'avg_games_per_player': np.mean(games_per_player) if games_per_player else 0,
            'median_games_per_player': np.median(games_per_player) if games_per_player else 0
        }
        
        all_records.extend(country_records)
        logger.info(f"Completed {country_code}: {len(country_records)} game records")
        
        # Save checkpoint after each country
        save_checkpoint(all_records, RAW_DATA_FILE)
    
    # Convert to DataFrame and process
    logger.info("Converting to DataFrame and processing...")
    df = pd.DataFrame(all_records)
    
    if df.empty:
        logger.error("No data collected!")
        return
    
    # Save raw data
    df.to_csv(RAW_DATA_FILE, index=False)
    logger.info(f"Raw data saved to {RAW_DATA_FILE}")
    
    # Clean and process data
    df_clean = df.dropna(subset=['rating']).copy()
    df_clean.sort_values(['username', 'date'], inplace=True)
    
    # Calculate rolling ELO
    logger.info("Calculating rolling statistics...")
    df_clean['rolling_elo'] = df_clean.groupby('username')['rating'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    )
    
    # Save processed data
    processed_file = os.path.join(PROCESSED_DATA_DIR, 'processed_chess_data.csv')
    df_clean.to_csv(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file}")
    
    # Generate summary statistics
    summary_stats = []
    
    # Overall statistics
    overall_elo_stats = calculate_statistics(df_clean['rolling_elo'], 'elo')
    overall_accuracy_stats = calculate_statistics(df_clean['accuracy'], 'accuracy')
    
    overall_summary = {
        'country': 'ALL',
        'total_players': len(df_clean['username'].unique()),
        'total_games': len(df_clean),
        **overall_elo_stats,
        **overall_accuracy_stats
    }
    summary_stats.append(overall_summary)
    
    # Country-specific statistics
    for country_code in COUNTRIES.keys():
        country_data = df_clean[df_clean['country'] == country_code]
        
        if country_data.empty:
            continue
            
        elo_stats = calculate_statistics(country_data['rolling_elo'], 'elo')
        accuracy_stats = calculate_statistics(country_data['accuracy'], 'accuracy')
        
        country_stat_summary = {
            'country': country_code,
            'total_players': len(country_data['username'].unique()),
            'total_games': len(country_data),
            **elo_stats,
            **accuracy_stats
        }
        summary_stats.append(country_stat_summary)
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Add games per player statistics
    for country_code in COUNTRIES.keys():
        if country_code in country_summary:
            mask = summary_df['country'] == country_code
            summary_df.loc[mask, 'avg_games_per_player'] = country_summary[country_code]['avg_games_per_player']
            summary_df.loc[mask, 'median_games_per_player'] = country_summary[country_code]['median_games_per_player']
    
    # Add overall games per player stats
    games_per_player_overall = df_clean.groupby('username').size()
    summary_df.loc[summary_df['country'] == 'ALL', 'avg_games_per_player'] = games_per_player_overall.mean()
    summary_df.loc[summary_df['country'] == 'ALL', 'median_games_per_player'] = games_per_player_overall.median()
    
    # Save summary statistics
    summary_df.to_csv(SUMMARY_STATS_FILE, index=False)
    
    logger.info(f"Summary statistics saved to {SUMMARY_STATS_FILE}")
    logger.info("\nSummary Preview:")
    print(summary_df.to_string(index=False))
    
    # Print country summaries
    logger.info("\nCountry Summaries:")
    for country_code, summary in country_summary.items():
        logger.info(f"{COUNTRIES[country_code]} ({country_code}):")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("")
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()