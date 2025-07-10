# Configuration
HEADERS = {'User-Agent': 'ChessDataAnalysisBot/1.0 (username: clluelless; contact: benjamin.goodair@bsg.ox.ac.uk'}
import os
import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import pickle
import gzip
import random
from datetime import datetime, date
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import gc

COUNTRIES = {'GB': 'Great Britain', 'FR': 'France', 'DE': 'Germany', 'US': 'United States', 'IN': 'India', 'RU': 'Russia', 'UA': 'Ukraine'}

# Configuration
MAX_PLAYERS_PER_COUNTRY = 25000
MAX_CONCURRENT_REQUESTS = 2
RATE_LIMIT_ARCHIVES = 2
MIN_YEAR = 2017  # For player qualification only
MIN_MONTHS_ACTIVE = 12
MAX_RETRIES = 3
BATCH_SIZE = 100
MAX_PLAYERS_IN_MEMORY = 5000

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'Data', 'processed')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for d in [PROCESSED_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'chess_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir=CHECKPOINTS_DIR):
        self.checkpoint_dir = checkpoint_dir
        
    def save_checkpoint(self, data, country_code, batch_num):
        try:
            checkpoint_data = {
                'processed_players': data.get('processed_players', []),
                'qualified_players': data.get('qualified_players', []),
                'batch_num': data.get('batch_num', batch_num),
                'country_code': country_code,
                'timestamp': datetime.now().isoformat()
            }
            
            filename = os.path.join(self.checkpoint_dir, f"{country_code}_batch_{batch_num}.pkl.gz")
            with gzip.open(filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {filename} (processed: {len(checkpoint_data['processed_players'])}, qualified: {len(checkpoint_data['qualified_players'])})")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(self, country_code):
        try:
            files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(f"{country_code}_batch_")]
            if not files:
                logger.info(f"No checkpoints found for {country_code}")
                return None, 0
            
            batch_nums = []
            for f in files:
                try:
                    batch_num = int(f.split('_batch_')[1].split('.')[0])
                    batch_nums.append(batch_num)
                except:
                    continue
            
            if not batch_nums:
                return None, 0
                
            latest_batch = max(batch_nums)
            filename = os.path.join(self.checkpoint_dir, f"{country_code}_batch_{latest_batch}.pkl.gz")
            
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded checkpoint: {country_code} batch {latest_batch}")
            return data, latest_batch
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {country_code}: {e}")
            return None, 0
    
    def cleanup_old_checkpoints(self, country_code, keep_last=2):
        try:
            files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(f"{country_code}_batch_")]
            if len(files) <= keep_last:
                return
                
            file_batches = []
            for f in files:
                try:
                    batch_num = int(f.split('_batch_')[1].split('.')[0])
                    file_batches.append((batch_num, f))
                except:
                    continue
            
            file_batches.sort()
            files_to_remove = file_batches[:-keep_last]
            
            for _, filename in files_to_remove:
                os.remove(os.path.join(self.checkpoint_dir, filename))
                logger.info(f"Removed old checkpoint: {filename}")
                
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")

class BatchGameCollector:
    def __init__(self):
        self.daily_games = {}
        self.daily_active_players = {}
        
    def add_game(self, date_str, rating, accuracy, player, result=None):
        if date_str not in self.daily_games:
            self.daily_games[date_str] = []
            self.daily_active_players[date_str] = set()
    
        self.daily_active_players[date_str].add(player)
    
        game_data = {'player': player}
        if rating and 0 < rating < 5000:
            game_data['rating'] = rating
        if accuracy and 0 <= accuracy <= 100:
            game_data['accuracy'] = accuracy
        if result in ['win', 'lose']:
            game_data['result'] = result
        
        self.daily_games[date_str].append(game_data)
    
    def compute_daily_stats(self, country_code):
        daily_records = []
        
        for date_str, games in self.daily_games.items():
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                ratings = [g['rating'] for g in games if 'rating' in g]
                accuracies = [g['accuracy'] for g in games if 'accuracy' in g]
                rating_players = set(g['player'] for g in games if 'rating' in g)
                accuracy_players = set(g['player'] for g in games if 'accuracy' in g)
                
                active_players_count = len(self.daily_active_players.get(date_str, set()))
                
                record = {'country': country_code, 'date': date_obj, 'active_players': active_players_count}
                
                # Rating stats
                if ratings:
                    ratings = np.array(ratings)
                    record.update({
                        'rating_count': len(ratings),
                        'rating_mean': np.mean(ratings),
                        'rating_std': np.std(ratings),
                        'rating_median': np.median(ratings),
                        'rating_p10': np.percentile(ratings, 10),
                        'rating_p90': np.percentile(ratings, 90),
                        'rating_players': len(rating_players)
                    })
                else:
                    record.update({
                        'rating_count': 0, 'rating_mean': 0, 'rating_std': 0,
                        'rating_median': 0, 'rating_p10': 0, 'rating_p90': 0, 'rating_players': 0
                    })
                
                # Accuracy stats
                if accuracies:
                    accuracies = np.array(accuracies)
                    record.update({
                        'accuracy_count': len(accuracies),
                        'accuracy_mean': np.mean(accuracies),
                        'accuracy_std': np.std(accuracies),
                        'accuracy_median': np.median(accuracies),
                        'accuracy_p10': np.percentile(accuracies, 10),
                        'accuracy_p90': np.percentile(accuracies, 90),
                        'accuracy_players': len(accuracy_players)
                    })
                else:
                    record.update({
                        'accuracy_count': 0, 'accuracy_mean': 0, 'accuracy_std': 0,
                        'accuracy_median': 0, 'accuracy_p10': 0, 'accuracy_p90': 0, 'accuracy_players': 0
                    })

                # Win/Loss stats
                results = [g['result'] for g in games if 'result' in g]
                wins = sum(1 for r in results if r == 'win')
                losses = sum(1 for r in results if r == 'lose')
                total_results = len(results)

                if total_results > 0:
                    record.update({
                        'win_count': wins,
                        'loss_count': losses,
                        'win_percentage': (wins / total_results) * 100,
                        'loss_percentage': (losses / total_results) * 100
                    })
                else:
                    record.update({
                        'win_count': 0, 'loss_count': 0,
                        'win_percentage': 0, 'loss_percentage': 0
                    })
                
                daily_records.append(record)
                
            except Exception as e:
                logger.error(f"Error processing date {date_str}: {e}")
                continue
        
        return daily_records
    
    def clear(self):
        self.daily_games.clear()
        self.daily_active_players.clear()
        gc.collect()

class ChessDataCollector:
    def __init__(self):
        self.session = None
        self.checkpoint_manager = CheckpointManager()
        self.processed_players = set()
        self.qualified_players = set()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

    async def create_session(self):
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS,
            limit_per_host=MAX_CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=90, connect=45)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout, 
            headers=HEADERS,
            raise_for_status=False
        )

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
        await asyncio.sleep(2)

    async def fetch_with_retry(self, url, max_retries=MAX_RETRIES):
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 429:
                        wait_time = min(120 * (2 ** attempt), 600)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status == 404:
                        return None
                    elif response.status >= 500:
                        if attempt < max_retries - 1:
                            wait_time = min(60 * (2 ** attempt), 300)
                            logger.warning(f"Server error {response.status}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"Server error {response.status} for {url}")
                            return None
                    elif response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                    
                    self.consecutive_failures = 0
                    await asyncio.sleep(RATE_LIMIT_ARCHIVES)
                    return await response.json()
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.consecutive_failures += 1
                if attempt < max_retries - 1:
                    wait_time = min(20 * (2 ** attempt), 120)
                    logger.warning(f"Connection error (attempt {attempt + 1}): {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.warning("Too many consecutive failures, recreating session")
                        await self.close_session()
                        await self.create_session()
                        self.consecutive_failures = 0
                    
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

    async def process_player(self, username, batch_collector):
        username_lower = username.lower()
        
        if username_lower in self.processed_players:
            return False
            
        try:
            archives_url = f'https://api.chess.com/pub/player/{username}/games/archives'
            archives_data = await self.fetch_with_retry(archives_url)
            if not archives_data:
                self.processed_players.add(username_lower)
                return False

            archives = archives_data.get('archives', [])
            
            # QUALIFICATION CHECK: Only check recent activity for player qualification
            months_active_recent = set()
            for archive_url in archives:
                try:
                    parts = archive_url.strip('/').split('/')
                    if len(parts) >= 2:
                        year, month = int(parts[-2]), int(parts[-1])
                        if year >= MIN_YEAR:  # Only count activity from MIN_YEAR onwards for qualification
                            months_active_recent.add((year, month))
                except (ValueError, IndexError):
                    continue
            
            years_active_recent = len(set(year for year, month in months_active_recent))
            is_qualified = (len(months_active_recent) >= MIN_MONTHS_ACTIVE and 
                           years_active_recent >= 2 and
                           any(year <= 2019 for year, month in months_active_recent))
            
            if not is_qualified:
                self.processed_players.add(username_lower)
                logger.debug(f"Player {username} skipped: insufficient recent activity ({len(months_active_recent)} months since {MIN_YEAR}, {years_active_recent} years)")
                return False

            self.qualified_players.add(username_lower)
            games_processed = 0
            
            # DATA COLLECTION: Process ALL archives (including pre-2017) for qualified players
            for archive_url in archives:
                # REMOVED THE YEAR FILTER - now processes all archives regardless of year
                archive_data = await self.fetch_with_retry(archive_url)
                if not archive_data:
                    continue

                for game in archive_data.get('games', []):
                    try:
                        timestamp = game.get('end_time', 0)
                        if not timestamp:
                            continue
                        game_date = datetime.utcfromtimestamp(timestamp).date().strftime('%Y-%m-%d')

                        for side in ['white', 'black']:
                            player = game.get(side, {})
                            if player.get('username', '').lower() != username_lower:
                                continue

                            rating = player.get('rating')
                            accuracy = game.get('accuracies', {}).get(side) if game.get('accuracies') else None

                            result = None
                            if side in ['white', 'black']:
                                player_result = game.get(side, {}).get('result')
                                if player_result == 'win':
                                    result = 'win'
                                elif player_result in ['lose', 'timeout', 'resigned', 'checkmated']:
                                    result = 'lose'

                            batch_collector.add_game(game_date, rating, accuracy, username_lower, result)
                            games_processed += 1

                    except Exception as e:
                        continue

            self.processed_players.add(username_lower)
            
            if len(self.processed_players) > MAX_PLAYERS_IN_MEMORY:
                logger.info(f"Clearing processed players set (was {len(self.processed_players)} players)")
                recent_players = list(self.processed_players)[-MAX_PLAYERS_IN_MEMORY//2:]
                self.processed_players = set(recent_players)
                recent_qualified = list(self.qualified_players)[-MAX_PLAYERS_IN_MEMORY//4:]
                self.qualified_players = set(recent_qualified)
                gc.collect()
            
            return games_processed > 0

        except Exception as e:
            logger.warning(f"Error processing player {username}: {e}")
            self.processed_players.add(username_lower)
            return False

    def merge_with_existing_data(self, new_records, country_code):
        try:
            output_file = os.path.join(PROCESSED_DATA_DIR, f'{country_code}_daily_stats.csv')
            
            new_df = pd.DataFrame(new_records)
            if new_df.empty:
                return
            
            new_df = new_df.sort_values('date')
            
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                existing_df['date'] = pd.to_datetime(existing_df['date']).dt.date
                combined_df = self.recalculate_aggregates(existing_df, new_df)
            else:
                combined_df = new_df
            
            combined_df = combined_df.sort_values('date')
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Updated {country_code} data: {len(combined_df)} daily records")
            
        except Exception as e:
            logger.error(f"Error merging data for {country_code}: {e}")

    def recalculate_aggregates(self, existing_df, new_df):
        try:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            aggregated = []
            for date, group in combined_df.groupby('date'):
                rating_count = group['rating_count'].sum()
                accuracy_count = group['accuracy_count'].sum()
                rating_players = group['rating_players'].sum()
                accuracy_players = group['accuracy_players'].sum()
                active_players = group['active_players'].sum()
                
                if rating_count > 0:
                    rating_mean = (group['rating_mean'] * group['rating_count']).sum() / rating_count
                else:
                    rating_mean = 0
                    
                if accuracy_count > 0:
                    accuracy_mean = (group['accuracy_mean'] * group['accuracy_count']).sum() / accuracy_count
                else:
                    accuracy_mean = 0

                win_count = group['win_count'].sum()
                loss_count = group['loss_count'].sum()
                total_results = win_count + loss_count

                if total_results > 0:
                    win_percentage = (win_count / total_results) * 100
                    loss_percentage = (loss_count / total_results) * 100
                else:
                    win_percentage = 0
                    loss_percentage = 0
                
                latest_row = group.iloc[-1]
                
                record = {
                    'country': latest_row['country'],
                    'date': date,
                    'active_players': active_players,
                    'rating_count': rating_count,
                    'rating_mean': rating_mean,
                    'rating_std': latest_row['rating_std'],
                    'rating_median': latest_row['rating_median'],
                    'rating_p10': latest_row['rating_p10'],
                    'rating_p90': latest_row['rating_p90'],
                    'rating_players': rating_players,
                    'accuracy_count': accuracy_count,
                    'accuracy_mean': accuracy_mean,
                    'accuracy_std': latest_row['accuracy_std'],
                    'accuracy_median': latest_row['accuracy_median'],
                    'accuracy_p10': latest_row['accuracy_p10'],
                    'accuracy_p90': latest_row['accuracy_p90'],
                    'accuracy_players': accuracy_players,
                    'win_count': win_count,
                    'loss_count': loss_count,
                    'win_percentage': win_percentage,
                    'loss_percentage': loss_percentage
                }
                aggregated.append(record)
            
            return pd.DataFrame(aggregated)
            
        except Exception as e:
            logger.error(f"Error recalculating aggregates: {e}")
            return pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['country', 'date'], keep='last')

def fetch_random_players(country_code, max_players=25000):
    try:
        players_url = f'https://api.chess.com/pub/country/{country_code}/players'
        all_players = []

        def fetch_page(page_num):
            try:
                resp = requests.get(
                    players_url, 
                    params={'page': page_num}, 
                    headers=HEADERS, 
                    timeout=45
                )
                resp.raise_for_status()
                players = resp.json().get('players', [])
                logger.debug(f"Page {page_num}: {len(players)} players")
                return players
            except Exception as e:
                logger.warning(f"Error fetching page {page_num} for {country_code}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fetch_page, i) for i in range(1, 51)]
            for future in as_completed(futures):
                players = future.result()
                if players:
                    all_players.extend(players)

        logger.info(f"Found {len(all_players)} total players for {country_code}")
        
        random.shuffle(all_players)
        initial_sample = min(len(all_players), max_players * 3)
        sampled_players = all_players[:initial_sample]
        
        logger.info(f"Sampled {len(sampled_players)} players for activity filtering")
        return sampled_players

    except Exception as e:
        logger.error(f"Error fetching players for {country_code}: {e}")
        return []

async def process_country(country_code):
    collector = None
    try:
        logger.info(f"Processing {COUNTRIES[country_code]} ({country_code})")
        
        collector = ChessDataCollector()
        
        checkpoint_data, last_batch = collector.checkpoint_manager.load_latest_checkpoint(country_code)
        start_batch = last_batch if checkpoint_data else 0
        
        if checkpoint_data:
            collector.processed_players = set(checkpoint_data.get('processed_players', []))
            collector.qualified_players = set(checkpoint_data.get('qualified_players', []))
            logger.info(f"Resumed from batch {last_batch}, {len(collector.processed_players)} players processed, {len(collector.qualified_players)} qualified")

        all_players = fetch_random_players(country_code, MAX_PLAYERS_PER_COUNTRY)
        if not all_players:
            return

        remaining_players = [p for p in all_players if p.lower() not in collector.processed_players]
        logger.info(f"Processing {len(remaining_players)} remaining players (targeting {MAX_PLAYERS_PER_COUNTRY} qualified players)")

        await collector.create_session()

        try:
            total_batches = (len(remaining_players) // BATCH_SIZE) + 1
            qualified_count = len(collector.qualified_players)
            
            for batch_num in range(start_batch, total_batches):
                if qualified_count >= MAX_PLAYERS_PER_COUNTRY:
                    logger.info(f"Reached target of {MAX_PLAYERS_PER_COUNTRY} qualified players")
                    break
                    
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(remaining_players))
                
                if start_idx >= len(remaining_players):
                    break
                    
                batch = remaining_players[start_idx:end_idx]
                logger.info(f"Processing batch {batch_num + 1}: players {start_idx + 1}-{end_idx} (qualified so far: {qualified_count})")

                batch_collector = BatchGameCollector()

                semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
                async def process_with_semaphore(player):
                    async with semaphore:
                        return await collector.process_player(player, batch_collector)

                tasks = [process_with_semaphore(player) for player in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful = sum(1 for r in results if r is True)
                new_qualified = len(collector.qualified_players) - qualified_count
                qualified_count = len(collector.qualified_players)
                
                logger.info(f"Batch {batch_num + 1}: {successful}/{len(batch)} players processed successfully, {new_qualified} new qualified players (total qualified: {qualified_count})")

                daily_records = batch_collector.compute_daily_stats(country_code)
                logger.info(f"Batch {batch_num + 1} generated {len(daily_records)} daily records")

                collector.merge_with_existing_data(daily_records, country_code)

                batch_collector.clear()
                del batch_collector

                checkpoint_data = {
                    'processed_players': list(collector.processed_players),
                    'qualified_players': list(collector.qualified_players),
                    'batch_num': batch_num + 1,
                    'country_code': country_code
                }
                collector.checkpoint_manager.save_checkpoint(checkpoint_data, country_code, batch_num + 1)
                
                if (batch_num + 1) % 3 == 0:
                    collector.checkpoint_manager.cleanup_old_checkpoints(country_code)

                if (batch_num + 1) % 5 == 0:
                    gc.collect()

                logger.info(f"Completed batch {batch_num + 1}/{total_batches} for {country_code}")

        finally:
            await collector.close_session()

        logger.info(f"Completed {country_code} with {len(collector.qualified_players)} qualified players")

    except Exception as e:
        logger.error(f"Error processing {country_code}: {e}")
        if collector:
            await collector.close_session()
        raise

async def main():
    try:
        logger.info("Starting chess data collection with full historic data for qualified players")

        for country_code in COUNTRIES.keys():
            try:
                await process_country(country_code)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error processing {country_code}: {e}")
                continue

        # Combine all country files
        all_dfs = []
        for country_code in COUNTRIES.keys():
            file_path = os.path.join(PROCESSED_DATA_DIR, f'{country_code}_daily_stats.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(['country', 'date'])
            final_file = os.path.join(PROCESSED_DATA_DIR, 'chess_national_daily_stats.csv')
            combined_df.to_csv(final_file, index=False)
            logger.info(f"Final dataset saved: {final_file} with {len(combined_df)} records")

        logger.info("Analysis completed!")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())