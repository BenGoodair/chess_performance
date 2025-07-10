import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, date
from tqdm import tqdm
from collections import defaultdict
import asyncio
import aiohttp
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gzip
import gc
import signal
import sys
from config import *
from utils import setup_logging, save_checkpoint, load_checkpoint

# Set up logging
logger = setup_logging(LOG_FILE)

class GracefulKiller:
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    kill_now = threading.Event()
    
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.kill_now.set()

def create_nested_dict():
    """Factory function for nested defaultdict - avoids lambda pickle issues"""
    return {'ratings': [], 'accuracies': []}

def create_daily_dict():
    """Factory function for daily data defaultdict - avoids lambda pickle issues"""
    return defaultdict(create_nested_dict)
    
    
class ChessDataCollector:
    def __init__(self, max_concurrent=MAX_CONCURRENT_REQUESTS, rate_limit_delay=RATE_LIMIT_ARCHIVES):
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.session = None
        # Use named functions instead of lambdas to avoid pickle issues
        self.daily_data = defaultdict(create_daily_dict)
        self.processed_players = set()
        self.failed_players = set()
        self.lock = threading.Lock()
        
        # Rate limiting tracking
        self.rate_limit_strikes = 0
        self.last_rate_limit_time = None
        self.cooldown_until = None
        self.max_strikes = 5  # Number of 429s before triggering cooldown
        self.cooldown_duration = 30 * 60  # 30 minutes in seconds
        self.strike_reset_time = 5 * 60  # Reset strikes after 5 minutes
        
        self.stats = {
            'players_processed': 0,
            'players_failed': 0,
            'archives_processed': 0,
            'games_found': 0,
            'start_time': time.time(),
            'cooldowns_triggered': 0,
            'total_cooldown_time': 0
        }
        
    async def create_session(self):
        """Create aiohttp session with optimized settings"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=HEADERS
        )
        logger.info(f"Created session with {self.max_concurrent} max concurrent connections")
    
    async def close_session(self):
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.1)  # Give time for connections to close
    
    def check_cooldown_status(self):
        """Check if we're in cooldown period"""
        current_time = time.time()
        
        # Reset strikes if enough time has passed since last rate limit
        if (self.last_rate_limit_time and 
            current_time - self.last_rate_limit_time > self.strike_reset_time):
            self.rate_limit_strikes = 0
            logger.info("Rate limit strikes reset")
        
        # Check if we're still in cooldown
        if self.cooldown_until and current_time < self.cooldown_until:
            remaining = int(self.cooldown_until - current_time)
            return True, remaining
        elif self.cooldown_until and current_time >= self.cooldown_until:
            # Cooldown just ended
            logger.info("Cooldown period ended, resuming requests")
            self.cooldown_until = None
            self.rate_limit_strikes = 0
            
        return False, 0
    
    async def handle_rate_limit(self, response_status=429):
        """Handle rate limiting with strike system and cooldown"""
        current_time = time.time()
        self.last_rate_limit_time = current_time
        self.rate_limit_strikes += 1
        
        logger.warning(f"Rate limit hit (strike {self.rate_limit_strikes}/{self.max_strikes})")
        
        if self.rate_limit_strikes >= self.max_strikes:
            # Trigger cooldown
            self.cooldown_until = current_time + self.cooldown_duration
            self.stats['cooldowns_triggered'] += 1
            
            cooldown_end_time = datetime.fromtimestamp(self.cooldown_until)
            logger.warning(f"RATE LIMIT COOLDOWN TRIGGERED! Pausing all requests until {cooldown_end_time}")
            logger.warning(f"Cooldown duration: {self.cooldown_duration/60:.1f} minutes")
            
            # Wait for the cooldown period
            await self.wait_for_cooldown()
        else:
            # Regular exponential backoff
            wait_time = min((2 ** self.rate_limit_strikes) * self.rate_limit_delay, 60)
            logger.info(f"Rate limited, waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
    
    async def wait_for_cooldown(self):
        """Wait for cooldown period with progress updates"""
        start_cooldown = time.time()
        
        while True:
            in_cooldown, remaining = self.check_cooldown_status()
            if not in_cooldown:
                break
                
            # Log progress every 5 minutes
            if remaining % 300 == 0 or remaining <= 60:
                logger.info(f"Cooldown in progress: {remaining//60}m {remaining%60}s remaining")
            
            # Sleep for 1 minute or remaining time, whichever is shorter
            sleep_time = min(60, remaining)
            await asyncio.sleep(sleep_time)
        
        end_cooldown = time.time()
        cooldown_duration = end_cooldown - start_cooldown
        self.stats['total_cooldown_time'] += cooldown_duration
        logger.info(f"Cooldown completed after {cooldown_duration/60:.1f} minutes")
    
    async def fetch_with_retry(self, url, max_retries=3):
        """Fetch URL with enhanced rate limiting and cooldown handling"""
        # Check if we're in cooldown before making any request
        in_cooldown, remaining = self.check_cooldown_status()
        if in_cooldown:
            logger.info(f"Skipping request due to cooldown ({remaining}s remaining)")
            await self.wait_for_cooldown()
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 429:  # Rate limited
                        await self.handle_rate_limit(response.status)
                        continue
                    elif response.status == 404:
                        return None  # Player/archive not found
                    elif response.status in [500, 502, 503, 504]:
                        # Server errors - treat similar to rate limiting but less severe
                        wait_time = min((2 ** attempt) * 5, 30)
                        logger.warning(f"Server error {response.status}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    
                    # Successful request - add small delay to be respectful
                    await asyncio.sleep(self.rate_limit_delay)
                    return await response.json()
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                if "429" in str(e):
                    await self.handle_rate_limit()
                    continue
                logger.warning(f"Client error fetching {url} (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.warning(f"Error fetching {url} (attempt {attempt + 1}): {e}")
                
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5
                await asyncio.sleep(wait_time)
        
        return None
    
    async def get_player_archives(self, username):
        """Get list of archive URLs for a player"""
        archives_url = f'https://api.chess.com/pub/player/{username}/games/archives'
        data = await self.fetch_with_retry(archives_url)
        return data.get('archives', []) if data else []
    
    async def process_archive(self, archive_url, username, country_code):
        """Process a single archive for a player"""
        data = await self.fetch_with_retry(archive_url)
        if not data:
            return 0
        
        games = data.get('games', [])
        games_processed = 0
        
        for game in games:
            try:
                timestamp = game.get('end_time', 0)
                if timestamp == 0:
                    continue
                    
                game_date = datetime.utcfromtimestamp(timestamp).date()
                
                for side in ['white', 'black']:
                    player = game.get(side, {})
                    if player.get('username', '').lower() != username.lower():
                        continue
                        
                    rating = player.get('rating')
                    accuracy = game.get('accuracies', {}).get(side) if game.get('accuracies') else None
                    
                    # Validate data types and ranges
                    if rating is not None:
                        try:
                            rating = float(rating)
                            if rating < 0 or rating > 5000:  # Sanity check
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    if accuracy is not None:
                        try:
                            accuracy = float(accuracy)
                            if accuracy < 0 or accuracy > 100:  # Sanity check
                                continue
                        except (ValueError, TypeError):
                            accuracy = None
                    
                    if rating is not None:
                        # Thread-safe update
                        with self.lock:
                            self.daily_data[country_code][game_date]['ratings'].append(rating)
                            if accuracy is not None:
                                self.daily_data[country_code][game_date]['accuracies'].append(accuracy)
                        
                        games_processed += 1
                        
            except Exception as e:
                logger.debug(f"Error processing game for {username}: {e}")
                continue
        
        with self.lock:
            self.stats['archives_processed'] += 1
            self.stats['games_found'] += games_processed
            
        return games_processed
    
    async def process_player(self, username, country_code, semaphore):
        """Process all games for a single player"""
        async with semaphore:
            if username in self.processed_players or username in self.failed_players:
                return False
                
            try:
                archives = await self.get_player_archives(username)
                if not archives:
                    with self.lock:
                        self.failed_players.add(username)
                        self.stats['players_failed'] += 1
                    return False
                
                # Process archives with controlled concurrency
                archive_semaphore = asyncio.Semaphore(3)  # Reduced from 5 to be more conservative
                
                async def process_single_archive(archive_url):
                    async with archive_semaphore:
                        return await self.process_archive(archive_url, username, country_code)
                
                # Process all archives for this player
                archive_tasks = [process_single_archive(url) for url in archives]
                results = await asyncio.gather(*archive_tasks, return_exceptions=True)
                
                # Count successful results
                total_games = sum(r for r in results if isinstance(r, int) and r > 0)
                
                # Only count as successful if we found some games
                success = total_games > 0
                
                with self.lock:
                    if success:
                        self.processed_players.add(username)
                        self.stats['players_processed'] += 1
                    else:
                        self.failed_players.add(username)
                        self.stats['players_failed'] += 1
                
                return success
                
            except Exception as e:
                logger.error(f"Error processing player {username}: {e}")
                with self.lock:
                    self.failed_players.add(username)
                    self.stats['players_failed'] += 1
                return False
    
    async def process_player_batch(self, players, country_code, batch_num):
        """Process a batch of players concurrently"""
        logger.info(f"Processing batch {batch_num} with {len(players)} players for {country_code}")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        killer = GracefulKiller()
        
        tasks = []
        for username in players:
            if killer.kill_now.is_set():
                logger.info("Graceful shutdown requested, stopping batch processing")
                break
            task = self.process_player(username, country_code, semaphore)
            tasks.append(task)
        
        # Process with progress tracking
        completed = 0
        successful = 0
        
        for task in asyncio.as_completed(tasks):
            if killer.kill_now.is_set():
                break
                
            try:
                result = await task
                completed += 1
                if result:
                    successful += 1
                    
                if completed % 50 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['players_processed'] / elapsed if elapsed > 0 else 0
                    cooldown_info = f", {self.stats['cooldowns_triggered']} cooldowns" if self.stats['cooldowns_triggered'] > 0 else ""
                    logger.info(f"Batch {batch_num}: {completed}/{len(players)} players, "
                              f"{successful} successful, {rate:.2f} players/sec{cooldown_info}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                completed += 1
        
        total_cooldown_mins = self.stats['total_cooldown_time'] / 60
        logger.info(f"Batch {batch_num} completed: {successful}/{completed} players successful. "
                   f"Total cooldown time: {total_cooldown_mins:.1f} minutes")
        return successful, completed
    
    def get_rate_limit_status(self):
        """Get current rate limiting status for monitoring"""
        in_cooldown, remaining = self.check_cooldown_status()
        return {
            'in_cooldown': in_cooldown,
            'cooldown_remaining_seconds': remaining,
            'rate_limit_strikes': self.rate_limit_strikes,
            'total_cooldowns': self.stats['cooldowns_triggered'],
            'total_cooldown_time_minutes': self.stats['total_cooldown_time'] / 60
        }
    
    def __getstate__(self):
        """Custom pickle state to handle defaultdict properly"""
        state = self.__dict__.copy()
        # Convert defaultdict to regular dict for pickling
        state['daily_data'] = {
            country: {
                date_key: {
                    'ratings': data['ratings'][:],  # Make copies
                    'accuracies': data['accuracies'][:]
                }
                for date_key, data in daily_dict.items()
            }
            for country, daily_dict in self.daily_data.items()
        }
        return state
    
    def __setstate__(self, state):
        """Custom unpickle state to restore defaultdict properly"""
        self.__dict__.update(state)
        # Restore defaultdict structure
        restored_data = defaultdict(create_daily_dict)
        for country, daily_dict in state['daily_data'].items():
            for date_key, data in daily_dict.items():
                restored_data[country][date_key] = data
        self.daily_data = restored_data
    
    def save_compressed_checkpoint(self, country_code, batch_num):
        """Save checkpoint with compression including rate limit state"""
        checkpoint_data = {
            'daily_data': dict(self.daily_data),
            'processed_players': list(self.processed_players),
            'failed_players': list(self.failed_players),
            'stats': self.stats.copy(),
            'batch_num': batch_num,
            'timestamp': datetime.now().isoformat(),
            'rate_limit_strikes': self.rate_limit_strikes,
            'last_rate_limit_time': self.last_rate_limit_time,
            'cooldown_until': self.cooldown_until
        }
        
        checkpoint_file = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{country_code}_batch_{batch_num}.pkl.gz")
        
        try:
            with gzip.open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return checkpoint_file
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_compressed_checkpoint(self, country_code, batch_num=None):
        """Load the latest checkpoint including rate limit state"""
        try:
            if batch_num:
                checkpoint_file = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{country_code}_batch_{batch_num}.pkl.gz")
            else:
                # Find latest checkpoint
                checkpoint_files = [f for f in os.listdir(CHECKPOINTS_DIR) 
                                  if f.startswith(f"checkpoint_{country_code}_batch_") and f.endswith('.pkl.gz')]
                if not checkpoint_files:
                    return 0
                
                # Sort by batch number
                checkpoint_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
                checkpoint_file = os.path.join(CHECKPOINTS_DIR, checkpoint_files[-1])
            
            with gzip.open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.daily_data = defaultdict(create_daily_dict)
            for country, daily_dict in checkpoint_data.get('daily_data', {}).items():
                for date_key, data in daily_dict.items():
                    self.daily_data[country][date_key] = data
            
            self.processed_players = set(checkpoint_data.get('processed_players', []))
            self.failed_players = set(checkpoint_data.get('failed_players', []))
            self.stats.update(checkpoint_data.get('stats', {}))
            
            # Restore rate limiting state
            self.rate_limit_strikes = checkpoint_data.get('rate_limit_strikes', 0)
            self.last_rate_limit_time = checkpoint_data.get('last_rate_limit_time')
            self.cooldown_until = checkpoint_data.get('cooldown_until')
            
            batch_num = checkpoint_data.get('batch_num', 0)
            logger.info(f"Loaded checkpoint: {checkpoint_file} (batch {batch_num})")
            logger.info(f"Processed players: {len(self.processed_players)}, Failed: {len(self.failed_players)}")
            
            # Check if we're resuming during a cooldown
            in_cooldown, remaining = self.check_cooldown_status()
            if in_cooldown:
                logger.warning(f"Resuming during cooldown period: {remaining}s remaining")
            
            return batch_num
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0

async def fetch_all_players_efficiently(country_code, max_workers=10):
    """Fetch all players using threading for pagination"""
    players_url = f'https://api.chess.com/pub/country/{country_code}/players'
    all_players = []
    
    def fetch_page(page_num):
        try:
            resp = requests.get(players_url, params={'page': page_num}, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            players = data.get('players', [])
            return page_num, players
        except Exception as e:
            logger.warning(f"Failed to fetch page {page_num} for {country_code}: {e}")
            return page_num, []
    
    logger.info(f"Fetching all players for {country_code}...")
    
    # Start with small batch to estimate
    with ThreadPoolExecutor(max_workers=5) as executor:
        initial_futures = [executor.submit(fetch_page, i) for i in range(1, 6)]
        for future in as_completed(initial_futures):
            page_num, players = future.result()
            if players:
                all_players.extend(players)
                logger.info(f"Page {page_num}: {len(players)} players")
    
    # Continue fetching if we got data
    if all_players:
        current_page = 6
        consecutive_empty = 0
        
        while consecutive_empty < 5 and current_page <= MAX_PAGES_PER_COUNTRY:
            chunk_size = min(max_workers, 10)
            page_chunk = list(range(current_page, current_page + chunk_size))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(fetch_page, page) for page in page_chunk]
                chunk_results = []
                
                for future in as_completed(futures):
                    page_num, players = future.result()
                    chunk_results.append((page_num, players))
            
            chunk_results.sort(key=lambda x: x[0])
            
            found_players_in_chunk = False
            for page_num, players in chunk_results:
                if players:
                    all_players.extend(players)
                    consecutive_empty = 0
                    found_players_in_chunk = True
                else:
                    consecutive_empty += 1
            
            if found_players_in_chunk:
                logger.info(f"Pages {current_page}-{current_page + chunk_size - 1}: {len(all_players)} total players")
            
            current_page += chunk_size
            
            if MAX_PLAYERS_PER_COUNTRY and len(all_players) >= MAX_PLAYERS_PER_COUNTRY:
                all_players = all_players[:MAX_PLAYERS_PER_COUNTRY]
                logger.info(f"Limited to {MAX_PLAYERS_PER_COUNTRY} players as configured")
                break
            
            time.sleep(0.2)  # Small delay between chunks
    
    logger.info(f"Total players found for {country_code}: {len(all_players)}")
    return all_players

async def process_country_optimized(country_code, resume_from_batch=None):
    """Process country with full optimization"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {COUNTRIES[country_code]} ({country_code}) - OPTIMIZED")
    logger.info(f"{'='*60}")
    
    # Initialize collector
    collector = ChessDataCollector()
    
    # Load checkpoint if resuming
    start_batch = 1
    if resume_from_batch:
        start_batch = collector.load_compressed_checkpoint(country_code, resume_from_batch) + 1
        logger.info(f"Resuming from batch {start_batch}")
    else:
        # Try to load latest checkpoint
        start_batch = collector.load_compressed_checkpoint(country_code) + 1
        if start_batch > 1:
            logger.info(f"Resuming from batch {start_batch}")
    
    # Fetch all players
    all_players = await fetch_all_players_efficiently(country_code)
    
    if not all_players:
        logger.warning(f"No players found for {country_code}")
        return {}
    
    # Filter out already processed players
    remaining_players = [p for p in all_players if p not in collector.processed_players]
    logger.info(f"Players remaining to process: {len(remaining_players)}")
    
    if not remaining_players:
        logger.info("All players already processed!")
        return dict(collector.daily_data)
    
    await collector.create_session()
    
    try:
        # Process in batches
        total_batches = (len(remaining_players) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_num in range(start_batch, total_batches + 1):
            start_idx = (batch_num - start_batch) * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(remaining_players))
            
            if start_idx >= len(remaining_players):
                break
                
            batch_players = remaining_players[start_idx:end_idx]
            
            logger.info(f"Starting batch {batch_num}/{total_batches} ({len(batch_players)} players)")
            
            successful, completed = await collector.process_player_batch(batch_players, country_code, batch_num)
            
            # Save checkpoint
            collector.save_compressed_checkpoint(country_code, batch_num)
            
            # Memory cleanup
            if batch_num % CLEANUP_FREQUENCY == 0:
                gc.collect()
                logger.info(f"Memory cleanup completed after batch {batch_num}")
            
            # Progress summary
            elapsed = time.time() - collector.stats['start_time']
            rate = collector.stats['players_processed'] / elapsed if elapsed > 0 else 0
            
            logger.info(f"Batch {batch_num} summary:")
            logger.info(f"  - Successful: {successful}/{completed}")
            logger.info(f"  - Total processed: {collector.stats['players_processed']}")
            logger.info(f"  - Rate: {rate:.2f} players/sec")
            logger.info(f"  - Games found: {collector.stats['games_found']}")
            logger.info(f"  - Elapsed: {elapsed/3600:.1f} hours")
    
    finally:
        await collector.close_session()
    
    logger.info(f"Country {country_code} processing completed!")
    logger.info(f"Final stats: {collector.stats}")
    
    return dict(collector.daily_data)

def safe_float_conversion(value):
    """Safely convert value to float, handling various edge cases"""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        if np.isfinite(value):  # Handles inf and -inf
            return float(value)
        else:
            return None
    
    if isinstance(value, str):
        try:
            result = float(value)
            if np.isfinite(result):
                return result
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    return None

def calculate_daily_statistics(values):
    """Calculate comprehensive statistics for a list of values with robust error handling"""
    if not values or len(values) == 0:
        return {}
    
    # Convert and filter values safely
    clean_values = []
    for v in values:
        clean_v = safe_float_conversion(v)
        if clean_v is not None:
            clean_values.append(clean_v)
    
    if not clean_values:
        return {}
    
    try:
        values_array = np.array(clean_values, dtype=np.float64)
        
        return {
            'count': len(values_array),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'q10': float(np.percentile(values_array, 10)),
            'q20': float(np.percentile(values_array, 20)),
            'q30': float(np.percentile(values_array, 30)),
            'q40': float(np.percentile(values_array, 40)),
            'q50': float(np.percentile(values_array, 50)),
            'q60': float(np.percentile(values_array, 60)),
            'q70': float(np.percentile(values_array, 70)),
            'q80': float(np.percentile(values_array, 80)),
            'q90': float(np.percentile(values_array, 90)),
        }
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {}

def create_country_summary_dataset(country_daily_data, country_code):
    """Convert daily aggregated data into summary statistics dataset"""
    summary_records = []
    
    for game_date, data in country_daily_data.items():
        rating_stats = calculate_daily_statistics(data['ratings'])
        accuracy_stats = calculate_daily_statistics(data['accuracies'])
        
        if rating_stats:
            record = {
                'country': country_code,
                'date': game_date,
                'games_count': rating_stats['count'],
                'rating_mean': rating_stats['mean'],
                'rating_median': rating_stats['median'],
                'rating_std': rating_stats['std'],
                'rating_min': rating_stats['min'],
                'rating_max': rating_stats['max'],
                'rating_q10': rating_stats['q10'],
                'rating_q20': rating_stats['q20'],
                'rating_q30': rating_stats['q30'],
                'rating_q40': rating_stats['q40'],
                'rating_q50': rating_stats['q50'],
                'rating_q60': rating_stats['q60'],
                'rating_q70': rating_stats['q70'],
                'rating_q80': rating_stats['q80'],
                'rating_q90': rating_stats['q90'],
            }
            
            if accuracy_stats:
                accuracy_fields = {
                    'accuracy_count': accuracy_stats['count'],
                    'accuracy_mean': accuracy_stats['mean'],
                    'accuracy_median': accuracy_stats['median'],
                    'accuracy_std': accuracy_stats['std'],
                    'accuracy_min': accuracy_stats['min'],
                    'accuracy_max': accuracy_stats['max'],
                    'accuracy_q10': accuracy_stats['q10'],
                    'accuracy_q20': accuracy_stats['q20'],
                    'accuracy_q30': accuracy_stats['q30'],
                    'accuracy_q40': accuracy_stats['q40'],
                    'accuracy_q50': accuracy_stats['q50'],
                    'accuracy_q60': accuracy_stats['q60'],
                    'accuracy_q70': accuracy_stats['q70'],
                    'accuracy_q80': accuracy_stats['q80'],
                    'accuracy_q90': accuracy_stats['q90'],
                }
                record.update(accuracy_fields)
            
            summary_records.append(record)
    
    return pd.DataFrame(summary_records)

def calculate_rolling_averages(df, window_days=7):
    """Calculate rolling averages for the summary statistics"""
    df_sorted = df.sort_values(['country', 'date']).copy()
    
    rolling_columns = ['rating_mean', 'rating_median', 'rating_std', 'accuracy_mean', 'accuracy_median', 'accuracy_std']
    
    for col in rolling_columns:
        if col in df_sorted.columns:
            df_sorted[f'{col}_rolling_{window_days}d'] = (
                df_sorted.groupby('country')[col]
                .rolling(window=window_days, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    
    return df_sorted

async def main():
    logger.info("Starting OPTIMIZED chess country-level daily aggregation analysis")
    logger.info(f"Configuration: {MAX_CONCURRENT_REQUESTS} concurrent, {BATCH_SIZE} batch size")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("psutil not available")
    
    all_country_summaries = []
    
    for country_code, country_name in COUNTRIES.items():
        try:
            logger.info(f"\nStarting {country_name} ({country_code})")
            
            country_daily_data = await process_country_optimized(country_code)
            
            if country_daily_data:
                country_summary_df = create_country_summary_dataset(country_daily_data, country_code)
                
                if not country_summary_df.empty:
                    country_summary_df = calculate_rolling_averages(country_summary_df)
                    
                    country_file = os.path.join(PROCESSED_DATA_DIR, f'{country_code}_daily_summary.csv')
                    country_summary_df.to_csv(country_file, index=False)
                    
                    all_country_summaries.append(country_summary_df)
                    
                    logger.info(f"Completed {country_code}: {len(country_summary_df)} daily records")
                    logger.info(f"Date range: {country_summary_df['date'].min()} to {country_summary_df['date'].max()}")
                    logger.info(f"Total games: {country_summary_df['games_count'].sum()}")
                    
        except Exception as e:
            logger.error(f"Error processing {country_code}: {e}")
            continue
    
    if all_country_summaries:
        logger.info("Combining all country summaries...")
        combined_df = pd.concat(all_country_summaries, ignore_index=True)
        combined_df = combined_df.sort_values(['country', 'date']).reset_index(drop=True)
        
        final_output_file = os.path.join(PROCESSED_DATA_DIR, 'chess_country_daily_analytics.csv')
        combined_df.to_csv(final_output_file, index=False)
        
        logger.info(f"Final dataset saved: {final_output_file}")
        logger.info(f"Total records: {len(combined_df)}")
        logger.info(f"Countries: {combined_df['country'].nunique()}")
    
    logger.info("OPTIMIZED analysis completed!")

if __name__ == "__main__":
    asyncio.run(main())