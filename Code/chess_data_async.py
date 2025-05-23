import os
import asyncio
import aiohttp
import pandas as pd
import time
import json
from datetime import datetime
from tqdm.asyncio import tqdm
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/chess_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChessDataFetcher:
    def __init__(self, max_concurrent: int = 100, rate_limit_delay: float = 0.1):
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.session = None
        self.headers = {
            'User-Agent': 'ChessDataAnalysisBot/1.0 (+https://example.com)'
        }
        self.country_code = 'GB'
        self.output_csv = os.getenv('OUTPUT_CSV', '/data/daily_chess_stats_GB.csv')
        self.checkpoint_file = '/data/checkpoint.json'
        self.processed_users_file = '/data/processed_users.txt'
        self.daily_aggregates = {}  # Store daily aggregated data
        self.raw_data_buffer = []   # Temporary buffer for raw data
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def load_checkpoint(self) -> Dict:
        """Load processing checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'last_page': 0, 'total_processed': 0}

    def save_checkpoint(self, page: int, total_processed: int):
        """Save processing checkpoint"""
        checkpoint = {
            'last_page': page,
            'total_processed': total_processed,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    def get_processed_users(self) -> set:
        """Get set of already processed users"""
        if os.path.exists(self.processed_users_file):
            with open(self.processed_users_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def add_processed_user(self, username: str):
        """Add user to processed list"""
        with open(self.processed_users_file, 'a') as f:
            f.write(f"{username}\n")

    async def fetch_json(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch JSON with error handling and rate limiting"""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    logger.warning(f"Rate limited, waiting 60s...")
                    await asyncio.sleep(60)
                    return await self.fetch_json(url, params)
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def fetch_all_players(self) -> List[str]:
        """Fetch all GB players with pagination"""
        checkpoint = self.load_checkpoint()
        start_page = checkpoint['last_page'] + 1
        
        players_url = f'https://api.chess.com/pub/country/{self.country_code}/players'
        all_players = []
        page = start_page
        
        logger.info(f"Starting from page {start_page}")
        
        while True:
            logger.info(f"Fetching page {page}...")
            data = await self.fetch_json(players_url, {'page': page})
            
            if not data or 'players' not in data:
                logger.info(f"No more players found at page {page}")
                break
                
            batch = data.get('players', [])
            if not batch:
                break
                
            all_players.extend(batch)
            self.save_checkpoint(page, len(all_players))
            
            logger.info(f"Page {page}: {len(batch)} players, total: {len(all_players)}")
            page += 1
            
            # Save batch to avoid memory issues
            if len(all_players) >= 10000:  # Process in batches of 10k
                yield all_players
                all_players = []
        
        if all_players:
            yield all_players

    async def process_player_games(self, username: str, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Process all games for a single player"""
        async with semaphore:
            try:
                # Check if already processed
                processed_users = self.get_processed_users()
                if username in processed_users:
                    return []

                archives_url = f'https://api.chess.com/pub/player/{username}/games/archives'
                archives_data = await self.fetch_json(archives_url)
                
                if not archives_data or 'archives' not in archives_data:
                    self.add_processed_user(username)
                    return []

                records = []
                archives = archives_data.get('archives', [])
                
                # Process recent archives first (last 6 months for efficiency)
                recent_archives = archives[-6:] if len(archives) > 6 else archives
                
                for archive_url in recent_archives:
                    games_data = await self.fetch_json(archive_url)
                    if not games_data or 'games' not in games_data:
                        continue
                        
                    games = games_data.get('games', [])
                    for game in games:
                        timestamp = game.get('end_time', 0)
                        if timestamp == 0:
                            continue
                            
                        date = datetime.utcfromtimestamp(timestamp)
                        
                        # Check both white and black sides
                        for side in ['white', 'black']:
                            player = game.get(side, {})
                            if player.get('username', '').lower() != username.lower():
                                continue
                                
                            rating = player.get('rating')
                            accuracy = game.get('accuracies', {}).get(side) if 'accuracies' in game else None
                            
                            if rating:  # Only record if we have rating data
                                records.append({
                                    'username': username,
                                    'date': date,
                                    'rating': rating,
                                    'accuracy': accuracy
                                })
                
                self.add_processed_user(username)
                return records
                
            except Exception as e:
                logger.error(f"Error processing player {username}: {e}")
                return []

    async def process_player_batch(self, players: List[str]) -> pd.DataFrame:
        """Process a batch of players concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        tasks = [
            self.process_player_games(player, semaphore) 
            for player in players
        ]
        
        all_records = []
        results = await tqdm.gather(*tasks, desc="Processing players")
        
        for records in results:
            all_records.extend(records)
        
        return pd.DataFrame(all_records)

    def aggregate_daily_stats(self, df: pd.DataFrame):
        """Aggregate raw data into daily statistics"""
        if df.empty:
            return
            
        # Filter out players with no ratings
        df = df.dropna(subset=['rating'])
        df = df.sort_values(['username', 'date'])
        
        # Convert date to date only (remove time)
        df['date_only'] = df['date'].dt.date
        
        # Compute rolling Elo per player (10-game window) before aggregating
        df['rolling_elo'] = df.groupby('username')['rating'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # Group by date and calculate daily statistics
        daily_stats = []
        
        for date, day_data in df.groupby('date_only'):
            stats = {
                'date': date,
                'total_games': len(day_data),
                'unique_players': day_data['username'].nunique()
            }
            
            # Rating statistics
            ratings = day_data['rating'].dropna()
            if not ratings.empty:
                stats.update({
                    'rating_mean': ratings.mean(),
                    'rating_median': ratings.median(),
                    'rating_std': ratings.std(),
                    'rating_p10': ratings.quantile(0.1),
                    'rating_p20': ratings.quantile(0.2),
                    'rating_p30': ratings.quantile(0.3),
                    'rating_p40': ratings.quantile(0.4),
                    'rating_p50': ratings.quantile(0.5),
                    'rating_p60': ratings.quantile(0.6),
                    'rating_p70': ratings.quantile(0.7),
                    'rating_p80': ratings.quantile(0.8),
                    'rating_p90': ratings.quantile(0.9),
                    'rating_count': len(ratings)
                })
            
            # Rolling Elo statistics
            rolling_elos = day_data['rolling_elo'].dropna()
            if not rolling_elos.empty:
                stats.update({
                    'rolling_elo_mean': rolling_elos.mean(),
                    'rolling_elo_median': rolling_elos.median(),
                    'rolling_elo_std': rolling_elos.std(),
                    'rolling_elo_p10': rolling_elos.quantile(0.1),
                    'rolling_elo_p20': rolling_elos.quantile(0.2),
                    'rolling_elo_p30': rolling_elos.quantile(0.3),
                    'rolling_elo_p40': rolling_elos.quantile(0.4),
                    'rolling_elo_p50': rolling_elos.quantile(0.5),
                    'rolling_elo_p60': rolling_elos.quantile(0.6),
                    'rolling_elo_p70': rolling_elos.quantile(0.7),
                    'rolling_elo_p80': rolling_elos.quantile(0.8),
                    'rolling_elo_p90': rolling_elos.quantile(0.9)
                })
            
            # Accuracy statistics (if available)
            accuracies = day_data['accuracy'].dropna()
            if not accuracies.empty:
                stats.update({
                    'accuracy_mean': accuracies.mean(),
                    'accuracy_median': accuracies.median(),
                    'accuracy_std': accuracies.std(),
                    'accuracy_p10': accuracies.quantile(0.1),
                    'accuracy_p20': accuracies.quantile(0.2),
                    'accuracy_p30': accuracies.quantile(0.3),
                    'accuracy_p40': accuracies.quantile(0.4),
                    'accuracy_p50': accuracies.quantile(0.5),
                    'accuracy_p60': accuracies.quantile(0.6),
                    'accuracy_p70': accuracies.quantile(0.7),
                    'accuracy_p80': accuracies.quantile(0.8),
                    'accuracy_p90': accuracies.quantile(0.9),
                    'accuracy_count': len(accuracies)
                })
            else:
                # Fill with NaN if no accuracy data
                accuracy_cols = ['accuracy_mean', 'accuracy_median', 'accuracy_std'] + \
                               [f'accuracy_p{i}0' for i in range(1, 10)] + ['accuracy_count']
                for col in accuracy_cols:
                    stats[col] = None
            
            daily_stats.append(stats)
        
        # Store in aggregates dict
        for stat in daily_stats:
            date_key = stat['date']
            if date_key not in self.daily_aggregates:
                self.daily_aggregates[date_key] = stat
            else:
                # Merge with existing data for this date
                existing = self.daily_aggregates[date_key]
                self.daily_aggregates[date_key] = self.merge_daily_stats(existing, stat)

    def merge_daily_stats(self, existing: Dict, new: Dict) -> Dict:
        """Merge two daily statistics dictionaries"""
        merged = existing.copy()
        
        # Simple additive fields
        merged['total_games'] += new['total_games']
        merged['unique_players'] = max(existing.get('unique_players', 0), new.get('unique_players', 0))
        
        # For statistical measures, we need to combine the underlying data
        # This is a simplified approach - for exact statistics, we'd need raw data
        # But this gives good approximations for large datasets
        
        def weighted_merge(field_base: str):
            """Merge statistical fields with weighted averages"""
            old_count = existing.get(f'{field_base}_count', 0)
            new_count = new.get(f'{field_base}_count', 0)
            total_count = old_count + new_count
            
            if total_count == 0:
                return
                
            if old_count == 0:
                for suffix in ['mean', 'median', 'std'] + [f'p{i}0' for i in range(1, 10)]:
                    key = f'{field_base}_{suffix}'
                    if key in new:
                        merged[key] = new[key]
                merged[f'{field_base}_count'] = new_count
            elif new_count == 0:
                pass  # Keep existing values
            else:
                # Weighted average for mean
                if f'{field_base}_mean' in existing and f'{field_base}_mean' in new:
                    old_mean = existing[f'{field_base}_mean']
                    new_mean = new[f'{field_base}_mean']
                    merged[f'{field_base}_mean'] = (old_mean * old_count + new_mean * new_count) / total_count
                
                # For median and percentiles, take weighted average (approximation)
                for suffix in ['median'] + [f'p{i}0' for i in range(1, 10)]:
                    key = f'{field_base}_{suffix}'
                    if key in existing and key in new:
                        old_val = existing[key]
                        new_val = new[key]
                        merged[key] = (old_val * old_count + new_val * new_count) / total_count
                
                # For std, use pooled standard deviation approximation
                if f'{field_base}_std' in existing and f'{field_base}_std' in new:
                    old_std = existing[f'{field_base}_std']
                    new_std = new[f'{field_base}_std']
                    old_mean = existing.get(f'{field_base}_mean', 0)
                    new_mean = new.get(f'{field_base}_mean', 0)
                    combined_mean = merged[f'{field_base}_mean']
                    
                    # Pooled variance approximation
                    old_var = old_std ** 2 + (old_mean - combined_mean) ** 2
                    new_var = new_std ** 2 + (new_mean - combined_mean) ** 2
                    combined_var = (old_var * old_count + new_var * new_count) / total_count
                    merged[f'{field_base}_std'] = combined_var ** 0.5
                
                merged[f'{field_base}_count'] = total_count
        
        # Apply weighted merge to each metric
        for field_base in ['rating', 'rolling_elo', 'accuracy']:
            if f'{field_base}_count' in existing or f'{field_base}_count' in new:
                weighted_merge(field_base)
        
        return merged

    def save_daily_aggregates_to_csv(self):
        """Save accumulated daily aggregates to CSV"""
        if not self.daily_aggregates:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(list(self.daily_aggregates.values()))
        df = df.sort_values('date')
        
        # Add rolling statistics over time (7-day and 30-day windows)
        numeric_cols = [col for col in df.columns if col not in ['date', 'unique_players']]
        
        for window in [7, 30]:
            for col in numeric_cols:
                if df[col].dtype in ['float64', 'int64'] and not col.endswith('_count'):
                    df[f'{col}_rolling_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Save to CSV
        df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved {len(df)} daily aggregates to {self.output_csv}")
        
        return df

    async def run(self):
        """Main execution function"""
        logger.info("Starting chess data analysis for 6M GB users...")
        
        total_processed = 0
        
        async for player_batch in self.fetch_all_players():
            logger.info(f"Processing batch of {len(player_batch)} players...")
            
            df = await self.process_player_batch(player_batch)
            
            if not df.empty:
                self.aggregate_daily_stats(df)
                total_processed += len(df)
                logger.info(f"Total records processed so far: {total_processed}")
                
                # Save aggregates periodically (every 50k records)
                if total_processed % 50000 < len(df):
                    self.save_daily_aggregates_to_csv()
            
            # Optional: Add delay between batches to be nice to the API
            await asyncio.sleep(5)
        
        # Final save
        final_df = self.save_daily_aggregates_to_csv()
        
        logger.info(f"Completed processing! Total records: {total_processed}")
        logger.info(f"Daily aggregates saved with {len(self.daily_aggregates)} unique dates")
        
        # Generate summary statistics
        if final_df is not None:
            self.generate_summary(final_df)

    def generate_summary(self, df: pd.DataFrame):
        """Generate summary statistics from daily aggregates"""
        try:
            logger.info("Generating summary statistics from daily aggregates...")
            
            summary_stats = {
                'total_days': len(df),
                'date_range': {
                    'min': df['date'].min(),
                    'max': df['date'].max()
                },
                'total_games_processed': df['total_games'].sum(),
                'peak_daily_games': df['total_games'].max(),
                'avg_daily_games': df['total_games'].mean(),
                'unique_players_peak': df['unique_players'].max(),
                'avg_daily_unique_players': df['unique_players'].mean(),
                'rating_stats': {
                    'overall_mean': df['rating_mean'].mean(),
                    'overall_median': df['rating_median'].mean(),
                    'mean_std': df['rating_std'].mean(),
                },
                'accuracy_stats': {
                    'overall_mean': df['accuracy_mean'].mean(),
                    'overall_median': df['accuracy_median'].mean(), 
                    'mean_std': df['accuracy_std'].mean(),
                } if 'accuracy_mean' in df.columns and df['accuracy_mean'].notna().any() else None
            }
            
            # Save summary
            summary_file = '/data/daily_aggregates_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"Summary: {summary_stats['total_days']} days, {summary_stats['total_games_processed']:,} total games")
            logger.info(f"Peak daily games: {summary_stats['peak_daily_games']:,}, Peak unique players: {summary_stats['unique_players_peak']:,}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


async def main():
    """Main entry point"""
    max_concurrent = int(os.getenv('CONCURRENT_REQUESTS', 100))
    
    async with ChessDataFetcher(max_concurrent=max_concurrent) as fetcher:
        await fetcher.run()


if __name__ == "__main__":
    asyncio.run(main())