import os

# API Configuration
HEADERS = {
    'User-Agent': 'ChessDataAnalysisBot/1.0 (+https://example.com)'
}

# Rate limiting (seconds between requests)
RATE_LIMIT_PLAYERS = 0.5
RATE_LIMIT_ARCHIVES = 0.2
RATE_LIMIT_PAGES = 0.5

# Countries to analyze
COUNTRIES = {
    'GB': 'Great Britain',
    'FR': 'France', 
    'DE': 'Germany'
}

# File paths (works on both local and AWS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SUMMARIES_DIR = os.path.join(DATA_DIR, 'summaries')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SUMMARIES_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Output file paths
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'all_countries_raw_data.csv')
SUMMARY_STATS_FILE = os.path.join(SUMMARIES_DIR, 'chess_statistics_summary.csv')
LOG_FILE = os.path.join(LOGS_DIR, 'chess_analysis.log')

# Processing limits (to prevent infinite loops)
MAX_PAGES_PER_COUNTRY = 1000
MAX_PLAYERS_PER_COUNTRY = None  # Set to a number to limit for testing

# Rolling window size for ELO calculation
ROLLING_WINDOW_SIZE = 10