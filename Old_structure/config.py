import os

# API Configuration - Optimized for Chess.com API limits
HEADERS = {
    'User-Agent': 'ChessDataAnalysisBot/1.0 (username: clluelless; contact: benjamin.goodair@bsg.ox.ac.uk)',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate'  # Enable compression for up to 80% bandwidth reduction
}

# Rate limiting - Optimized based on Chess.com's official limits
# Chess.com allows unlimited serial requests but limits parallel requests
# Official limit: max 2 concurrent requests, 3 archives per second for bulk operations
RATE_LIMIT_PLAYERS = 0.0     # No delay needed for serial requests
RATE_LIMIT_ARCHIVES = 0.34   # ~3 requests per second (1/3 = 0.33, rounded up for safety)
RATE_LIMIT_PAGES = 0.0       # No delay needed for serial requests

# Concurrent request limits - Chess.com's hard limit
MAX_CONCURRENT_REQUESTS = 2   # Chess.com blocks at >2 concurrent requests
BATCH_SIZE = 50               # Smaller batches to reduce individual request load
CLEANUP_FREQUENCY = 10        # Less frequent cleanup since we're being more conservative

# Countries to analyze - Extended list
COUNTRIES = {
    'GB': 'Great Britain',
    'FR': 'France',
    'DE': 'Germany',
    'US': 'United States',      # USA
    'IN': 'India',              # India
    'RU': 'Russia',             # Russia
    'UA': 'Ukraine',            # Ukraine
    'CA': 'Canada',             # Canada
    'BR': 'Brazil',             # Brazil
    'IT': 'Italy',              # Italy
    'ES': 'Spain',              # Strong chess tradition
    'NL': 'Netherlands',        # Strong chess culture
    'NO': 'Norway',             # Magnus Carlsen's country
    'AR': 'Argentina',          # Strong South American presence
    'CN': 'China',              # Growing chess power
    'PL': 'Poland',             # Strong European chess
    'SE': 'Sweden',             # Nordic chess strength
    'DK': 'Denmark',            # Nordic region
    'AU': 'Australia',          # Oceania representation
    'MX': 'Mexico',             # North American diversity
    'TR': 'Turkey',             # Bridge between Europe/Asia
    'IR': 'Iran',               # Strong Middle Eastern chess
    'AM': 'Armenia',            # Chess powerhouse (small but mighty)
    'GE': 'Georgia',            # Strong chess tradition
    'IS': 'Iceland',            # Bobby Fischer connection
    'CZ': 'Czech Republic',     # Strong Central European chess
    'HU': 'Hungary',            # Historic chess strength
    'RO': 'Romania',            # Eastern European chess
    'GR': 'Greece',             # Mediterranean representation
    'IL': 'Israel',             # Strong chess culture
    'JP': 'Japan',              # Asian representation
    'KR': 'South Korea',        # Growing Asian chess market
    'CL': 'Chile',              # South American diversity
    'CO': 'Colombia',           # Growing chess scene
    'PE': 'Peru',               # Andean chess representation
    'VE': 'Venezuela',          # Caribbean/South American chess
    'EG': 'Egypt',              # African chess representation
    'ZA': 'South Africa',       # Southern African chess
    'NG': 'Nigeria',            # West African chess
    'MA': 'Morocco',            # North African chess
    'PH': 'Philippines',        # Southeast Asian chess
    'TH': 'Thailand',           # Southeast Asian representation
    'MY': 'Malaysia',           # Southeast Asian diversity
    'SG': 'Singapore',          # Southeast Asian hub
    'ID': 'Indonesia',          # Large Southeast Asian nation
    'VN': 'Vietnam',            # Growing Asian chess scene
    'BD': 'Bangladesh',         # South Asian representation
    'PK': 'Pakistan',           # South Asian chess
    'LK': 'Sri Lanka',          # Island nation chess
    'NZ': 'New Zealand',        # Oceania completion
    'FI': 'Finland',            # Nordic completion
    'EE': 'Estonia',            # Baltic chess
    'LV': 'Latvia',             # Baltic representation
    'LT': 'Lithuania',          # Baltic completion
    'BY': 'Belarus',            # Eastern European chess
    'MD': 'Moldova',            # Eastern European diversity
    'BG': 'Bulgaria',           # Southeastern European chess
    'RS': 'Serbia',             # Balkan chess strength
    'HR': 'Croatia',            # Balkan representation
    'SI': 'Slovenia',           # Alpine chess
    'SK': 'Slovakia',           # Central European chess
    'AT': 'Austria',            # Central European representation
    'CH': 'Switzerland',        # Alpine chess culture
    'BE': 'Belgium',            # Western European chess
    'LU': 'Luxembourg',         # Small European nation
    'PT': 'Portugal',           # Iberian Peninsula completion
    'IE': 'Ireland',            # Celtic chess
    'MT': 'Malta',              # Mediterranean island chess
    'CY': 'Cyprus',             # Eastern Mediterranean
}

# File paths (works on both local and AWS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SUMMARIES_DIR = os.path.join(DATA_DIR, 'summaries')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SUMMARIES_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Output file paths
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'all_countries_raw_data.csv')
SUMMARY_STATS_FILE = os.path.join(SUMMARIES_DIR, 'chess_statistics_summary.csv')
LOG_FILE = os.path.join(LOGS_DIR, 'chess_analysis.log')

# Processing limits - Restored to original ambitious targets
MAX_PAGES_PER_COUNTRY = 1000
MAX_PLAYERS_PER_COUNTRY = 10000

# Rolling window size for ELO calculation
ROLLING_WINDOW_SIZE = 10

# Debug settings
DEBUG_MODE = True
VERBOSE_LOGGING = True

# API endpoint base URLs
CHESS_COM_API_BASE = 'https://api.chess.com/pub'

# Timeout settings - Increased for potentially slower responses
REQUEST_TIMEOUT = 90          # Increased for archive requests which can be large
CONNECT_TIMEOUT = 30          # Increased connection timeout

# Retry settings - Enhanced for better resilience
MAX_RETRIES = 5               # More retries since we're being more conservative
RETRY_BACKOFF_FACTOR = 3      # Longer backoff to avoid triggering rate limits
MIN_RETRY_DELAY = 2           # Minimum 2 seconds between retries
MAX_RETRY_DELAY = 300         # Max 5 minutes between retries

# Cache settings - Leverage Chess.com's 12-hour cache cycle
USE_ETAG_CACHING = True       # Use ETag headers for efficient caching
CACHE_INVALIDATION_HOURS = 12 # Chess.com updates cache every 12 hours max

# Request strategy settings
PREFER_SERIAL_REQUESTS = True # Use serial requests when possible for unlimited rate
ARCHIVE_REQUEST_DELAY = 0.34  # Specific delay for archive requests (3/second limit)
BURST_PROTECTION = True       # Implement burst protection to avoid abnormal activity detection

# Error handling
HANDLE_429_RETRY_AFTER = True # Check Retry-After header in 429 responses
EXPONENTIAL_BACKOFF_429 = True # Use exponential backoff for 429 errors
MAX_429_RETRIES = 10          # Maximum retries for rate limit errors