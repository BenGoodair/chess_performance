import pandas as pd
import numpy as np
import logging

def setup_logging(log_file):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_statistics(data, column_name):
    """Calculate comprehensive statistics for a column"""
    if len(data) == 0:
        return {}
    
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return {}
    
    stats = {
        f'{column_name}_mean': clean_data.mean(),
        f'{column_name}_median': clean_data.median(),
        f'{column_name}_std': clean_data.std(),
        f'{column_name}_count': len(clean_data)
    }
    
    # Calculate deciles (10th, 20th, ..., 90th percentiles)
    for i in range(1, 10):
        percentile = i * 10
        stats[f'{column_name}_p{percentile}'] = clean_data.quantile(i/10)
    
    return stats

def save_checkpoint(data, filename):
    """Save data as checkpoint in case of interruption"""
    try:
        df = pd.DataFrame(data)
        checkpoint_file = filename.replace('.csv', '_checkpoint.csv')
        df.to_csv(checkpoint_file, index=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        return False

def load_checkpoint(filename):
    """Load data from checkpoint file"""
    try:
        checkpoint_file = filename.replace('.csv', '_checkpoint.csv')
        df = pd.read_csv(checkpoint_file)
        return df.to_dict('records')
    except Exception:
        return None