import pandas as pd
import numpy as np
import logging
import pickle
import gzip
import os
import glob
import re

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

def save_checkpoint(data, batch_num, checkpoint_dir="checkpoints"):
    """Save data as compressed pickle checkpoint"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = f"{checkpoint_dir}/checkpoint_GB_batch_{batch_num}.pkl.gz"
        
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logging.info(f"Checkpoint saved: {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        return False

def load_checkpoint(batch_num, checkpoint_dir="checkpoints"):
    """Load data from specific checkpoint file"""
    try:
        filename = f"{checkpoint_dir}/checkpoint_GB_batch_{batch_num}.pkl.gz"
        
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
        
        logging.info(f"Checkpoint loaded: {filename}")
        return data
    except Exception as e:
        logging.error(f"Failed to load checkpoint {filename}: {e}")
        return None

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint file and return the batch number"""
    try:
        # Look for checkpoint files
        pattern = f"{checkpoint_dir}/checkpoint_GB_batch_*.pkl.gz"
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            logging.info("No checkpoint files found")
            return None, None
        
        # Extract batch numbers and find the maximum
        batch_numbers = []
        for file in checkpoint_files:
            match = re.search(r'checkpoint_GB_batch_(\d+)\.pkl\.gz', file)
            if match:
                batch_numbers.append(int(match.group(1)))
        
        if not batch_numbers:
            return None, None
        
        latest_batch = max(batch_numbers)
        latest_file = f"{checkpoint_dir}/checkpoint_GB_batch_{latest_batch}.pkl.gz"
        
        logging.info(f"Latest checkpoint found: batch {latest_batch}")
        return latest_batch, latest_file
        
    except Exception as e:
        logging.error(f"Error finding latest checkpoint: {e}")
        return None, None

def load_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Load the most recent checkpoint data"""
    latest_batch, latest_file = find_latest_checkpoint(checkpoint_dir)
    
    if latest_batch is None:
        logging.info("No checkpoints found, starting from beginning")
        return None, 0
    
    data = load_checkpoint(latest_batch, checkpoint_dir)
    if data is not None:
        logging.info(f"Resumed from checkpoint batch {latest_batch}")
        return data, latest_batch
    else:
        logging.error(f"Failed to load latest checkpoint")
        return None, 0

def cleanup_old_checkpoints(checkpoint_dir="checkpoints", keep_last_n=5):
    """Remove old checkpoint files, keeping only the last N"""
    try:
        pattern = f"{checkpoint_dir}/checkpoint_GB_batch_*.pkl.gz"
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= keep_last_n:
            return
        
        # Sort files by batch number
        file_batches = []
        for file in checkpoint_files:
            match = re.search(r'checkpoint_GB_batch_(\d+)\.pkl\.gz', file)
            if match:
                file_batches.append((int(match.group(1)), file))
        
        file_batches.sort(key=lambda x: x[0])  # Sort by batch number
        
        # Remove older files
        files_to_remove = file_batches[:-keep_last_n]
        for batch_num, file_path in files_to_remove:
            os.remove(file_path)
            logging.info(f"Removed old checkpoint: {file_path}")
            
    except Exception as e:
        logging.error(f"Error cleaning up checkpoints: {e}")

# Legacy CSV functions (kept for backward compatibility)
def save_checkpoint_csv(data, filename):
    """Save data as checkpoint in CSV format (legacy)"""
    try:
        df = pd.DataFrame(data)
        checkpoint_file = filename.replace('.csv', '_checkpoint.csv')
        df.to_csv(checkpoint_file, index=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save CSV checkpoint: {e}")
        return False

def load_checkpoint_csv(filename):
    """Load data from CSV checkpoint file (legacy)"""
    try:
        checkpoint_file = filename.replace('.csv', '_checkpoint.csv')
        df = pd.read_csv(checkpoint_file)
        return df.to_dict('records')
    except Exception:
        return None