import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def analyze_chess_data(csv_path='/data/daily_chess_stats_GB.csv'):
    """Analyze and visualize the daily chess statistics"""
    
    # Load the data
    print("Loading daily chess statistics...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print(f"Data loaded: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    print(f"Total games processed: {df['total_games'].sum():,}")
    print(f"Peak daily games: {df['total_games'].max():,}")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Chess.com GB Players - Daily Statistics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Daily Games Over Time
    axes[0, 0].plot(df['date'], df['total_games'], linewidth=1, alpha=0.7)
    axes[0, 0].plot(df['date'], df['total_games'].rolling(30).mean(), 'r-', linewidth=2, label='30-day avg')
    axes[0, 0].set_title('Daily Games Played')
    axes[0, 0].set_ylabel('Number of Games')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Unique Players Over Time
    axes[0, 1].plot(df['date'], df['unique_players'], linewidth=1, alpha=0.7)
    axes[0, 1].plot(df['date'], df['unique_players'].rolling(30).mean(), 'r-', linewidth=2, label='30-day avg')
    axes[0, 1].set_title('Daily Unique Players')
    axes[0, 1].set_ylabel('Number of Players')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rating Distribution Evolution
    axes[1, 0].plot(df['date'], df['rating_mean'], label='Mean', linewidth=2)
    axes[1, 0].plot(df['date'], df['rating_median'], label='Median', linewidth=2)
    axes[1, 0].fill_between(df['date'], 
                           df['rating_mean'] - df['rating_std'], 
                           df['rating_mean'] + df['rating_std'], 
                           alpha=0.2, label='±1 Std Dev')
    axes[1, 0].set_title('Rating Statistics Over Time')
    axes[1, 0].set_ylabel('Rating')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rating Percentiles
    percentiles = ['rating_p10', 'rating_p25', 'rating_p50', 'rating_p75', 'rating_p90']
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    
    for i, (pct, color) in enumerate(zip(percentiles, colors)):
        if pct in df.columns:
            label = f"{pct.split('_p')[1]}th percentile"
            axes[1, 1].plot(df['date'], df[pct], color=color, label=label, linewidth=2)
    
    axes[1, 1].set_title('Rating Percentiles Over Time')
    axes[1, 1].set_ylabel('Rating')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Accuracy Statistics (if available)
    if 'accuracy_mean' in df.columns and df['accuracy_mean'].notna().any():
        axes[2, 0].plot(df['date'], df['accuracy_mean'], label='Mean', linewidth=2)
        axes[2, 0].plot(df['date'], df['accuracy_median'], label='Median', linewidth=2)
        axes[2, 0].fill_between(df['date'], 
                               df['accuracy_mean'] - df['accuracy_std'], 
                               df['accuracy_mean'] + df['accuracy_std'], 
                               alpha=0.2, label='±1 Std Dev')
        axes[2, 0].set_title('Accuracy Statistics Over Time')
        axes[2, 0].set_ylabel('Accuracy (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Accuracy Percentiles
        acc_percentiles = ['accuracy_p10', 'accuracy_p25', 'accuracy_p50', 'accuracy_p75', 'accuracy_p90']
        for i, (pct, color) in enumerate(zip(acc_percentiles, colors)):
            if pct in df.columns:
                label = f"{pct.split('_p')[1]}th percentile"
                axes[2, 1].plot(df['date'], df[pct], color=color, label=label, linewidth=2)
        
        axes[2, 1].set_title('Accuracy Percentiles Over Time')
        axes[2, 1].set_ylabel('Accuracy (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'No Accuracy Data Available', 
                       transform=axes[2, 0].transAxes, ha='center', va='center', fontsize=14)
        axes[2, 1].text(0.5, 0.5, 'No Accuracy Data Available', 
                       transform=axes[2, 1].transAxes, ha='center', va='center', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('/app/Figures/chess_daily_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('/app/Figures/chess_daily_analysis.pdf', bbox_inches='tight')
    
    print("Visualizations saved to /app/Figures/")
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total days: {len(df)}")
    print(f"Total games: {df['total_games'].sum():,}")
    print(f"Average daily games: {df['total_games'].mean():.0f}")
    print(f"Peak unique players in a day: {df['unique_players'].max():,}")
    print(f"Average rating: {df['rating_mean'].mean():.0f}")
    print(f"Rating standard deviation: {df['rating_std'].mean():.0f}")
    
    if 'accuracy_mean' in df.columns and df['accuracy_mean'].notna().any():
        print(f"Average accuracy: {df['accuracy_mean'].mean():.1f}%")
    
    # Trend analysis
    recent_30_days = df.tail(30)
    older_30_days = df.iloc[-(60+30):-30] if len(df) >= 60 else df.head(30)
    
    if len(recent_30_days) > 0 and len(older_30_days) > 0:
        rating_trend = recent_30_days['rating_mean'].mean() - older_30_days['rating_mean'].mean()
        games_trend = recent_30_days['total_games'].mean() - older_30_days['total_games'].mean()
        
        print(f"\n=== 30-DAY TRENDS ===")
        print(f"Rating trend: {rating_trend:+.1f} points")
        print(f"Daily games trend: {games_trend:+.0f} games")
    
    return df

if __name__ == "__main__":
    analyze_chess_data()