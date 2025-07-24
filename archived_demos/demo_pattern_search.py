"""
Demo script for Pattern Search functionality

This script demonstrates:
1. Uploading custom patterns
2. Finding similar patterns across tickers
3. Pattern-based backtesting
4. Creating and checking pattern alerts
5. Pattern library management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.dashboard.search import (
    PatternSearchEngine,
    quick_pattern_upload,
    quick_pattern_search,
    quick_pattern_backtest
)

def generate_sample_data(ticker: str, days: int = 100) -> pd.DataFrame:
    """Generate sample price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    np.random.seed(hash(ticker) % 1000)
    returns = np.random.normal(0.001, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Add some patterns
    if ticker == "AAPL":
        # Add a head and shoulders pattern around day 50
        pattern_days = 20
        pattern_start = 40
        for i in range(pattern_days):
            if i < 5:
                price[pattern_start + i] *= 1.02
            elif i < 10:
                price[pattern_start + i] *= 0.98
            elif i < 15:
                price[pattern_start + i] *= 1.03
            else:
                price[pattern_start + i] *= 0.97
    
    return pd.DataFrame({
        'open': price * np.random.uniform(0.99, 1.01, days),
        'high': price * np.random.uniform(1.01, 1.03, days),
        'low': price * np.random.uniform(0.97, 0.99, days),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)

def create_sample_patterns():
    """Create some sample patterns for testing"""
    patterns = []
    
    # 1. Bull flag pattern
    bull_flag = np.array([
        100, 102, 104, 106, 108,  # Initial rise
        107, 106.5, 106, 106.5, 107,  # Consolidation
        108, 110, 112, 114, 116  # Breakout
    ])
    
    # 2. Double bottom pattern
    double_bottom = np.array([
        100, 98, 96, 94, 92,  # First decline
        93, 94, 95, 96, 97,   # First recovery
        96, 95, 94, 93, 92,   # Second decline
        93, 95, 97, 99, 101   # Final recovery
    ])
    
    # 3. Ascending triangle pattern
    ascending_triangle = np.array([
        100, 98, 99, 97, 98.5,
        96.5, 98, 97, 99, 98,
        99.5, 98.5, 100, 99, 101
    ])
    
    return {
        'bull_flag': bull_flag,
        'double_bottom': double_bottom,
        'ascending_triangle': ascending_triangle
    }

def demo_pattern_upload():
    """Demo: Upload custom patterns"""
    print("\n=== Pattern Upload Demo ===")
    
    patterns = create_sample_patterns()
    uploaded_patterns = []
    
    for name, data in patterns.items():
        pattern = quick_pattern_upload(
            name=name,
            data=data,
            description=f"Sample {name.replace('_', ' ')} pattern",
            tags=['technical', 'price_pattern']
        )
        uploaded_patterns.append(pattern)
        print(f"Uploaded pattern: {pattern.name} (ID: {pattern.id})")
        print(f"  - Length: {len(pattern.data)} bars")
        print(f"  - Mean: {pattern.metadata['mean']:.2f}")
        print(f"  - Std: {pattern.metadata['std']:.2f}")
    
    return uploaded_patterns

def demo_pattern_search(pattern_id: str):
    """Demo: Search for similar patterns"""
    print("\n=== Pattern Search Demo ===")
    
    # Generate sample data for multiple tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    ticker_data = {ticker: generate_sample_data(ticker) for ticker in tickers}
    
    # Search for pattern
    matches = quick_pattern_search(
        pattern_id=pattern_id,
        ticker_data=ticker_data,
        threshold=0.7
    )
    
    print(f"Found {len(matches)} matches with similarity >= 0.7")
    
    # Display top matches
    for i, match in enumerate(matches[:5]):
        print(f"\nMatch {i+1}:")
        print(f"  - Ticker: {match.ticker}")
        print(f"  - Date: {match.timestamp.strftime('%Y-%m-%d')}")
        print(f"  - Similarity: {match.similarity_score:.3f}")
        print(f"  - Pattern: {match.metadata['pattern_name']}")
    
    return matches, ticker_data

def demo_pattern_backtest(pattern_id: str, ticker_data: dict):
    """Demo: Backtest a pattern"""
    print("\n=== Pattern Backtest Demo ===")
    
    result = quick_pattern_backtest(
        pattern_id=pattern_id,
        ticker_data=ticker_data,
        entry_threshold=0.75
    )
    
    print(f"Backtest Results:")
    print(f"  - Total trades: {result.total_trades}")
    print(f"  - Win rate: {result.win_rate:.1%}")
    print(f"  - Total return: {result.total_return:.2%}")
    print(f"  - Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"  - Max drawdown: {result.max_drawdown:.2%}")
    print(f"  - Profit factor: {result.profit_factor:.2f}")
    
    # Show trade distribution
    if result.trades:
        returns = [t['net_return'] for t in result.trades]
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
        plt.xlabel('Return per Trade')
        plt.ylabel('Frequency')
        plt.title('Trade Return Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pattern_backtest_distribution.png', dpi=150)
        plt.close()
        print("\nTrade distribution saved to pattern_backtest_distribution.png")
    
    return result

def demo_pattern_alerts():
    """Demo: Create and check pattern alerts"""
    print("\n=== Pattern Alerts Demo ===")
    
    engine = PatternSearchEngine()
    
    # Get first pattern
    patterns = list(engine.library.patterns.values())
    if not patterns:
        print("No patterns found. Please run pattern upload demo first.")
        return
    
    pattern = patterns[0]
    
    # Create alerts
    alert1 = engine.create_alert(
        pattern_id=pattern.id,
        ticker="AAPL",
        condition="match",
        threshold=0.85
    )
    
    alert2 = engine.create_alert(
        pattern_id=pattern.id,
        ticker="GOOGL",
        condition="match",
        threshold=0.80
    )
    
    print(f"Created {len(engine.alerts)} alerts:")
    for alert in engine.alerts.values():
        print(f"  - Alert {alert.id[:8]}...")
        print(f"    Pattern: {alert.metadata['pattern_name']}")
        print(f"    Ticker: {alert.ticker}")
        print(f"    Threshold: {alert.threshold}")
    
    # Check alerts with sample data
    ticker_data = {
        'AAPL': generate_sample_data('AAPL', 10),
        'GOOGL': generate_sample_data('GOOGL', 10)
    }
    
    triggered = engine.check_alerts(ticker_data)
    print(f"\nTriggered alerts: {len(triggered)}")
    
    return engine.alerts

def demo_pattern_library():
    """Demo: Pattern library management"""
    print("\n=== Pattern Library Demo ===")
    
    engine = PatternSearchEngine()
    
    # Search patterns by criteria
    technical_patterns = engine.library.search_patterns(tags=['technical'])
    print(f"Technical patterns: {len(technical_patterns)}")
    
    # Get pattern statistics
    if engine.library.patterns:
        pattern_id = list(engine.library.patterns.keys())[0]
        stats = engine.get_pattern_statistics(pattern_id)
        
        print(f"\nPattern Statistics for {stats['pattern_info']['name']}:")
        print(f"  - Created: {stats['pattern_info']['created_at']}")
        print(f"  - Length: {stats['pattern_info']['length']} bars")
        print(f"  - Tags: {', '.join(stats['pattern_info']['tags'])}")
        
        if stats['performance_metrics']:
            print(f"  - Performance:")
            for metric, value in stats['performance_metrics'].items():
                print(f"    {metric}: {value:.3f}")
    
    # Export pattern
    if engine.library.patterns:
        pattern_id = list(engine.library.patterns.keys())[0]
        json_export = engine.export_pattern(pattern_id, format='json')
        print(f"\nExported pattern to JSON ({len(json_export)} chars)")
        
        # Save to file
        with open('exported_pattern.json', 'w') as f:
            f.write(json_export)
        print("Pattern saved to exported_pattern.json")

def visualize_pattern_matches(matches, ticker_data):
    """Visualize pattern matches on price charts"""
    if not matches:
        return
    
    # Group matches by ticker
    matches_by_ticker = {}
    for match in matches[:10]:  # Limit to first 10 matches
        if match.ticker not in matches_by_ticker:
            matches_by_ticker[match.ticker] = []
        matches_by_ticker[match.ticker].append(match)
    
    # Create subplots
    n_tickers = len(matches_by_ticker)
    if n_tickers == 0:
        return
    
    fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4*n_tickers))
    if n_tickers == 1:
        axes = [axes]
    
    for idx, (ticker, ticker_matches) in enumerate(matches_by_ticker.items()):
        ax = axes[idx]
        df = ticker_data[ticker]
        
        # Plot price
        ax.plot(df.index, df['close'], 'b-', alpha=0.7, label='Close Price')
        
        # Highlight matches
        for match in ticker_matches:
            start_idx, end_idx = match.location
            match_dates = df.index[start_idx:end_idx]
            match_prices = df['close'].iloc[start_idx:end_idx]
            
            ax.plot(match_dates, match_prices, 'r-', linewidth=3, 
                   label=f'Match (sim={match.similarity_score:.2f})')
            
            # Add shaded region
            ax.axvspan(match_dates[0], match_dates[-1], 
                      alpha=0.2, color='red')
        
        ax.set_title(f'{ticker} - Pattern Matches')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pattern_matches_visualization.png', dpi=150)
    plt.close()
    print("\nPattern matches visualization saved to pattern_matches_visualization.png")

def main():
    """Run all demos"""
    print("Pattern Search Functionality Demo")
    print("=" * 50)
    
    # 1. Upload patterns
    patterns = demo_pattern_upload()
    
    if patterns:
        # 2. Search for patterns
        pattern_id = patterns[0].id
        matches, ticker_data = demo_pattern_search(pattern_id)
        
        # 3. Visualize matches
        if matches:
            visualize_pattern_matches(matches, ticker_data)
        
        # 4. Backtest pattern
        demo_pattern_backtest(pattern_id, ticker_data)
        
        # 5. Create alerts
        demo_pattern_alerts()
        
        # 6. Library management
        demo_pattern_library()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nGenerated files:")
    print("  - pattern_backtest_distribution.png")
    print("  - pattern_matches_visualization.png")
    print("  - exported_pattern.json")
    print("\nPattern library stored in: data/pattern_library/")
    print("Pattern cache stored in: data/pattern_cache/")

if __name__ == "__main__":
    main()
