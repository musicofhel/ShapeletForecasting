"""
Demo script for the time series visualization component
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.dashboard.visualizations.timeseries import TimeSeriesVisualizer, create_timeseries_component
import plotly.io as pio

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_realistic_price_data(ticker: str, start_date: str, end_date: str, freq: str = '1H'):
    """Generate realistic price data with trends and volatility"""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Base prices
    base_prices = {
        'BTC': 40000,
        'ETH': 2500,
        'SOL': 100
    }
    
    base_price = base_prices.get(ticker, 1000)
    data = []
    
    # Generate price movement with trends
    trend = 0
    prices = []
    
    for i, date in enumerate(dates):
        # Add some trend changes
        if i % 100 == 0:
            trend = random.uniform(-0.0005, 0.0005)
        
        # Calculate price with trend and random walk
        if i == 0:
            price = base_price
        else:
            volatility = 0.01 if ticker == 'BTC' else 0.015
            price = prices[-1] * (1 + trend + random.gauss(0, volatility))
        
        prices.append(price)
        
        # Generate OHLC data
        open_price = price * (1 + random.uniform(-0.002, 0.002))
        high_price = max(open_price, price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, price) * (1 - random.uniform(0, 0.005))
        close_price = price
        
        # Volume with some correlation to price movement
        base_volume = 1000000 if ticker == 'BTC' else 500000
        volume = base_volume * (1 + abs(close_price - open_price) / open_price * 10) * random.uniform(0.8, 1.2)
        
        data.append({
            'timestamp': date,
            'ticker': ticker,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': int(volume)
        })
    
    return pd.DataFrame(data)

def generate_patterns(price_data: pd.DataFrame):
    """Generate sample patterns based on price data"""
    patterns = []
    
    # Head and Shoulders pattern
    patterns.append({
        'pattern_type': 'head_shoulders',
        'start_time': '2024-01-10 08:00:00',
        'end_time': '2024-01-15 16:00:00',
        'quality': 0.85,
        'statistics': {
            'confidence': 0.92,
            'strength': 2.5,
            'duration': 128,
            'price_change': -0.045,
            'volume_ratio': 1.8,
            'success_rate': 0.75,
            'key_points': {
                'left_shoulder': {'time': '2024-01-11 10:00:00', 'price': 41200},
                'head': {'time': '2024-01-13 02:00:00', 'price': 42100},
                'right_shoulder': {'time': '2024-01-14 18:00:00', 'price': 41150}
            }
        }
    })
    
    # Double Bottom pattern
    patterns.append({
        'pattern_type': 'double_bottom',
        'start_time': '2024-01-25 00:00:00',
        'end_time': '2024-02-02 12:00:00',
        'quality': 0.78,
        'statistics': {
            'confidence': 0.85,
            'strength': 2.2,
            'duration': 204,
            'price_change': 0.082,
            'volume_ratio': 2.1,
            'success_rate': 0.68,
            'peaks': [
                {'time': '2024-01-27 14:00:00', 'price': 38500},
                {'time': '2024-01-31 08:00:00', 'price': 38650}
            ]
        }
    })
    
    # Triangle pattern
    patterns.append({
        'pattern_type': 'triangle',
        'start_time': '2024-02-10 00:00:00',
        'end_time': '2024-02-18 00:00:00',
        'quality': 0.65,
        'statistics': {
            'confidence': 0.75,
            'strength': 1.8,
            'duration': 192,
            'price_change': 0.035,
            'volume_ratio': 0.8,
            'success_rate': 0.62
        }
    })
    
    # Support/Resistance levels
    patterns.append({
        'pattern_type': 'support_resistance',
        'start_time': '2024-02-20 00:00:00',
        'end_time': '2024-02-28 00:00:00',
        'quality': 0.90,
        'statistics': {
            'confidence': 0.95,
            'strength': 3.0,
            'duration': 192,
            'price_change': 0.002,
            'volume_ratio': 1.0,
            'success_rate': 0.85,
            'levels': [
                {'type': 'resistance', 'price': 42500, 'touches': 4},
                {'type': 'support', 'price': 40800, 'touches': 3}
            ]
        }
    })
    
    return patterns

def generate_predictions(last_price: float, periods: int = 24):
    """Generate prediction data with confidence bands"""
    timestamps = pd.date_range(start='2024-03-01', periods=periods, freq='1H')
    
    # Generate predictions with increasing uncertainty
    values = [last_price]
    for i in range(1, periods):
        trend = 0.0005  # Slight upward trend
        uncertainty = i * 0.0002  # Increasing uncertainty
        values.append(values[-1] * (1 + trend + random.gauss(0, uncertainty)))
    
    # Calculate confidence bands (wider as we go further)
    confidence_multiplier = np.linspace(0.01, 0.04, periods)
    lower_bound = [v * (1 - cm) for v, cm in zip(values, confidence_multiplier)]
    upper_bound = [v * (1 + cm) for v, cm in zip(values, confidence_multiplier)]
    
    return {
        'timestamps': timestamps,
        'values': values,
        'confidence_lower': lower_bound,
        'confidence_upper': upper_bound
    }

def main():
    """Run the time series visualization demo"""
    print("Generating sample data...")
    
    # Generate price data for multiple tickers
    btc_data = generate_realistic_price_data('BTC', '2024-01-01', '2024-03-01')
    eth_data = generate_realistic_price_data('ETH', '2024-01-01', '2024-03-01')
    
    # Combine data
    price_data = pd.concat([btc_data, eth_data], ignore_index=True)
    
    # Generate patterns
    patterns = generate_patterns(price_data)
    
    # Generate predictions based on last BTC price
    last_btc_price = btc_data['close'].iloc[-1]
    predictions = generate_predictions(last_btc_price)
    
    print("\nCreating visualizations...")
    
    # Create visualizer
    visualizer = TimeSeriesVisualizer()
    
    # 1. Main time series plot with all features
    print("\n1. Creating main time series plot...")
    main_fig = visualizer.create_timeseries_plot(
        price_data=price_data,
        patterns=patterns,
        predictions=predictions,
        selected_tickers=['BTC'],
        show_volume=True,
        height=800
    )
    
    # Save as HTML
    pio.write_html(main_fig, 'timeseries_main_plot.html', config=visualizer.default_config)
    print("   Saved: timeseries_main_plot.html")
    
    # 2. Multi-ticker view
    print("\n2. Creating multi-ticker view...")
    multi_fig = visualizer.create_timeseries_plot(
        price_data=price_data,
        patterns=patterns,
        predictions=None,  # No predictions for multi-ticker view
        selected_tickers=['BTC', 'ETH'],
        show_volume=True,
        height=800
    )
    multi_fig.update_layout(title={'text': 'Multi-Ticker Pattern Analysis'})
    
    pio.write_html(multi_fig, 'timeseries_multi_ticker.html', config=visualizer.default_config)
    print("   Saved: timeseries_multi_ticker.html")
    
    # 3. Pattern focus view
    print("\n3. Creating pattern focus view...")
    focus_pattern = patterns[0]  # Focus on head and shoulders
    focus_fig = visualizer.create_pattern_focus_plot(
        price_data=btc_data,
        pattern=focus_pattern,
        context_periods=50
    )
    
    pio.write_html(focus_fig, 'timeseries_pattern_focus.html', config=visualizer.default_config)
    print("   Saved: timeseries_pattern_focus.html")
    
    # 4. Pattern comparison
    print("\n4. Creating pattern comparison charts...")
    
    # Quality comparison
    quality_fig = visualizer.create_pattern_comparison_plot(patterns, metric='quality')
    pio.write_html(quality_fig, 'pattern_comparison_quality.html')
    print("   Saved: pattern_comparison_quality.html")
    
    # Confidence comparison
    confidence_fig = visualizer.create_pattern_comparison_plot(patterns, metric='confidence')
    confidence_fig.update_layout(title='Pattern Comparison by Confidence')
    pio.write_html(confidence_fig, 'pattern_comparison_confidence.html')
    print("   Saved: pattern_comparison_confidence.html")
    
    # 5. Clean view without volume
    print("\n5. Creating clean view without volume...")
    clean_fig = visualizer.create_timeseries_plot(
        price_data=btc_data,
        patterns=patterns,
        predictions=predictions,
        show_volume=False,
        height=600
    )
    clean_fig.update_layout(title={'text': 'BTC Price Patterns - Clean View'})
    
    pio.write_html(clean_fig, 'timeseries_clean_view.html', config=visualizer.default_config)
    print("   Saved: timeseries_clean_view.html")
    
    # Print summary
    print("\n" + "="*50)
    print("DEMO SUMMARY")
    print("="*50)
    print(f"Generated {len(price_data)} price records for {price_data['ticker'].nunique()} tickers")
    print(f"Created {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"  - {pattern['pattern_type']}: Quality {pattern['quality']:.2%}")
    print(f"Generated {len(predictions['timestamps'])} prediction points")
    print("\nVisualization files created:")
    print("  - timeseries_main_plot.html: Main interactive plot with all features")
    print("  - timeseries_multi_ticker.html: Multi-ticker comparison")
    print("  - timeseries_pattern_focus.html: Focused view of a specific pattern")
    print("  - pattern_comparison_quality.html: Pattern quality comparison")
    print("  - pattern_comparison_confidence.html: Pattern confidence comparison")
    print("  - timeseries_clean_view.html: Clean view without volume")
    print("\nOpen any HTML file in a browser to interact with the visualizations!")

if __name__ == "__main__":
    main()
