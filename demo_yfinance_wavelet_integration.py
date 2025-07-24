"""
YFinance Integration with Wavelet Analysis
==========================================

This demo shows how to:
1. Pull data from yfinance
2. Prepare it for wavelet analysis
3. Integrate with your existing pattern matching system
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import your existing modules
from src.dashboard.data_utils import DataManager
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer

def fetch_and_prepare_data(ticker="AAPL", period_days=90):
    """
    Fetch data from yfinance and prepare for wavelet analysis
    """
    print(f"\nFetching {ticker} data for the last {period_days} days...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Prepare data in the format expected by your system
    prepared_data = pd.DataFrame({
        'timestamp': data.index,
        'price': data['Close'].values,
        'volume': data['Volume'].values,
        'high': data['High'].values,
        'low': data['Low'].values,
        'open': data['Open'].values
    })
    
    # Reset index for compatibility
    prepared_data = prepared_data.reset_index(drop=True)
    
    print(f"Data fetched: {len(prepared_data)} data points")
    print(f"Date range: {prepared_data['timestamp'].min()} to {prepared_data['timestamp'].max()}")
    
    return prepared_data

def analyze_with_wavelets(data, ticker="AAPL"):
    """
    Perform wavelet analysis on the fetched data
    """
    print("\nPerforming wavelet analysis...")
    
    # Initialize the wavelet analyzer
    analyzer = WaveletSequenceAnalyzer()
    
    # Analyze the price data
    analysis_results = analyzer.analyze_sequence(
        data['price'].values,
        timestamps=data['timestamp'].values
    )
    
    # Extract key patterns
    patterns = analysis_results.get('patterns', [])
    print(f"\nFound {len(patterns)} patterns in {ticker} data")
    
    # Show pattern summary
    if patterns:
        print("\nPattern Summary:")
        for i, pattern in enumerate(patterns[:5]):  # Show first 5
            print(f"  Pattern {i+1}:")
            print(f"    - Type: {pattern.get('type', 'Unknown')}")
            print(f"    - Confidence: {pattern.get('confidence', 0):.2%}")
            print(f"    - Duration: {pattern.get('duration', 0)} periods")
    
    return analysis_results

def classify_patterns(data, ticker="AAPL"):
    """
    Use the pattern classifier on the data
    """
    print(f"\nClassifying patterns for {ticker}...")
    
    # Initialize classifier
    classifier = PatternClassifier()
    
    # Classify the price series
    classification = classifier.classify(
        data['price'].values,
        timestamps=data['timestamp'].values
    )
    
    print(f"\nClassification Results:")
    print(f"  Primary Pattern: {classification.get('primary_pattern', 'Unknown')}")
    print(f"  Confidence: {classification.get('confidence', 0):.2%}")
    print(f"  Secondary Patterns: {', '.join(classification.get('secondary_patterns', []))}")
    
    return classification

def real_time_monitoring_demo():
    """
    Demo real-time pattern monitoring with live data
    """
    print("\n" + "=" * 60)
    print("REAL-TIME MONITORING DEMO")
    print("=" * 60)
    
    # List of tickers to monitor
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
    
    print(f"\nMonitoring patterns for: {', '.join(tickers)}")
    
    results = {}
    
    for ticker in tickers:
        print(f"\n--- Analyzing {ticker} ---")
        
        try:
            # Fetch recent data (last 30 days for speed)
            data = fetch_and_prepare_data(ticker, period_days=30)
            
            # Quick pattern classification
            classification = classify_patterns(data, ticker)
            
            # Store results
            results[ticker] = {
                'latest_price': data['price'].iloc[-1],
                'price_change': (data['price'].iloc[-1] - data['price'].iloc[0]) / data['price'].iloc[0] * 100,
                'pattern': classification.get('primary_pattern', 'Unknown'),
                'confidence': classification.get('confidence', 0)
            }
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    # Display summary
    print("\n" + "=" * 60)
    print("PATTERN MONITORING SUMMARY")
    print("=" * 60)
    
    for ticker, result in results.items():
        if 'error' in result:
            print(f"\n{ticker}: Error - {result['error']}")
        else:
            print(f"\n{ticker}:")
            print(f"  Latest Price: ${result['latest_price']:.2f}")
            print(f"  30-Day Change: {result['price_change']:+.2f}%")
            print(f"  Pattern: {result['pattern']}")
            print(f"  Confidence: {result['confidence']:.2%}")

def integration_with_data_manager():
    """
    Demo integration with your DataManager class
    """
    print("\n" + "=" * 60)
    print("DATA MANAGER INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize DataManager
    dm = DataManager()
    
    # Fetch data using yfinance
    ticker = "SPY"
    print(f"\nFetching {ticker} data via yfinance...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Get data from yfinance
    yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Convert to DataManager format
    print("Converting to DataManager format...")
    
    # Create a dataset entry
    dataset_info = {
        'ticker': ticker,
        'source': 'yfinance',
        'data': {
            'timestamps': yf_data.index.tolist(),
            'prices': yf_data['Close'].tolist(),
            'volumes': yf_data['Volume'].tolist()
        },
        'metadata': {
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date()),
            'total_points': len(yf_data)
        }
    }
    
    print(f"\nDataset prepared:")
    print(f"  Ticker: {dataset_info['ticker']}")
    print(f"  Points: {dataset_info['metadata']['total_points']}")
    print(f"  Period: {dataset_info['metadata']['start_date']} to {dataset_info['metadata']['end_date']}")
    
    return dataset_info

def main():
    """
    Run the complete integration demo
    """
    print("YFINANCE + WAVELET ANALYSIS INTEGRATION DEMO")
    print("=" * 60)
    
    try:
        # Demo 1: Basic data fetch and analysis
        print("\nDemo 1: Basic Analysis")
        data = fetch_and_prepare_data("AAPL", period_days=60)
        wavelet_results = analyze_with_wavelets(data, "AAPL")
        
        # Demo 2: Pattern classification
        print("\nDemo 2: Pattern Classification")
        classification = classify_patterns(data, "AAPL")
        
        # Demo 3: Real-time monitoring
        real_time_monitoring_demo()
        
        # Demo 4: DataManager integration
        integration_with_data_manager()
        
        print("\n" + "=" * 60)
        print("INTEGRATION DEMO COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in demo: {e}")
        print("\nMake sure all required modules are installed:")
        print("  pip install yfinance pandas numpy matplotlib")
        print("\nAnd that the wavelet analysis modules are available in src/dashboard/")

if __name__ == "__main__":
    main()
