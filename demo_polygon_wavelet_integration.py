"""
Polygon + Wavelet Analysis Integration
======================================

This demo shows how to:
1. Fetch data from Polygon API
2. Integrate with your wavelet analysis system
3. Perform pattern classification on real-time data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Import your wavelet analysis modules
from src.dashboard.data_utils import DataManager
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer

# Polygon API configuration
API_KEY = "rzbt8GG8mqwLo1EMuSzbkC34uxA_Df1R"
BASE_URL = "https://api.polygon.io"

class PolygonDataFetcher:
    """Fetches and prepares data from Polygon for wavelet analysis"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_call_time = 0
        self.call_count = 0
        
    def _rate_limit(self):
        """Simple rate limiting - 5 calls per minute"""
        current_time = time.time()
        if current_time - self.last_call_time < 12:  # 12 seconds between calls (5 per minute)
            wait_time = 12 - (current_time - self.last_call_time)
            print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        self.last_call_time = time.time()
        
    def fetch_stock_data(self, ticker, days=90):
        """Fetch stock data for wavelet analysis"""
        self._rate_limit()
        
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000
        }
        
        print(f"Fetching {ticker} data from {from_date} to {to_date}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK" and data.get("results"):
                df = pd.DataFrame(data["results"])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                
                # Prepare for wavelet analysis
                prepared_data = pd.DataFrame({
                    'timestamp': df['timestamp'],
                    'price': df['c'],
                    'volume': df['v'],
                    'high': df['h'],
                    'low': df['l'],
                    'open': df['o']
                })
                
                print(f"Successfully fetched {len(prepared_data)} data points")
                return prepared_data
            else:
                print(f"No data available for {ticker}")
                return None
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

def analyze_single_stock(ticker="AAPL"):
    """Demo: Analyze a single stock with wavelet analysis"""
    print("\n" + "=" * 60)
    print(f"ANALYZING {ticker} WITH WAVELET ANALYSIS")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = PolygonDataFetcher(API_KEY)
    
    # Fetch data
    data = fetcher.fetch_stock_data(ticker, days=60)
    
    if data is None:
        print("Failed to fetch data")
        return
    
    # Initialize analyzers
    classifier = PatternClassifier()
    sequence_analyzer = WaveletSequenceAnalyzer()
    
    # Perform pattern classification
    print(f"\nClassifying patterns in {ticker}...")
    classification = classifier.classify(
        data['price'].values,
        timestamps=data['timestamp'].values
    )
    
    print(f"\nPattern Classification Results:")
    print(f"  Primary Pattern: {classification.get('primary_pattern', 'Unknown')}")
    print(f"  Confidence: {classification.get('confidence', 0):.2%}")
    print(f"  Pattern Features:")
    for feature, value in classification.get('features', {}).items():
        print(f"    - {feature}: {value}")
    
    # Perform sequence analysis
    print(f"\nAnalyzing wavelet sequences...")
    sequence_results = sequence_analyzer.analyze_sequence(
        data['price'].values,
        timestamps=data['timestamp'].values
    )
    
    patterns = sequence_results.get('patterns', [])
    print(f"\nFound {len(patterns)} wavelet patterns")
    
    if patterns:
        print("\nTop Patterns:")
        for i, pattern in enumerate(patterns[:3]):
            print(f"\n  Pattern {i+1}:")
            print(f"    Type: {pattern.get('type', 'Unknown')}")
            print(f"    Confidence: {pattern.get('confidence', 0):.2%}")
            print(f"    Start: {pattern.get('start_date', 'N/A')}")
            print(f"    End: {pattern.get('end_date', 'N/A')}")
            print(f"    Duration: {pattern.get('duration', 0)} days")
    
    return data, classification, sequence_results

def multi_stock_pattern_scan():
    """Demo: Scan multiple stocks for patterns"""
    print("\n" + "=" * 60)
    print("MULTI-STOCK PATTERN SCAN")
    print("=" * 60)
    
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    fetcher = PolygonDataFetcher(API_KEY)
    classifier = PatternClassifier()
    
    results = {}
    
    print(f"\nScanning {len(tickers)} stocks for patterns...")
    print("(Rate limited to 5 API calls per minute)")
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        
        # Fetch recent data (30 days for speed)
        data = fetcher.fetch_stock_data(ticker, days=30)
        
        if data is not None:
            # Quick classification
            classification = classifier.classify(
                data['price'].values,
                timestamps=data['timestamp'].values
            )
            
            results[ticker] = {
                'latest_price': data['price'].iloc[-1],
                'price_change': ((data['price'].iloc[-1] - data['price'].iloc[0]) / data['price'].iloc[0]) * 100,
                'pattern': classification.get('primary_pattern', 'Unknown'),
                'confidence': classification.get('confidence', 0),
                'trend': classification.get('features', {}).get('trend', 'Unknown')
            }
            
            print(f"  Pattern: {results[ticker]['pattern']}")
            print(f"  Confidence: {results[ticker]['confidence']:.2%}")
            print(f"  30-day Change: {results[ticker]['price_change']:+.2f}%")
        else:
            print(f"  Failed to fetch data")
    
    # Summary report
    print("\n" + "=" * 60)
    print("PATTERN SCAN SUMMARY")
    print("=" * 60)
    
    # Group by pattern
    pattern_groups = {}
    for ticker, result in results.items():
        pattern = result['pattern']
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(ticker)
    
    print("\nStocks by Pattern Type:")
    for pattern, tickers in pattern_groups.items():
        print(f"\n{pattern}:")
        for ticker in tickers:
            r = results[ticker]
            print(f"  {ticker}: ${r['latest_price']:.2f} ({r['price_change']:+.2f}%) - Confidence: {r['confidence']:.2%}")

def real_time_monitoring_simulation():
    """Demo: Simulate real-time pattern monitoring"""
    print("\n" + "=" * 60)
    print("REAL-TIME PATTERN MONITORING SIMULATION")
    print("=" * 60)
    
    ticker = "SPY"  # S&P 500 ETF
    fetcher = PolygonDataFetcher(API_KEY)
    classifier = PatternClassifier()
    
    print(f"\nMonitoring {ticker} for pattern changes...")
    print("(Simulating with historical data)")
    
    # Get 90 days of data
    full_data = fetcher.fetch_stock_data(ticker, days=90)
    
    if full_data is None:
        print("Failed to fetch data")
        return
    
    # Simulate real-time by analyzing progressively
    window_size = 30  # 30-day rolling window
    
    print(f"\nAnalyzing rolling {window_size}-day windows...")
    
    pattern_history = []
    
    for i in range(window_size, len(full_data), 5):  # Step by 5 days
        # Get window of data
        window_data = full_data.iloc[i-window_size:i]
        
        # Classify pattern
        classification = classifier.classify(
            window_data['price'].values,
            timestamps=window_data['timestamp'].values
        )
        
        current_pattern = classification.get('primary_pattern', 'Unknown')
        confidence = classification.get('confidence', 0)
        date = window_data['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        price = window_data['price'].iloc[-1]
        
        # Check for pattern change
        if pattern_history and current_pattern != pattern_history[-1]['pattern']:
            print(f"\n🔄 PATTERN CHANGE on {date}:")
            print(f"   {pattern_history[-1]['pattern']} → {current_pattern}")
            print(f"   Price: ${price:.2f}")
            print(f"   Confidence: {confidence:.2%}")
        
        pattern_history.append({
            'date': date,
            'pattern': current_pattern,
            'confidence': confidence,
            'price': price
        })
    
    # Summary
    print(f"\n\nPattern History Summary:")
    print(f"Total observations: {len(pattern_history)}")
    
    # Count pattern occurrences
    pattern_counts = {}
    for entry in pattern_history:
        pattern = entry['pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("\nPattern Distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(pattern_history)) * 100
        print(f"  {pattern}: {count} occurrences ({percentage:.1f}%)")

def save_analysis_results(ticker, data, classification, filename=None):
    """Save analysis results for later use"""
    if filename is None:
        filename = f"polygon_{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.json"
    
    results = {
        'ticker': ticker,
        'analysis_date': datetime.now().isoformat(),
        'data_points': len(data),
        'date_range': {
            'start': data['timestamp'].min().isoformat(),
            'end': data['timestamp'].max().isoformat()
        },
        'classification': {
            'primary_pattern': classification.get('primary_pattern'),
            'confidence': classification.get('confidence'),
            'features': classification.get('features', {})
        },
        'price_summary': {
            'latest': float(data['price'].iloc[-1]),
            'high': float(data['price'].max()),
            'low': float(data['price'].min()),
            'average': float(data['price'].mean())
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis results saved to {filename}")

def main():
    """Run the integration demos"""
    print("POLYGON + WAVELET ANALYSIS INTEGRATION")
    print("======================================")
    print(f"API Key: {API_KEY[:10]}...")
    print("Rate Limit: 5 calls per minute\n")
    
    try:
        # Demo 1: Single stock analysis
        data, classification, sequence_results = analyze_single_stock("AAPL")
        
        # Save results
        if data is not None:
            save_analysis_results("AAPL", data, classification)
        
        # Demo 2: Multi-stock pattern scan
        multi_stock_pattern_scan()
        
        # Demo 3: Real-time monitoring simulation
        real_time_monitoring_simulation()
        
        print("\n" + "=" * 60)
        print("INTEGRATION DEMO COMPLETE!")
        print("=" * 60)
        
        print("\nNext Steps:")
        print("1. Use the saved analysis results in your dashboard")
        print("2. Set up scheduled pattern scans")
        print("3. Implement real-time alerts for pattern changes")
        print("4. Integrate with your trading strategies")
        
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
