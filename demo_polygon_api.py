"""
Polygon.io API Demo for Financial Data
======================================

This demo shows how to:
1. Fetch stock data using Polygon API
2. Handle rate limiting (5 calls/minute for free tier)
3. Integrate with your wavelet analysis system
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Your Polygon API key
API_KEY = "rzbt8GG8mqwLo1EMuSzbkC34uxA_Df1R"
BASE_URL = "https://api.polygon.io"

class PolygonClient:
    """Simple Polygon API client with rate limiting"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_call_time = 0
        self.call_count = 0
        self.rate_limit_window = 60  # 60 seconds
        self.max_calls_per_window = 5  # 5 calls per minute
        
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_call_time > self.rate_limit_window:
            self.call_count = 0
            
        # If we've hit the limit, wait
        if self.call_count >= self.max_calls_per_window:
            wait_time = self.rate_limit_window - (current_time - self.last_call_time)
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)
                self.call_count = 0
                
        self.last_call_time = time.time()
        self.call_count += 1
        
    def get_aggregates(self, ticker, multiplier=1, timespan="day", from_date=None, to_date=None):
        """Get aggregate bars for a ticker"""
        self._rate_limit()
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
            
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
            
    def get_ticker_details(self, ticker):
        """Get details about a ticker"""
        self._rate_limit()
        
        url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ticker details: {e}")
            return None
            
    def get_last_quote(self, ticker):
        """Get the last quote for a ticker"""
        self._rate_limit()
        
        url = f"{BASE_URL}/v2/last/nbbo/{ticker}"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching last quote: {e}")
            return None

def demo_basic_data_fetch():
    """Demo 1: Basic stock data fetching"""
    print("=" * 60)
    print("DEMO 1: Basic Stock Data Fetch")
    print("=" * 60)
    
    client = PolygonClient(API_KEY)
    ticker = "AAPL"
    
    print(f"\nFetching {ticker} data for the last 30 days...")
    
    # Get aggregate data
    data = client.get_aggregates(ticker)
    
    if data and data.get("status") == "OK":
        results = data.get("results", [])
        print(f"Successfully fetched {len(results)} data points")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap'
        })
        
        # Display summary
        print("\nData Summary:")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Latest Close: ${df['close'].iloc[-1]:.2f}")
        print(f"30-day High: ${df['high'].max():.2f}")
        print(f"30-day Low: ${df['low'].min():.2f}")
        
        print("\nLast 5 days:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail())
        
        return df
    else:
        print("Failed to fetch data")
        return None

def demo_ticker_details():
    """Demo 2: Get ticker details"""
    print("\n" + "=" * 60)
    print("DEMO 2: Ticker Details")
    print("=" * 60)
    
    client = PolygonClient(API_KEY)
    ticker = "MSFT"
    
    print(f"\nFetching details for {ticker}...")
    
    details = client.get_ticker_details(ticker)
    
    if details and details.get("status") == "OK":
        results = details.get("results", {})
        
        print(f"\n{ticker} Details:")
        print(f"  Name: {results.get('name', 'N/A')}")
        print(f"  Market Cap: ${results.get('market_cap', 0):,.0f}")
        print(f"  Description: {results.get('description', 'N/A')[:100]}...")
        print(f"  Primary Exchange: {results.get('primary_exchange', 'N/A')}")
        print(f"  Currency: {results.get('currency_name', 'N/A')}")
        print(f"  Type: {results.get('type', 'N/A')}")
        
        return results
    else:
        print("Failed to fetch ticker details")
        return None

def demo_multiple_tickers():
    """Demo 3: Fetch data for multiple tickers with rate limiting"""
    print("\n" + "=" * 60)
    print("DEMO 3: Multiple Tickers (Rate Limited)")
    print("=" * 60)
    
    client = PolygonClient(API_KEY)
    tickers = ["AAPL", "GOOGL", "TSLA"]
    
    print(f"\nFetching data for: {', '.join(tickers)}")
    print("Note: Rate limited to 5 calls per minute")
    
    results = {}
    
    for ticker in tickers:
        print(f"\nFetching {ticker}...")
        
        # Get last 7 days of data
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        data = client.get_aggregates(ticker, from_date=from_date, to_date=to_date)
        
        if data and data.get("status") == "OK":
            bars = data.get("results", [])
            if bars:
                latest = bars[-1]
                first = bars[0]
                
                results[ticker] = {
                    'latest_close': latest['c'],
                    'change': latest['c'] - first['c'],
                    'change_pct': ((latest['c'] - first['c']) / first['c']) * 100,
                    'volume': latest['v']
                }
                
                print(f"  Success: ${latest['c']:.2f} ({results[ticker]['change_pct']:+.2f}%)")
            else:
                print(f"  No data available")
        else:
            print(f"  Failed to fetch data")
    
    # Summary
    if results:
        print("\n" + "-" * 40)
        print("SUMMARY:")
        for ticker, data in results.items():
            print(f"{ticker}: ${data['latest_close']:.2f} ({data['change_pct']:+.2f}%)")
    
    return results

def demo_crypto_data():
    """Demo 4: Cryptocurrency data"""
    print("\n" + "=" * 60)
    print("DEMO 4: Cryptocurrency Data")
    print("=" * 60)
    
    client = PolygonClient(API_KEY)
    
    # Polygon uses X: prefix for crypto
    crypto_ticker = "X:BTCUSD"
    
    print(f"\nFetching Bitcoin (BTC/USD) data...")
    
    # Get last 7 days
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    
    data = client.get_aggregates(crypto_ticker, from_date=from_date, to_date=to_date)
    
    if data and data.get("status") == "OK":
        results = data.get("results", [])
        
        if results:
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            
            print(f"\nBitcoin - Last 7 days:")
            print(f"Latest Price: ${df['c'].iloc[-1]:,.2f}")
            print(f"7-day High: ${df['h'].max():,.2f}")
            print(f"7-day Low: ${df['l'].min():,.2f}")
            
            # Calculate volatility
            df['returns'] = df['c'].pct_change()
            volatility = df['returns'].std() * np.sqrt(365) * 100
            print(f"Annualized Volatility: {volatility:.2f}%")
            
            print("\nDaily Prices:")
            for _, row in df.iterrows():
                print(f"  {row['timestamp'].strftime('%Y-%m-%d')}: ${row['c']:,.2f}")
    else:
        print("Failed to fetch crypto data")

def prepare_for_wavelet_analysis(df):
    """Prepare Polygon data for wavelet analysis"""
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR WAVELET ANALYSIS")
    print("=" * 60)
    
    # Ensure proper format
    prepared_data = pd.DataFrame({
        'timestamp': df['timestamp'],
        'price': df['close'],
        'volume': df['volume'],
        'high': df['high'],
        'low': df['low'],
        'open': df['open']
    })
    
    # Reset index
    prepared_data = prepared_data.reset_index(drop=True)
    
    print(f"\nData prepared for wavelet analysis:")
    print(f"  Shape: {prepared_data.shape}")
    print(f"  Columns: {list(prepared_data.columns)}")
    print(f"  Date range: {prepared_data['timestamp'].min()} to {prepared_data['timestamp'].max()}")
    
    return prepared_data

def main():
    """Run all demos"""
    print("POLYGON.IO API DEMO")
    print("===================")
    print(f"Using API Key: {API_KEY[:10]}...")
    print("Rate Limit: 5 calls per minute")
    print()
    
    try:
        # Demo 1: Basic data fetch
        df = demo_basic_data_fetch()
        
        # Demo 2: Ticker details
        demo_ticker_details()
        
        # Demo 3: Multiple tickers
        demo_multiple_tickers()
        
        # Demo 4: Crypto data
        demo_crypto_data()
        
        # Prepare data for wavelet analysis
        if df is not None:
            prepared_data = prepare_for_wavelet_analysis(df)
            
            # Save to CSV for later use
            prepared_data.to_csv('polygon_data_sample.csv', index=False)
            print("\nData saved to polygon_data_sample.csv")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in demo: {e}")

if __name__ == "__main__":
    main()
