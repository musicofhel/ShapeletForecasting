"""
Simple YFinance Demo with Error Handling
========================================

A basic demo that shows how to pull data from yfinance
with proper error handling and rate limiting.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def demo_single_ticker():
    """Demo pulling data for a single ticker"""
    print("SIMPLE YFINANCE DEMO")
    print("=" * 50)
    
    # Use a popular ticker
    ticker = "SPY"  # S&P 500 ETF - usually more reliable
    
    print(f"\nFetching data for {ticker}...")
    
    try:
        # Method 1: Using download function
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if not data.empty:
            print(f"\nSuccessfully downloaded {len(data)} days of data")
            print("\nLast 5 days:")
            print(data.tail())
            
            # Calculate some basic stats
            print("\nBasic Statistics:")
            print(f"Latest Close: ${data['Close'].iloc[-1]:.2f}")
            print(f"30-day High: ${data['High'].max():.2f}")
            print(f"30-day Low: ${data['Low'].min():.2f}")
            print(f"Average Volume: {data['Volume'].mean():,.0f}")
            
            # Calculate returns
            data['Daily_Return'] = data['Close'].pct_change()
            print(f"\nAverage Daily Return: {data['Daily_Return'].mean():.4%}")
            print(f"Volatility (Std Dev): {data['Daily_Return'].std():.4%}")
            
        else:
            print("No data received")
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        
    # Add delay to avoid rate limiting
    time.sleep(2)
    
    print("\n" + "-" * 50)
    
    try:
        # Method 2: Using Ticker object
        print(f"\nFetching info using Ticker object for {ticker}...")
        ticker_obj = yf.Ticker(ticker)
        
        # Get historical data
        hist = ticker_obj.history(period="1mo")
        
        if not hist.empty:
            print(f"\nGot {len(hist)} days of historical data")
            print("\nRecent price action:")
            recent = hist.tail(3)[['Open', 'High', 'Low', 'Close', 'Volume']]
            print(recent.round(2))
            
        # Try to get info (may fail due to rate limiting)
        try:
            info = ticker_obj.info
            if info:
                print(f"\nTicker Info Available:")
                print(f"  Name: {info.get('shortName', 'N/A')}")
                print(f"  Exchange: {info.get('exchange', 'N/A')}")
                print(f"  Currency: {info.get('currency', 'N/A')}")
        except:
            print("\nTicker info not available (rate limited)")
            
    except Exception as e:
        print(f"Error with Ticker object: {e}")

def demo_crypto():
    """Demo cryptocurrency data"""
    print("\n" + "=" * 50)
    print("CRYPTOCURRENCY DEMO")
    print("=" * 50)
    
    crypto = "BTC-USD"
    
    try:
        # Get last 7 days of Bitcoin data
        btc = yf.download(crypto, period="7d", interval="1d", progress=False)
        
        if not btc.empty:
            print(f"\nBitcoin (BTC-USD) - Last 7 days:")
            print(btc[['Open', 'High', 'Low', 'Close']].round(2))
            
            # Calculate daily changes
            btc['Change'] = btc['Close'].diff()
            btc['Change%'] = btc['Close'].pct_change() * 100
            
            print("\nDaily Changes:")
            print(btc[['Close', 'Change', 'Change%']].tail().round(2))
            
    except Exception as e:
        print(f"Error fetching crypto data: {e}")

def demo_multiple_tickers_safe():
    """Demo multiple tickers with delays"""
    print("\n" + "=" * 50)
    print("MULTIPLE TICKERS DEMO (WITH DELAYS)")
    print("=" * 50)
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = {}
    
    for ticker in tickers:
        print(f"\nFetching {ticker}...")
        try:
            # Get last 5 days
            data = yf.download(ticker, period="5d", progress=False)
            
            if not data.empty:
                latest_close = data['Close'].iloc[-1]
                change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                change_pct = (change / data['Close'].iloc[0]) * 100
                
                results[ticker] = {
                    'price': latest_close,
                    'change': change,
                    'change_pct': change_pct
                }
                print(f"  Success: ${latest_close:.2f} ({change_pct:+.2f}%)")
            else:
                print(f"  No data available")
                
        except Exception as e:
            print(f"  Error: {e}")
            
        # Delay between requests
        time.sleep(1)
    
    # Summary
    if results:
        print("\nSUMMARY:")
        print("-" * 40)
        for ticker, data in results.items():
            print(f"{ticker}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")

def main():
    """Run the demos"""
    try:
        # Single ticker demo
        demo_single_ticker()
        
        # Crypto demo
        demo_crypto()
        
        # Multiple tickers with delays
        demo_multiple_tickers_safe()
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETE!")
        print("=" * 50)
        
        print("\nNOTE: If you're getting rate limit errors, try:")
        print("  1. Add delays between requests")
        print("  2. Use fewer requests")
        print("  3. Try again later")
        print("  4. Use a VPN or different network")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
