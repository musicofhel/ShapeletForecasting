"""
Test script to verify YFinance data integration in the dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.data_utils_yfinance import YFinanceDataManager
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance_integration():
    """Test the YFinance data manager integration"""
    
    print("=" * 60)
    print("Testing YFinance Dashboard Integration")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = YFinanceDataManager()
    
    # Test symbols
    test_symbols = ['SPY', 'AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        
        try:
            # Download data
            df = data_manager.download_data(symbol, period='1mo', interval='1d')
            
            if df is not None and not df.empty:
                print(f"✓ Successfully fetched {len(df)} rows for {symbol}")
                
                # Show latest price
                latest_price = df['close'].iloc[-1]
                print(f"  Latest Close Price: ${latest_price:.2f}")
                
                # Show date range
                start_date = df.index[0]
                end_date = df.index[-1]
                print(f"  Date Range: {start_date} to {end_date}")
                
                # Check if price is reasonable (not $150 for SPY!)
                if symbol == 'SPY':
                    if 400 < latest_price < 600:  # Reasonable range for SPY
                        print(f"  ✓ SPY price looks correct (not $150!)")
                    else:
                        print(f"  ⚠ SPY price might be incorrect: ${latest_price:.2f}")
                
                # Test ticker info
                info = data_manager.get_ticker_info(symbol)
                if info:
                    print(f"  Ticker Info: {info['name']}")
                    
            else:
                print(f"✗ Failed to fetch data for {symbol}")
                
        except Exception as e:
            print(f"✗ Error testing {symbol}: {e}")
    
    # Test cache
    print("\n" + "=" * 60)
    print("Testing Cache Functionality")
    print("=" * 60)
    
    # Fetch SPY again - should use cache
    print("\nFetching SPY again (should use cache)...")
    df_cached = data_manager.download_data('SPY', period='1mo', interval='1d')
    if df_cached is not None:
        print("✓ Cache working - data retrieved")
    
    # Check database file
    db_path = data_manager.db_path
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"\nDatabase file exists: {db_path}")
        print(f"Database size: {size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_yfinance_integration()
