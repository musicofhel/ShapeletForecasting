"""
Test script to verify YFinance data is being fetched and displayed correctly
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_yfinance_symbols():
    """Test that YFinance can fetch data for our symbols"""
    
    symbols = {
        'BTCUSD': 'BTC-USD',
        'ETHUSD': 'ETH-USD', 
        'SPY': 'SPY',
        'AAPL': 'AAPL'
    }
    
    print("Testing YFinance data fetching...")
    print("=" * 50)
    
    for display_name, yf_symbol in symbols.items():
        try:
            # Fetch data
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="5d", interval="1h")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                print(f"\n{display_name} ({yf_symbol}):")
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Data Points: {len(hist)}")
                print(f"  Latest Time: {hist.index[-1]}")
                print("  ✓ Data fetched successfully")
            else:
                print(f"\n{display_name} ({yf_symbol}): ✗ No data returned")
                
        except Exception as e:
            print(f"\n{display_name} ({yf_symbol}): ✗ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_yfinance_symbols()
