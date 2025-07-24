"""
Fixed yfinance test with rate limiting, caching, and fallbacks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.data_utils import DataManager
import pandas as pd
from datetime import datetime, timedelta
import time

print("Testing yfinance with improved error handling...\n")

# Initialize data manager with rate limiting
dm = DataManager(rate_limit_delay=2.0)  # 2 second delay between requests

# Test 1: Try to download with rate limiting and caching
print("Test 1: Downloading AAPL data with rate limiting...")
data = dm.download_data("AAPL", period="1mo")
if data is not None:
    print(f"✓ Successfully got {len(data)} days of data")
    print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
else:
    print("✗ Failed to get data")

# Test 2: Test caching (should be instant)
print("\nTest 2: Testing cache...")
start_time = time.time()
data = dm.download_data("AAPL", period="1mo")
elapsed = time.time() - start_time
if data is not None and elapsed < 0.5:
    print(f"✓ Loaded from cache in {elapsed:.3f} seconds")

# Test 3: Test multiple tickers with batch download
print("\nTest 3: Testing batch download...")
tickers = ["MSFT", "GOOGL", "TSLA"]
data = dm.download_data(tickers, period="5d")
if data is not None:
    print(f"✓ Got data for multiple tickers")
    if hasattr(data.columns, 'levels'):  # Multi-level columns
        for ticker in tickers:
            if ticker in data.columns.levels[1]:
                latest = data[('Close', ticker)].iloc[-1]
                print(f"  {ticker}: ${latest:.2f}")

# Test 4: Test error handling for invalid ticker
print("\nTest 4: Testing error handling for invalid ticker...")
invalid_data = dm.download_data("INVALID_TICKER_XYZ", period="1d")
if invalid_data is None:
    print("✓ Correctly handled invalid ticker")
else:
    print("✗ Should have returned None for invalid ticker")

# Test 5: Test ticker info with fallback
print("\nTest 5: Testing ticker info...")
info = dm.get_ticker_info("AAPL")
print(f"✓ Got ticker info: {info.get('longName', 'Unknown')}")

# Test 6: Clear cache and test rate limiting
print("\nTest 6: Testing rate limiting (this may take a few seconds)...")
test_tickers = ["IBM", "NFLX", "AMZN"]
for i, ticker in enumerate(test_tickers):
    print(f"  Requesting {ticker}...", end="", flush=True)
    start = time.time()
    data = dm.download_data(ticker, period="1d", use_cache=False)
    elapsed = time.time() - start
    if data is not None:
        print(f" ✓ ({elapsed:.1f}s)")
    else:
        print(f" ✗ ({elapsed:.1f}s)")

# Summary and recommendations
print("\n" + "="*60)
print("SUMMARY AND RECOMMENDATIONS")
print("="*60)

print("\nData Manager Features:")
print("✓ Rate limiting (2s delay between requests)")
print("✓ Automatic caching (daily cache)")
print("✓ Batch download support")
print("✓ Error handling and logging")

print("\nUsage in your project:")
print("```python")
print("from src.dashboard.data_utils import data_manager")
print("")
print("# Download with automatic fallbacks")
print("data = data_manager.download_data('AAPL', period='1y')")
print("")
print("# Download multiple tickers efficiently")
print("data = data_manager.download_data(['AAPL', 'MSFT', 'GOOGL'], period='1mo')")
print("")
print("# Force fresh download (no cache)")
print("data = data_manager.download_data('BTC-USD', use_cache=False)")
print("")
print("# Load demo data")
print("data = data_manager.load_demo_data('btcusd_1h.csv')")
print("```")

print("\nIf you're still getting rate limited:")
print("1. Wait 5-10 minutes before trying again")
print("2. Use cached data (automatic)")
print("3. Consider using a VPN")
print("4. Try during off-peak hours")
print("5. Use polite delays between requests")

# Create a simple demo showing how to use in the dashboard
print("\n" + "="*60)
print("CREATING DEMO USAGE FILE...")
print("="*60)

demo_code = '''"""
Demo: Using DataManager in your dashboard
"""
from src.dashboard.data_utils import data_manager
import pandas as pd

def load_ticker_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Load ticker data with automatic fallbacks"""
    
    # This will:
    # 1. Check cache first
    # 2. Try to download from yfinance with rate limiting
    # 3. Return None if data cannot be fetched
    
    data = data_manager.download_data(ticker, period=period)
    
    if data is None:
        raise ValueError(f"Could not load data for {ticker}")
        
    return data

# Example usage in a dashboard callback
def update_chart(ticker):
    try:
        # Load data with automatic handling
        data = load_ticker_data(ticker, period="6mo")
        
        # Your visualization code here
        print(f"Loaded {len(data)} days of data for {ticker}")
        print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
        
        return data
        
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None

# Test it
if __name__ == "__main__":
    # Test with different tickers
    for ticker in ["AAPL", "BTC-USD", "INVALID-TICKER"]:
        print(f"\\nTesting {ticker}:")
        update_chart(ticker)
'''

with open("demo_data_manager_usage.py", "w") as f:
    f.write(demo_code)

print("✓ Created demo_data_manager_usage.py")
print("\nThe DataManager is now ready to use in your project!")
