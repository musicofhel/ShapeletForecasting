"""
Test yfinance connectivity and functionality
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing yfinance connectivity...\n")

# Test 1: Try to download a popular ticker
print("Test 1: Downloading AAPL data...")
try:
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print(f"✓ Successfully connected to yfinance")
    print(f"  Company: {info.get('longName', 'N/A')}")
    print(f"  Current Price: ${info.get('currentPrice', 'N/A')}")
except Exception as e:
    print(f"✗ Failed to get ticker info: {e}")

# Test 2: Try to download historical data
print("\nTest 2: Downloading historical data...")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = yf.download("AAPL", start=start_date, end=end_date, progress=False)
    
    if not data.empty:
        print(f"✓ Successfully downloaded {len(data)} days of data")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("✗ Downloaded data is empty")
except Exception as e:
    print(f"✗ Failed to download historical data: {e}")

# Test 3: Try alternative tickers
print("\nTest 3: Testing multiple tickers...")
test_tickers = ["MSFT", "GOOGL", "BTC-USD", "^GSPC"]  # Stocks, crypto, index
for symbol in test_tickers:
    try:
        data = yf.download(symbol, period="1d", progress=False)
        if not data.empty:
            print(f"✓ {symbol}: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"✗ {symbol}: No data")
    except Exception as e:
        print(f"✗ {symbol}: {str(e)[:50]}...")

# Test 4: Check for common issues
print("\nDiagnostics:")
print(f"- yfinance version: {yf.__version__}")

# Provide solutions
print("\nIf tests are failing, try these solutions:")
print("1. Check internet connection")
print("2. Try using a VPN if Yahoo Finance is blocked in your region")
print("3. Update yfinance: pip install --upgrade yfinance")
print("4. Clear yfinance cache: yf.Ticker('AAPL').history(period='1d', actions=False, auto_adjust=True, back_adjust=False)")
print("5. Use a different network (some corporate networks block Yahoo Finance)")
print("\nNote: yfinance is FREE and requires NO API key!")
