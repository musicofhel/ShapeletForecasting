# yfinance Guide - No API Key Required!

## Overview
yfinance is a **FREE** Python library that downloads financial data from Yahoo Finance. It does **NOT** require any API key, registration, or payment.

## Common Issues and Solutions

### 1. Rate Limiting (429 Too Many Requests)
**Your current issue** - Yahoo Finance limits the number of requests to prevent abuse.

**Solutions:**
- Wait a few minutes before trying again
- Add delays between requests: `time.sleep(1)`
- Use batch downloads instead of individual requests
- Download data less frequently (cache results)

### 2. Network/Firewall Issues
Some networks block Yahoo Finance access.

**Solutions:**
- Try a different network (home vs work)
- Use a VPN
- Check if your firewall is blocking finance.yahoo.com

### 3. Data Availability
Some tickers might be delisted or have limited data.

**Solutions:**
- Verify ticker symbols on finance.yahoo.com
- Use major tickers for testing (AAPL, MSFT, GOOGL)
- Check if the market is open

## Best Practices for yfinance

### 1. Batch Downloads
Instead of:
```python
# Bad - multiple requests
aapl = yf.download("AAPL", period="1y")
msft = yf.download("MSFT", period="1y")
googl = yf.download("GOOGL", period="1y")
```

Do this:
```python
# Good - single request
data = yf.download(["AAPL", "MSFT", "GOOGL"], period="1y")
```

### 2. Add Delays
```python
import time

tickers = ["AAPL", "MSFT", "GOOGL"]
for ticker in tickers:
    data = yf.download(ticker, period="1d", progress=False)
    time.sleep(1)  # Wait 1 second between requests
```

### 3. Cache Data Locally
```python
import os
import pickle
from datetime import datetime, timedelta

def get_cached_data(ticker, cache_dir="data/cache"):
    cache_file = f"{cache_dir}/{ticker}_{datetime.now().date()}.pkl"
    
    # Check if cached data exists for today
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Download fresh data
    data = yf.download(ticker, period="1y", progress=False)
    
    # Cache it
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data
```

### 4. Error Handling
```python
def safe_download(ticker, **kwargs):
    """Download data with error handling"""
    try:
        data = yf.download(ticker, progress=False, **kwargs)
        if data.empty:
            print(f"No data available for {ticker}")
            return None
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None
```

## Alternative Data Sources

If yfinance is consistently blocked:

1. **Alpha Vantage** (free tier with API key)
   ```python
   # Requires: pip install alpha_vantage
   # Get free API key from: https://www.alphavantage.co/support/#api-key
   ```

2. **IEX Cloud** (free tier with API key)
   ```python
   # Requires: pip install pyEX
   # Get free API key from: https://iexcloud.io/
   ```

3. **Local CSV Files** (for testing)
   ```python
   # Use pre-downloaded data for development
   data = pd.read_csv('data/demo/btcusd_1h.csv', index_col='timestamp', parse_dates=True)
   ```

## Testing Without Live Data

For development and testing, you can use synthetic data:

```python
import numpy as np
import pandas as pd

def generate_synthetic_ohlcv(days=365, ticker="TEST"):
    """Generate synthetic OHLCV data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    # Random walk for price
    returns = np.random.normal(0.001, 0.02, days)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_price = close + np.random.normal(0, 0.5, days)
    volume = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df
```

## Summary

- yfinance is **FREE** and requires **NO API KEY**
- Rate limiting is temporary - wait and retry
- Use batch downloads and caching to minimize requests
- Have fallback options for development (synthetic or cached data)
- The error you're seeing is likely temporary rate limiting, not a permanent issue
