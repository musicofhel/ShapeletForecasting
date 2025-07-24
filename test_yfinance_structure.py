"""
Test yfinance data structure to understand the issue
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test fetching data
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print("Testing yfinance data structure...")
print("=" * 60)

# Download data
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

print(f"\nData type: {type(data)}")
print(f"Data shape: {data.shape}")
print(f"Data columns: {list(data.columns)}")
print(f"Data index type: {type(data.index)}")
print(f"Data index name: {data.index.name}")

# Check column structure
print("\nColumn structure:")
for col in data.columns:
    print(f"  {col}: {type(data[col])}")
    if hasattr(data[col], 'shape'):
        print(f"    Shape: {data[col].shape}")

# Test different ways to access data
print("\nTesting data access methods:")

# Method 1: Direct column access
try:
    close_values = data['Close'].values
    print(f"✓ data['Close'].values works - shape: {close_values.shape}")
except Exception as e:
    print(f"✗ data['Close'].values failed: {e}")

# Method 2: Using iloc
try:
    close_iloc = data.iloc[:, data.columns.get_loc('Close')].values
    print(f"✓ Using iloc works - shape: {close_iloc.shape}")
except Exception as e:
    print(f"✗ Using iloc failed: {e}")

# Method 3: Check if columns are MultiIndex
print(f"\nIs MultiIndex? {isinstance(data.columns, pd.MultiIndex)}")
if isinstance(data.columns, pd.MultiIndex):
    print("Column levels:", data.columns.levels)
    print("Column names:", data.columns.names)

# Try to create DataFrame
print("\nTesting DataFrame creation:")

# Method 1: Using index directly
try:
    df1 = pd.DataFrame({
        'timestamp': data.index,
        'price': data['Close'].values
    })
    print(f"✓ Method 1 works - shape: {df1.shape}")
except Exception as e:
    print(f"✗ Method 1 failed: {e}")

# Method 2: Flattening columns if MultiIndex
try:
    if isinstance(data.columns, pd.MultiIndex):
        # For MultiIndex columns, we need to access differently
        close_data = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close']
        df2 = pd.DataFrame({
            'timestamp': data.index,
            'price': close_data.values
        })
    else:
        df2 = pd.DataFrame({
            'timestamp': data.index,
            'price': data['Close'].values
        })
    print(f"✓ Method 2 works - shape: {df2.shape}")
except Exception as e:
    print(f"✗ Method 2 failed: {e}")

# Show sample data
print("\nSample data:")
print(data.head())
