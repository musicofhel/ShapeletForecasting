"""
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
        print(f"\nTesting {ticker}:")
        update_chart(ticker)
