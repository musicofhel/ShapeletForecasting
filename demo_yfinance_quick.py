"""
Quick yfinance Demo - Basic Data Retrieval
==========================================

This script demonstrates basic yfinance functionality:
1. Downloading stock data
2. Getting ticker info
3. Retrieving historical data
4. Accessing financial statements
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

def demo_basic_download():
    """Demo 1: Basic stock data download"""
    print("=" * 60)
    print("DEMO 1: Basic Stock Data Download")
    print("=" * 60)
    
    # Download Apple stock data for the last month
    ticker = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nDownloading {ticker} data from {start_date.date()} to {end_date.date()}")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    print(f"\nData shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
    
    return data

def demo_ticker_info():
    """Demo 2: Get detailed ticker information"""
    print("\n" + "=" * 60)
    print("DEMO 2: Ticker Information")
    print("=" * 60)
    
    # Create ticker object
    ticker = yf.Ticker("MSFT")
    
    # Get stock info
    info = ticker.info
    
    # Display key information
    print("\nMicrosoft (MSFT) Information:")
    key_info = {
        "Company Name": info.get('longName', 'N/A'),
        "Sector": info.get('sector', 'N/A'),
        "Industry": info.get('industry', 'N/A'),
        "Market Cap": f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else 'N/A',
        "PE Ratio": info.get('trailingPE', 'N/A'),
        "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
        "Current Price": info.get('currentPrice', 'N/A'),
        "Volume": f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A'
    }
    
    for key, value in key_info.items():
        print(f"{key}: {value}")
    
    return ticker

def demo_multiple_tickers():
    """Demo 3: Download data for multiple tickers"""
    print("\n" + "=" * 60)
    print("DEMO 3: Multiple Tickers Download")
    print("=" * 60)
    
    # Download data for multiple tech stocks
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nDownloading data for: {', '.join(tickers)}")
    print(f"Period: Last 7 days")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Show closing prices
    print("\nClosing Prices:")
    print(data['Close'].tail())
    
    # Calculate daily returns
    returns = data['Close'].pct_change()
    print("\nDaily Returns (%):")
    print((returns.tail() * 100).round(2))
    
    return data

def demo_financial_data():
    """Demo 4: Get financial statements"""
    print("\n" + "=" * 60)
    print("DEMO 4: Financial Statements")
    print("=" * 60)
    
    ticker = yf.Ticker("AAPL")
    
    # Get quarterly financials
    print("\nApple Quarterly Income Statement (Last 4 Quarters):")
    quarterly_financials = ticker.quarterly_financials
    if not quarterly_financials.empty:
        # Show revenue and net income
        if 'Total Revenue' in quarterly_financials.index:
            print("\nTotal Revenue:")
            print(quarterly_financials.loc['Total Revenue'].apply(lambda x: f"${x:,.0f}"))
        
        if 'Net Income' in quarterly_financials.index:
            print("\nNet Income:")
            print(quarterly_financials.loc['Net Income'].apply(lambda x: f"${x:,.0f}"))
    
    # Get balance sheet
    print("\n\nBalance Sheet Summary (Most Recent):")
    balance_sheet = ticker.quarterly_balance_sheet
    if not balance_sheet.empty:
        latest_quarter = balance_sheet.columns[0]
        key_items = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity']
        
        for item in key_items:
            if item in balance_sheet.index:
                value = balance_sheet.loc[item, latest_quarter]
                print(f"{item}: ${value:,.0f}")

def demo_options_and_dividends():
    """Demo 5: Options and dividend data"""
    print("\n" + "=" * 60)
    print("DEMO 5: Options and Dividends")
    print("=" * 60)
    
    ticker = yf.Ticker("AAPL")
    
    # Get dividend history
    print("\nApple Dividend History (Last 5):")
    dividends = ticker.dividends
    if not dividends.empty:
        print(dividends.tail())
    
    # Get options expiration dates
    print("\nOptions Expiration Dates:")
    try:
        options_dates = ticker.options
        print(f"Available expiration dates: {len(options_dates)}")
        print(f"Next 5 expirations: {options_dates[:5]}")
        
        # Get options chain for the first expiration
        if options_dates:
            opt_chain = ticker.option_chain(options_dates[0])
            print(f"\nOptions chain for {options_dates[0]}:")
            print(f"Calls available: {len(opt_chain.calls)}")
            print(f"Puts available: {len(opt_chain.puts)}")
    except Exception as e:
        print(f"Options data not available: {e}")

def demo_crypto_data():
    """Demo 6: Cryptocurrency data"""
    print("\n" + "=" * 60)
    print("DEMO 6: Cryptocurrency Data")
    print("=" * 60)
    
    # Download Bitcoin data
    crypto_ticker = "BTC-USD"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nDownloading {crypto_ticker} data for the last week")
    btc_data = yf.download(crypto_ticker, start=start_date, end=end_date, progress=False)
    
    print("\nBitcoin (BTC-USD) - Last 7 days:")
    print(btc_data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Calculate volatility
    returns = btc_data['Close'].pct_change()
    volatility = returns.std() * (365 ** 0.5) * 100  # Annualized volatility
    print(f"\nAnnualized Volatility: {volatility:.2f}%")

def main():
    """Run all demos"""
    print("YFINANCE QUICK DEMO")
    print("===================")
    print("Demonstrating various yfinance capabilities\n")
    
    try:
        # Run demos
        demo_basic_download()
        demo_ticker_info()
        demo_multiple_tickers()
        demo_financial_data()
        demo_options_and_dividends()
        demo_crypto_data()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure you have yfinance installed: pip install yfinance")

if __name__ == "__main__":
    main()
