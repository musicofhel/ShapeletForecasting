"""
S&P 500 Production Data Collection Script
Collects comprehensive historical data from all S&P 500 companies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import requests
from bs4 import BeautifulSoup
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SP500DataCollector:
    """Collect comprehensive data from S&P 500 companies"""
    
    def __init__(self, output_path: str = "data/sp500_production_data.pkl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get S&P 500 tickers
        self.tickers = self.get_sp500_tickers()
        logger.info(f"Found {len(self.tickers)} S&P 500 tickers")
        
        # Time periods for comprehensive coverage
        self.periods = [
            ('2018-01-01', '2019-01-01'),  # Pre-volatility period
            ('2019-01-01', '2020-01-01'),  # Normal market
            ('2020-01-01', '2021-01-01'),  # COVID crash and recovery
            ('2021-01-01', '2022-01-01'),  # Bull market
            ('2022-01-01', '2023-01-01'),  # Bear market/inflation
            ('2023-01-01', '2024-01-01'),  # Recovery period
            ('2024-01-01', datetime.now().strftime('%Y-%m-%d'))  # Current
        ]
        
        # Track progress
        self.successful_downloads = 0
        self.failed_downloads = []
        
    def get_sp500_tickers(self) -> List[str]:
        """Fetch current S&P 500 ticker list from Wikipedia"""
        try:
            # Get S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table
            table = soup.find('table', {'id': 'constituents'})
            if not table:
                # Fallback to first table
                table = soup.find('table')
            
            # Extract tickers
            tickers = []
            for row in table.findAll('tr')[1:]:  # Skip header
                cells = row.findAll('td')
                if cells:
                    ticker = cells[0].text.strip()
                    # Clean up ticker (remove footnotes, etc.)
                    ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B for yfinance
                    tickers.append(ticker)
            
            logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")
            return sorted(tickers)
            
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500 list: {e}")
            logger.info("Using fallback ticker list")
            
            # Fallback list of major S&P 500 companies
            return [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'NVDA',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
                'NFLX', 'CRM', 'XOM', 'CVX', 'PFE', 'CSCO', 'TMO', 'ABBV', 'ACN',
                'COST', 'NKE', 'WMT', 'ABT', 'MRK', 'AVGO', 'PEP', 'INTC', 'CMCSA',
                'VZ', 'T', 'ORCL', 'DHR', 'QCOM', 'NEE', 'TXN', 'PM', 'UNP', 'HON',
                'RTX', 'IBM', 'SPGI', 'AMD', 'GS', 'CAT', 'BA', 'SBUX', 'AMGN',
                'BLK', 'INTU', 'CVS', 'GILD', 'MDLZ', 'AXP', 'DE', 'BKNG', 'MMC',
                'TJX', 'LMT', 'MO', 'ISRG', 'ZTS', 'SYK', 'ADI', 'LRCX', 'TMUS',
                'ATVI', 'ADP', 'VRTX', 'REGN', 'MS', 'PLD', 'CI', 'CB', 'ETN',
                'BSX', 'EQIX', 'CME', 'BDX', 'CSX', 'SO', 'HUM', 'CL', 'PANW',
                'FIS', 'AON', 'MU', 'ITW', 'SHW', 'GE', 'MCO', 'DUK', 'ICE'
            ]
    
    def collect_ticker_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Collect data for a single ticker with robust error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Download with specific parameters for production
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    interval='1d',  # Daily data for stability
                    auto_adjust=True,  # Adjust for splits/dividends
                    prepost=False,
                    threads=False
                )
                
                if len(data) > 50:  # Minimum data requirement
                    # Calculate technical indicators
                    data['returns'] = data['Close'].pct_change()
                    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    data['volatility'] = data['returns'].rolling(20).std()
                    data['volume_ma'] = data['Volume'].rolling(20).mean()
                    data['price_ma_10'] = data['Close'].rolling(10).mean()
                    data['price_ma_50'] = data['Close'].rolling(50).mean()
                    
                    # RSI
                    data['rsi'] = self.calculate_rsi(data['Close'])
                    
                    # MACD
                    data['macd'], data['macd_signal'] = self.calculate_macd(data['Close'])
                    
                    # Bollinger Bands
                    data['bb_upper'], data['bb_lower'] = self.calculate_bollinger_bands(data['Close'])
                    
                    return data
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def collect_all_data(self, max_tickers: int = None) -> Dict[str, Any]:
        """Collect data for all S&P 500 tickers"""
        all_data = {
            'ticker_data': {},
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'tickers': self.tickers[:max_tickers] if max_tickers else self.tickers,
                'periods': self.periods,
                'interval': '1d',
                'total_tickers': len(self.tickers)
            }
        }
        
        total_sequences = 0
        successful_tickers = []
        tickers_to_process = self.tickers[:max_tickers] if max_tickers else self.tickers
        
        print(f"\nCollecting data for {len(tickers_to_process)} tickers...")
        print("This will take approximately 30-45 minutes for all S&P 500 stocks.")
        print("Progress will be shown every 10 tickers.\n")
        
        for i, ticker in enumerate(tickers_to_process):
            ticker_sequences = []
            
            for start, end in self.periods:
                data = self.collect_ticker_data(ticker, start, end)
                
                if len(data) > 50:
                    ticker_sequences.append({
                        'data': data,
                        'period': f"{start}_{end}",
                        'ticker': ticker,
                        'length': len(data),
                        'sector': self.get_sector(ticker)  # Add sector info
                    })
                    total_sequences += 1
                
                # Rate limiting
                time.sleep(0.1)
            
            if ticker_sequences:
                all_data['ticker_data'][ticker] = ticker_sequences
                successful_tickers.append(ticker)
                self.successful_downloads += 1
            else:
                self.failed_downloads.append(ticker)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(tickers_to_process)} tickers processed")
                logger.info(f"Successful: {self.successful_downloads}, Failed: {len(self.failed_downloads)}")
        
        all_data['metadata']['total_sequences'] = total_sequences
        all_data['metadata']['successful_tickers'] = successful_tickers
        all_data['metadata']['failed_tickers'] = self.failed_downloads
        
        logger.info(f"\nCollection Summary:")
        logger.info(f"Total tickers processed: {len(successful_tickers)}/{len(tickers_to_process)}")
        logger.info(f"Total sequences collected: {total_sequences}")
        logger.info(f"Failed downloads: {len(self.failed_downloads)}")
        
        return all_data
    
    def get_sector(self, ticker: str) -> str:
        """Get sector information for a ticker (simplified)"""
        # In production, you'd fetch this from a proper data source
        # For now, return a placeholder
        return "Unknown"
    
    def save_data(self, data: Dict[str, Any]):
        """Save collected data with compression"""
        logger.info(f"Saving data to {self.output_path}")
        
        # Save as compressed pickle
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save summary
        summary_path = self.output_path.parent / 'sp500_data_summary.json'
        summary = {
            'collection_date': data['metadata']['collection_date'],
            'total_sequences': data['metadata']['total_sequences'],
            'successful_tickers': len(data['metadata']['successful_tickers']),
            'failed_tickers': len(data['metadata'].get('failed_tickers', [])),
            'periods': data['metadata']['periods'],
            'file_size_mb': self.output_path.stat().st_size / (1024 * 1024)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data saved successfully ({summary['file_size_mb']:.1f} MB)")
    
    def run(self, max_tickers: int = None):
        """Run the complete data collection pipeline"""
        logger.info("Starting S&P 500 data collection...")
        logger.info(f"Target tickers: {len(self.tickers)}")
        logger.info(f"Time periods: {len(self.periods)}")
        
        if max_tickers:
            logger.info(f"Limiting to first {max_tickers} tickers")
        
        # Collect data
        data = self.collect_all_data(max_tickers)
        
        # Save data
        self.save_data(data)
        
        return data


def main():
    """Run S&P 500 data collection"""
    print("S&P 500 PRODUCTION DATA COLLECTION")
    print("=" * 60)
    print("\nThis will collect comprehensive historical data from S&P 500 companies.")
    print("\nOptions:")
    print("1. Quick test (first 10 tickers) - ~2 minutes")
    print("2. Medium dataset (first 100 tickers) - ~15 minutes")
    print("3. Full S&P 500 (all ~500 tickers) - ~45 minutes")
    print("\nPress Ctrl+C to cancel at any time.\n")
    
    try:
        # For production, we'll collect first 100 tickers as a good balance
        # You can change this to None for all 500
        collector = SP500DataCollector()
        
        # Collect first 100 S&P 500 companies
        data = collector.run(max_tickers=100)
        
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"\nCollected {data['metadata']['total_sequences']} sequences")
        print(f"From {len(data['metadata']['successful_tickers'])} tickers")
        print(f"Failed: {len(data['metadata'].get('failed_tickers', []))} tickers")
        print(f"\nData saved to: {collector.output_path}")
        print(f"File size: {collector.output_path.stat().st_size / (1024*1024):.1f} MB")
        print("\nNext steps:")
        print("1. Process with wavelet pipeline")
        print("2. Re-train models with this comprehensive dataset")
        print("3. Expect significantly better production performance!")
        
    except KeyboardInterrupt:
        print("\n\nCollection cancelled by user.")
    except Exception as e:
        print(f"\n\nError during collection: {e}")
        raise


if __name__ == "__main__":
    main()
