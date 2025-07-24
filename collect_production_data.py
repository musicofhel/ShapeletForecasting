"""
Production Data Collection Script
Collects extensive historical data from multiple tickers for robust model training
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDataCollector:
    """Collect extensive data for production-ready models"""
    
    def __init__(self, output_path: str = "data/production_sequences.pkl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Production ticker list - diverse set of liquid assets
        self.tickers = [
            # Major indices
            'SPY', 'QQQ', 'DIA', 'IWM',
            # Tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'WFC',
            # Energy
            'XOM', 'CVX', 'COP',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'CVS',
            # Consumer
            'WMT', 'HD', 'PG', 'KO',
            # Crypto-related
            'COIN', 'MSTR', 'RIOT',
            # Commodities ETFs
            'GLD', 'SLV', 'USO', 'UNG',
            # Volatility
            'VXX', 'UVXY',
            # International
            'EWJ', 'EWZ', 'FXI', 'EEM'
        ]
        
        # Time periods for different market conditions
        self.periods = [
            ('2019-01-01', '2019-12-31'),  # Pre-COVID normal market
            ('2020-01-01', '2020-12-31'),  # COVID crash and recovery
            ('2021-01-01', '2021-12-31'),  # Bull market
            ('2022-01-01', '2022-12-31'),  # Bear market
            ('2023-01-01', '2024-01-01'),  # Recovery
            ('2024-01-01', datetime.now().strftime('%Y-%m-%d'))  # Recent
        ]
        
    def collect_ticker_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Collect data for a single ticker with retries"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {ticker} from {start} to {end}")
                
                # Download with progress disabled to reduce API load
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    interval='1h'  # Hourly data for more patterns
                )
                
                if len(data) > 0:
                    # Add returns and volatility
                    data['returns'] = data['Close'].pct_change()
                    data['volatility'] = data['returns'].rolling(24).std()
                    data['volume_ma'] = data['Volume'].rolling(24).mean()
                    
                    # Add technical indicators
                    data['rsi'] = self.calculate_rsi(data['Close'])
                    data['macd'], data['macd_signal'] = self.calculate_macd(data['Close'])
                    
                    return data
                else:
                    logger.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to download {ticker} after {max_retries} attempts")
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
    
    def collect_all_data(self) -> Dict[str, Any]:
        """Collect data for all tickers and periods"""
        all_data = {
            'ticker_data': {},
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'tickers': self.tickers,
                'periods': self.periods,
                'interval': '1h'
            }
        }
        
        total_sequences = 0
        successful_tickers = []
        
        for ticker in self.tickers:
            ticker_sequences = []
            
            for start, end in self.periods:
                data = self.collect_ticker_data(ticker, start, end)
                
                if len(data) > 100:  # Minimum data requirement
                    ticker_sequences.append({
                        'data': data,
                        'period': f"{start}_{end}",
                        'ticker': ticker,
                        'length': len(data)
                    })
                    total_sequences += 1
                
                # Rate limiting
                time.sleep(0.5)
            
            if ticker_sequences:
                all_data['ticker_data'][ticker] = ticker_sequences
                successful_tickers.append(ticker)
                logger.info(f"Collected {len(ticker_sequences)} sequences for {ticker}")
        
        all_data['metadata']['total_sequences'] = total_sequences
        all_data['metadata']['successful_tickers'] = successful_tickers
        
        logger.info(f"\nCollection Summary:")
        logger.info(f"Total tickers processed: {len(successful_tickers)}/{len(self.tickers)}")
        logger.info(f"Total sequences collected: {total_sequences}")
        
        return all_data
    
    def save_data(self, data: Dict[str, Any]):
        """Save collected data"""
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {self.output_path}")
        
        # Save summary
        summary_path = self.output_path.parent / 'production_data_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Production Data Collection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Collection Date: {data['metadata']['collection_date']}\n")
            f.write(f"Total Sequences: {data['metadata']['total_sequences']}\n")
            f.write(f"Successful Tickers: {len(data['metadata']['successful_tickers'])}\n\n")
            f.write("Tickers:\n")
            for ticker in data['metadata']['successful_tickers']:
                f.write(f"  - {ticker}: {len(data['ticker_data'][ticker])} sequences\n")
    
    def run(self):
        """Run the complete data collection pipeline"""
        logger.info("Starting production data collection...")
        logger.info(f"Target tickers: {len(self.tickers)}")
        logger.info(f"Time periods: {len(self.periods)}")
        
        # Collect data
        data = self.collect_all_data()
        
        # Save data
        self.save_data(data)
        
        return data


def main():
    """Run production data collection"""
    print("PRODUCTION DATA COLLECTION")
    print("=" * 50)
    print("\nThis will collect extensive historical data for production model training.")
    print(f"Estimated time: 10-15 minutes")
    print("\nPress Ctrl+C to cancel at any time.\n")
    
    try:
        collector = ProductionDataCollector()
        data = collector.run()
        
        print("\n" + "=" * 50)
        print("COLLECTION COMPLETE!")
        print("=" * 50)
        print(f"\nCollected {data['metadata']['total_sequences']} sequences")
        print(f"From {len(data['metadata']['successful_tickers'])} tickers")
        print(f"\nData saved to: {collector.output_path}")
        print("\nNext steps:")
        print("1. Run wavelet_pattern_pipeline.py with --input data/production_sequences.pkl")
        print("2. Re-train models with the larger dataset")
        print("3. Expect significantly better performance!")
        
    except KeyboardInterrupt:
        print("\n\nCollection cancelled by user.")
    except Exception as e:
        print(f"\n\nError during collection: {e}")
        raise


if __name__ == "__main__":
    main()
