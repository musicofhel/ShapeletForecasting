"""
Data utilities using Polygon API for financial data
Handles data fetching, caching, and preprocessing
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonDataManager:
    """Manages data fetching from Polygon API with caching"""
    
    def __init__(self, api_key: str = None, cache_dir: str = "data/cache"):
        """Initialize Polygon data manager
        
        Args:
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
            cache_dir: Directory for SQLite cache
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY', 'rzbt8GG8mqwLo1EMuSzbkC34uxA_Df1R')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "polygon_cache.db"
        self._init_cache()
        
    def _init_cache(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timestamp INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_info (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                market_cap REAL,
                exchange TEXT,
                type TEXT,
                last_updated INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _is_crypto(self, ticker: str) -> bool:
        """Check if ticker is a cryptocurrency"""
        crypto_suffixes = ['USD', 'USDT', 'USDC', 'EUR', 'GBP']
        return any(ticker.endswith(suffix) for suffix in crypto_suffixes)
        
    def _format_crypto_ticker(self, ticker: str) -> str:
        """Format cryptocurrency ticker for Polygon API"""
        # Convert BTC-USD to X:BTCUSD format
        if '-' in ticker:
            base, quote = ticker.split('-')
            return f"X:{base}{quote}"
        return ticker
        
    def download_data(self, ticker: str, period: str = "1mo", 
                     interval: str = "1d") -> Optional[pd.DataFrame]:
        """Download data from Polygon API
        
        Args:
            ticker: Stock/crypto ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (only '1d' supported for now)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Check cache first
            cached_data = self._get_cached_data(ticker, period)
            if cached_data is not None and len(cached_data) > 0:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
            
            # Calculate date range
            end_date = datetime.now()
            period_days = self._period_to_days(period)
            start_date = end_date - timedelta(days=period_days)
            
            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Determine if crypto or stock
            if self._is_crypto(ticker):
                formatted_ticker = self._format_crypto_ticker(ticker)
                url = f"https://api.polygon.io/v2/aggs/ticker/{formatted_ticker}/range/1/day/{start_str}/{end_str}"
            else:
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
            
            params = {
                "apiKey": self.api_key,
                "adjusted": "true",
                "sort": "asc"
            }
            
            logger.info(f"Fetching {ticker} from Polygon API...")
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                # Convert to DataFrame
                results = data["results"]
                df = pd.DataFrame(results)
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    't': 'timestamp'
                })
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                
                # Calculate returns
                df['returns'] = df['close'].pct_change().fillna(0)
                
                # Cache the data
                self._cache_data(ticker, df)
                
                logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
                return df
                
            else:
                logger.warning(f"No data returned for {ticker}: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
            
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get ticker information from Polygon API"""
        try:
            # Check cache
            cached_info = self._get_cached_ticker_info(ticker)
            if cached_info:
                return cached_info
                
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
            params = {"apiKey": self.api_key}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") == "OK":
                results = data.get("results", {})
                info = {
                    'ticker': ticker,
                    'name': results.get('name', ticker),
                    'market_cap': results.get('market_cap', 0),
                    'exchange': results.get('primary_exchange', 'Unknown'),
                    'type': results.get('type', 'Unknown')
                }
                
                # Cache the info
                self._cache_ticker_info(info)
                return info
                
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {e}")
            
        return None
        
    def _period_to_days(self, period: str) -> int:
        """Convert period string to number of days"""
        period_map = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            'max': 3650
        }
        return period_map.get(period, 30)
        
    def _get_cached_data(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available and recent"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            period_days = self._period_to_days(period)
            start_date = end_date - timedelta(days=period_days)
            
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_data
                WHERE ticker = ? AND date >= ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(ticker, start_date.strftime("%Y-%m-%d")))
            conn.close()
            
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['returns'] = df['close'].pct_change().fillna(0)
                
                # Check if cache is recent (within last hour)
                last_date = df.index[-1]
                if (datetime.now() - last_date).total_seconds() < 3600:
                    return df
                    
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            
        return None
        
    def _cache_data(self, ticker: str, df: pd.DataFrame):
        """Cache price data to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            cache_df = df.reset_index()
            cache_df['ticker'] = ticker
            cache_df['date'] = cache_df['date'].dt.strftime("%Y-%m-%d")
            cache_df['timestamp'] = int(time.time())
            
            # Insert or replace data
            cache_df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'timestamp']].to_sql(
                'price_data', conn, if_exists='replace', index=False
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            
    def _get_cached_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get ticker info from cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, market_cap, exchange, type
                FROM ticker_info
                WHERE ticker = ? AND last_updated > ?
            """, (ticker, int(time.time()) - 86400))  # Cache for 24 hours
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'ticker': ticker,
                    'name': row[0],
                    'market_cap': row[1],
                    'exchange': row[2],
                    'type': row[3]
                }
                
        except Exception as e:
            logger.error(f"Ticker info cache read error: {e}")
            
        return None
        
    def _cache_ticker_info(self, info: Dict):
        """Cache ticker info to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO ticker_info
                (ticker, name, market_cap, exchange, type, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                info['ticker'],
                info['name'],
                info['market_cap'],
                info['exchange'],
                info['type'],
                int(time.time())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ticker info cache write error: {e}")
            


# Create a global instance for backward compatibility
data_manager = PolygonDataManager()


# Backward compatibility function
def load_financial_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Legacy function for loading financial data"""
    days = (end_date - start_date).days
    
    if days <= 7:
        period = "5d"
    elif days <= 30:
        period = "1mo"
    elif days <= 90:
        period = "3mo"
    elif days <= 180:
        period = "6mo"
    elif days <= 365:
        period = "1y"
    else:
        period = "2y"
        
    return data_manager.download_data(ticker, period=period)
