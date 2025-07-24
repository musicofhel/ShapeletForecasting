"""
YFinance MVP with Smart Backoff and SQLite Storage
Sprint 2: Minimal implementation with exponential backoff
"""

import yfinance as yf
import sqlite3
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Minimal YFinance fetcher with exponential backoff and SQLite storage"""
    
    def __init__(self, db_path: str = "price_data.db"):
        self.db_path = db_path
        self.backoff_times = [1, 2, 4, 8, 16]  # Exponential backoff in seconds
        self.max_retries = len(self.backoff_times)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with price_data table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _check_cache(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Check if data exists in cache for given ticker and date range"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
            
            if not df.empty:
                logger.info(f"Found {len(df)} cached records for {ticker}")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            
        return None
    
    def _save_to_cache(self, ticker: str, data: pd.DataFrame):
        """Save fetched data to SQLite cache"""
        timestamp = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (ticker, date, open, high, low, close, volume, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    timestamp
                ))
            conn.commit()
        
        logger.info(f"Saved {len(data)} records for {ticker} to cache")
    
    def fetch_with_backoff(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with exponential backoff on rate limits"""
        
        # Check cache first
        cached_data = self._check_cache(ticker, start_date, end_date)
        if cached_data is not None:
            return cached_data
        
        # Fetch from API with backoff
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching {ticker} from YFinance (attempt {attempt + 1}/{self.max_retries})")
                
                # Create ticker object and fetch data
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                
                # Save to cache
                self._save_to_cache(ticker, data)
                
                logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                return data
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit error
                if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = self.backoff_times[attempt]
                        logger.warning(f"Rate limit detected. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached. Rate limit persists.")
                        return None
                else:
                    logger.error(f"Error fetching {ticker}: {e}")
                    return None
        
        return None
    
    def test_backoff(self, ticker: str = "AAPL"):
        """Test backoff mechanism by making rapid requests"""
        logger.info("Testing backoff mechanism with rapid requests...")
        
        end_date = datetime.now()
        results = []
        
        # Make 5 rapid requests to trigger rate limit
        for i in range(5):
            start_date = end_date - timedelta(days=30 * (i + 1))
            logger.info(f"\nRequest {i + 1}: Fetching {ticker} for period ending {start_date.strftime('%Y-%m-%d')}")
            
            start_time = time.time()
            data = self.fetch_with_backoff(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            elapsed = time.time() - start_time
            
            results.append({
                'request': i + 1,
                'success': data is not None,
                'records': len(data) if data is not None else 0,
                'elapsed_time': round(elapsed, 2)
            })
            
            # Small delay between requests
            time.sleep(0.5)
        
        return results


def main():
    """Main function to demonstrate YFinance MVP"""
    
    # Initialize fetcher
    fetcher = YFinanceFetcher()
    
    # Test 1: Fetch AAPL for 1 month
    logger.info("\n=== Test 1: Fetch AAPL for 1 month ===")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if data is not None:
        logger.info(f"\nFetched data summary:")
        logger.info(f"Records: {len(data)}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Columns: {list(data.columns)}")
        logger.info(f"\nFirst 5 rows:")
        print(data.head())
        logger.info(f"\nLast 5 rows:")
        print(data.tail())
    
    # Test 2: Verify data is cached
    logger.info("\n=== Test 2: Verify data is cached ===")
    cached_data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if cached_data is not None:
        logger.info("Successfully retrieved data from cache (no API call)")
    
    # Test 3: Test backoff mechanism
    logger.info("\n=== Test 3: Test backoff mechanism ===")
    backoff_results = fetcher.test_backoff("AAPL")
    
    logger.info("\nBackoff test results:")
    for result in backoff_results:
        logger.info(f"Request {result['request']}: "
                   f"Success={result['success']}, "
                   f"Records={result['records']}, "
                   f"Time={result['elapsed_time']}s")
    
    # Test 4: Check database contents
    logger.info("\n=== Test 4: Database contents ===")
    with sqlite3.connect(fetcher.db_path) as conn:
        cursor = conn.execute("""
            SELECT ticker, COUNT(*) as record_count, 
                   MIN(date) as earliest_date, 
                   MAX(date) as latest_date
            FROM price_data
            GROUP BY ticker
        """)
        
        logger.info("\nDatabase summary:")
        for row in cursor.fetchall():
            logger.info(f"Ticker: {row[0]}, Records: {row[1]}, "
                       f"Date range: {row[2]} to {row[3]}")


if __name__ == "__main__":
    main()
