"""
YFinance MVP with Smart Backoff and SQLite Storage - Fixed Version
Sprint 2: Minimal implementation with exponential backoff
"""

import yfinance as yf
import sqlite3
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Minimal YFinance fetcher with exponential backoff and SQLite storage"""
    
    def __init__(self, db_path: str = "price_data.db", polite_delay: int = 3):
        self.db_path = db_path
        self.backoff_times = [1, 2, 4, 8, 16]  # Exponential backoff in seconds
        self.max_retries = len(self.backoff_times)
        self.polite_delay = polite_delay  # Delay between requests to be polite
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
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is related to rate limiting"""
        rate_limit_indicators = [
            '429', 'rate limit', 'too many requests', 
            'exceeded', 'quota', 'throttle'
        ]
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
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
                
                # Try to get info first to check if ticker is valid
                try:
                    info = stock.info
                    if not info or 'symbol' not in info:
                        logger.warning(f"Ticker {ticker} may not be valid")
                except:
                    pass
                
                # Fetch historical data
                data = stock.history(start=start_date, end=end_date, auto_adjust=True, prepost=False)
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                
                # Save to cache
                self._save_to_cache(ticker, data)
                
                logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                
                # Add polite delay after successful fetch to avoid rate limits
                if self.polite_delay > 0:
                    logger.info(f"Waiting {self.polite_delay}s before next request (polite delay)...")
                    time.sleep(self.polite_delay)
                
                return data
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a 429 error in the exception chain
                import traceback
                full_error = traceback.format_exc()
                
                # Check for rate limit error in both the message and full traceback
                if self._is_rate_limit_error(error_msg) or self._is_rate_limit_error(full_error):
                    if attempt < self.max_retries - 1:
                        wait_time = self.backoff_times[attempt]
                        logger.warning(f"Rate limit detected (429 error). Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached. Rate limit persists.")
                        return None
                else:
                    # For non-rate-limit errors, don't retry
                    logger.error(f"Non-recoverable error for {ticker}: {e}")
                    return None
        
        return None
    
    def test_backoff(self, ticker: str = "SPY"):
        """Test backoff mechanism by making rapid requests"""
        logger.info("Testing backoff mechanism with rapid requests...")
        
        end_date = datetime.now()
        results = []
        
        # Make 5 rapid requests to potentially trigger rate limit
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
    
    def simulate_rate_limit(self):
        """Simulate rate limit scenario for testing"""
        logger.info("\n=== Simulating Rate Limit Scenario ===")
        
        class MockRateLimitError(Exception):
            pass
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Simulated request (attempt {attempt + 1}/{self.max_retries})")
                
                # Simulate rate limit on first 3 attempts
                if attempt < 3:
                    raise MockRateLimitError("429 Too Many Requests")
                
                # Success on 4th attempt
                logger.info("Request successful!")
                return True
                
            except MockRateLimitError as e:
                error_msg = str(e)
                
                if self._is_rate_limit_error(error_msg) and attempt < self.max_retries - 1:
                    wait_time = self.backoff_times[attempt]
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached")
                    return False
        
        return False
    
    def fetch_batch(self, tickers: List[str], start_date: str, end_date: str, 
                   save_csv: bool = False, csv_path: str = "stock_data.csv") -> pd.DataFrame:
        """Fetch multiple tickers with polite delays, following blog post approach"""
        all_data = pd.DataFrame()
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"Fetching {ticker} ({i+1}/{len(tickers)})...")
                
                # Check cache first
                data = self._check_cache(ticker, start_date, end_date)
                
                if data is None:
                    # Need to fetch from API
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date)
                    
                    if not data.empty:
                        self._save_to_cache(ticker, data)
                        # Add ticker column for batch data
                        data['ticker'] = ticker
                        all_data = pd.concat([all_data, data])
                        
                        # Polite delay between API calls
                        if i < len(tickers) - 1:  # Don't delay after last ticker
                            logger.info(f"Waiting {self.polite_delay}s before next request...")
                            time.sleep(self.polite_delay)
                else:
                    # Data from cache
                    data['ticker'] = ticker
                    all_data = pd.concat([all_data, data])
                    
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Hit 429 error on {ticker}. Waiting 10 seconds...")
                    time.sleep(10)
                    # Try once more after extended wait
                    try:
                        stock = yf.Ticker(ticker)
                        data = stock.history(start=start_date, end=end_date)
                        if not data.empty:
                            self._save_to_cache(ticker, data)
                            data['ticker'] = ticker
                            all_data = pd.concat([all_data, data])
                    except Exception as e2:
                        logger.error(f"Failed to fetch {ticker} after retry: {e2}")
                else:
                    logger.error(f"Error with {ticker}: {e}")
        
        # Save to CSV if requested
        if save_csv and not all_data.empty:
            all_data.to_csv(csv_path)
            logger.info(f"Saved batch data to {csv_path}")
        
        return all_data


def main():
    """Main function to demonstrate YFinance MVP"""
    
    # Initialize fetcher
    fetcher = YFinanceFetcher()
    
    # Test 1: Try with a more reliable ticker (SPY)
    logger.info("\n=== Test 1: Fetch SPY for 1 month ===")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = fetcher.fetch_with_backoff(
        "SPY",
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
        "SPY",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if cached_data is not None:
        logger.info("Successfully retrieved data from cache (no API call)")
    
    # Test 3: Simulate rate limit scenario
    fetcher.simulate_rate_limit()
    
    # Test 4: Try multiple tickers
    logger.info("\n=== Test 4: Try multiple tickers ===")
    test_tickers = ["MSFT", "GOOGL", "TSLA"]
    
    for ticker in test_tickers:
        logger.info(f"\nTrying {ticker}...")
        test_data = fetcher.fetch_with_backoff(
            ticker,
            (end_date - timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if test_data is not None:
            logger.info(f"✓ {ticker}: Fetched {len(test_data)} records")
        else:
            logger.info(f"✗ {ticker}: Failed to fetch data")
    
    # Test 5: Check database contents
    logger.info("\n=== Test 5: Database contents ===")
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
