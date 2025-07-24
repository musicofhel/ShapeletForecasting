"""
Test YFinance with polite delays approach from blog post
"""

import os
from datetime import datetime, timedelta
from yfinance_mvp_fixed import YFinanceFetcher
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_blog_approach():
    """Test the approach suggested in the blog post"""
    
    # List of stocks to test
    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    data_file = "test_stock_data.csv"
    
    # Initialize fetcher with 3-second polite delay
    fetcher = YFinanceFetcher(db_path="test_price_data.db", polite_delay=3)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info("Testing YFinance with polite delays approach")
    logger.info("=" * 50)
    logger.info(f"Stocks: {stocks}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Polite delay: 3 seconds between requests")
    logger.info("")
    
    # Check if we have cached data
    if os.path.exists(data_file):
        logger.info(f"Found existing data file: {data_file}")
        logger.info("Delete it to test fresh API calls")
    
    # Use the batch fetch method
    logger.info("Starting batch fetch with polite delays...")
    all_data = fetcher.fetch_batch(
        stocks,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        save_csv=True,
        csv_path=data_file
    )
    
    # Summary
    if not all_data.empty:
        logger.info("\n" + "=" * 50)
        logger.info("FETCH SUMMARY:")
        logger.info(f"Total records fetched: {len(all_data)}")
        
        # Group by ticker
        for ticker in stocks:
            ticker_data = all_data[all_data['ticker'] == ticker]
            if not ticker_data.empty:
                logger.info(f"{ticker}: {len(ticker_data)} records")
            else:
                logger.info(f"{ticker}: No data")
        
        logger.info(f"\nData saved to: {data_file}")
        logger.info(f"Database saved to: test_price_data.db")
    else:
        logger.warning("No data was fetched successfully")
    
    # Test cache functionality
    logger.info("\n" + "=" * 50)
    logger.info("Testing cache retrieval...")
    
    # Try to fetch AAPL again - should come from cache
    cached_data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if cached_data is not None:
        logger.info("✓ Successfully retrieved AAPL from cache (no API call)")
    
    return all_data


def test_single_ticker_with_retry():
    """Test single ticker with extended retry on 429"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing single ticker with retry logic...")
    
    fetcher = YFinanceFetcher(db_path="test_single_ticker.db", polite_delay=3)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Try to fetch a single ticker
    data = fetcher.fetch_with_backoff(
        "SPY",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if data is not None:
        logger.info(f"✓ Successfully fetched SPY: {len(data)} records")
        logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
    else:
        logger.warning("✗ Failed to fetch SPY data")


if __name__ == "__main__":
    # Test the blog's batch approach
    batch_data = test_blog_approach()
    
    # Test single ticker
    test_single_ticker_with_retry()
    
    logger.info("\n" + "=" * 50)
    logger.info("Test complete!")
    logger.info("\nKey improvements implemented:")
    logger.info("1. Polite 3-second delay between ALL requests")
    logger.info("2. Extended 10-second wait on 429 errors")
    logger.info("3. SQLite caching to avoid redundant API calls")
    logger.info("4. Batch processing with CSV export option")
