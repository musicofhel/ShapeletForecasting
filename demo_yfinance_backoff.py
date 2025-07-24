"""
Demonstration of YFinance MVP with Backoff Mechanism
Shows how the exponential backoff works when rate limits are encountered
"""

import sqlite3
import time
from datetime import datetime, timedelta
import logging
from yfinance_mvp_fixed import YFinanceFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_cache_functionality(fetcher):
    """Demonstrate cache-first approach with real tickers"""
    logger.info("\n=== Demonstrating Cache Functionality ===")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # First fetch - will hit API or cache
    logger.info("\nFetching AAPL...")
    start_time = time.time()
    data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    first_time = time.time() - start_time
    
    if data is not None:
        logger.info(f"✓ Retrieved {len(data)} records in {first_time:.3f}s")
        logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
    
    # Second fetch - should be from cache
    logger.info("\nFetching AAPL again (should be from cache)...")
    start_time = time.time()
    data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    cache_time = time.time() - start_time
    
    if data is not None:
        logger.info(f"✓ Retrieved {len(data)} records from cache in {cache_time:.3f}s")
        if cache_time < first_time / 10:  # Cache should be much faster
            logger.info("✓ Cache retrieval is significantly faster!")


def demonstrate_backoff_simulation(fetcher):
    """Demonstrate the backoff mechanism with simulation"""
    logger.info("\n=== Demonstrating Exponential Backoff ===")
    logger.info("Simulating rate limit scenario...")
    
    # Show the backoff times
    logger.info(f"\nBackoff schedule: {fetcher.backoff_times} seconds")
    logger.info("Total wait time if all retries used: {}s".format(sum(fetcher.backoff_times)))
    
    # Run the simulation
    success = fetcher.simulate_rate_limit()
    
    if success:
        logger.info("\n✓ Backoff mechanism successfully handled rate limit!")
    else:
        logger.info("\n✗ Max retries exceeded")


def demonstrate_real_world_usage(fetcher):
    """Demonstrate real-world usage with popular tickers"""
    logger.info("\n=== Real-World Usage Demo ===")
    
    # Popular tickers to test
    tickers = ["SPY", "QQQ", "MSFT", "GOOGL", "TSLA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Just last week
    
    logger.info(f"Fetching last week's data for popular tickers...")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    results = []
    for ticker in tickers:
        logger.info(f"\nFetching {ticker}...")
        start_time = time.time()
        
        data = fetcher.fetch_with_backoff(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        elapsed = time.time() - start_time
        
        if data is not None:
            results.append({
                'ticker': ticker,
                'success': True,
                'records': len(data),
                'latest_close': data['Close'].iloc[-1],
                'time': elapsed
            })
            logger.info(f"✓ Success: {len(data)} records, latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            results.append({
                'ticker': ticker,
                'success': False,
                'records': 0,
                'latest_close': 0,
                'time': elapsed
            })
            logger.info(f"✗ Failed to fetch data")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY:")
    successful = sum(1 for r in results if r['success'])
    logger.info(f"Successfully fetched: {successful}/{len(tickers)} tickers")
    
    if successful > 0:
        logger.info("\nSuccessful fetches:")
        for r in results:
            if r['success']:
                logger.info(f"  {r['ticker']}: ${r['latest_close']:.2f} ({r['records']} records in {r['time']:.2f}s)")


def demonstrate_database_operations(fetcher):
    """Demonstrate database operations"""
    logger.info("\n=== Database Operations ===")
    
    with sqlite3.connect(fetcher.db_path) as conn:
        # Show summary statistics
        cursor = conn.execute("""
            SELECT 
                ticker,
                COUNT(*) as records,
                MIN(date) as start_date,
                MAX(date) as end_date,
                ROUND(AVG(close), 2) as avg_close,
                ROUND(MIN(low), 2) as min_low,
                ROUND(MAX(high), 2) as max_high
            FROM price_data
            GROUP BY ticker
            ORDER BY ticker
        """)
        
        results = cursor.fetchall()
        
        if results:
            logger.info("\nCached data summary:")
            logger.info("Ticker | Records | Start Date | End Date   | Avg Close | Min Low | Max High")
            logger.info("-" * 80)
            
            for row in results:
                logger.info(f"{row[0]:6} | {row[1]:7} | {row[2]} | {row[3]} | "
                           f"{row[4]:9} | {row[5]:7} | {row[6]:8}")
        else:
            logger.info("No data in cache yet")


def main():
    """Main demonstration function"""
    logger.info("YFinance MVP with Real Data Demonstration")
    logger.info("=" * 50)
    
    # Create fetcher with demo database
    fetcher = YFinanceFetcher(db_path="demo_real_data.db", polite_delay=3)
    
    # Demonstrate cache functionality
    demonstrate_cache_functionality(fetcher)
    
    # Demonstrate backoff simulation
    demonstrate_backoff_simulation(fetcher)
    
    # Demonstrate real-world usage
    demonstrate_real_world_usage(fetcher)
    
    # Show database contents
    demonstrate_database_operations(fetcher)
    
    logger.info("\n" + "=" * 50)
    logger.info("Demonstration complete!")
    logger.info("\nKey Features Demonstrated:")
    logger.info("✓ Real market data fetching")
    logger.info("✓ SQLite cache with fast retrieval")
    logger.info("✓ Exponential backoff (1s → 2s → 4s → 8s → 16s)")
    logger.info("✓ Polite delays between requests")
    logger.info("✓ Database persistence")
    
    logger.info("\nThe YFinance MVP is ready for production use!")
    logger.info("Note: Using polite delays and caching to avoid rate limits.")


if __name__ == "__main__":
    main()
