"""
Test script for YFinance MVP
Verifies core functionality: fetching, caching, and backoff
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from yfinance_mvp import YFinanceFetcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_fetch():
    """Test basic data fetching"""
    logger.info("\n=== Testing Basic Fetch ===")
    
    # Clean up any existing database
    db_path = "test_price_data.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    fetcher = YFinanceFetcher(db_path=db_path)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = fetcher.fetch_with_backoff(
        "AAPL",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    assert data is not None, "Failed to fetch data"
    assert len(data) > 0, "No data returned"
    assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), "Missing columns"
    
    logger.info(f"✓ Fetched {len(data)} records successfully")
    return True


def test_cache_functionality():
    """Test that cache is working properly"""
    logger.info("\n=== Testing Cache Functionality ===")
    
    db_path = "test_price_data.db"
    fetcher = YFinanceFetcher(db_path=db_path)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # First fetch - should hit API
    start_time = time.time()
    data1 = fetcher.fetch_with_backoff(
        "MSFT",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    api_time = time.time() - start_time
    
    # Second fetch - should hit cache
    start_time = time.time()
    data2 = fetcher.fetch_with_backoff(
        "MSFT",
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    cache_time = time.time() - start_time
    
    assert data1 is not None and data2 is not None, "Failed to fetch data"
    assert len(data1) == len(data2), "Cache returned different data"
    assert cache_time < api_time, "Cache should be faster than API"
    
    logger.info(f"✓ API fetch time: {api_time:.2f}s")
    logger.info(f"✓ Cache fetch time: {cache_time:.2f}s")
    logger.info(f"✓ Cache is {api_time/cache_time:.1f}x faster")
    
    return True


def test_database_structure():
    """Test database structure and data integrity"""
    logger.info("\n=== Testing Database Structure ===")
    
    db_path = "test_price_data.db"
    
    with sqlite3.connect(db_path) as conn:
        # Check table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='price_data'
        """)
        assert cursor.fetchone() is not None, "price_data table not found"
        
        # Check columns
        cursor = conn.execute("PRAGMA table_info(price_data)")
        columns = [row[1] for row in cursor.fetchall()]
        expected_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"
        
        # Check data
        cursor = conn.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        assert count > 0, "No data in database"
        
        logger.info(f"✓ Database has {count} records")
        logger.info(f"✓ All required columns present")
    
    return True


def test_error_handling():
    """Test error handling with invalid ticker"""
    logger.info("\n=== Testing Error Handling ===")
    
    db_path = "test_price_data.db"
    fetcher = YFinanceFetcher(db_path=db_path)
    
    # Test with invalid ticker
    data = fetcher.fetch_with_backoff(
        "INVALID_TICKER_XYZ",
        "2024-01-01",
        "2024-01-31"
    )
    
    # Should handle gracefully and return None
    assert data is None or len(data) == 0, "Should handle invalid ticker gracefully"
    logger.info("✓ Invalid ticker handled gracefully")
    
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("Starting YFinance MVP Tests")
    logger.info("=" * 50)
    
    tests = [
        test_basic_fetch,
        test_cache_functionality,
        test_database_structure,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, "PASSED"))
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            results.append((test.__name__, f"FAILED: {e}"))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    for test_name, status in results:
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    # Clean up test database
    if os.path.exists("test_price_data.db"):
        os.remove("test_price_data.db")
        logger.info("\nCleaned up test database")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
