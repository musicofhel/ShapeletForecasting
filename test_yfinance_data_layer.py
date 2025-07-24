"""
Test script to verify YFinance data layer implementation
Tests the new data_utils_yfinance module and integration
"""

import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_yfinance_data_layer():
    """Test the YFinance data layer implementation"""
    
    print("=" * 60)
    print("Testing YFinance Data Layer Implementation")
    print("=" * 60)
    
    # Test 1: Import the module
    print("\n1. Testing module import...")
    try:
        from src.dashboard.data_utils import data_manager, DataManager, load_financial_data
        print("✓ Successfully imported data_utils module")
        print(f"  - data_manager type: {type(data_manager).__name__}")
        print(f"  - DataManager type: {DataManager.__name__}")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return
    
    # Test 2: Create data manager instance
    print("\n2. Testing data manager instantiation...")
    try:
        dm = DataManager()
        print("✓ Successfully created DataManager instance")
        print(f"  - Cache path: {dm.db_path}")
        print(f"  - Backoff times: {dm.backoff_times}")
    except Exception as e:
        print(f"✗ Failed to create instance: {e}")
        return
    
    # Test 3: Test data download
    print("\n3. Testing data download...")
    test_tickers = ["AAPL", "MSFT", "BTC-USD"]
    
    for ticker in test_tickers:
        print(f"\n  Testing {ticker}...")
        try:
            data = dm.download_data(ticker, period="1mo")
            if data is not None:
                print(f"  ✓ Successfully downloaded {ticker}")
                print(f"    - Records: {len(data)}")
                print(f"    - Date range: {data.index[0]} to {data.index[-1]}")
                print(f"    - Columns: {list(data.columns)}")
                
                # Verify expected columns
                expected_cols = ['open', 'high', 'low', 'close', 'volume', 'returns']
                missing_cols = [col for col in expected_cols if col not in data.columns]
                if missing_cols:
                    print(f"    ⚠ Missing columns: {missing_cols}")
                else:
                    print(f"    ✓ All expected columns present")
            else:
                print(f"  ✗ No data returned for {ticker}")
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")
    
    # Test 4: Test ticker info
    print("\n4. Testing ticker info retrieval...")
    for ticker in ["AAPL", "GOOGL"]:
        try:
            info = dm.get_ticker_info(ticker)
            if info:
                print(f"  ✓ Got info for {ticker}:")
                print(f"    - Name: {info.get('name', 'N/A')}")
                print(f"    - Exchange: {info.get('exchange', 'N/A')}")
                print(f"    - Type: {info.get('type', 'N/A')}")
            else:
                print(f"  ✗ No info returned for {ticker}")
        except Exception as e:
            print(f"  ✗ Error getting info for {ticker}: {e}")
    
    # Test 5: Test cache functionality
    print("\n5. Testing cache functionality...")
    ticker = "AAPL"
    try:
        # First call (should hit API or use existing cache)
        import time
        start_time = time.time()
        data1 = dm.download_data(ticker, period="5d")
        time1 = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        data2 = dm.download_data(ticker, period="5d")
        time2 = time.time() - start_time
        
        if data1 is not None and data2 is not None:
            print(f"  ✓ Cache test successful")
            print(f"    - First call: {time1:.3f}s")
            print(f"    - Second call: {time2:.3f}s")
            if time2 < time1:
                print(f"    ✓ Cache is faster ({time1/time2:.1f}x speedup)")
            
            # Verify data consistency
            if len(data1) == len(data2):
                print(f"    ✓ Data consistency verified ({len(data1)} records)")
            else:
                print(f"    ⚠ Data inconsistency: {len(data1)} vs {len(data2)} records")
    except Exception as e:
        print(f"  ✗ Cache test failed: {e}")
    
    # Test 6: Test legacy function
    print("\n6. Testing legacy load_financial_data function...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = load_financial_data("AAPL", start_date, end_date)
        if data is not None:
            print(f"  ✓ Legacy function works")
            print(f"    - Records: {len(data)}")
        else:
            print(f"  ✗ Legacy function returned None")
    except Exception as e:
        print(f"  ✗ Legacy function failed: {e}")
    
    # Test 7: Test None handling
    print("\n7. Testing None handling for invalid ticker...")
    try:
        data = dm.download_data("INVALID_TICKER_XYZ", period="1mo")
        if data is None:
            print(f"  ✓ Correctly returned None for invalid ticker")
        else:
            print(f"  ⚠ Unexpected data returned for invalid ticker")
    except Exception as e:
        print(f"  ✗ Error handling invalid ticker: {e}")
    
    print("\n" + "=" * 60)
    print("YFinance Data Layer Testing Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_yfinance_data_layer()
