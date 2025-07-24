"""
Test script to verify dashboard can start with YFinance data layer
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing dashboard startup with YFinance data layer...")
print("=" * 60)

# Test 1: Import data utils
try:
    from src.dashboard.data_utils import data_manager
    print("✓ Successfully imported data_utils")
    print(f"  Data manager type: {type(data_manager).__name__}")
except Exception as e:
    print(f"✗ Failed to import data_utils: {e}")
    sys.exit(1)

# Test 2: Test data fetching
try:
    print("\nTesting data fetch...")
    data = data_manager.download_data("AAPL", period="5d")
    if data is not None:
        print(f"✓ Successfully fetched data: {len(data)} rows")
    else:
        print("⚠ No data returned (could be rate limited)")
except Exception as e:
    print(f"✗ Error fetching data: {e}")

# Test 3: Import dashboard components
try:
    print("\nTesting dashboard imports...")
    from src.dashboard.forecast_app import app
    print("✓ Successfully imported forecast_app")
    
    # Check if app is configured correctly
    if hasattr(app, 'server'):
        print("✓ Dash app configured correctly")
    else:
        print("⚠ Dash app may not be configured correctly")
        
except Exception as e:
    print(f"✗ Failed to import forecast_app: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Dashboard startup test complete")
print("\nTo run the dashboard, use: python src/dashboard/forecast_app.py")
print("Then open: http://localhost:8050")
