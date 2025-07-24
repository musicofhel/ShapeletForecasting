"""
Simple test script for pattern search functionality
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test imports
try:
    from src.dashboard.search import (
        PatternSearchEngine,
        quick_pattern_upload,
        quick_pattern_search,
        quick_pattern_backtest
    )
    print("✓ Successfully imported pattern search modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test pattern upload
try:
    # Create a simple pattern
    test_pattern = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110])
    
    pattern = quick_pattern_upload(
        name="test_pattern",
        data=test_pattern,
        description="Test pattern for verification",
        tags=['test']
    )
    
    print(f"✓ Pattern uploaded successfully")
    print(f"  - ID: {pattern.id}")
    print(f"  - Name: {pattern.name}")
    print(f"  - Length: {len(pattern.data)}")
except Exception as e:
    print(f"✗ Pattern upload error: {e}")
    exit(1)

# Test pattern search
try:
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    test_data = {
        'TEST': pd.DataFrame({
            'close': 100 + np.random.randn(50).cumsum(),
            'open': 100 + np.random.randn(50).cumsum(),
            'high': 102 + np.random.randn(50).cumsum(),
            'low': 98 + np.random.randn(50).cumsum(),
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
    }
    
    matches = quick_pattern_search(
        pattern_id=pattern.id,
        ticker_data=test_data,
        threshold=0.5
    )
    
    print(f"✓ Pattern search completed")
    print(f"  - Found {len(matches)} matches")
except Exception as e:
    print(f"✗ Pattern search error: {e}")
    exit(1)

# Test pattern backtest
try:
    result = quick_pattern_backtest(
        pattern_id=pattern.id,
        ticker_data=test_data,
        entry_threshold=0.5
    )
    
    print(f"✓ Pattern backtest completed")
    print(f"  - Total trades: {result.total_trades}")
    print(f"  - Win rate: {result.win_rate:.1%}")
except Exception as e:
    print(f"✗ Pattern backtest error: {e}")
    exit(1)

# Test pattern library
try:
    engine = PatternSearchEngine()
    
    # Check if pattern was saved
    saved_pattern = engine.library.get_pattern(pattern.id)
    if saved_pattern:
        print(f"✓ Pattern library working correctly")
        print(f"  - Patterns in library: {len(engine.library.patterns)}")
    else:
        print("✗ Pattern not found in library")
except Exception as e:
    print(f"✗ Pattern library error: {e}")
    exit(1)

# Test alerts
try:
    alert = engine.create_alert(
        pattern_id=pattern.id,
        ticker="TEST",
        condition="match",
        threshold=0.7
    )
    
    print(f"✓ Alert created successfully")
    print(f"  - Alert ID: {alert.id[:20]}...")
    print(f"  - Total alerts: {len(engine.alerts)}")
except Exception as e:
    print(f"✗ Alert creation error: {e}")
    exit(1)

print("\n✓ All tests passed! Pattern search functionality is working correctly.")
