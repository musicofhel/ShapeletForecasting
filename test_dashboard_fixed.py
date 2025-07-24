"""
Test script to run the fixed dashboard and identify any issues
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        # Test Dash imports
        import dash
        from dash import Dash, html, dcc
        import dash_bootstrap_components as dbc
        print("✓ Dash imports successful")
        
        # Test Plotly imports
        import plotly.graph_objects as go
        import plotly.express as px
        print("✓ Plotly imports successful")
        
        # Test data libraries
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("✓ Data library imports successful")
        
        # Test custom modules
        from src.dashboard.layouts.forecast_layout import create_forecast_layout
        print("✓ Layout module import successful")
        
        from src.dashboard.callbacks.prediction_callbacks import register_prediction_callbacks
        print("✓ Callbacks module import successful")
        
        from src.dashboard.pattern_predictor import PatternPredictor
        print("✓ PatternPredictor import successful")
        
        from src.dashboard.pattern_matcher import PatternMatcher
        print("✓ PatternMatcher import successful")
        
        from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
        print("✓ WaveletSequenceAnalyzer import successful")
        
        from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer
        print("✓ PatternSequenceVisualizer import successful")
        
        from src.dashboard.pattern_classifier import PatternClassifier
        print("✓ PatternClassifier import successful")
        
        from src.dashboard.data_utils_yfinance import YFinanceDataManager
        print("✓ YFinanceDataManager import successful")
        
        from src.models.transformer_predictor import TransformerPredictor
        print("✓ TransformerPredictor import successful")
        
        from src.models.xgboost_predictor import XGBoostPredictor
        print("✓ XGBoostPredictor import successful")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False

def test_yfinance_connection():
    """Test YFinance data connection"""
    print("\nTesting YFinance connection...")
    
    try:
        import yfinance as yf
        
        # Test with a simple ticker
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info:
            print(f"✓ YFinance connection successful")
            print(f"  Retrieved data for: {info.get('longName', 'AAPL')}")
            return True
        else:
            print("✗ YFinance returned no data")
            return False
            
    except Exception as e:
        print(f"✗ YFinance connection error: {e}")
        return False

def test_dashboard_initialization():
    """Test dashboard initialization"""
    print("\nTesting dashboard initialization...")
    
    try:
        from src.dashboard.forecast_app_fixed import app, data_manager
        
        # Check if app was created
        if app:
            print("✓ Dashboard app created successfully")
        
        # Check if data manager was initialized
        if data_manager:
            print("✓ Data manager initialized successfully")
            
        # Check if layout was set
        if app.layout:
            print("✓ Dashboard layout set successfully")
            
        return True
        
    except Exception as e:
        print(f"✗ Dashboard initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from src.dashboard.forecast_app_fixed import data_manager
        
        # Test loading data for BTC
        df = data_manager.load_data('BTCUSD', '1 Day')
        
        if df is not None and not df.empty:
            print(f"✓ Data loaded successfully")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print("✗ No data loaded")
            return False
            
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Dashboard Testing Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("YFinance Connection Test", test_yfinance_connection),
        ("Dashboard Initialization Test", test_dashboard_initialization),
        ("Data Loading Test", test_data_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Dashboard is ready to run.")
        print("\nTo start the dashboard, run:")
        print("  python src/dashboard/forecast_app_fixed.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues before running the dashboard.")

if __name__ == "__main__":
    main()
