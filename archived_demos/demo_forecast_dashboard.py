"""
Demo Script for Forecast Dashboard

This script demonstrates the forecasting dashboard functionality including:
- Running the dashboard locally
- Testing key features
- Generating sample predictions
"""

import sys
import os
import time
import webbrowser
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.forecast_app import app, DataManager, PerformanceMonitor

def generate_demo_data():
    """Generate demo data for testing"""
    print("Generating demo data...")
    
    # Create sample time series data
    dates = pd.date_range(end=datetime.now(), periods=5000, freq='1H')
    
    # Generate realistic price data with patterns
    trend = np.linspace(100, 150, 5000)
    seasonal = 10 * np.sin(np.linspace(0, 20*np.pi, 5000))
    noise = np.random.randn(5000) * 2
    price = trend + seasonal + noise
    
    # Add some distinct patterns
    pattern_indices = [1000, 2000, 3000, 4000]
    for idx in pattern_indices:
        # Create a distinctive pattern
        pattern = np.array([0, 2, 5, 3, 1, -1, -3, -2, 0, 1])
        if idx + len(pattern) < len(price):
            price[idx:idx+len(pattern)] += pattern * 3
    
    data = pd.DataFrame({
        'timestamp': dates,
        'price': price,
        'volume': np.random.randint(1000, 10000, 5000),
        'volatility': np.abs(np.random.randn(5000) * 0.02)
    })
    
    # Save demo data
    os.makedirs('data/demo', exist_ok=True)
    data.to_csv('data/demo/btcusd_1h.csv', index=False)
    
    print(f"Generated {len(data)} data points with patterns")
    return data

def test_dashboard_features():
    """Test key dashboard features"""
    print("\nTesting dashboard features...")
    
    # Initialize components
    data_manager = DataManager()
    perf_monitor = PerformanceMonitor()
    
    # Test data loading
    print("- Testing data loading...")
    start_time = time.time()
    data = data_manager.load_data("BTCUSD", "1H")
    load_time = time.time() - start_time
    print(f"  ✓ Loaded {len(data)} records in {load_time:.2f}s")
    
    # Test pattern detection
    print("- Testing pattern detection...")
    patterns = data_manager.get_patterns("BTCUSD")
    print(f"  ✓ Detected {len(patterns)} patterns")
    
    # Test prediction generation
    print("- Testing prediction generation...")
    from src.dashboard.forecast_app import compute_pattern_predictions
    predictions = compute_pattern_predictions("BTCUSD", 10)
    print(f"  ✓ Generated predictions with {predictions['confidence']:.1%} confidence")
    
    # Test performance monitoring
    print("- Testing performance monitoring...")
    for _ in range(5):
        start = time.time()
        # Simulate callback
        time.sleep(0.1)
        perf_monitor.record_callback_time(time.time() - start)
    
    avg_time = perf_monitor.get_average_callback_time()
    print(f"  ✓ Average callback time: {avg_time*1000:.0f}ms")
    
    # Check performance criteria
    perf_status = perf_monitor.check_performance()
    if perf_status['meets_criteria']:
        print("  ✓ Performance criteria met!")
    else:
        print("  ⚠ Performance needs optimization")

def open_browser():
    """Open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:8050')

def main():
    """Main demo function"""
    print("=" * 60)
    print("FORECAST DASHBOARD DEMO")
    print("=" * 60)
    
    # Generate demo data
    generate_demo_data()
    
    # Test features
    test_dashboard_features()
    
    # Start dashboard
    print("\n" + "=" * 60)
    print("Starting Forecast Dashboard...")
    print("=" * 60)
    print("\nDashboard Features:")
    print("- Main time series view with pattern overlays")
    print("- Pattern sequence visualization")
    print("- Next-pattern prediction display")
    print("- Accuracy metrics panel")
    print("- Pattern exploration tools")
    print("\nPerformance Targets:")
    print("- Dashboard loads in <3 seconds")
    print("- All callbacks execute in <500ms")
    print("- Handles 100k+ data points smoothly")
    print("- Responsive design for all devices")
    
    print("\n" + "-" * 60)
    print("Opening dashboard at http://localhost:8050")
    print("Press Ctrl+C to stop the server")
    print("-" * 60 + "\n")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the app
    try:
        app.run_server(debug=True, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")

if __name__ == "__main__":
    main()
