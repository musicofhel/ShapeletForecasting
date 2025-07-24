"""
Dashboard runner script with proper path setup
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now we can import and run the dashboard
try:
    from src.dashboard.forecast_app_fixed import app
    
    print("=" * 60)
    print("Starting Wavelet Forecast Dashboard with Real YFinance Data")
    print("=" * 60)
    print("\nDashboard will be available at: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("\nNOTE: This dashboard now pulls REAL data from YFinance!")
    print("      - Data is cached in SQLite database")
    print("      - Prices shown are actual market prices")
    print("      - Patterns are detected from real price movements")
    print("=" * 60)
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    
except Exception as e:
    print(f"\nError starting dashboard: {e}")
    import traceback
    traceback.print_exc()
