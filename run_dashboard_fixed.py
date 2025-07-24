#!/usr/bin/env python
"""
Run the fixed dashboard application
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the dashboard
from src.dashboard.forecast_app_fixed import app

if __name__ == '__main__':
    print("Starting Wavelet Forecast Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the server")
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050)
