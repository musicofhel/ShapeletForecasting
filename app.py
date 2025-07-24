#!/usr/bin/env python3
"""
Financial Wavelet Forecasting Dashboard - Single Entry Point
Unified application that consolidates all demo functionality into one dashboard
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment(mode='dev'):
    """Setup environment variables based on mode"""
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    if mode == 'demo':
        os.environ['DEMO_MODE'] = 'true'
        os.environ['USE_SAMPLE_DATA'] = 'true'
    elif mode == 'prod':
        os.environ['DASH_DEBUG'] = 'false'
    else:  # dev
        os.environ['DASH_DEBUG'] = 'true'

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import dash
        import dash_bootstrap_components as dbc
        import plotly
        import pandas as pd
        import numpy as np
        import scipy
        import sklearn
        logger.info("All dependencies verified successfully")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
        return False

def create_demo_data():
    """Create sample data for demo mode"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data directory
    data_dir = Path('data/demo')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample BTC data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
    btc_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.5) + np.abs(np.random.randn(1000)),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.5) - np.abs(np.random.randn(1000)),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Ensure high > low
    btc_data['high'] = np.maximum(btc_data['high'], btc_data['low'] + 0.1)
    btc_data['close'] = np.clip(btc_data['close'], btc_data['low'], btc_data['high'])
    
    btc_data.to_csv(data_dir / 'btcusd_1h.csv', index=False)
    logger.info(f"Created demo data: {data_dir}/btcusd_1h.csv")
    
    # Create sample patterns
    patterns = []
    for i in range(20):
        patterns.append({
            'id': f'pattern_{i}',
            'name': f'Pattern {chr(65 + i % 26)}',
            'type': ['bullish', 'bearish', 'neutral'][i % 3],
            'frequency': np.random.randint(5, 50),
            'avg_return': np.random.uniform(-2, 5),
            'confidence': np.random.uniform(0.6, 0.95),
            'last_seen': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat()
        })
    
    import json
    with open(data_dir / 'sample_patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    logger.info(f"Created demo patterns: {data_dir}/sample_patterns.json")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description='Financial Wavelet Forecasting Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Start in development mode
  python app.py --mode demo         # Start with demo data
  python app.py --mode prod --port 8080  # Production mode on port 8080
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['dev', 'demo', 'prod'],
        default='dev',
        help='Run mode: dev (development), demo (with sample data), prod (production)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the application on (default: 8050)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind the application to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment(args.mode)
    
    # Create demo data if in demo mode
    if args.mode == 'demo':
        create_demo_data()
    
    # Import and run the dashboard
    try:
        from src.dashboard.forecast_app import app
        
        logger.info(f"Starting Financial Wavelet Forecasting Dashboard")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        
        if args.mode == 'prod':
            app.run_server(
                debug=False,
                host=args.host,
                port=args.port,
                threaded=True
            )
        else:
            app.run_server(
                debug=args.debug or args.mode == 'dev',
                host=args.host,
                port=args.port,
                dev_tools_hot_reload=True
            )
            
    except ImportError as e:
        logger.error(f"Failed to import dashboard: {e}")
        logger.error("Please ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
