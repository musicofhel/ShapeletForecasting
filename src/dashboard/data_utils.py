"""
Data utilities - now using YFinance instead of Polygon API
This module provides backward compatibility while using YFinance
"""

# Import everything from the YFinance implementation
from .data_utils_yfinance import *

# The data_utils_yfinance module already provides:
# - YFinanceDataManager class
# - data_manager instance
# - load_financial_data function
# - All the functionality needed for the dashboard

# Additional utility functions for compatibility
def prepare_data_for_analysis(df):
    """Prepare data for wavelet analysis"""
    if df is None or df.empty:
        return df
        
    # Ensure we have required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Missing column: {col}")
            
    # Add technical indicators
    if 'close' in df.columns:
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
    return df


def validate_data(df):
    """Validate that data is suitable for analysis"""
    if df is None or df.empty:
        return False
        
    # Check minimum length
    if len(df) < 20:
        logger.warning("Insufficient data points (minimum 20 required)")
        return False
        
    # Check for required columns
    required = ['close', 'volume']
    if not all(col in df.columns for col in required):
        logger.warning(f"Missing required columns. Found: {df.columns.tolist()}")
        return False
        
    # Check for NaN values in critical columns
    if df['close'].isna().any():
        logger.warning("NaN values found in close prices")
        return False
        
    return True


# For backward compatibility with the old DataManager class
DataManager = YFinanceDataManager
