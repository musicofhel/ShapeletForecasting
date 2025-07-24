# Pattern Search Functionality Summary

## Overview
The Pattern Search module provides comprehensive pattern search capabilities for the financial dashboard, including custom pattern upload, similarity search across tickers, pattern-based backtesting, alerts, and library management.

## Key Features

### 1. Custom Pattern Upload
- Upload custom price patterns for analysis
- Automatic pattern metadata extraction (mean, std, min, max)
- Support for tags and categorization
- Unique pattern ID generation

### 2. Pattern Similarity Search
- Find similar patterns across multiple tickers
- Multiple similarity metrics (correlation, euclidean, cosine, DTW)
- Parallel search for performance
- Configurable similarity thresholds
- Result caching for improved performance

### 3. Pattern-Based Backtesting
- Backtest patterns across historical data
- Configurable entry/exit rules
- Comprehensive performance metrics:
  - Win rate
  - Sharpe ratio
  - Max drawdown
  - Profit factor
  - Total return
- Trade-level analysis

### 4. Pattern Alerts
- Create alerts based on pattern matches
- Configurable alert conditions and thresholds
- Real-time alert checking
- Alert history tracking

### 5. Pattern Library Management
- Persistent pattern storage
- Search patterns by tags, ticker, timeframe
- Pattern statistics and performance tracking
- Import/export functionality (JSON, CSV, pickle)

## Core Components

### Classes

#### `Pattern`
- Represents a financial pattern with metadata
- Stores pattern data, tags, performance metrics
- Serializable to/from dictionary

#### `PatternMatch`
- Represents a pattern match result
- Contains similarity score, location, timestamp
- Links to original pattern and ticker

#### `BacktestResult`
- Comprehensive backtesting results
- Trade-level details
- Aggregate performance metrics

#### `PatternAlert`
- Pattern-based alert configuration
- Tracks alert triggers and history
- Supports multiple alert conditions

#### `PatternLibrary`
- Manages pattern storage and retrieval
- File-based persistence
- Search and filter capabilities

#### `PatternSearchEngine`
- Main engine for pattern operations
- Integrates all functionality
- Manages caching and optimization

### Convenience Functions
- `quick_pattern_upload()` - Quick pattern upload
- `quick_pattern_search()` - Quick pattern search
- `quick_pattern_backtest()` - Quick backtesting

## Usage Examples

### Upload a Pattern
```python
from src.dashboard.search import quick_pattern_upload

# Upload a bull flag pattern
pattern = quick_pattern_upload(
    name="bull_flag",
    data=[100, 102, 104, 106, 108, 107, 106.5, 106, 106.5, 107, 108, 110, 112, 114, 116],
    description="Bull flag continuation pattern",
    tags=['bullish', 'continuation']
)
```

### Search for Similar Patterns
```python
from src.dashboard.search import quick_pattern_search

# Search across multiple tickers
matches = quick_pattern_search(
    pattern_id=pattern.id,
    ticker_data={'AAPL': aapl_df, 'GOOGL': googl_df},
    threshold=0.8
)

# Process matches
for match in matches[:5]:
    print(f"{match.ticker}: {match.timestamp} (similarity: {match.similarity_score:.3f})")
```

### Backtest a Pattern
```python
from src.dashboard.search import quick_pattern_backtest

# Run backtest
result = quick_pattern_backtest(
    pattern_id=pattern.id,
    ticker_data=ticker_data,
    entry_threshold=0.85
)

print(f"Win rate: {result.win_rate:.1%}")
print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
```

### Create Pattern Alerts
```python
from src.dashboard.search import PatternSearchEngine

engine = PatternSearchEngine()

# Create alert
alert = engine.create_alert(
    pattern_id=pattern.id,
    ticker="AAPL",
    condition="match",
    threshold=0.9
)

# Check alerts
triggered = engine.check_alerts(live_data)
```

## File Structure
```
src/dashboard/search/
├── __init__.py           # Module exports
└── pattern_search.py     # Main implementation

data/
├── pattern_library/      # Pattern storage
│   └── patterns.json
└── pattern_cache/        # Search cache
    └── alerts.json
```

## Performance Considerations

### Optimization Features
- Parallel pattern search across tickers
- Result caching to avoid redundant searches
- Normalized pattern comparison
- Efficient sliding window implementation

### Scalability
- ThreadPoolExecutor for concurrent searches
- Configurable worker threads
- Memory-efficient pattern storage
- Lazy loading of pattern data

## Integration Points

### With Existing Modules
- Uses DTWEngine for DTW-based similarity
- Uses WaveletAnalyzer for wavelet features
- Compatible with dashboard visualization components
- Integrates with real-time monitoring

### Data Requirements
- Expects DataFrame with 'close' column minimum
- Supports OHLCV data format
- Handles multiple timeframes

## Configuration Options

### Search Parameters
- `similarity_threshold`: Minimum similarity score (0-1)
- `max_results`: Maximum number of matches to return
- `use_cache`: Enable/disable result caching

### Backtest Parameters
- `entry_threshold`: Minimum similarity for trade entry
- `exit_rules`: Dictionary of exit conditions
  - `take_profit`: Profit target percentage
  - `stop_loss`: Stop loss percentage
  - `max_holding_period`: Maximum bars to hold
- `commission`: Trading commission rate

### Alert Parameters
- `condition`: Alert trigger condition
- `threshold`: Similarity threshold for alerts
- `enabled`: Alert active status

## Future Enhancements

### Planned Features
1. Machine learning-based pattern recognition
2. Multi-timeframe pattern analysis
3. Pattern combination strategies
4. Advanced risk management rules
5. Real-time pattern discovery
6. Pattern morphing analysis
7. Statistical significance testing
8. Pattern clustering and categorization

### Performance Improvements
1. GPU acceleration for similarity calculations
2. Distributed pattern search
3. Advanced caching strategies
4. Incremental pattern updates

## Demo Script
Run `demo_pattern_search.py` to see all features in action:
```bash
python demo_pattern_search.py
```

This generates:
- Pattern backtest distribution chart
- Pattern matches visualization
- Exported pattern JSON file
- Pattern library and cache directories

## Dependencies
- numpy: Numerical computations
- pandas: Data manipulation
- scipy: Statistical functions
- matplotlib: Visualization (for demo)
- pickle: Object serialization
- json: Data persistence
- hashlib: ID generation
- concurrent.futures: Parallel processing

## Error Handling
- Graceful handling of missing patterns
- Validation of input data formats
- Cache corruption recovery
- Alert system fault tolerance

## Best Practices
1. Normalize patterns before uploading
2. Use meaningful pattern names and descriptions
3. Tag patterns appropriately for easy search
4. Set reasonable similarity thresholds
5. Monitor backtest results before live trading
6. Regularly clean up old cache files
7. Export important patterns for backup
