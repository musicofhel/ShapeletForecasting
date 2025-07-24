"""
Example integration of cache optimization with dashboard components
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .cache_manager import cache_manager


class OptimizedDashboard:
    """Example dashboard with cache optimization integrated"""
    
    def __init__(self):
        self.cache = cache_manager
        
    @cache_manager.cached(ttl=3600)
    def load_market_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Load market data with caching"""
        # Simulate expensive data loading
        print(f"Loading {ticker} data for {period}...")
        
        # In real implementation, this would query a database
        dates = pd.date_range(end='2024-01-01', periods=1000, freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(1000).cumsum() * 0.5,
            'high': 100 + np.random.randn(1000).cumsum() * 0.5 + 1,
            'low': 100 + np.random.randn(1000).cumsum() * 0.5 - 1,
            'close': 100 + np.random.randn(1000).cumsum() * 0.5,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        return data
    
    @cache_manager.cached(ttl=7200, cache_type='pattern')
    def discover_patterns(self, data: pd.DataFrame, pattern_type: str) -> List[Dict[str, Any]]:
        """Discover patterns with caching"""
        print(f"Discovering {pattern_type} patterns...")
        
        # Simulate pattern discovery
        patterns = []
        for i in range(np.random.randint(3, 8)):
            patterns.append({
                'id': f'{pattern_type}_{i}',
                'type': pattern_type,
                'confidence': np.random.uniform(0.7, 0.95),
                'start_idx': np.random.randint(0, len(data) - 100),
                'length': np.random.randint(50, 200)
            })
        
        return patterns
    
    def get_optimized_visualization_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Get visualization data optimized for WebGL"""
        return self.cache.optimize_for_webgl(data, max_points=5000)
    
    def create_lazy_dataset(self, data_source, total_rows: int):
        """Create lazy-loading dataset"""
        return self.cache.lazy_load_dataframe(
            data_source=data_source,
            total_rows=total_rows,
            columns=['timestamp', 'value', 'volume']
        )
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health metrics"""
        stats = self.cache.get_cache_stats()
        
        return {
            'total_memory_mb': stats['total_memory_mb'],
            'cache_efficiency': {
                'memory': stats['memory_cache']['hit_rate'],
                'patterns': stats['pattern_cache']['hit_rate'],
                'visualizations': stats['visualization_cache']['hit_rate']
            },
            'entries': {
                'memory': stats['memory_cache']['size'],
                'patterns': stats['pattern_cache']['size'],
                'visualizations': stats['visualization_cache']['size']
            }
        }


# Integration with existing components
def integrate_with_sidebar():
    """Example: Integrate caching with sidebar component"""
    
    @cache_manager.cached(ttl=300)  # 5 minute cache
    def get_filter_options(data_type: str) -> List[str]:
        """Get cached filter options"""
        # This would normally query available options
        if data_type == 'tickers':
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        elif data_type == 'patterns':
            return ['head_and_shoulders', 'double_top', 'triangle', 'flag']
        elif data_type == 'timeframes':
            return ['1h', '4h', '1d', '1w', '1m']
        return []
    
    return get_filter_options


def integrate_with_pattern_gallery():
    """Example: Integrate caching with pattern gallery"""
    
    @cache_manager.cached(ttl=1800, cache_type='pattern')
    def get_pattern_thumbnails(pattern_ids: List[str]) -> Dict[str, Any]:
        """Get cached pattern thumbnails"""
        thumbnails = {}
        
        for pattern_id in pattern_ids:
            # Simulate thumbnail generation
            thumbnails[pattern_id] = {
                'image_data': np.random.randn(100, 100),
                'metadata': {
                    'confidence': np.random.uniform(0.7, 0.95),
                    'type': 'head_and_shoulders'
                }
            }
        
        return thumbnails
    
    return get_pattern_thumbnails


def integrate_with_realtime_monitor():
    """Example: Integrate progressive loading with realtime monitor"""
    
    async def stream_realtime_data(callback):
        """Stream data with progressive loading"""
        
        def data_source(offset: int, limit: int) -> pd.DataFrame:
            """Generate realtime data chunks"""
            timestamps = pd.date_range(
                start='2024-01-01',
                periods=limit,
                freq='1s'
            ) + pd.Timedelta(seconds=offset)
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'price': 100 + np.random.randn(limit).cumsum() * 0.01,
                'volume': np.random.randint(100, 1000, limit)
            })
        
        # Use progressive loading
        await cache_manager.progressive_load(
            data_source=data_source,
            callback=callback,
            chunk_size=100
        )
    
    return stream_realtime_data


def integrate_with_analytics():
    """Example: Integrate caching with analytics calculations"""
    
    @cache_manager.cached(ttl=3600)
    def calculate_technical_indicators(data: pd.DataFrame, indicators: List[str]) -> Dict[str, pd.Series]:
        """Calculate technical indicators with caching"""
        results = {}
        
        for indicator in indicators:
            if indicator == 'sma_20':
                results[indicator] = data['close'].rolling(20).mean()
            elif indicator == 'rsi':
                # Simplified RSI calculation
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                results[indicator] = 100 - (100 / (1 + rs))
            elif indicator == 'bollinger_bands':
                sma = data['close'].rolling(20).mean()
                std = data['close'].rolling(20).std()
                results[f'{indicator}_upper'] = sma + (std * 2)
                results[f'{indicator}_lower'] = sma - (std * 2)
        
        return results
    
    return calculate_technical_indicators


def integrate_with_export():
    """Example: Integrate caching with export functionality"""
    
    @cache_manager.cached(ttl=600, cache_type='visualization')
    def generate_export_preview(data: pd.DataFrame, format: str) -> Dict[str, Any]:
        """Generate export preview with caching"""
        preview = {
            'format': format,
            'rows': len(data),
            'columns': list(data.columns),
            'file_size_estimate': len(data) * len(data.columns) * 8 / 1024  # KB
        }
        
        if format == 'csv':
            preview['sample'] = data.head(5).to_csv(index=False)
        elif format == 'json':
            preview['sample'] = data.head(5).to_json(orient='records')
        elif format == 'excel':
            preview['sheets'] = ['data', 'metadata', 'charts']
        
        return preview
    
    return generate_export_preview


# Performance monitoring integration
class CachePerformanceMonitor:
    """Monitor cache performance for dashboard optimization"""
    
    def __init__(self):
        self.metrics = []
        
    def record_cache_hit(self, cache_type: str, key: str, response_time: float):
        """Record cache hit metrics"""
        self.metrics.append({
            'timestamp': pd.Timestamp.now(),
            'type': 'hit',
            'cache_type': cache_type,
            'key': key,
            'response_time': response_time
        })
    
    def record_cache_miss(self, cache_type: str, key: str, computation_time: float):
        """Record cache miss metrics"""
        self.metrics.append({
            'timestamp': pd.Timestamp.now(),
            'type': 'miss',
            'cache_type': cache_type,
            'key': key,
            'computation_time': computation_time
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get cache performance summary"""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        
        return {
            'total_requests': len(df),
            'hit_rate': len(df[df['type'] == 'hit']) / len(df),
            'avg_hit_time': df[df['type'] == 'hit']['response_time'].mean(),
            'avg_miss_time': df[df['type'] == 'miss']['computation_time'].mean(),
            'cache_efficiency': {
                cache_type: len(df[(df['cache_type'] == cache_type) & (df['type'] == 'hit')]) / 
                           len(df[df['cache_type'] == cache_type])
                for cache_type in df['cache_type'].unique()
            }
        }


# Example usage
if __name__ == "__main__":
    # Create optimized dashboard
    dashboard = OptimizedDashboard()
    
    # Load data with caching
    data = dashboard.load_market_data('AAPL', '1d')
    print(f"Loaded {len(data)} rows of market data")
    
    # Discover patterns with caching
    patterns = dashboard.discover_patterns(data, 'head_and_shoulders')
    print(f"Found {len(patterns)} patterns")
    
    # Get cache health
    health = dashboard.get_cache_health()
    print(f"Cache health: {health}")
    
    # Test integrations
    filter_options = integrate_with_sidebar()
    print(f"Available tickers: {filter_options('tickers')}")
    
    # Performance monitoring
    monitor = CachePerformanceMonitor()
    monitor.record_cache_hit('memory', 'test_key', 0.001)
    monitor.record_cache_miss('pattern', 'test_key2', 2.5)
    print(f"Performance summary: {monitor.get_performance_summary()}")
