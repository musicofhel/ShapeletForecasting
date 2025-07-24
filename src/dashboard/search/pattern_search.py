"""
Pattern Search Functionality for Financial Dashboard

This module provides comprehensive pattern search capabilities including:
- Custom pattern upload and management
- Pattern similarity search across tickers
- Pattern-based backtesting
- Custom pattern alerts
- Pattern library management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import pickle
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from collections import defaultdict

# Import DTW and wavelet analysis modules
try:
    from src.dtw.dtw_engine import DTWEngine
    from src.wavelet_analysis.wavelet_analyzer import WaveletAnalyzer
except ImportError:
    # Fallback for testing
    DTWEngine = None
    WaveletAnalyzer = None

warnings.filterwarnings('ignore')


@dataclass
class Pattern:
    """Represents a financial pattern"""
    id: str
    name: str
    description: str
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    ticker: Optional[str] = None
    timeframe: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'data': self.data.tolist(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'ticker': self.ticker,
            'timeframe': self.timeframe,
            'performance_metrics': self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Create pattern from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            data=np.array(data['data']),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            tags=data.get('tags', []),
            ticker=data.get('ticker'),
            timeframe=data.get('timeframe'),
            performance_metrics=data.get('performance_metrics', {})
        )


@dataclass
class PatternMatch:
    """Represents a pattern match result"""
    pattern_id: str
    ticker: str
    timestamp: datetime
    similarity_score: float
    location: Tuple[int, int]  # start, end indices
    price_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Represents backtesting results for a pattern"""
    pattern_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PatternAlert:
    """Represents a pattern-based alert"""
    id: str
    pattern_id: str
    ticker: str
    condition: str  # 'match', 'similarity_threshold', 'performance_threshold'
    threshold: float
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternLibrary:
    """Manages pattern library storage and retrieval"""
    
    def __init__(self, library_path: str = "data/pattern_library"):
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, Pattern] = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Load patterns from disk"""
        pattern_file = self.library_path / "patterns.json"
        if pattern_file.exists():
            with open(pattern_file, 'r') as f:
                data = json.load(f)
                for pattern_data in data:
                    pattern = Pattern.from_dict(pattern_data)
                    self.patterns[pattern.id] = pattern
    
    def save_patterns(self):
        """Save patterns to disk"""
        pattern_file = self.library_path / "patterns.json"
        data = [pattern.to_dict() for pattern in self.patterns.values()]
        with open(pattern_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_pattern(self, pattern: Pattern):
        """Add pattern to library"""
        self.patterns[pattern.id] = pattern
        self.save_patterns()
    
    def remove_pattern(self, pattern_id: str):
        """Remove pattern from library"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            self.save_patterns()
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def search_patterns(self, 
                       tags: Optional[List[str]] = None,
                       ticker: Optional[str] = None,
                       timeframe: Optional[str] = None,
                       min_performance: Optional[float] = None) -> List[Pattern]:
        """Search patterns by criteria"""
        results = []
        
        for pattern in self.patterns.values():
            # Check tags
            if tags and not any(tag in pattern.tags for tag in tags):
                continue
            
            # Check ticker
            if ticker and pattern.ticker != ticker:
                continue
            
            # Check timeframe
            if timeframe and pattern.timeframe != timeframe:
                continue
            
            # Check performance
            if min_performance:
                avg_return = pattern.performance_metrics.get('avg_return', 0)
                if avg_return < min_performance:
                    continue
            
            results.append(pattern)
        
        return results


class PatternSearchEngine:
    """Main pattern search engine"""
    
    def __init__(self, 
                 library_path: str = "data/pattern_library",
                 cache_dir: str = "data/pattern_cache"):
        self.library = PatternLibrary(library_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtw_engine = DTWEngine() if DTWEngine else None
        self.wavelet_analyzer = WaveletAnalyzer() if WaveletAnalyzer else None
        self.alerts: Dict[str, PatternAlert] = {}
        self.load_alerts()
    
    def upload_pattern(self, 
                      name: str,
                      description: str,
                      data: Union[np.ndarray, pd.Series],
                      tags: Optional[List[str]] = None,
                      ticker: Optional[str] = None,
                      timeframe: Optional[str] = None) -> Pattern:
        """Upload a custom pattern"""
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            data = data.values
        
        # Generate unique ID
        pattern_id = self._generate_pattern_id(name, data)
        
        # Create pattern
        pattern = Pattern(
            id=pattern_id,
            name=name,
            description=description,
            data=data,
            tags=tags or [],
            ticker=ticker,
            timeframe=timeframe
        )
        
        # Add metadata
        pattern.metadata['length'] = len(data)
        pattern.metadata['mean'] = float(np.mean(data))
        pattern.metadata['std'] = float(np.std(data))
        pattern.metadata['min'] = float(np.min(data))
        pattern.metadata['max'] = float(np.max(data))
        
        # Add to library
        self.library.add_pattern(pattern)
        
        return pattern
    
    def find_similar_patterns(self,
                            pattern_id: str,
                            data: Dict[str, pd.DataFrame],
                            similarity_threshold: float = 0.8,
                            max_results: int = 50,
                            use_cache: bool = True) -> List[PatternMatch]:
        """Find similar patterns across multiple tickers"""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        # Check cache
        cache_key = self._get_cache_key(pattern_id, list(data.keys()))
        if use_cache:
            cached_results = self._load_from_cache(cache_key)
            if cached_results:
                return cached_results
        
        results = []
        
        # Search across all tickers in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for ticker, df in data.items():
                if 'close' in df.columns:
                    price_data = df['close'].values
                    future = executor.submit(
                        self._search_pattern_in_series,
                        pattern,
                        price_data,
                        ticker,
                        df.index,
                        similarity_threshold
                    )
                    futures[future] = ticker
            
            # Collect results
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    matches = future.result()
                    results.extend(matches)
                except Exception as e:
                    print(f"Error searching {ticker}: {e}")
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        results = results[:max_results]
        
        # Cache results
        if use_cache:
            self._save_to_cache(cache_key, results)
        
        return results
    
    def _search_pattern_in_series(self,
                                pattern: Pattern,
                                price_data: np.ndarray,
                                ticker: str,
                                timestamps: pd.DatetimeIndex,
                                similarity_threshold: float) -> List[PatternMatch]:
        """Search for pattern in a single price series"""
        matches = []
        pattern_len = len(pattern.data)
        
        # Normalize pattern
        pattern_norm = self._normalize_data(pattern.data)
        
        # Sliding window search
        for i in range(len(price_data) - pattern_len + 1):
            window = price_data[i:i + pattern_len]
            window_norm = self._normalize_data(window)
            
            # Calculate similarity
            similarity = self._calculate_similarity(pattern_norm, window_norm)
            
            if similarity >= similarity_threshold:
                match = PatternMatch(
                    pattern_id=pattern.id,
                    ticker=ticker,
                    timestamp=timestamps[i],
                    similarity_score=similarity,
                    location=(i, i + pattern_len),
                    price_data=window,
                    metadata={
                        'pattern_name': pattern.name,
                        'end_timestamp': timestamps[i + pattern_len - 1]
                    }
                )
                matches.append(match)
        
        return matches
    
    def backtest_pattern(self,
                        pattern_id: str,
                        data: Dict[str, pd.DataFrame],
                        entry_threshold: float = 0.85,
                        exit_rules: Dict[str, Any] = None,
                        position_size: float = 1.0,
                        commission: float = 0.001) -> BacktestResult:
        """Backtest a pattern across multiple tickers"""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        # Default exit rules
        if exit_rules is None:
            exit_rules = {
                'take_profit': 0.02,  # 2% profit
                'stop_loss': 0.01,    # 1% loss
                'max_holding_period': 10  # bars
            }
        
        trades = []
        
        # Find all pattern matches
        matches = self.find_similar_patterns(
            pattern_id, 
            data, 
            similarity_threshold=entry_threshold,
            max_results=1000
        )
        
        # Simulate trades for each match
        for match in matches:
            ticker_data = data[match.ticker]
            
            # Get entry point
            entry_idx = match.location[1]
            if entry_idx >= len(ticker_data):
                continue
            
            entry_price = ticker_data.iloc[entry_idx]['close']
            entry_time = ticker_data.index[entry_idx]
            
            # Find exit point
            exit_idx, exit_price, exit_reason = self._find_exit_point(
                ticker_data,
                entry_idx,
                entry_price,
                exit_rules
            )
            
            if exit_idx is None:
                continue
            
            # Calculate trade metrics
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * commission
            
            trade = {
                'ticker': match.ticker,
                'entry_time': entry_time,
                'exit_time': ticker_data.index[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_return,
                'net_return': net_return,
                'exit_reason': exit_reason,
                'similarity_score': match.similarity_score
            }
            trades.append(trade)
        
        # Calculate backtest metrics
        result = self._calculate_backtest_metrics(trades, pattern_id)
        
        # Update pattern performance metrics
        pattern.performance_metrics = {
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'avg_return': result.total_return / max(result.total_trades, 1),
            'sharpe_ratio': result.sharpe_ratio,
            'profit_factor': result.profit_factor
        }
        self.library.save_patterns()
        
        return result
    
    def create_alert(self,
                    pattern_id: str,
                    ticker: str,
                    condition: str = 'match',
                    threshold: float = 0.9) -> PatternAlert:
        """Create a pattern-based alert"""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        alert_id = f"{pattern_id}_{ticker}_{condition}_{int(datetime.now().timestamp())}"
        
        alert = PatternAlert(
            id=alert_id,
            pattern_id=pattern_id,
            ticker=ticker,
            condition=condition,
            threshold=threshold,
            metadata={
                'pattern_name': pattern.name,
                'pattern_description': pattern.description
            }
        )
        
        self.alerts[alert_id] = alert
        self.save_alerts()
        
        return alert
    
    def check_alerts(self, 
                    data: Dict[str, pd.DataFrame]) -> List[Tuple[PatternAlert, PatternMatch]]:
        """Check all active alerts"""
        triggered_alerts = []
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            # Get latest data for ticker
            if alert.ticker not in data:
                continue
            
            ticker_data = data[alert.ticker]
            
            # Check for pattern match
            if alert.condition == 'match':
                matches = self._search_pattern_in_series(
                    self.library.get_pattern(alert.pattern_id),
                    ticker_data['close'].values,
                    alert.ticker,
                    ticker_data.index,
                    alert.threshold
                )
                
                # Check if any recent matches
                for match in matches:
                    if match.timestamp >= datetime.now() - timedelta(hours=1):
                        alert.last_triggered = datetime.now()
                        alert.trigger_count += 1
                        triggered_alerts.append((alert, match))
        
        self.save_alerts()
        return triggered_alerts
    
    def get_pattern_statistics(self, pattern_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a pattern"""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        stats = {
            'pattern_info': {
                'id': pattern.id,
                'name': pattern.name,
                'description': pattern.description,
                'created_at': pattern.created_at.isoformat(),
                'tags': pattern.tags,
                'length': len(pattern.data)
            },
            'data_statistics': pattern.metadata,
            'performance_metrics': pattern.performance_metrics,
            'usage_statistics': {
                'total_searches': 0,  # Would need to track this
                'total_matches': 0,   # Would need to track this
                'active_alerts': sum(1 for a in self.alerts.values() 
                                   if a.pattern_id == pattern_id and a.enabled)
            }
        }
        
        # Add wavelet features if available
        if self.wavelet_analyzer:
            try:
                wavelet_features = self._extract_wavelet_features(pattern.data)
                stats['wavelet_features'] = wavelet_features
            except:
                pass
        
        return stats
    
    def export_pattern(self, pattern_id: str, format: str = 'json') -> Union[str, bytes]:
        """Export a pattern in various formats"""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        if format == 'json':
            return json.dumps(pattern.to_dict(), indent=2)
        elif format == 'csv':
            df = pd.DataFrame({
                'index': range(len(pattern.data)),
                'value': pattern.data
            })
            return df.to_csv(index=False)
        elif format == 'pickle':
            return pickle.dumps(pattern)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_pattern(self, data: Union[str, bytes], format: str = 'json') -> Pattern:
        """Import a pattern from various formats"""
        if format == 'json':
            pattern_dict = json.loads(data)
            pattern = Pattern.from_dict(pattern_dict)
        elif format == 'pickle':
            pattern = pickle.loads(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.library.add_pattern(pattern)
        return pattern
    
    # Helper methods
    
    def _generate_pattern_id(self, name: str, data: np.ndarray) -> str:
        """Generate unique pattern ID"""
        content = f"{name}_{data.tobytes()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 0:
            return (data - min_val) / (max_val - min_val)
        return data - min_val
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        # Use multiple similarity metrics
        
        # 1. Correlation
        corr, _ = pearsonr(pattern1, pattern2)
        
        # 2. Euclidean distance (inverted)
        euclidean_dist = euclidean(pattern1, pattern2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. Cosine similarity
        cosine_sim = 1 - cosine(pattern1, pattern2)
        
        # 4. DTW similarity if available
        if self.dtw_engine:
            try:
                dtw_dist = self.dtw_engine.compute_distance(pattern1, pattern2)
                dtw_sim = 1 / (1 + dtw_dist)
            except:
                dtw_sim = 0
        else:
            dtw_sim = 0
        
        # Weighted average
        weights = [0.3, 0.2, 0.3, 0.2]
        similarities = [corr, euclidean_sim, cosine_sim, dtw_sim]
        
        return sum(w * s for w, s in zip(weights, similarities))
    
    def _find_exit_point(self,
                        data: pd.DataFrame,
                        entry_idx: int,
                        entry_price: float,
                        exit_rules: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """Find exit point based on rules"""
        take_profit = exit_rules.get('take_profit', 0.02)
        stop_loss = exit_rules.get('stop_loss', 0.01)
        max_holding = exit_rules.get('max_holding_period', 10)
        
        for i in range(entry_idx + 1, min(entry_idx + max_holding + 1, len(data))):
            current_price = data.iloc[i]['close']
            return_pct = (current_price - entry_price) / entry_price
            
            # Check take profit
            if return_pct >= take_profit:
                return i, current_price, 'take_profit'
            
            # Check stop loss
            if return_pct <= -stop_loss:
                return i, current_price, 'stop_loss'
        
        # Max holding period reached
        if entry_idx + max_holding < len(data):
            return entry_idx + max_holding, data.iloc[entry_idx + max_holding]['close'], 'max_holding'
        
        return None, None, None
    
    def _calculate_backtest_metrics(self, trades: List[Dict[str, Any]], pattern_id: str) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        if not trades:
            return BacktestResult(
                pattern_id=pattern_id,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                trades=[]
            )
        
        returns = [t['net_return'] for t in trades]
        winning_trades = [t for t in trades if t['net_return'] > 0]
        losing_trades = [t for t in trades if t['net_return'] <= 0]
        
        # Calculate metrics
        total_return = sum(returns)
        win_rate = len(winning_trades) / len(trades)
        
        avg_win = np.mean([t['net_return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_return'] for t in losing_trades]) if losing_trades else 0
        
        # Sharpe ratio (assuming daily returns)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        gross_profit = sum(t['net_return'] for t in winning_trades)
        gross_loss = abs(sum(t['net_return'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            pattern_id=pattern_id,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades
        )
    
    def _extract_wavelet_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract wavelet features from pattern"""
        if not self.wavelet_analyzer:
            return {}
        
        # Perform wavelet decomposition
        coeffs = self.wavelet_analyzer.decompose(data)
        
        # Extract features
        features = {
            'dominant_scale': self._find_dominant_scale(coeffs),
            'energy_distribution': self._calculate_energy_distribution(coeffs),
            'wavelet_entropy': self._calculate_wavelet_entropy(coeffs)
        }
        
        return features
    
    def _find_dominant_scale(self, coeffs: List[np.ndarray]) -> int:
        """Find dominant wavelet scale"""
        energies = [np.sum(c**2) for c in coeffs[1:]]  # Skip approximation
        return np.argmax(energies) + 1
    
    def _calculate_energy_distribution(self, coeffs: List[np.ndarray]) -> List[float]:
        """Calculate energy distribution across scales"""
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies)
        return [e / total_energy for e in energies] if total_energy > 0 else energies
    
    def _calculate_wavelet_entropy(self, coeffs: List[np.ndarray]) -> float:
        """Calculate wavelet entropy"""
        energy_dist = self._calculate_energy_distribution(coeffs)
        entropy = -sum(p * np.log2(p) for p in energy_dist if p > 0)
        return entropy
    
    def _get_cache_key(self, pattern_id: str, tickers: List[str]) -> str:
        """Generate cache key"""
        content = f"{pattern_id}_{'_'.join(sorted(tickers))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[PatternMatch]]:
        """Load results from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, results: List[PatternMatch]):
        """Save results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except:
            pass
    
    def load_alerts(self):
        """Load alerts from disk"""
        alerts_file = self.cache_dir / "alerts.json"
        if alerts_file.exists():
            try:
                with open(alerts_file, 'r') as f:
                    data = json.load(f)
                    for alert_data in data:
                        alert = PatternAlert(
                            id=alert_data['id'],
                            pattern_id=alert_data['pattern_id'],
                            ticker=alert_data['ticker'],
                            condition=alert_data['condition'],
                            threshold=alert_data['threshold'],
                            enabled=alert_data.get('enabled', True),
                            created_at=datetime.fromisoformat(alert_data['created_at']),
                            last_triggered=datetime.fromisoformat(alert_data['last_triggered']) 
                                         if alert_data.get('last_triggered') else None,
                            trigger_count=alert_data.get('trigger_count', 0),
                            metadata=alert_data.get('metadata', {})
                        )
                        self.alerts[alert.id] = alert
            except:
                pass
    
    def save_alerts(self):
        """Save alerts to disk"""
        alerts_file = self.cache_dir / "alerts.json"
        data = []
        for alert in self.alerts.values():
            alert_data = {
                'id': alert.id,
                'pattern_id': alert.pattern_id,
                'ticker': alert.ticker,
                'condition': alert.condition,
                'threshold': alert.threshold,
                'enabled': alert.enabled,
                'created_at': alert.created_at.isoformat(),
                'last_triggered': alert.last_triggered.isoformat() if alert.last_triggered else None,
                'trigger_count': alert.trigger_count,
                'metadata': alert.metadata
            }
            data.append(alert_data)
        
        with open(alerts_file, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions for quick pattern search operations

def quick_pattern_upload(name: str, 
                        data: Union[np.ndarray, pd.Series, List[float]],
                        description: str = "",
                        tags: Optional[List[str]] = None) -> Pattern:
    """Quick function to upload a pattern"""
    engine = PatternSearchEngine()
    
    # Convert data to numpy array
    if isinstance(data, list):
        data = np.array(data)
    
    return engine.upload_pattern(
        name=name,
        description=description or f"Pattern: {name}",
        data=data,
        tags=tags
    )


def quick_pattern_search(pattern_id: str,
                        ticker_data: Dict[str, pd.DataFrame],
                        threshold: float = 0.8) -> List[PatternMatch]:
    """Quick function to search for a pattern"""
    engine = PatternSearchEngine()
    return engine.find_similar_patterns(
        pattern_id=pattern_id,
        data=ticker_data,
        similarity_threshold=threshold
    )


def quick_pattern_backtest(pattern_id: str,
                          ticker_data: Dict[str, pd.DataFrame],
                          entry_threshold: float = 0.85) -> BacktestResult:
    """Quick function to backtest a pattern"""
    engine = PatternSearchEngine()
    return engine.backtest_pattern(
        pattern_id=pattern_id,
        data=ticker_data,
        entry_threshold=entry_threshold
    )
