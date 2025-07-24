"""
Real-time Pattern Detection Monitor

This module provides real-time pattern detection capabilities that:
1. Monitors live price data streams
2. Detects patterns as they form in real-time
3. Generates alerts for significant patterns
4. Updates predictions dynamically
5. Shows pattern formation progress
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
from collections import deque
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternAlert:
    """Alert for detected pattern"""
    timestamp: datetime
    pattern_type: str
    confidence: float
    ticker: str
    timeframe: str
    price_level: float
    expected_move: float
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternProgress:
    """Track pattern formation progress"""
    pattern_id: str
    pattern_type: str
    start_time: datetime
    completion: float  # 0-100%
    confidence: float
    expected_duration: timedelta
    key_levels: List[float]
    current_stage: str


class RealTimePatternMonitor:
    """
    Real-time pattern detection and monitoring system
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 update_interval: float = 1.0,
                 min_confidence: float = 0.7,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize real-time pattern monitor
        
        Args:
            window_size: Size of rolling window for analysis
            update_interval: Update frequency in seconds
            min_confidence: Minimum confidence for pattern detection
            alert_callback: Callback function for alerts
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.min_confidence = min_confidence
        self.alert_callback = alert_callback
        
        # Data storage
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.pattern_history: Dict[str, List[Dict]] = {}
        self.active_patterns: Dict[str, PatternProgress] = {}
        
        # Pattern detection parameters
        self.wavelet = 'db4'
        self.scales = np.arange(1, 32)
        self.scaler = StandardScaler()
        
        # Threading components
        self.data_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        
        # Pattern templates
        self._initialize_pattern_templates()
        
    def _initialize_pattern_templates(self):
        """Initialize pattern detection templates"""
        self.pattern_templates = {
            'breakout': {
                'min_duration': 10,
                'key_features': ['resistance_break', 'volume_surge'],
                'risk_level': 'medium'
            },
            'reversal': {
                'min_duration': 15,
                'key_features': ['trend_exhaustion', 'divergence'],
                'risk_level': 'high'
            },
            'continuation': {
                'min_duration': 8,
                'key_features': ['flag_formation', 'volume_decline'],
                'risk_level': 'low'
            },
            'triangle': {
                'min_duration': 20,
                'key_features': ['converging_lines', 'decreasing_volatility'],
                'risk_level': 'medium'
            },
            'head_shoulders': {
                'min_duration': 25,
                'key_features': ['three_peaks', 'neckline'],
                'risk_level': 'high'
            }
        }
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.start()
            print("Real-time pattern monitoring started")
            
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Real-time pattern monitoring stopped")
        
    def add_price_data(self, ticker: str, price: float, volume: float, 
                      timestamp: Optional[datetime] = None):
        """
        Add new price data point
        
        Args:
            ticker: Stock ticker symbol
            price: Current price
            volume: Current volume
            timestamp: Data timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        self.data_queue.put({
            'ticker': ticker,
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Process incoming data
                self._process_data_queue()
                
                # Detect patterns for each ticker
                for ticker in self.price_buffers:
                    if len(self.price_buffers[ticker]) >= self.window_size // 2:
                        self._detect_patterns(ticker)
                        self._update_active_patterns(ticker)
                        
                # Process alerts
                self._process_alerts()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                
    def _process_data_queue(self):
        """Process incoming data from queue"""
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                ticker = data['ticker']
                
                # Initialize buffers if needed
                if ticker not in self.price_buffers:
                    self.price_buffers[ticker] = deque(maxlen=self.window_size)
                    self.volume_buffers[ticker] = deque(maxlen=self.window_size)
                    self.pattern_history[ticker] = []
                    
                # Add data to buffers
                self.price_buffers[ticker].append(data['price'])
                self.volume_buffers[ticker].append(data['volume'])
                
            except queue.Empty:
                break
                
    def _detect_patterns(self, ticker: str):
        """Detect patterns in price data"""
        prices = np.array(self.price_buffers[ticker])
        volumes = np.array(self.volume_buffers[ticker])
        
        if len(prices) < 20:
            return
            
        # Perform wavelet analysis
        coeffs = self._wavelet_transform(prices)
        
        # Extract features
        features = self._extract_features(prices, volumes, coeffs)
        
        # Check each pattern type
        for pattern_type, template in self.pattern_templates.items():
            confidence = self._check_pattern(pattern_type, features, prices)
            
            if confidence >= self.min_confidence:
                # Check if pattern is already being tracked
                pattern_id = f"{ticker}_{pattern_type}_{datetime.now().timestamp()}"
                
                if not self._is_duplicate_pattern(ticker, pattern_type):
                    # Create new pattern progress
                    progress = PatternProgress(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        start_time=datetime.now(),
                        completion=self._calculate_completion(pattern_type, features),
                        confidence=confidence,
                        expected_duration=timedelta(minutes=template['min_duration']),
                        key_levels=self._identify_key_levels(prices),
                        current_stage=self._identify_stage(pattern_type, features)
                    )
                    
                    self.active_patterns[pattern_id] = progress
                    
                    # Generate alert if pattern is significant
                    if confidence >= 0.8:
                        self._generate_alert(ticker, pattern_type, confidence, 
                                           prices[-1], template['risk_level'])
                        
    def _wavelet_transform(self, prices: np.ndarray) -> np.ndarray:
        """Perform wavelet transform on price data"""
        # Continuous wavelet transform
        coeffs, _ = pywt.cwt(prices, self.scales, self.wavelet)
        return coeffs
        
    def _extract_features(self, prices: np.ndarray, volumes: np.ndarray, 
                         coeffs: np.ndarray) -> Dict[str, float]:
        """Extract features for pattern detection"""
        features = {}
        
        # Price features
        features['price_change'] = (prices[-1] - prices[0]) / prices[0]
        features['volatility'] = np.std(prices) / np.mean(prices)
        features['trend_strength'] = self._calculate_trend_strength(prices)
        
        # Volume features
        features['volume_ratio'] = volumes[-1] / np.mean(volumes)
        features['volume_trend'] = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        # Wavelet features
        features['dominant_scale'] = self.scales[np.argmax(np.mean(np.abs(coeffs), axis=1))]
        features['energy_concentration'] = np.max(np.abs(coeffs)) / np.sum(np.abs(coeffs))
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(prices)
        features['macd_signal'] = self._calculate_macd_signal(prices)
        
        # Support/Resistance
        features['near_resistance'] = self._check_near_resistance(prices)
        features['near_support'] = self._check_near_support(prices)
        
        return features
        
    def _check_pattern(self, pattern_type: str, features: Dict[str, float], 
                      prices: np.ndarray) -> float:
        """Check if features match pattern template"""
        confidence = 0.0
        
        if pattern_type == 'breakout':
            confidence = self._check_breakout_pattern(features, prices)
        elif pattern_type == 'reversal':
            confidence = self._check_reversal_pattern(features, prices)
        elif pattern_type == 'continuation':
            confidence = self._check_continuation_pattern(features, prices)
        elif pattern_type == 'triangle':
            confidence = self._check_triangle_pattern(features, prices)
        elif pattern_type == 'head_shoulders':
            confidence = self._check_head_shoulders_pattern(features, prices)
            
        return confidence
        
    def _check_breakout_pattern(self, features: Dict[str, float], 
                               prices: np.ndarray) -> float:
        """Check for breakout pattern"""
        confidence = 0.0
        
        # Check for resistance break
        if features['near_resistance'] and features['price_change'] > 0.02:
            confidence += 0.3
            
        # Check for volume surge
        if features['volume_ratio'] > 1.5:
            confidence += 0.3
            
        # Check for trend strength
        if features['trend_strength'] > 0.7:
            confidence += 0.2
            
        # Check for momentum
        if features['rsi'] > 60 and features['rsi'] < 80:
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _check_reversal_pattern(self, features: Dict[str, float], 
                               prices: np.ndarray) -> float:
        """Check for reversal pattern"""
        confidence = 0.0
        
        # Check for trend exhaustion
        if abs(features['price_change']) > 0.05 and features['volatility'] > 0.02:
            confidence += 0.25
            
        # Check for divergence
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        if (price_trend > 0 and features['rsi'] > 70) or \
           (price_trend < 0 and features['rsi'] < 30):
            confidence += 0.35
            
        # Check for volume pattern
        if features['volume_trend'] * price_trend < 0:  # Opposite trends
            confidence += 0.2
            
        # Check for support/resistance
        if features['near_support'] or features['near_resistance']:
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _check_continuation_pattern(self, features: Dict[str, float], 
                                   prices: np.ndarray) -> float:
        """Check for continuation pattern"""
        confidence = 0.0
        
        # Check for flag formation
        if features['volatility'] < 0.015 and abs(features['price_change']) < 0.02:
            confidence += 0.4
            
        # Check for volume decline
        if features['volume_ratio'] < 0.8:
            confidence += 0.3
            
        # Check for trend persistence
        if features['trend_strength'] > 0.6:
            confidence += 0.3
            
        return min(confidence, 1.0)
        
    def _check_triangle_pattern(self, features: Dict[str, float], 
                               prices: np.ndarray) -> float:
        """Check for triangle pattern"""
        confidence = 0.0
        
        # Check for converging price action
        highs = self._find_local_extrema(prices, 'high')
        lows = self._find_local_extrema(prices, 'low')
        
        if len(highs) >= 2 and len(lows) >= 2:
            high_slope = np.polyfit(highs[:, 0], highs[:, 1], 1)[0]
            low_slope = np.polyfit(lows[:, 0], lows[:, 1], 1)[0]
            
            # Check for convergence
            if high_slope < 0 and low_slope > 0:
                confidence += 0.5
                
        # Check for decreasing volatility
        if features['volatility'] < 0.02:
            confidence += 0.3
            
        # Check for volume pattern
        if features['volume_trend'] < 0:
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _check_head_shoulders_pattern(self, features: Dict[str, float], 
                                     prices: np.ndarray) -> float:
        """Check for head and shoulders pattern"""
        confidence = 0.0
        
        # Find peaks
        peaks = self._find_local_extrema(prices, 'high')
        
        if len(peaks) >= 3:
            # Check for three peaks with middle highest
            if peaks[1, 1] > peaks[0, 1] and peaks[1, 1] > peaks[2, 1]:
                confidence += 0.4
                
                # Check for similar shoulder heights
                shoulder_diff = abs(peaks[0, 1] - peaks[2, 1]) / peaks[1, 1]
                if shoulder_diff < 0.05:
                    confidence += 0.3
                    
            # Check for neckline
            valleys = self._find_local_extrema(prices, 'low')
            if len(valleys) >= 2:
                neckline_slope = abs(np.polyfit(valleys[:2, 0], valleys[:2, 1], 1)[0])
                if neckline_slope < 0.001:
                    confidence += 0.3
                    
        return min(confidence, 1.0)
        
    def _calculate_completion(self, pattern_type: str, 
                            features: Dict[str, float]) -> float:
        """Calculate pattern completion percentage"""
        template = self.pattern_templates[pattern_type]
        
        # Check key features
        completed_features = 0
        for feature in template['key_features']:
            if self._is_feature_present(feature, features):
                completed_features += 1
                
        completion = (completed_features / len(template['key_features'])) * 100
        
        # Adjust based on pattern-specific criteria
        if pattern_type == 'head_shoulders':
            # Need at least 3 peaks for completion
            if 'three_peaks' not in features or features.get('peak_count', 0) < 3:
                completion = min(completion, 60)
                
        return completion
        
    def _identify_stage(self, pattern_type: str, 
                       features: Dict[str, float]) -> str:
        """Identify current stage of pattern formation"""
        if pattern_type == 'breakout':
            if features.get('near_resistance', False):
                return 'approaching_resistance'
            elif features.get('price_change', 0) > 0.01:
                return 'breaking_out'
            else:
                return 'consolidating'
                
        elif pattern_type == 'reversal':
            if features.get('rsi', 50) > 70 or features.get('rsi', 50) < 30:
                return 'extreme_reached'
            elif features.get('divergence', False):
                return 'divergence_forming'
            else:
                return 'momentum_slowing'
                
        elif pattern_type == 'triangle':
            if features.get('volatility', 1) < 0.01:
                return 'apex_approaching'
            else:
                return 'converging'
                
        return 'forming'
        
    def _identify_key_levels(self, prices: np.ndarray) -> List[float]:
        """Identify key price levels"""
        levels = []
        
        # Recent high and low
        levels.append(np.max(prices))
        levels.append(np.min(prices))
        
        # Support and resistance levels
        peaks = self._find_local_extrema(prices, 'high')
        valleys = self._find_local_extrema(prices, 'low')
        
        if len(peaks) > 0:
            levels.extend(peaks[:, 1].tolist())
        if len(valleys) > 0:
            levels.extend(valleys[:, 1].tolist())
            
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        
        return levels[:5]  # Return top 5 levels
        
    def _is_duplicate_pattern(self, ticker: str, pattern_type: str) -> bool:
        """Check if pattern is already being tracked"""
        for pattern_id, progress in self.active_patterns.items():
            if ticker in pattern_id and progress.pattern_type == pattern_type:
                # Check if pattern is still active (not expired)
                age = datetime.now() - progress.start_time
                if age < progress.expected_duration * 1.5:
                    return True
                    
        return False
        
    def _update_active_patterns(self, ticker: str):
        """Update progress of active patterns"""
        patterns_to_remove = []
        
        for pattern_id, progress in self.active_patterns.items():
            if ticker in pattern_id:
                # Update completion
                prices = np.array(self.price_buffers[ticker])
                volumes = np.array(self.volume_buffers[ticker])
                coeffs = self._wavelet_transform(prices)
                features = self._extract_features(prices, volumes, coeffs)
                
                progress.completion = self._calculate_completion(
                    progress.pattern_type, features
                )
                progress.current_stage = self._identify_stage(
                    progress.pattern_type, features
                )
                
                # Check if pattern has expired
                age = datetime.now() - progress.start_time
                if age > progress.expected_duration * 2:
                    patterns_to_remove.append(pattern_id)
                    
                # Check if pattern completed
                elif progress.completion >= 90:
                    self._handle_pattern_completion(ticker, progress)
                    patterns_to_remove.append(pattern_id)
                    
        # Remove expired/completed patterns
        for pattern_id in patterns_to_remove:
            del self.active_patterns[pattern_id]
            
    def _handle_pattern_completion(self, ticker: str, progress: PatternProgress):
        """Handle completed pattern"""
        # Add to history
        if ticker not in self.pattern_history:
            self.pattern_history[ticker] = []
            
        self.pattern_history[ticker].append({
            'pattern_type': progress.pattern_type,
            'start_time': progress.start_time,
            'end_time': datetime.now(),
            'confidence': progress.confidence,
            'key_levels': progress.key_levels
        })
        
        # Generate completion alert
        self._generate_alert(
            ticker=ticker,
            pattern_type=progress.pattern_type,
            confidence=progress.confidence,
            price_level=self.price_buffers[ticker][-1],
            risk_level=self.pattern_templates[progress.pattern_type]['risk_level'],
            metadata={'completion': True, 'pattern_id': progress.pattern_id}
        )
        
    def _generate_alert(self, ticker: str, pattern_type: str, confidence: float,
                       price_level: float, risk_level: str, 
                       metadata: Optional[Dict] = None):
        """Generate pattern alert"""
        # Calculate expected move based on pattern type
        expected_move = self._calculate_expected_move(pattern_type, price_level)
        
        alert = PatternAlert(
            timestamp=datetime.now(),
            pattern_type=pattern_type,
            confidence=confidence,
            ticker=ticker,
            timeframe='1m',  # Could be dynamic
            price_level=price_level,
            expected_move=expected_move,
            risk_level=risk_level,
            metadata=metadata or {}
        )
        
        self.alert_queue.put(alert)
        
    def _calculate_expected_move(self, pattern_type: str, 
                                price_level: float) -> float:
        """Calculate expected price move for pattern"""
        # Pattern-specific move calculations
        move_percentages = {
            'breakout': 0.03,
            'reversal': -0.05,
            'continuation': 0.02,
            'triangle': 0.04,
            'head_shoulders': -0.06
        }
        
        percentage = move_percentages.get(pattern_type, 0.02)
        return price_level * percentage
        
    def _process_alerts(self):
        """Process and dispatch alerts"""
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                
                # Call alert callback if provided
                if self.alert_callback:
                    self.alert_callback(alert)
                    
                # Log alert
                print(f"ALERT: {alert.pattern_type} pattern detected for {alert.ticker} "
                      f"at ${alert.price_level:.2f} (confidence: {alert.confidence:.2%})")
                      
            except queue.Empty:
                break
                
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(prices) < 2:
            return 0.0
            
        # Linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return abs(r_squared)
        
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
            
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _calculate_macd_signal(self, prices: np.ndarray) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
            
        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        macd = ema12 - ema26
        signal = self._calculate_ema(macd, 9)
        
        return (macd[-1] - signal[-1]) / prices[-1]
        
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def _check_near_resistance(self, prices: np.ndarray, 
                              threshold: float = 0.01) -> bool:
        """Check if price is near resistance"""
        if len(prices) < 20:
            return False
            
        # Find recent peaks
        peaks = signal.find_peaks(prices, distance=5)[0]
        
        if len(peaks) > 0:
            recent_peak = prices[peaks[-1]]
            current_price = prices[-1]
            
            # Check if within threshold of recent peak
            if abs(current_price - recent_peak) / recent_peak < threshold:
                return True
                
        return False
        
    def _check_near_support(self, prices: np.ndarray, 
                           threshold: float = 0.01) -> bool:
        """Check if price is near support"""
        if len(prices) < 20:
            return False
            
        # Find recent valleys
        valleys = signal.find_peaks(-prices, distance=5)[0]
        
        if len(valleys) > 0:
            recent_valley = prices[valleys[-1]]
            current_price = prices[-1]
            
            # Check if within threshold of recent valley
            if abs(current_price - recent_valley) / recent_valley < threshold:
                return True
                
        return False
        
    def _find_local_extrema(self, prices: np.ndarray, 
                           extrema_type: str = 'high') -> np.ndarray:
        """Find local extrema in price data"""
        if extrema_type == 'high':
            indices = signal.find_peaks(prices, distance=5)[0]
        else:
            indices = signal.find_peaks(-prices, distance=5)[0]
            
        if len(indices) == 0:
            return np.array([])
            
        extrema = np.column_stack((indices, prices[indices]))
        return extrema
        
    def _is_feature_present(self, feature_name: str, 
                           features: Dict[str, float]) -> bool:
        """Check if a key feature is present"""
        feature_checks = {
            'resistance_break': lambda f: f.get('near_resistance', False) and 
                                        f.get('price_change', 0) > 0.01,
            'volume_surge': lambda f: f.get('volume_ratio', 0) > 1.5,
            'trend_exhaustion': lambda f: f.get('rsi', 50) > 70 or 
                                        f.get('rsi', 50) < 30,
            'divergence': lambda f: abs(f.get('macd_signal', 0)) > 0.001,
            'flag_formation': lambda f: f.get('volatility', 1) < 0.015,
            'volume_decline': lambda f: f.get('volume_ratio', 1) < 0.8,
            'converging_lines': lambda f: f.get('volatility', 1) < 0.02,
            'decreasing_volatility': lambda f: f.get('volatility', 1) < 0.015,
            'three_peaks': lambda f: f.get('peak_count', 0) >= 3,
            'neckline': lambda f: f.get('has_neckline', False)
        }
        
        check_func = feature_checks.get(feature_name)
        if check_func:
            return check_func(features)
            
        return False
        
    def get_active_patterns(self, ticker: Optional[str] = None) -> Dict[str, PatternProgress]:
        """Get currently active patterns"""
        if ticker:
            return {k: v for k, v in self.active_patterns.items() if ticker in k}
        return self.active_patterns.copy()
        
    def get_pattern_history(self, ticker: str, 
                           hours: int = 24) -> List[Dict]:
        """Get pattern history for ticker"""
        if ticker not in self.pattern_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [p for p in self.pattern_history[ticker] 
                if p['start_time'] >= cutoff_time]
        
    def get_pattern_statistics(self, ticker: str) -> Dict[str, Any]:
        """Get pattern statistics for ticker"""
        history = self.pattern_history.get(ticker, [])
        
        if not history:
            return {
                'total_patterns': 0,
                'pattern_counts': {},
                'avg_confidence': 0,
                'success_rate': 0
            }
            
        pattern_counts = {}
        total_confidence = 0
        
        for pattern in history:
            pattern_type = pattern['pattern_type']
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            total_confidence += pattern['confidence']
            
        return {
            'total_patterns': len(history),
            'pattern_counts': pattern_counts,
            'avg_confidence': total_confidence / len(history),
            'success_rate': self._calculate_success_rate(history)
        }
        
    def _calculate_success_rate(self, history: List[Dict]) -> float:
        """Calculate pattern success rate"""
        # This would need actual outcome data
        # For now, return a placeholder
        return 0.65
        
    def export_alerts(self, hours: int = 24) -> pd.DataFrame:
        """Export recent alerts as DataFrame"""
        # This would need to store alerts
        # For now, return empty DataFrame
        return pd.DataFrame(columns=[
            'timestamp', 'ticker', 'pattern_type', 
            'confidence', 'price_level', 'risk_level'
        ])


def create_demo_monitor():
    """Create demo pattern monitor"""
    
    def alert_handler(alert: PatternAlert):
        """Handle pattern alerts"""
        print(f"\n{'='*60}")
        print(f"🚨 PATTERN ALERT - {alert.pattern_type.upper()}")
        print(f"{'='*60}")
        print(f"Ticker: {alert.ticker}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Confidence: {alert.confidence:.1%}")
        print(f"Price: ${alert.price_level:.2f}")
        print(f"Expected Move: ${alert.expected_move:.2f}")
        print(f"Risk Level: {alert.risk_level}")
        print(f"Stage: {alert.metadata.get('stage', 'N/A')}")
        print(f"{'='*60}\n")
    
    # Create monitor instance
    monitor = RealTimePatternMonitor(
        window_size=100,
        update_interval=1.0,
        min_confidence=0.7,
        alert_callback=alert_handler
    )
    
    return monitor


def simulate_live_data(monitor: RealTimePatternMonitor, 
                      ticker: str = "AAPL",
                      base_price: float = 150.0,
                      duration_seconds: int = 300):
    """
    Simulate live price data for testing
    
    Args:
        monitor: Pattern monitor instance
        ticker: Stock ticker
        base_price: Starting price
        duration_seconds: Simulation duration
    """
    import random
    
    print(f"Starting live data simulation for {ticker}")
    print(f"Base price: ${base_price:.2f}")
    print(f"Duration: {duration_seconds} seconds")
    print("-" * 60)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate data
    start_time = time.time()
    price = base_price
    
    # Pattern scenarios
    scenarios = [
        'normal',      # Random walk
        'breakout',    # Upward breakout
        'reversal',    # Trend reversal
        'triangle',    # Triangle formation
    ]
    
    current_scenario = random.choice(scenarios)
    scenario_start = time.time()
    
    print(f"Starting scenario: {current_scenario}")
    
    try:
        while time.time() - start_time < duration_seconds:
            # Switch scenarios periodically
            if time.time() - scenario_start > 60:
                current_scenario = random.choice(scenarios)
                scenario_start = time.time()
                print(f"\nSwitching to scenario: {current_scenario}")
            
            # Generate price based on scenario
            if current_scenario == 'normal':
                # Random walk
                change = random.gauss(0, 0.002) * price
                volume = random.uniform(1000000, 2000000)
                
            elif current_scenario == 'breakout':
                # Gradual increase with volume surge
                change = random.gauss(0.001, 0.001) * price
                volume = random.uniform(2000000, 4000000)
                
            elif current_scenario == 'reversal':
                # Trend change
                elapsed = time.time() - scenario_start
                if elapsed < 30:
                    change = random.gauss(0.002, 0.001) * price
                else:
                    change = random.gauss(-0.002, 0.001) * price
                volume = random.uniform(1500000, 3000000)
                
            elif current_scenario == 'triangle':
                # Converging price action
                elapsed = time.time() - scenario_start
                volatility = max(0.001, 0.003 - elapsed * 0.00005)
                change = random.gauss(0, volatility) * price
                volume = random.uniform(800000, 1500000)
            
            # Update price
            price += change
            price = max(price, base_price * 0.9)  # Floor at -10%
            price = min(price, base_price * 1.1)  # Cap at +10%
            
            # Add data to monitor
            monitor.add_price_data(ticker, price, volume)
            
            # Display current price periodically
            if int(time.time()) % 10 == 0:
                active_patterns = monitor.get_active_patterns(ticker)
                if active_patterns:
                    print(f"\nActive patterns for {ticker}:")
                    for pattern_id, progress in active_patterns.items():
                        print(f"  - {progress.pattern_type}: "
                              f"{progress.completion:.0f}% complete, "
                              f"stage: {progress.current_stage}")
            
            time.sleep(0.1)  # 10 updates per second
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Display statistics
        stats = monitor.get_pattern_statistics(ticker)
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE - Pattern Statistics for {ticker}")
        print(f"{'='*60}")
        print(f"Total patterns detected: {stats['total_patterns']}")
        print(f"Average confidence: {stats['avg_confidence']:.1%}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print("\nPattern counts:")
        for pattern_type, count in stats['pattern_counts'].items():
            print(f"  - {pattern_type}: {count}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Create and run demo
    monitor = create_demo_monitor()
    
    # Simulate live data
    simulate_live_data(
        monitor,
        ticker="AAPL",
        base_price=150.0,
        duration_seconds=120  # 2 minutes
    )
