"""
Multi-timeframe analysis for comprehensive market understanding.
Analyzes patterns across multiple time horizons simultaneously.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pywt
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    name: str
    window_size: int
    wavelet: str
    decomposition_level: int
    weight: float


class MultiTimeframeAnalyzer:
    """
    Analyzes financial data across multiple timeframes to capture
    both short-term and long-term patterns.
    """
    
    def __init__(self, timeframe_configs: Optional[List[TimeframeConfig]] = None):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            timeframe_configs: List of timeframe configurations
        """
        if timeframe_configs is None:
            # Default configurations for different timeframes
            self.timeframe_configs = [
                TimeframeConfig("intraday", 60, "db4", 3, 0.2),      # 1-hour
                TimeframeConfig("daily", 1440, "db6", 4, 0.3),       # 1-day
                TimeframeConfig("weekly", 10080, "db8", 5, 0.3),     # 1-week
                TimeframeConfig("monthly", 43200, "db10", 6, 0.2)    # 1-month
            ]
        else:
            self.timeframe_configs = timeframe_configs
            
        self.timeframe_features = {}
        self.combined_features = None
        
    def analyze_timeframe(self, data: np.ndarray, config: TimeframeConfig) -> Dict[str, np.ndarray]:
        """
        Analyze data for a specific timeframe.
        
        Args:
            data: Input time series data
            config: Timeframe configuration
            
        Returns:
            Dictionary of features for this timeframe
        """
        features = {}
        
        # Resample data to timeframe window
        if len(data) > config.window_size:
            resampled = signal.resample(data, config.window_size)
        else:
            resampled = data
            
        # Wavelet decomposition
        coeffs = pywt.wavedec(resampled, config.wavelet, level=config.decomposition_level)
        
        # Extract features from each decomposition level
        for i, coeff in enumerate(coeffs):
            level_name = f"{config.name}_level_{i}"
            
            # Statistical features
            features[f"{level_name}_mean"] = np.mean(coeff)
            features[f"{level_name}_std"] = np.std(coeff)
            features[f"{level_name}_energy"] = np.sum(coeff ** 2)
            features[f"{level_name}_entropy"] = -np.sum(coeff ** 2 * np.log(coeff ** 2 + 1e-10))
            
            # Frequency domain features
            fft_coeff = np.fft.fft(coeff)
            features[f"{level_name}_dominant_freq"] = np.argmax(np.abs(fft_coeff[:len(fft_coeff)//2]))
            features[f"{level_name}_spectral_centroid"] = np.sum(np.arange(len(fft_coeff)) * np.abs(fft_coeff)) / np.sum(np.abs(fft_coeff))
            
        # Trend features
        features[f"{config.name}_trend_strength"] = self._calculate_trend_strength(resampled)
        features[f"{config.name}_volatility"] = np.std(np.diff(resampled))
        
        return features
    
    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """Calculate trend strength using linear regression."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        y_fit = np.polyval(coeffs, x)
        r_squared = 1 - (np.sum((data - y_fit) ** 2) / np.sum((data - np.mean(data)) ** 2))
        return abs(r_squared)
    
    def analyze_all_timeframes(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Analyze data across all configured timeframes.
        
        Args:
            data: Input time series data
            
        Returns:
            Dictionary of features for each timeframe
        """
        self.timeframe_features = {}
        
        for config in self.timeframe_configs:
            logger.info(f"Analyzing {config.name} timeframe...")
            self.timeframe_features[config.name] = self.analyze_timeframe(data, config)
            
        return self.timeframe_features
    
    def combine_timeframe_features(self, 
                                 timeframe_features: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                                 method: str = 'weighted') -> np.ndarray:
        """
        Combine features from multiple timeframes.
        
        Args:
            timeframe_features: Features from each timeframe
            method: Combination method ('weighted', 'concatenate', 'pca')
            
        Returns:
            Combined feature vector
        """
        if timeframe_features is None:
            timeframe_features = self.timeframe_features
            
        if method == 'weighted':
            # Weighted average based on timeframe importance
            combined = []
            
            for config in self.timeframe_configs:
                tf_features = timeframe_features.get(config.name, {})
                feature_vector = np.array(list(tf_features.values()))
                
                # Normalize features
                if len(feature_vector) > 0:
                    feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
                    weighted_features = feature_vector * config.weight
                    combined.extend(weighted_features)
                    
            self.combined_features = np.array(combined)
            
        elif method == 'concatenate':
            # Simple concatenation
            combined = []
            for tf_name, features in timeframe_features.items():
                combined.extend(list(features.values()))
            self.combined_features = np.array(combined)
            
        elif method == 'pca':
            # PCA-based combination (requires sklearn)
            from sklearn.decomposition import PCA
            
            all_features = []
            for tf_name, features in timeframe_features.items():
                all_features.append(list(features.values()))
                
            all_features = np.array(all_features).T
            pca = PCA(n_components=min(10, all_features.shape[1]))
            self.combined_features = pca.fit_transform(all_features).flatten()
            
        return self.combined_features
    
    def get_timeframe_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between different timeframes.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Correlation matrix between timeframes
        """
        correlations = {}
        
        for i, config1 in enumerate(self.timeframe_configs):
            for j, config2 in enumerate(self.timeframe_configs):
                if i <= j:
                    # Resample data to each timeframe
                    data1 = signal.resample(data.values.flatten(), config1.window_size)
                    data2 = signal.resample(data.values.flatten(), config2.window_size)
                    
                    # Calculate correlation
                    min_len = min(len(data1), len(data2))
                    corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                    
                    correlations[f"{config1.name}_vs_{config2.name}"] = corr
                    
        return pd.DataFrame([correlations])
    
    def predict_multi_timeframe(self, 
                              current_data: np.ndarray,
                              models: Dict[str, any]) -> Dict[str, np.ndarray]:
        """
        Generate predictions for each timeframe.
        
        Args:
            current_data: Current market data
            models: Dictionary of models for each timeframe
            
        Returns:
            Predictions for each timeframe
        """
        predictions = {}
        
        for config in self.timeframe_configs:
            if config.name in models:
                # Extract features for this timeframe
                features = self.analyze_timeframe(current_data, config)
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                
                # Generate prediction
                model = models[config.name]
                pred = model.predict(feature_vector)
                predictions[config.name] = pred
                
        return predictions
    
    def aggregate_predictions(self, 
                            predictions: Dict[str, np.ndarray],
                            method: str = 'weighted') -> np.ndarray:
        """
        Aggregate predictions from multiple timeframes.
        
        Args:
            predictions: Predictions from each timeframe
            method: Aggregation method
            
        Returns:
            Aggregated prediction
        """
        if method == 'weighted':
            # Weighted average based on timeframe weights
            weighted_sum = 0
            total_weight = 0
            
            for config in self.timeframe_configs:
                if config.name in predictions:
                    weighted_sum += predictions[config.name] * config.weight
                    total_weight += config.weight
                    
            return weighted_sum / total_weight if total_weight > 0 else np.zeros_like(list(predictions.values())[0])
            
        elif method == 'voting':
            # Majority voting for classification
            votes = []
            for config in self.timeframe_configs:
                if config.name in predictions:
                    votes.append(predictions[config.name])
                    
            votes = np.array(votes)
            return np.median(votes, axis=0)
            
        elif method == 'max_confidence':
            # Select prediction with highest confidence
            max_pred = None
            max_conf = -np.inf
            
            for config in self.timeframe_configs:
                if config.name in predictions:
                    conf = np.max(np.abs(predictions[config.name]))
                    if conf > max_conf:
                        max_conf = conf
                        max_pred = predictions[config.name]
                        
            return max_pred
