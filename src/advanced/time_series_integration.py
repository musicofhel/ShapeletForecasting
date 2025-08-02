"""
Advanced Time Series Integration Module

Integrates SAX, MultiRocket, HIVECOTEV2, and other advanced time series techniques
into the financial wavelet prediction system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

# Try to import advanced libraries
try:
    from aeon.classification.hybrid import HIVECOTEV2
    from aeon.classification.convolution_based import MultiRocketClassifier
    from aeon.transformations.collection import TimeSeriesScaler
    HAS_AEON = True
except ImportError:
    HAS_AEON = False
    warnings.warn("aeon toolkit not available. Some advanced features will be disabled.", UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class SAXConfig:
    """Configuration for SAX transformation"""
    n_segments: int = 20  # Number of segments for PAA
    alphabet_size: int = 5  # Size of SAX alphabet
    z_threshold: float = 0.01  # Z-normalization threshold
    
    
@dataclass
class SimilaritySearchConfig:
    """Configuration for similarity search"""
    method: str = 'dtw'  # 'dtw', 'euclidean', 'sax', 'shapelet'
    top_k: int = 10  # Number of similar patterns to retrieve
    min_similarity: float = 0.7  # Minimum similarity threshold
    

class SAXTransformer:
    """
    Symbolic Aggregate approXimation (SAX) implementation for time series discretization.
    Based on: https://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html
    """
    
    def __init__(self, config: SAXConfig):
        self.config = config
        self._setup_breakpoints()
        
    def _setup_breakpoints(self):
        """Setup SAX alphabet breakpoints based on normal distribution"""
        # Gaussian breakpoints for different alphabet sizes
        self.breakpoints = {
            3: [-0.43, 0.43],
            4: [-0.67, 0, 0.67],
            5: [-0.84, -0.25, 0.25, 0.84],
            6: [-0.97, -0.43, 0, 0.43, 0.97],
            7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
            8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
            9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
            10: [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]
        }
        
        if self.config.alphabet_size not in self.breakpoints:
            # Generate custom breakpoints for other alphabet sizes
            self.breakpoints[self.config.alphabet_size] = [
                stats.norm.ppf((i+1) / (self.config.alphabet_size + 1))
                for i in range(self.config.alphabet_size - 1)
            ]
            
    def transform(self, series: np.ndarray) -> str:
        """
        Transform time series to SAX representation
        
        Args:
            series: Input time series
            
        Returns:
            SAX string representation
        """
        # Z-normalize
        normalized = self._znormalize(series)
        
        # PAA transformation
        paa = self._paa_transform(normalized)
        
        # Convert to SAX symbols
        sax_string = self._to_sax_string(paa)
        
        return sax_string
        
    def _znormalize(self, series: np.ndarray) -> np.ndarray:
        """Z-normalize the time series"""
        mean = np.mean(series)
        std = np.std(series)
        
        if std < self.config.z_threshold:
            return np.zeros_like(series)
            
        return (series - mean) / std
        
    def _paa_transform(self, series: np.ndarray) -> np.ndarray:
        """Piecewise Aggregate Approximation"""
        n = len(series)
        segment_len = n / self.config.n_segments
        
        paa = np.zeros(self.config.n_segments)
        for i in range(self.config.n_segments):
            start = int(i * segment_len)
            end = int((i + 1) * segment_len)
            paa[i] = np.mean(series[start:end])
            
        return paa
        
    def _to_sax_string(self, paa: np.ndarray) -> str:
        """Convert PAA to SAX string"""
        sax_chars = []
        breakpoints = self.breakpoints[self.config.alphabet_size]
        
        for value in paa:
            # Find appropriate symbol
            symbol_idx = 0
            for bp in breakpoints:
                if value > bp:
                    symbol_idx += 1
                else:
                    break
                    
            # Convert to letter (a, b, c, ...)
            sax_chars.append(chr(ord('a') + symbol_idx))
            
        return ''.join(sax_chars)
        
    def inverse_transform(self, sax_string: str, original_length: int) -> np.ndarray:
        """
        Approximate inverse transformation from SAX to time series
        
        Args:
            sax_string: SAX representation
            original_length: Length of original time series
            
        Returns:
            Reconstructed time series
        """
        # Convert SAX string back to PAA values
        paa = []
        breakpoints = self.breakpoints[self.config.alphabet_size]
        
        for char in sax_string:
            symbol_idx = ord(char) - ord('a')
            
            # Get midpoint of the symbol's range
            if symbol_idx == 0:
                value = breakpoints[0] - 0.5
            elif symbol_idx == len(breakpoints):
                value = breakpoints[-1] + 0.5
            else:
                value = (breakpoints[symbol_idx-1] + breakpoints[symbol_idx]) / 2
                
            paa.append(value)
            
        # Interpolate PAA back to original length
        paa = np.array(paa)
        segment_len = original_length / len(paa)
        
        reconstructed = np.zeros(original_length)
        for i in range(len(paa)):
            start = int(i * segment_len)
            end = int((i + 1) * segment_len)
            reconstructed[start:end] = paa[i]
            
        return reconstructed


class TimeSeriesSimilaritySearch:
    """
    Advanced similarity search for time series data
    Based on: https://www.geeksforgeeks.org/machine-learning/similarity-search-for-time-series-data/
    """
    
    def __init__(self, config: SimilaritySearchConfig):
        self.config = config
        self.index = {}  # Storage for indexed patterns
        self.sax_transformer = SAXTransformer(SAXConfig())
        
    def index_patterns(self, patterns: List[Dict[str, Any]]):
        """
        Index patterns for fast similarity search
        
        Args:
            patterns: List of pattern dictionaries with 'data' and 'metadata'
        """
        logger.info(f"Indexing {len(patterns)} patterns")
        
        for i, pattern in enumerate(patterns):
            data = pattern['data']
            
            # Store original data
            self.index[i] = {
                'data': data,
                'metadata': pattern.get('metadata', {}),
                'features': self._extract_features(data),
                'sax': self.sax_transformer.transform(data)
            }
            
    def _extract_features(self, series: np.ndarray) -> Dict[str, float]:
        """Extract statistical features for fast filtering"""
        return {
            'mean': float(np.mean(series)),
            'std': float(np.std(series)),
            'skew': float(stats.skew(series)),
            'kurtosis': float(stats.kurtosis(series)),
            'trend': float(np.polyfit(range(len(series)), series, 1)[0]),
            'length': len(series)
        }
        
    def search(self, query: np.ndarray, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar patterns
        
        Args:
            query: Query time series
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of similar patterns with similarity scores
        """
        results = []
        query_features = self._extract_features(query)
        query_sax = self.sax_transformer.transform(query)
        
        # Pre-filter based on features
        candidates = self._prefilter_candidates(query_features, filter_criteria)
        
        # Compute similarities
        for idx in candidates:
            pattern = self.index[idx]
            
            if self.config.method == 'sax':
                similarity = self._sax_similarity(query_sax, pattern['sax'])
            elif self.config.method == 'euclidean':
                similarity = self._euclidean_similarity(query, pattern['data'])
            elif self.config.method == 'dtw':
                similarity = self._dtw_similarity(query, pattern['data'])
            else:
                similarity = 0.0
                
            if similarity >= self.config.min_similarity:
                results.append({
                    'pattern_id': idx,
                    'similarity': similarity,
                    'data': pattern['data'],
                    'metadata': pattern['metadata']
                })
                
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:self.config.top_k]
        
    def _prefilter_candidates(self, query_features: Dict[str, float],
                             filter_criteria: Optional[Dict] = None) -> List[int]:
        """Pre-filter candidates based on features"""
        candidates = []
        
        for idx, pattern in self.index.items():
            # Feature-based filtering
            feature_diff = sum(
                abs(query_features[k] - pattern['features'][k]) 
                for k in ['mean', 'std', 'trend']
            )
            
            if feature_diff < 2.0:  # Threshold for feature similarity
                # Apply additional filters if provided
                if filter_criteria:
                    metadata = pattern['metadata']
                    if all(metadata.get(k) == v for k, v in filter_criteria.items()):
                        candidates.append(idx)
                else:
                    candidates.append(idx)
                    
        return candidates
        
    def _sax_similarity(self, sax1: str, sax2: str) -> float:
        """Compute SAX-based similarity"""
        if len(sax1) != len(sax2):
            return 0.0
            
        matches = sum(1 for a, b in zip(sax1, sax2) if a == b)
        return matches / len(sax1)
        
    def _euclidean_similarity(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Compute Euclidean-based similarity"""
        # Resample if different lengths
        if len(series1) != len(series2):
            series2 = np.interp(
                np.linspace(0, 1, len(series1)),
                np.linspace(0, 1, len(series2)),
                series2
            )
            
        distance = np.linalg.norm(series1 - series2)
        # Convert to similarity (0-1)
        return 1.0 / (1.0 + distance)
        
    def _dtw_similarity(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Compute DTW-based similarity"""
        # Import DTW calculator
        from ..dtw.dtw_calculator import DTWCalculator
        
        dtw_calc = DTWCalculator(normalize=True)
        result = dtw_calc.compute(series1, series2)
        
        # Convert distance to similarity
        return 1.0 / (1.0 + result.normalized_distance)


class MultiRocketIntegration:
    """
    Integration of MultiRocket for time series classification
    Based on: https://github.com/ChangWeiTan/MultiRocket
    """
    
    def __init__(self, n_kernels: int = 10000, n_features: int = 50):
        self.n_kernels = n_kernels
        self.n_features = n_features
        self.classifier = None
        self.scaler = StandardScaler()
        
        if HAS_AEON:
            self.rocket = MultiRocketClassifier(
                num_kernels=n_kernels,
                n_jobs=-1,
                random_state=42
            )
        else:
            logger.warning("MultiRocket not available, using fallback implementation")
            self.rocket = None
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit MultiRocket classifier
        
        Args:
            X: Training time series data (n_samples, n_timepoints)
            y: Training labels
        """
        if self.rocket is not None:
            # Use aeon's MultiRocket
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])  # Convert to 3D
            self.rocket.fit(X_3d, y)
        else:
            # Fallback: extract manual features
            features = self._extract_rocket_features(X)
            features_scaled = self.scaler.fit_transform(features)
            
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.classifier.fit(features_scaled, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using MultiRocket"""
        if self.rocket is not None:
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])
            return self.rocket.predict(X_3d)
        else:
            features = self._extract_rocket_features(X)
            features_scaled = self.scaler.transform(features)
            return self.classifier.predict(features_scaled)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using MultiRocket"""
        if self.rocket is not None:
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])
            return self.rocket.predict_proba(X_3d)
        else:
            features = self._extract_rocket_features(X)
            features_scaled = self.scaler.transform(features)
            return self.classifier.predict_proba(features_scaled)
            
    def _extract_rocket_features(self, X: np.ndarray) -> np.ndarray:
        """
        Fallback feature extraction mimicking ROCKET approach
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_features))
        
        for i in range(n_samples):
            series = X[i]
            
            # Random convolutional features
            for j in range(self.n_features):
                # Random kernel length
                kernel_length = np.random.choice([7, 9, 11])
                
                if len(series) >= kernel_length:
                    # Random weights
                    weights = np.random.normal(0, 1, kernel_length)
                    
                    # Convolution
                    conv = np.convolve(series, weights, mode='valid')
                    
                    # Random pooling (max or mean)
                    if np.random.rand() > 0.5:
                        features[i, j] = np.max(conv)
                    else:
                        features[i, j] = np.mean(conv)
                else:
                    features[i, j] = 0
                    
        return features


class HIVECOTEV2Integration:
    """
    Integration of HIVECOTE V2 ensemble classifier
    Based on: https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.hybrid.HIVECOTEV2.html
    """
    
    def __init__(self, time_limit_minutes: int = 60):
        self.time_limit_minutes = time_limit_minutes
        
        if HAS_AEON:
            self.hivecote = HIVECOTEV2(
                time_limit_in_minutes=time_limit_minutes,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
        else:
            logger.warning("HIVECOTE V2 not available, using ensemble fallback")
            self.hivecote = None
            self._setup_fallback_ensemble()
            
    def _setup_fallback_ensemble(self):
        """Setup fallback ensemble when aeon is not available"""
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100)),
                ('svm', SVC(probability=True)),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ],
            voting='soft'
        )
        self.feature_extractor = AdvancedFeatureExtractor()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit HIVECOTE V2 classifier
        
        Args:
            X: Training time series data
            y: Training labels
        """
        if self.hivecote is not None:
            # Convert to 3D format for aeon
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])
            self.hivecote.fit(X_3d, y)
        else:
            # Fallback: extract features and use ensemble
            features = self.feature_extractor.extract_all_features(X)
            self.ensemble.fit(features, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using HIVECOTE V2"""
        if self.hivecote is not None:
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])
            return self.hivecote.predict(X_3d)
        else:
            features = self.feature_extractor.extract_all_features(X)
            return self.ensemble.predict(features)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using HIVECOTE V2"""
        if self.hivecote is not None:
            X_3d = X.reshape(X.shape[0], 1, X.shape[1])
            return self.hivecote.predict_proba(X_3d)
        else:
            features = self.feature_extractor.extract_all_features(X)
            return self.ensemble.predict_proba(features)


class AdvancedFeatureExtractor:
    """
    Extract advanced features for time series classification
    """
    
    def __init__(self):
        self.sax_transformer = SAXTransformer(SAXConfig())
        
    def extract_all_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature set from time series
        
        Args:
            X: Time series data (n_samples, n_timepoints)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        n_samples = X.shape[0]
        features_list = []
        
        for i in range(n_samples):
            series = X[i]
            features = []
            
            # Statistical features
            features.extend(self._extract_statistical_features(series))
            
            # Spectral features
            features.extend(self._extract_spectral_features(series))
            
            # Shape features
            features.extend(self._extract_shape_features(series))
            
            # SAX features
            features.extend(self._extract_sax_features(series))
            
            features_list.append(features)
            
        return np.array(features_list)
        
    def _extract_statistical_features(self, series: np.ndarray) -> List[float]:
        """Extract statistical features"""
        return [
            np.mean(series),
            np.std(series),
            np.min(series),
            np.max(series),
            stats.skew(series),
            stats.kurtosis(series),
            np.percentile(series, 25),
            np.percentile(series, 75),
            np.median(series)
        ]
        
    def _extract_spectral_features(self, series: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        fft = np.fft.fft(series)
        power_spectrum = np.abs(fft) ** 2
        
        # Find dominant frequencies
        freqs = np.fft.fftfreq(len(series))
        dominant_freq_idx = np.argmax(power_spectrum[1:len(series)//2]) + 1
        
        return [
            float(freqs[dominant_freq_idx]),  # Dominant frequency
            float(power_spectrum[dominant_freq_idx]),  # Power at dominant freq
            float(np.sum(power_spectrum)),  # Total power
            float(np.std(power_spectrum))  # Power variance
        ]
        
    def _extract_shape_features(self, series: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        # Find peaks and valleys
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(series)
        valleys, _ = find_peaks(-series)
        
        # Trend
        x = np.arange(len(series))
        trend = np.polyfit(x, series, 1)[0]
        
        return [
            float(len(peaks)),  # Number of peaks
            float(len(valleys)),  # Number of valleys
            float(trend),  # Linear trend
            float(np.mean(np.diff(series))),  # Mean difference
            float(np.std(np.diff(series)))  # Std of differences
        ]
        
    def _extract_sax_features(self, series: np.ndarray) -> List[float]:
        """Extract SAX-based features"""
        sax_string = self.sax_transformer.transform(series)
        
        # Character frequency
        char_counts = {}
        for char in sax_string:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Entropy
        probs = np.array(list(char_counts.values())) / len(sax_string)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Transitions
        transitions = sum(1 for i in range(len(sax_string)-1) 
                         if sax_string[i] != sax_string[i+1])
        
        return [
            float(len(set(sax_string))),  # Unique symbols
            float(entropy),  # Symbol entropy
            float(transitions / (len(sax_string) - 1))  # Transition rate
        ]


class AdvancedTimeSeriesClassifier:
    """
    Unified interface for advanced time series classification
    """
    
    def __init__(self, method: str = 'hivecote', **kwargs):
        """
        Initialize classifier
        
        Args:
            method: Classification method ('hivecote', 'multirocket', 'ensemble')
            **kwargs: Additional arguments for specific classifiers
        """
        self.method = method
        
        if method == 'hivecote':
            self.classifier = HIVECOTEV2Integration(**kwargs)
        elif method == 'multirocket':
            self.classifier = MultiRocketIntegration(**kwargs)
        elif method == 'ensemble':
            self._setup_ensemble(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.similarity_search = TimeSeriesSimilaritySearch(
            SimilaritySearchConfig()
        )
        
    def _setup_ensemble(self, **kwargs):
        """Setup ensemble of multiple classifiers"""
        self.classifiers = {
            'hivecote': HIVECOTEV2Integration(time_limit_minutes=30),
            'multirocket': MultiRocketIntegration(n_kernels=5000)
        }
        self.weights = kwargs.get('weights', {'hivecote': 0.6, 'multirocket': 0.4})
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            pattern_metadata: Optional[List[Dict]] = None):
        """
        Fit the classifier
        
        Args:
            X: Training time series data
            y: Training labels
            pattern_metadata: Optional metadata for each pattern
        """
        logger.info(f"Training {self.method} classifier on {len(X)} samples")
        
        if self.method == 'ensemble':
            for name, clf in self.classifiers.items():
                logger.info(f"Training {name}...")
                clf.fit(X, y)
        else:
            self.classifier.fit(X, y)
            
        # Index patterns for similarity search
        if pattern_metadata:
            patterns = [
                {'data': X[i], 'metadata': pattern_metadata[i]}
                for i in range(len(X))
            ]
            self.similarity_search.index_patterns(patterns)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.method == 'ensemble':
            # Weighted ensemble prediction
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
        else:
            return self.classifier.predict(X)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.method == 'ensemble':
            # Weighted average of probabilities
            all_probas = []
            for name, clf in self.classifiers.items():
                probas = clf.predict_proba(X)
                all_probas.append(probas * self.weights[name])
                
            return np.sum(all_probas, axis=0) / sum(self.weights.values())
        else:
            return self.classifier.predict_proba(X)
            
    def find_similar_patterns(self, query: np.ndarray, 
                            filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """
        Find similar patterns to a query
        
        Args:
            query: Query time series
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of similar patterns
        """
        return self.similarity_search.search(query, filter_criteria)
