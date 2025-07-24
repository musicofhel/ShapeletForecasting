"""
Feature Engineering Pipeline

This module provides a unified pipeline for feature extraction, transformation,
and scaling for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path
import pickle
import h5py
from datetime import datetime

# Import feature components
from .pattern_feature_extractor import PatternFeatureExtractor
from .technical_indicators import TechnicalIndicators
from .transition_matrix import TransitionMatrixBuilder

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Unified feature engineering pipeline for financial time series.
    
    Combines:
    - Pattern-based features (wavelets, DTW, clusters)
    - Technical indicators
    - Transition matrix features
    - Feature scaling and transformation
    """
    
    def __init__(self,
                 # Pattern feature parameters
                 use_pattern_features: bool = True,
                 wavelet: str = 'morl',
                 n_patterns: int = 20,
                 pattern_length: int = 30,
                 
                 # Technical indicator parameters
                 use_technical_indicators: bool = True,
                 price_col: str = 'close',
                 
                 # Transition matrix parameters
                 use_transition_features: bool = True,
                 max_transition_order: int = 2,
                 
                 # Scaling parameters
                 scaler_type: str = 'standard',
                 handle_missing: str = 'mean',
                 
                 # General parameters
                 feature_window: int = 252,  # 1 year of trading days
                 prediction_horizon: int = 5):  # 5 days ahead
        """
        Initialize feature pipeline.
        
        Parameters:
        -----------
        use_pattern_features : bool
            Whether to include pattern-based features
        wavelet : str
            Wavelet type for pattern extraction
        n_patterns : int
            Number of reference patterns
        pattern_length : int
            Length of patterns
        use_technical_indicators : bool
            Whether to include technical indicators
        price_col : str
            Column name for price data
        use_transition_features : bool
            Whether to include transition matrix features
        max_transition_order : int
            Maximum order for transition matrices
        scaler_type : str
            Type of scaler ('standard', 'robust', 'minmax')
        handle_missing : str
            How to handle missing values ('mean', 'median', 'drop')
        feature_window : int
            Window size for feature extraction
        prediction_horizon : int
            Prediction horizon for target creation
        """
        self.use_pattern_features = use_pattern_features
        self.use_technical_indicators = use_technical_indicators
        self.use_transition_features = use_transition_features
        
        self.feature_window = feature_window
        self.prediction_horizon = prediction_horizon
        self.price_col = price_col
        
        # Initialize components
        if use_pattern_features:
            self.pattern_extractor = PatternFeatureExtractor(
                wavelet=wavelet,
                n_patterns=n_patterns,
                pattern_length=pattern_length
            )
        
        if use_technical_indicators:
            self.technical_calculator = TechnicalIndicators(
                price_col=price_col
            )
            
        if use_transition_features:
            self.transition_builder = TransitionMatrixBuilder(
                n_patterns=n_patterns,
                pattern_length=pattern_length,
                max_order=max_transition_order
            )
            
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        # Initialize imputer
        if handle_missing == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif handle_missing == 'median':
            self.imputer = SimpleImputer(strategy='median')
        else:
            self.imputer = None
            
        self.handle_missing = handle_missing
        
        # Storage for fitted state
        self.is_fitted = False
        self.feature_names = None
        self.selected_features = None
        
    def fit(self, df: pd.DataFrame, 
            target_col: Optional[str] = None) -> 'FeaturePipeline':
        """
        Fit the feature pipeline on training data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with OHLCV columns
        target_col : str, optional
            Target column for supervised feature extraction
            
        Returns:
        --------
        self : FeaturePipeline
        """
        logger.info("Fitting feature pipeline...")
        
        # Extract windows for pattern learning
        if self.use_pattern_features or self.use_transition_features:
            windows, labels = self._extract_windows(df, target_col)
            
            if self.use_pattern_features:
                # Fit pattern extractor
                self.pattern_extractor.fit(windows, labels)
                
            if self.use_transition_features:
                # Extract pattern sequences
                sequences = self._extract_pattern_sequences(windows)
                conditions = self._extract_market_conditions(df)
                
                # Fit transition builder
                self.transition_builder.fit(sequences, conditions)
                
        # Compute all features for scaling
        all_features = self.transform(df, fit_scaler=True)
        
        self.is_fitted = True
        logger.info(f"Feature pipeline fitted with {len(self.feature_names)} features")
        
        return self
        
    def transform(self, df: pd.DataFrame, 
                 fit_scaler: bool = False) -> pd.DataFrame:
        """
        Transform data into features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
        fit_scaler : bool
            Whether to fit the scaler (only during training)
            
        Returns:
        --------
        features : pd.DataFrame
            Extracted and scaled features
        """
        if not fit_scaler and not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        features_list = []
        feature_names_list = []
        
        # 1. Technical indicators
        if self.use_technical_indicators:
            tech_features = self.technical_calculator.compute_all_indicators(df)
            features_list.append(tech_features)
            feature_names_list.extend(tech_features.columns.tolist())
            
        # 2. Pattern-based features
        if self.use_pattern_features:
            pattern_features = self._compute_pattern_features(df)
            features_list.append(pattern_features)
            feature_names_list.extend(pattern_features.columns.tolist())
            
        # 3. Transition features
        if self.use_transition_features:
            transition_features = self._compute_transition_features(df)
            features_list.append(transition_features)
            feature_names_list.extend(transition_features.columns.tolist())
            
        # 4. Price-based features
        price_features = self._compute_price_features(df)
        features_list.append(price_features)
        feature_names_list.extend(price_features.columns.tolist())
        
        # 5. Time-based features
        time_features = self._compute_time_features(df)
        features_list.append(time_features)
        feature_names_list.extend(time_features.columns.tolist())
        
        # Combine all features
        all_features = pd.concat(features_list, axis=1)
        
        # Store feature names
        if fit_scaler:
            self.feature_names = feature_names_list
            
        # Handle missing values
        if self.handle_missing == 'drop':
            all_features = all_features.dropna()
        elif self.imputer is not None:
            if fit_scaler:
                all_features.iloc[:, :] = self.imputer.fit_transform(all_features)
            else:
                all_features.iloc[:, :] = self.imputer.transform(all_features)
                
        # Scale features
        if fit_scaler:
            all_features.iloc[:, :] = self.scaler.fit_transform(all_features)
        else:
            all_features.iloc[:, :] = self.scaler.transform(all_features)
            
        return all_features
        
    def fit_transform(self, df: pd.DataFrame, 
                     target_col: Optional[str] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df)
        
    def _extract_windows(self, df: pd.DataFrame, 
                        target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract sliding windows from time series."""
        price_data = df[self.price_col].values
        
        windows = []
        labels = []
        
        for i in range(self.feature_window, len(price_data) - self.prediction_horizon):
            window = price_data[i - self.feature_window:i]
            windows.append(window)
            
            if target_col is not None and target_col in df.columns:
                # Binary classification: price goes up or down
                future_price = price_data[i + self.prediction_horizon]
                current_price = price_data[i]
                label = 1 if future_price > current_price else 0
                labels.append(label)
                
        windows = np.array(windows)
        labels = np.array(labels) if labels else None
        
        return windows, labels
        
    def _extract_pattern_sequences(self, windows: np.ndarray) -> List[List[int]]:
        """Extract pattern sequences from windows."""
        sequences = []
        
        for window in windows:
            # Divide window into patterns
            patterns = []
            pattern_length = self.pattern_extractor.pattern_length
            
            for i in range(0, len(window) - pattern_length + 1, pattern_length // 2):
                pattern = window[i:i + pattern_length]
                
                # Find closest reference pattern
                min_dist = float('inf')
                best_idx = 0
                
                for j, ref_pattern in enumerate(self.pattern_extractor.reference_patterns):
                    result = self.pattern_extractor.dtw_calculator.compute(pattern, ref_pattern)
                    if result.normalized_distance < min_dist:
                        min_dist = result.normalized_distance
                        best_idx = j
                        
                patterns.append(best_idx)
                
            sequences.append(patterns)
            
        return sequences
        
    def _extract_market_conditions(self, df: pd.DataFrame) -> Optional[List[List[str]]]:
        """Extract market conditions for conditional transitions."""
        if 'volume' not in df.columns:
            return None
            
        conditions_list = []
        
        # Simple market regime classification
        returns = df[self.price_col].pct_change()
        volatility = returns.rolling(20).std()
        volume_ma = df['volume'].rolling(20).mean()
        
        for i in range(self.feature_window, len(df) - self.prediction_horizon):
            window_conditions = []
            
            for j in range(i - self.feature_window, i):
                # Classify market condition
                if j < 20:  # Not enough data for indicators
                    condition = 'unknown'
                elif volatility.iloc[j] > volatility.quantile(0.75):
                    condition = 'high_volatility'
                elif df['volume'].iloc[j] > 2 * volume_ma.iloc[j]:
                    condition = 'high_volume'
                elif returns.iloc[j] > returns.quantile(0.8):
                    condition = 'strong_uptrend'
                elif returns.iloc[j] < returns.quantile(0.2):
                    condition = 'strong_downtrend'
                else:
                    condition = 'normal'
                    
                window_conditions.append(condition)
                
            conditions_list.append(window_conditions)
            
        return conditions_list
        
    def _compute_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute pattern-based features."""
        windows, _ = self._extract_windows(df)
        pattern_features = self.pattern_extractor.transform(windows)
        
        # Align with original dataframe index
        start_idx = self.feature_window
        end_idx = len(df) - self.prediction_horizon
        pattern_features.index = df.index[start_idx:end_idx]
        
        # Fill missing rows with NaN
        full_features = pd.DataFrame(
            index=df.index,
            columns=pattern_features.columns
        )
        full_features.loc[pattern_features.index] = pattern_features
        
        return full_features
        
    def _compute_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute transition matrix features."""
        windows, _ = self._extract_windows(df)
        sequences = self._extract_pattern_sequences(windows)
        conditions = self._extract_market_conditions(df)
        
        transition_features = self.transition_builder.transform(sequences, conditions)
        
        # Align with original dataframe index
        start_idx = self.feature_window
        end_idx = len(df) - self.prediction_horizon
        transition_features.index = df.index[start_idx:end_idx]
        
        # Fill missing rows with NaN
        full_features = pd.DataFrame(
            index=df.index,
            columns=transition_features.columns
        )
        full_features.loc[transition_features.index] = transition_features
        
        return full_features
        
    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features."""
        features = pd.DataFrame(index=df.index)
        
        close = df[self.price_col]
        high = df.get('high', close)
        low = df.get('low', close)
        open_price = df.get('open', close)
        
        # Returns
        features['returns_1d'] = close.pct_change()
        features['returns_5d'] = close.pct_change(5)
        features['returns_20d'] = close.pct_change(20)
        
        # Log returns
        features['log_returns_1d'] = np.log(close / close.shift(1))
        features['log_returns_5d'] = np.log(close / close.shift(5))
        
        # Price ratios
        features['high_low_ratio'] = high / low
        features['close_open_ratio'] = close / open_price
        
        # Price position
        features['price_position_20d'] = (close - close.rolling(20).min()) / \
                                        (close.rolling(20).max() - close.rolling(20).min())
        features['price_position_50d'] = (close - close.rolling(50).min()) / \
                                        (close.rolling(50).max() - close.rolling(50).min())
        
        # Volatility
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        features['volatility_50d'] = features['returns_1d'].rolling(50).std()
        
        # Skewness and kurtosis
        features['returns_skew_20d'] = features['returns_1d'].rolling(20).skew()
        features['returns_kurt_20d'] = features['returns_1d'].rolling(20).kurt()
        
        return features
        
    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features."""
        features = pd.DataFrame(index=df.index)
        
        # Extract datetime components
        if isinstance(df.index, pd.DatetimeIndex):
            features['day_of_week'] = df.index.dayofweek
            features['day_of_month'] = df.index.day
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            
            # Cyclical encoding
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Trading day features
            features['is_monday'] = (features['day_of_week'] == 0).astype(int)
            features['is_friday'] = (features['day_of_week'] == 4).astype(int)
            features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
            features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
            
        return features
        
    def create_target(self, df: pd.DataFrame, 
                     target_type: str = 'classification') -> pd.Series:
        """
        Create target variable for prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with price information
        target_type : str
            Type of target ('classification', 'regression')
            
        Returns:
        --------
        target : pd.Series
            Target variable
        """
        close = df[self.price_col]
        
        if target_type == 'classification':
            # Binary classification: 1 if price goes up, 0 if down
            future_returns = close.shift(-self.prediction_horizon) / close - 1
            target = (future_returns > 0).astype(int)
            target.name = f'target_{self.prediction_horizon}d_direction'
            
        elif target_type == 'regression':
            # Regression: predict future returns
            target = close.shift(-self.prediction_horizon) / close - 1
            target.name = f'target_{self.prediction_horizon}d_return'
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
        return target
        
    def save(self, filepath: str):
        """Save fitted pipeline to file."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
            
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save components
        data = {
            'config': {
                'use_pattern_features': self.use_pattern_features,
                'use_technical_indicators': self.use_technical_indicators,
                'use_transition_features': self.use_transition_features,
                'feature_window': self.feature_window,
                'prediction_horizon': self.prediction_horizon,
                'price_col': self.price_col,
                'handle_missing': self.handle_missing
            },
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'imputer': self.imputer
        }
        
        # Save main data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Save components separately
        base_path = Path(filepath).parent
        
        if self.use_pattern_features:
            self.pattern_extractor.save_reference_patterns(
                str(base_path / 'pattern_extractor.pkl')
            )
            
        if self.use_transition_features:
            self.transition_builder.save(
                str(base_path / 'transition_builder.pkl')
            )
            
        logger.info(f"Saved feature pipeline to {filepath}")
        
    def load(self, filepath: str):
        """Load fitted pipeline from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Restore configuration
        config = data['config']
        self.use_pattern_features = config['use_pattern_features']
        self.use_technical_indicators = config['use_technical_indicators']
        self.use_transition_features = config['use_transition_features']
        self.feature_window = config['feature_window']
        self.prediction_horizon = config['prediction_horizon']
        self.price_col = config['price_col']
        self.handle_missing = config['handle_missing']
        
        # Restore fitted state
        self.feature_names = data['feature_names']
        self.selected_features = data['selected_features']
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        
        # Load components
        base_path = Path(filepath).parent
        
        if self.use_pattern_features:
            self.pattern_extractor = PatternFeatureExtractor()
            self.pattern_extractor.load_reference_patterns(
                str(base_path / 'pattern_extractor.pkl')
            )
            
        if self.use_technical_indicators:
            self.technical_calculator = TechnicalIndicators(price_col=self.price_col)
            
        if self.use_transition_features:
            self.transition_builder = TransitionMatrixBuilder(n_patterns=1)  # Dummy init
            self.transition_builder.load(
                str(base_path / 'transition_builder.pkl')
            )
            
        self.is_fitted = True
        logger.info(f"Loaded feature pipeline from {filepath}")
        
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if self.feature_names is None:
            raise ValueError("Pipeline must be fitted to get feature names")
        return self.feature_names.copy()
