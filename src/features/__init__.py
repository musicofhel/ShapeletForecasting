"""
Feature Engineering Module for Financial Wavelet Prediction

This module provides comprehensive feature extraction and engineering capabilities
for financial time series analysis, including:
- Pattern-based features from wavelets and DTW
- Technical indicators
- Feature transformation and scaling
- Feature selection and importance analysis
"""

from .pattern_feature_extractor import PatternFeatureExtractor
from .technical_indicators import TechnicalIndicators
from .feature_pipeline import FeaturePipeline
from .feature_selector import FeatureSelector
from .transition_matrix import TransitionMatrixBuilder

__all__ = [
    'PatternFeatureExtractor',
    'TechnicalIndicators',
    'FeaturePipeline',
    'FeatureSelector',
    'TransitionMatrixBuilder'
]

__version__ = '0.1.0'
