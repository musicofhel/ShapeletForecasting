"""
Wavelet Analysis Module

This module provides tools for wavelet analysis of financial time series data,
including:
- Continuous Wavelet Transform (CWT) analysis
- Motif discovery for recurring patterns
- Shapelet extraction for discriminative subsequences
- Pattern visualization tools
"""

from .wavelet_analyzer import WaveletAnalyzer
from .motif_discovery import MotifDiscovery
from .shapelet_extractor import ShapeletExtractor
from .pattern_visualizer import PatternVisualizer

__all__ = [
    'WaveletAnalyzer',
    'MotifDiscovery',
    'ShapeletExtractor',
    'PatternVisualizer'
]
