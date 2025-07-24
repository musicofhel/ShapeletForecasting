"""
Pattern Search Module for Financial Dashboard

This module provides comprehensive pattern search capabilities.
"""

from .pattern_search import (
    # Core classes
    Pattern,
    PatternMatch,
    BacktestResult,
    PatternAlert,
    PatternLibrary,
    PatternSearchEngine,
    
    # Convenience functions
    quick_pattern_upload,
    quick_pattern_search,
    quick_pattern_backtest
)

__all__ = [
    'Pattern',
    'PatternMatch',
    'BacktestResult',
    'PatternAlert',
    'PatternLibrary',
    'PatternSearchEngine',
    'quick_pattern_upload',
    'quick_pattern_search',
    'quick_pattern_backtest'
]
