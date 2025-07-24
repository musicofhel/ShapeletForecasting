"""
Evaluation and Backtesting Module for Financial Wavelet Prediction

This module provides comprehensive tools for evaluating trading strategies,
performing backtests, and analyzing risk metrics.
"""

from .backtest_engine import BacktestEngine, WalkForwardBacktest
from .trading_simulator import TradingSimulator, Position, Trade
from .risk_analyzer import RiskAnalyzer, RiskMetrics
from .performance_reporter import PerformanceReporter, BacktestReport
from .market_regime_analyzer import MarketRegimeAnalyzer, MarketRegime

__all__ = [
    'BacktestEngine',
    'WalkForwardBacktest',
    'TradingSimulator',
    'Position',
    'Trade',
    'RiskAnalyzer',
    'RiskMetrics',
    'PerformanceReporter',
    'BacktestReport',
    'MarketRegimeAnalyzer',
    'MarketRegime'
]
