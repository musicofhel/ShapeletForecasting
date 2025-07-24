"""
Tests for Forecast Backtester
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.dashboard.evaluation.forecast_backtester import (
    ForecastBacktester, 
    BacktestResult, 
    WalkForwardResult
)


class TestBacktestResult:
    """Test suite for BacktestResult dataclass"""
    
    def test_backtest_result_creation(self):
        """Test basic BacktestResult creation"""
        result = BacktestResult(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=-0.01,
            profit_factor=2.0,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            total_return=0.15,
            annualized_return=0.12,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            pattern_accuracy={'head_and_shoulders': 0.75},
            forecast_accuracy=0.65
        )
        
        assert result.total_trades == 100
        assert result.win_rate == 0.6
        assert result.pattern_accuracy['head_and_shoulders'] == 0.75


class TestWalkForwardResult:
    """Test suite for WalkForwardResult dataclass"""
    
    def test_walk_forward_result_creation(self):
        """Test basic WalkForwardResult creation"""
        result = WalkForwardResult(
            window_start=datetime(2020, 1, 1),
            window_end=datetime(2020, 6, 30),
            training_accuracy=0.70,
            testing_accuracy=0.65,
            pattern_performance={'head_and_shoulders': {'accuracy': 0.75}},
            market_regime="bull",
            volatility_regime="normal"
        )
        
        assert result.training_accuracy == 0.70
        assert result.testing_accuracy == 0.65
        assert result.market_regime == "bull"


class TestForecastBacktester:
    """Test suite for ForecastBacktester"""
    
    @pytest.fixture
    def backtester(self):
        """Create ForecastBacktester instance"""
        return ForecastBacktester(
            lookback_days=252,
            forecast_horizon=5,
            min_pattern_occurrences=5
        )
    
    def test_initialization(self, backtester):
        """Test backtester initialization"""
        assert backtester.lookback_days == 252
        assert backtester.forecast_horizon == 5
        assert backtester.min_pattern_occurrences == 5
        assert backtester.classifier is not None
        assert backtester.predictor is not None
        assert backtester.analyzer is not None
    
    def test_empty_backtest_result(self, backtester):
        """Test empty backtest result handling"""
        result = backtester._empty_backtest_result()
        
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.total_return == 0.0
    
    def test_calculate_max_drawdown(self, backtester):
        """Test max drawdown calculation"""
        returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01]
        drawdown = backtester._calculate_max_drawdown(returns)
        
        assert isinstance(drawdown, float)
        assert drawdown >= 0
    
    def test_calculate_sharpe_ratio(self, backtester):
        """Test Sharpe ratio calculation"""
        returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01]
        sharpe = backtester._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
    
    def test_calculate_consecutive_stats(self, backtester):
        """Test consecutive wins/losses calculation"""
        returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.01, 0.01]
        max_wins, max_losses = backtester._calculate_consecutive_stats(returns)
        
        assert isinstance(max_wins, int)
        assert isinstance(max_losses, int)
        assert max_wins >= 0
        assert max_losses >= 0
    
    def test_calculate_trend(self, backtester):
        """Test trend calculation"""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 30),
            'returns': np.random.normal(0.001, 0.02, 30),
            'volume': np.random.normal(1000000, 100000, 30)
        }, index=dates)
        
        trend = backtester._calculate_trend(data)
        
        assert trend in ["upward", "downward", "sideways"]
    
    def test_determine_market_regime(self, backtester):
        """Test market regime determination"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 100),
            'returns': np.random.normal(0.001, 0.02, 100),
            'volume': np.random.normal(1000000, 100000, 100)
        }, index=dates)
        
        regime = backtester._determine_market_regime(data)
        
        assert regime in ["high_volatility", "low_volatility", "normal_volatility", "neutral"]
    
    def test_determine_volatility_regime(self, backtester):
        """Test volatility regime determination"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 100),
            'returns': np.random.normal(0.001, 0.02, 100),
            'volume': np.random.normal(1000000, 100000, 100)
        }, index=dates)
        
        regime = backtester._determine_volatility_regime(data)
        
        assert regime in ["elevated", "compressed", "normal"]
    
    def test_calculate_pattern_accuracy(self, backtester):
        """Test pattern accuracy calculation"""
        forecasts = [
            {
                'pattern': {'type': 'head_and_shoulders'},
                'expected_return': 0.02,
                'actual_return': 0.015
            },
            {
                'pattern': {'type': 'head_and_shoulders'},
                'expected_return': 0.01,
                'actual_return': 0.008
            },
            {
                'pattern': {'type': 'triangle'},
                'expected_return': 0.015,
                'actual_return': -0.005
            }
        ]
        
        accuracy = backtester._calculate_pattern_accuracy(forecasts)
        
        assert isinstance(accuracy, dict)
        assert 'head_and_shoulders' in accuracy
        assert 'triangle' in accuracy
    
    def test_calculate_pattern_accuracy_with_min_occurrences(self, backtester):
        """Test pattern accuracy with minimum occurrences filter"""
        forecasts = [
            {
                'pattern': {'type': 'head_and_shoulders'},
                'expected_return': 0.02,
                'actual_return': 0.015
            }
        ]
        
        accuracy = backtester._calculate_pattern_accuracy(forecasts)
        
        # Should be empty due to min_occurrences filter
        assert accuracy == {}
    
    def test_generate_recommendations(self, backtester):
        """Test recommendation generation"""
        result = BacktestResult(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            total_trades=100,
            winning_trades=70,
            losing_trades=30,
            win_rate=0.7,
            avg_win=0.02,
            avg_loss=-0.01,
            profit_factor=2.0,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            total_return=0.15,
            annualized_return=0.12,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            pattern_accuracy={'head_and_shoulders': 0.75},
            forecast_accuracy=0.65
        )
        
        walk_forward = [
            WalkForwardResult(
                window_start=datetime(2020, 1, 1),
                window_end=datetime(2020, 6, 30),
                training_accuracy=0.70,
                testing_accuracy=0.65,
                pattern_performance={'head_and_shoulders': {'accuracy': 0.75}},
                market_regime="bull",
                volatility_regime="normal"
            )
        ]
        
        recommendations = backtester._generate_recommendations(result, walk_forward)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_create_performance_dashboard(self, backtester):
        """Test performance dashboard creation"""
        result = BacktestResult(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=-0.01,
            profit_factor=2.0,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            total_return=0.15,
            annualized_return=0.12,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            pattern_accuracy={'head_and_shoulders': 0.75},
            forecast_accuracy=0.65
        )
        
        walk_forward = [
            WalkForwardResult(
                window_start=datetime(2020, 1, 1),
                window_end=datetime(2020, 6, 30),
                training_accuracy=0.70,
                testing_accuracy=0.65,
                pattern_performance={'head_and_shoulders': {'accuracy': 0.75}},
                market_regime="bull",
                volatility_regime="normal"
            )
        ]
        
        fig = backtester.create_performance_dashboard(result, walk_forward)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_generate_backtest_report(self, backtester):
        """Test backtest report generation"""
        result = BacktestResult(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=-0.01,
            profit_factor=2.0,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            total_return=0.15,
            annualized_return=0.12,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            pattern_accuracy={'head_and_shoulders': 0.75},
            forecast_accuracy=0.65
        )
        
        walk_forward = [
            WalkForwardResult(
                window_start=datetime(2020, 1, 1),
                window_end=datetime(2020, 6, 30),
                training_accuracy=0.70,
                testing_accuracy=0.65,
                pattern_performance={'head_and_shoulders': {'accuracy': 0.75}},
                market_regime="bull",
                volatility_regime="normal"
            )
        ]
        
        report = backtester.generate_backtest_report("AAPL", result, walk_forward)
        
        assert isinstance(report, dict)
        assert 'ticker' in report
        assert 'performance_summary' in report
        assert 'recommendations' in report


class TestForecastBacktesterIntegration:
    """Integration tests for ForecastBacktester"""
    
    def test_full_workflow(self):
        """Test complete backtesting workflow"""
        backtester = ForecastBacktester()
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(100, 150, 500) + np.random.normal(0, 5, 500),
            'returns': np.random.normal(0.001, 0.02, 500),
            'volume': np.random.normal(1000000, 100000, 500)
        }, index=dates)
        
        # Mock the data loading
        def mock_load_financial_data(ticker, start, end):
            return data[(data.index >= start) & (data.index <= end)]
        
        # Monkey patch for testing
        import src.dashboard.evaluation.forecast_backtester as fb
        fb.load_financial_data = mock_load_financial_data
        
        # Run backtest
        result = backtester.run_backtest("TEST", dates[0], dates[-1])
        
        assert isinstance(result, BacktestResult)
        
        # Run walk-forward
        walk_forward = backtester.run_walk_forward_analysis(
            "TEST", dates[0], dates[-1], window_size=100, step_size=50
        )
        
        assert isinstance(walk_forward, list)
        
        # Generate report
        report = backtester.generate_backtest_report("TEST", result, walk_forward)
        
        assert isinstance(report, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
