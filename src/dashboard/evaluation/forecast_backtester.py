"""
Forecast Backtester for Wavelet Pattern Forecasting Dashboard
Provides comprehensive backtesting capabilities for pattern-based forecasts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from ..pattern_classifier import PatternClassifier
from ..pattern_predictor import PatternPredictor
from ..wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from ..data_utils import data_manager


@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_return: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    annualized_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    pattern_accuracy: Dict[str, float]
    forecast_accuracy: float


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    window_start: datetime
    window_end: datetime
    training_accuracy: float
    testing_accuracy: float
    pattern_performance: Dict[str, Dict[str, float]]
    market_regime: str
    volatility_regime: str


class ForecastBacktester:
    """Comprehensive backtesting engine for pattern forecasts"""
    
    def __init__(self, 
                 lookback_days: int = 252,
                 forecast_horizon: int = 5,
                 min_pattern_occurrences: int = 5):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.min_pattern_occurrences = min_pattern_occurrences
        
        self.classifier = PatternClassifier()
        self.predictor = PatternPredictor()
        self.analyzer = WaveletSequenceAnalyzer()
        
    def run_backtest(self,
                    ticker: str,
                    start_date: datetime,
                    end_date: datetime,
                    pattern_types: List[str] = None) -> BacktestResult:
        """Run complete backtest for given ticker and date range"""
        
        # Load data using data_manager
        # Convert datetime to period string
        days_diff = (end_date - start_date).days
        if days_diff <= 7:
            period = "5d"
        elif days_diff <= 30:
            period = "1mo"
        elif days_diff <= 90:
            period = "3mo"
        elif days_diff <= 180:
            period = "6mo"
        else:
            period = "1y"
            
        data = data_manager.download_data(ticker, period=period)
        if data is None:
            return self._empty_backtest_result()
        
        # Extract patterns
        patterns = self.analyzer.extract_patterns(data)
        
        # Filter patterns if specified
        if pattern_types:
            patterns = [p for p in patterns if p['type'] in pattern_types]
        
        # Generate forecasts
        forecasts = self._generate_forecasts(patterns, data)
        
        # Evaluate performance
        results = self._evaluate_performance(forecasts, data)
        
        return results
    
    def _generate_forecasts(self, 
                         patterns: List[Dict], 
                         data: pd.DataFrame) -> List[Dict]:
        """Generate forecasts for extracted patterns"""
        
        forecasts = []
        
        for pattern in patterns:
            # Get market context
            pattern_end = pattern['end_date']
            context_data = data[data.index <= pattern_end].tail(50)
            
            market_context = {
                'trend': self._calculate_trend(context_data),
                'volatility': context_data['returns'].std() * np.sqrt(252),
                'volume': context_data['volume'].mean()
            }
            
            # Generate prediction
            prediction = self.predictor.predict_next_patterns(
                [pattern['type']],
                market_context
            )
            
            if prediction:
                forecast = {
                    'pattern': pattern,
                    'prediction': prediction[0],
                    'expected_return': prediction[0]['expected_return'],
                    'confidence': prediction[0]['confidence'],
                    'forecast_date': pattern_end,
                    'actual_return': self._calculate_actual_return(
                        data, pattern_end, self.forecast_horizon
                    )
                }
                forecasts.append(forecast)
        
        return forecasts
    
    def _calculate_actual_return(self,
                              data: pd.DataFrame,
                              start_date: datetime,
                              horizon: int) -> float:
        """Calculate actual return over forecast horizon"""
        
        start_idx = data.index.get_loc(start_date)
        end_idx = min(start_idx + horizon, len(data) - 1)
        
        if start_idx >= len(data) - 1:
            return 0.0
        
        start_price = data.iloc[start_idx]['close']
        end_price = data.iloc[end_idx]['close']
        
        return (end_price - start_price) / start_price
    
    def _evaluate_performance(self,
                         forecasts: List[Dict],
                         data: pd.DataFrame) -> BacktestResult:
        """Evaluate forecast performance"""
        
        if not forecasts:
            return self._empty_backtest_result()
        
        # Calculate metrics
        total_trades = len(forecasts)
        winning_trades = sum(1 for f in forecasts if f['actual_return'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        returns = [f['actual_return'] for f in forecasts]
        expected_returns = [f['expected_return'] for f in forecasts]
        
        # Performance metrics
        total_return = sum(returns)
        annualized_return = total_return * (252 / self.forecast_horizon) / total_trades
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Win/loss metrics
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Pattern accuracy
        pattern_accuracy = self._calculate_pattern_accuracy(forecasts)
        
        # Forecast accuracy
        forecast_accuracy = accuracy_score(
            [r > 0 for r in returns],
            [e > 0 for e in expected_returns]
        )
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(returns)
        
        # Calculate average return
        avg_return = np.mean(returns) if returns else 0.0
        
        return BacktestResult(
            start_date=min(f['forecast_date'] for f in forecasts),
            end_date=max(f['forecast_date'] for f in forecasts),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_return=avg_return,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            pattern_accuracy=pattern_accuracy,
            forecast_accuracy=forecast_accuracy
        )
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252 / self.forecast_horizon)
    
    def _calculate_consecutive_stats(self, returns: List[float]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not returns:
            return 0, 0
        
        wins = [r > 0 for r in returns]
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for win in wins:
            if win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_pattern_accuracy(self, forecasts: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy by pattern type"""
        pattern_groups = {}
        
        for forecast in forecasts:
            pattern_type = forecast['pattern']['type']
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            
            predicted = forecast['expected_return'] > 0
            actual = forecast['actual_return'] > 0
            pattern_groups[pattern_type].append(predicted == actual)
        
        return {
            pattern_type: np.mean(accuracies)
            for pattern_type, accuracies in pattern_groups.items()
            if len(accuracies) >= self.min_pattern_occurrences
        }
    
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """Calculate market trend from data"""
        if len(data) < 20:
            return "neutral"
        
        returns = data['returns'].values
        slope = np.polyfit(range(len(returns)), returns, 1)[0]
        
        if slope > 0.001:
            return "upward"
        elif slope < -0.001:
            return "downward"
        else:
            return "sideways"
    
    def _empty_backtest_result(self) -> BacktestResult:
        """Return empty backtest result"""
        return BacktestResult(
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_return=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            total_return=0.0,
            annualized_return=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            pattern_accuracy={},
            forecast_accuracy=0.0
        )
    
    def run_walk_forward_analysis(self,
                               ticker: str,
                               start_date: datetime,
                               end_date: datetime,
                               window_size: int = 252,
                               step_size: int = 63) -> List[WalkForwardResult]:
        """Run walk-forward analysis"""
        
        # Load data using data_manager
        days_diff = (end_date - start_date).days
        if days_diff <= 7:
            period = "5d"
        elif days_diff <= 30:
            period = "1mo"
        elif days_diff <= 90:
            period = "3mo"
        elif days_diff <= 180:
            period = "6mo"
        else:
            period = "1y"
            
        data = data_manager.download_data(ticker, period=period)
        if data is None:
            return []
        results = []
        
        # Create windows
        current_start = start_date
        while current_start + timedelta(days=window_size) <= end_date:
            window_end = current_start + timedelta(days=window_size)
            
            # Training period
            training_data = data[data.index < window_end - timedelta(days=step_size)]
            
            # Testing period
            testing_data = data[
                (data.index >= window_end - timedelta(days=step_size)) &
                (data.index < window_end)
            ]
            
            if len(training_data) < 50 or len(testing_data) < 10:
                current_start += timedelta(days=step_size)
                continue
            
            # Run backtest on training data
            training_result = self.run_backtest(
                ticker,
                training_data.index[0],
                training_data.index[-1]
            )
            
            # Run backtest on testing data
            testing_result = self.run_backtest(
                ticker,
                testing_data.index[0],
                testing_data.index[-1]
            )
            
            # Determine market regime
            market_regime = self._determine_market_regime(training_data)
            volatility_regime = self._determine_volatility_regime(training_data)
            
            # Extract pattern performance
            pattern_performance = self._extract_pattern_performance(training_result)
            
            result = WalkForwardResult(
                window_start=current_start,
                window_end=window_end,
                training_accuracy=training_result.forecast_accuracy,
                testing_accuracy=testing_result.forecast_accuracy,
                pattern_performance=pattern_performance,
                market_regime=market_regime,
                volatility_regime=volatility_regime
            )
            
            results.append(result)
            current_start += timedelta(days=step_size)
        
        return results
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Determine current market regime"""
        if len(data) < 50:
            return "neutral"
        
        returns = data['returns'].values
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility > 0.25:
            return "high_volatility"
        elif volatility < 0.10:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def _determine_volatility_regime(self, data: pd.DataFrame) -> str:
        """Determine volatility regime"""
        returns = data['returns'].values
        current_vol = np.std(returns[-20:]) * np.sqrt(252)
        historical_vol = np.std(returns) * np.sqrt(252)
        
        if current_vol > historical_vol * 1.5:
            return "elevated"
        elif current_vol < historical_vol * 0.7:
            return "compressed"
        else:
            return "normal"
    
    def _extract_pattern_performance(self, result: BacktestResult) -> Dict[str, Dict[str, float]]:
        """Extract pattern-specific performance metrics"""
        return {
            pattern_type: {
                'accuracy': accuracy,
                'win_rate': result.win_rate,
                'avg_return': result.avg_return
            }
            for pattern_type, accuracy in result.pattern_accuracy.items()
        }
    
    def create_performance_dashboard(self,
                                   backtest_result: BacktestResult,
                                   walk_forward_results: List[WalkForwardResult]) -> go.Figure:
        """Create comprehensive performance dashboard"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Win/Loss Distribution',
                'Pattern Accuracy', 'Walk-Forward Performance',
                'Risk Metrics', 'Pattern Performance'
            ),
            specs=[[{"secondary_y": False}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Cumulative returns
        returns = [0.01] * backtest_result.total_trades  # Placeholder
        cumulative = np.cumsum(returns)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative))),
                y=cumulative,
                mode='lines',
                name='Cumulative Returns'
            ),
            row=1, col=1
        )
        
        # Win/Loss distribution
        wins = [0.02] * backtest_result.winning_trades
        losses = [-0.01] * backtest_result.losing_trades
        fig.add_trace(
            go.Box(y=wins + losses, name='Returns'),
            row=1, col=2
        )
        
        # Pattern accuracy
        if backtest_result.pattern_accuracy:
            patterns = list(backtest_result.pattern_accuracy.keys())
            accuracies = list(backtest_result.pattern_accuracy.values())
            fig.add_trace(
                go.Bar(x=patterns, y=accuracies, name='Pattern Accuracy'),
                row=2, col=1
            )
        
        # Walk-forward performance
        if walk_forward_results:
            dates = [r.window_end for r in walk_forward_results]
            accuracies = [r.testing_accuracy for r in walk_forward_results]
            fig.add_trace(
                go.Scatter(x=dates, y=accuracies, mode='lines+markers',
                          name='Walk-Forward Accuracy'),
                row=2, col=2
            )
        
        # Risk metrics
        risk_metrics = ['Max Drawdown', 'Sharpe Ratio', 'Win Rate']
        risk_values = [
            backtest_result.max_drawdown,
            backtest_result.sharpe_ratio,
            backtest_result.win_rate
        ]
        fig.add_trace(
            go.Bar(x=risk_metrics, y=risk_values, name='Risk Metrics'),
            row=3, col=1
        )
        
        # Pattern performance
        if backtest_result.pattern_accuracy:
            patterns = list(backtest_result.pattern_accuracy.keys())
            performance = [backtest_result.pattern_accuracy[p] for p in patterns]
            fig.add_trace(
                go.Bar(x=patterns, y=performance, name='Performance'),
                row=3, col=2
            )
        
        fig.update_layout(
            height=900,
            title_text="Forecast Backtesting Performance Dashboard",
            showlegend=False
        )
        
        return fig
    
    def generate_backtest_report(self,
                               ticker: str,
                               backtest_result: BacktestResult,
                               walk_forward_results: List[WalkForwardResult]) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        
        report = {
            'ticker': ticker,
            'backtest_period': {
                'start': backtest_result.start_date.isoformat(),
                'end': backtest_result.end_date.isoformat(),
                'total_days': (backtest_result.end_date - backtest_result.start_date).days
            },
            'performance_summary': {
                'total_trades': backtest_result.total_trades,
                'win_rate': backtest_result.win_rate,
                'profit_factor': backtest_result.profit_factor,
                'total_return': backtest_result.total_return,
                'annualized_return': backtest_result.annualized_return,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown
            },
            'pattern_analysis': {
                'pattern_accuracy': backtest_result.pattern_accuracy,
                'forecast_accuracy': backtest_result.forecast_accuracy,
                'best_performing_patterns': dict(
                    sorted(backtest_result.pattern_accuracy.items(), 
                          key=lambda x: x[1], reverse=True)[:3]
                )
            },
            'walk_forward_analysis': [
                {
                    'window': f"{r.window_start.strftime('%Y-%m-%d')} to {r.window_end.strftime('%Y-%m-%d')}",
                    'training_accuracy': r.training_accuracy,
                    'testing_accuracy': r.testing_accuracy,
                    'market_regime': r.market_regime,
                    'volatility_regime': r.volatility_regime
                }
                for r in walk_forward_results
            ],
            'recommendations': self._generate_recommendations(backtest_result, walk_forward_results)
        }
        
        return report
    
    def _generate_recommendations(self,
                               backtest_result: BacktestResult,
                               walk_forward_results: List[WalkForwardResult]) -> List[str]:
        """Generate trading recommendations based on backtest results"""
        
        recommendations = []
        
        if backtest_result.win_rate > 0.6:
            recommendations.append("High win rate suggests strategy is effective")
        
        if backtest_result.sharpe_ratio > 1.0:
            recommendations.append("Good risk-adjusted returns (Sharpe > 1.0)")
        
        if backtest_result.max_drawdown < 0.1:
            recommendations.append("Low maximum drawdown indicates good risk management")
        
        if backtest_result.pattern_accuracy:
            best_pattern = max(backtest_result.pattern_accuracy.items(), 
                             key=lambda x: x[1])
            recommendations.append(f"Focus on {best_pattern[0]} patterns (accuracy: {best_pattern[1]:.2%})")
        
        if walk_forward_results:
            avg_testing_accuracy = np.mean([r.testing_accuracy for r in walk_forward_results])
            if avg_testing_accuracy > 0.65:
                recommendations.append("Walk-forward analysis confirms strategy robustness")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    backtester = ForecastBacktester()
    
    # Run backtest
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    result = backtester.run_backtest("AAPL", start_date, end_date)
    
    # Run walk-forward analysis
    walk_forward = backtester.run_walk_forward_analysis(
        "AAPL", start_date, end_date
    )
    
    # Create dashboard
    fig = backtester.create_performance_dashboard(result, walk_forward)
    
    # Generate report
    report = backtester.generate_backtest_report("AAPL", result, walk_forward)
    
    print("Backtest Report:")
    print(json.dumps(report, indent=2, default=str))
    
    fig.show()
