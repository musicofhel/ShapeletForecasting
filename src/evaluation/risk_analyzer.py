"""
Risk Analysis Module for Trading Strategies

This module provides comprehensive risk metrics and analysis tools
for evaluating trading strategy performance and risk characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    downside_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    average_drawdown: float
    recovery_factor: float
    
    # Risk metrics
    value_at_risk: float  # VaR
    conditional_value_at_risk: float  # CVaR
    beta: float
    alpha: float
    
    # Higher moments
    skewness: float
    kurtosis: float
    
    # Tail risk
    tail_ratio: float
    omega_ratio: float
    
    # Additional metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    kelly_criterion: float


class RiskAnalyzer:
    """
    Comprehensive risk analysis for trading strategies
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 periods_per_year: int = 252):
        """
        Initialize risk analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate
            confidence_level: Confidence level for VaR/CVaR
            periods_per_year: Number of trading periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.periods_per_year = periods_per_year
        
    def analyze(self, 
                returns: Union[pd.Series, np.ndarray],
                benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                trades: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Perform comprehensive risk analysis
        
        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns for relative metrics
            trades: Optional trade history for trade-based metrics
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        # Convert to numpy array if needed
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = returns[~np.isnan(returns)]
        
        # Basic return metrics
        total_return = self._calculate_total_return(returns)
        annual_return = self._annualize_return(returns)
        volatility = self._calculate_volatility(returns)
        downside_vol = self._calculate_downside_volatility(returns)
        
        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio(returns, volatility)
        sortino = self._calculate_sortino_ratio(returns, downside_vol)
        
        # Drawdown analysis
        equity_curve = self._returns_to_equity_curve(returns)
        drawdowns = self._calculate_drawdowns(equity_curve)
        max_dd = drawdowns['max_drawdown']
        max_dd_duration = drawdowns['max_duration']
        avg_dd = drawdowns['average_drawdown']
        
        calmar = self._calculate_calmar_ratio(annual_return, max_dd)
        recovery = self._calculate_recovery_factor(total_return, max_dd)
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None:
            info_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            beta, alpha = self._calculate_beta_alpha(returns, benchmark_returns)
        else:
            info_ratio = 0.0
            beta = 1.0
            alpha = 0.0
            
        # Risk metrics
        var = self._calculate_value_at_risk(returns)
        cvar = self._calculate_conditional_value_at_risk(returns)
        
        # Higher moments
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Tail risk
        tail_ratio = self._calculate_tail_ratio(returns)
        omega = self._calculate_omega_ratio(returns)
        
        # Trade-based metrics
        if trades is not None and len(trades) > 0:
            trade_metrics = self._analyze_trades(trades)
            win_rate = trade_metrics['win_rate']
            profit_factor = trade_metrics['profit_factor']
            expectancy = trade_metrics['expectancy']
            kelly = self._calculate_kelly_criterion(win_rate, trade_metrics['avg_win'], trade_metrics['avg_loss'])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            expectancy = 0.0
            kelly = 0.0
            
        return RiskMetrics(
            total_return=total_return,
            annualized_return=annual_return,
            volatility=volatility,
            downside_volatility=downside_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            average_drawdown=avg_dd,
            recovery_factor=recovery,
            value_at_risk=var,
            conditional_value_at_risk=cvar,
            beta=beta,
            alpha=alpha,
            skewness=skew,
            kurtosis=kurt,
            tail_ratio=tail_ratio,
            omega_ratio=omega,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly
        )
        
    def calculate_rolling_metrics(self,
                                returns: pd.Series,
                                window: int = 252,
                                min_periods: int = 30) -> pd.DataFrame:
        """
        Calculate rolling risk metrics
        
        Args:
            returns: Returns series with datetime index
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            DataFrame with rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window, min_periods=min_periods).apply(
            lambda x: self._calculate_total_return(x)
        )
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window, min_periods=min_periods).std() * np.sqrt(self.periods_per_year)
        
        # Rolling Sharpe
        rolling_metrics['rolling_sharpe'] = returns.rolling(window, min_periods=min_periods).apply(
            lambda x: self._calculate_sharpe_ratio(x, x.std() * np.sqrt(self.periods_per_year))
        )
        
        # Rolling maximum drawdown
        equity_curve = self._returns_to_equity_curve(returns)
        rolling_metrics['rolling_max_drawdown'] = equity_curve.rolling(window, min_periods=min_periods).apply(
            lambda x: self._calculate_drawdowns(x)['max_drawdown']
        )
        
        # Rolling VaR
        rolling_metrics['rolling_var'] = returns.rolling(window, min_periods=min_periods).apply(
            lambda x: self._calculate_value_at_risk(x)
        )
        
        return rolling_metrics
        
    def stress_test(self,
                   returns: pd.Series,
                   scenarios: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing on the strategy
        
        Args:
            returns: Historical returns
            scenarios: Custom stress scenarios (name -> return multiplier)
            
        Returns:
            Dictionary of scenario results
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                'market_crash': -0.20,  # 20% crash
                'flash_crash': -0.10,   # 10% flash crash
                'volatility_spike': 3.0,  # 3x volatility
                'correlation_breakdown': -1.0,  # Correlation flip
                'black_swan': -0.30  # 30% black swan event
            }
            
        results = {}
        
        for scenario_name, shock in scenarios.items():
            if scenario_name == 'volatility_spike':
                # Increase volatility
                stressed_returns = returns * shock
            elif scenario_name == 'correlation_breakdown':
                # Flip returns
                stressed_returns = returns * shock
            else:
                # Apply shock
                stressed_returns = returns.copy()
                stressed_returns.iloc[len(stressed_returns)//2] = shock
                
            # Calculate metrics under stress
            stress_metrics = self.analyze(stressed_returns)
            
            results[scenario_name] = {
                'total_return': stress_metrics.total_return,
                'max_drawdown': stress_metrics.max_drawdown,
                'sharpe_ratio': stress_metrics.sharpe_ratio,
                'var_95': stress_metrics.value_at_risk,
                'cvar_95': stress_metrics.conditional_value_at_risk
            }
            
        return results
        
    def calculate_risk_contribution(self,
                                  portfolio_returns: pd.DataFrame,
                                  weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate risk contribution of each asset in portfolio
        
        Args:
            portfolio_returns: DataFrame with asset returns
            weights: Portfolio weights (equal weight if None)
            
        Returns:
            Dictionary with risk contributions
        """
        n_assets = len(portfolio_returns.columns)
        
        if weights is None:
            weights = np.ones(n_assets) / n_assets
            
        # Calculate covariance matrix
        cov_matrix = portfolio_returns.cov() * self.periods_per_year
        
        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        
        # Component risk contribution
        component_contrib = weights * marginal_contrib
        
        # Percentage risk contribution
        pct_contrib = component_contrib / portfolio_vol
        
        return {
            'marginal_contribution': marginal_contrib,
            'component_contribution': component_contrib,
            'percentage_contribution': pct_contrib,
            'portfolio_volatility': portfolio_vol
        }
        
    def optimize_risk_parity(self, 
                           returns: pd.DataFrame,
                           target_vol: Optional[float] = None) -> np.ndarray:
        """
        Calculate risk parity portfolio weights
        
        Args:
            returns: Asset returns DataFrame
            target_vol: Target portfolio volatility
            
        Returns:
            Optimal weights array
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * self.periods_per_year
        
        # Objective: minimize variance of risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            return np.var(risk_contrib)
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, x0, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            
            # Scale to target volatility if specified
            if target_vol is not None:
                current_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                weights *= target_vol / current_vol
                
            return weights
        else:
            logger.warning("Risk parity optimization failed, returning equal weights")
            return x0
            
    # Private helper methods
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total cumulative return"""
        return np.prod(1 + returns) - 1
        
    def _annualize_return(self, returns: np.ndarray) -> float:
        """Annualize returns"""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        return (1 + total_return) ** (self.periods_per_year / n_periods) - 1
        
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(self.periods_per_year)
        
    def _calculate_downside_volatility(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate downside volatility"""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        return np.std(downside_returns) * np.sqrt(self.periods_per_year)
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0:
            return 0.0
        annual_return = self._annualize_return(returns)
        return (annual_return - self.risk_free_rate) / volatility
        
    def _calculate_sortino_ratio(self, returns: np.ndarray, downside_vol: float) -> float:
        """Calculate Sortino ratio"""
        if downside_vol == 0:
            return 0.0
        annual_return = self._annualize_return(returns)
        return (annual_return - self.risk_free_rate) / downside_vol
        
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0.0
        return annual_return / abs(max_drawdown)
        
    def _calculate_information_ratio(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        # Ensure both arrays have the same length
        min_len = min(len(returns), len(benchmark_returns))
        returns_aligned = returns[:min_len]
        benchmark_aligned = benchmark_returns[:min_len]
        
        active_returns = returns_aligned - benchmark_aligned
        tracking_error = np.std(active_returns) * np.sqrt(self.periods_per_year)
        if tracking_error == 0:
            return 0.0
        annual_active_return = self._annualize_return(active_returns)
        return annual_active_return / tracking_error
        
    def _calculate_beta_alpha(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha relative to benchmark"""
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 1.0
        else:
            beta = covariance / benchmark_variance
            
        # Calculate alpha
        strategy_return = self._annualize_return(returns)
        benchmark_return = self._annualize_return(benchmark_returns)
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return beta, alpha
        
    def _returns_to_equity_curve(self, returns: np.ndarray) -> np.ndarray:
        """Convert returns to equity curve"""
        return np.cumprod(1 + returns)
        
    def _calculate_drawdowns(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown statistics"""
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown series
        drawdowns = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = abs(np.min(drawdowns))
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = abs(np.mean(negative_drawdowns)) if len(negative_drawdowns) > 0 else 0
        
        # Maximum drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start + 1
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                current_duration = 0
                
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': avg_drawdown,
            'max_duration': max_duration
        }
        
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor"""
        if max_drawdown == 0:
            return np.inf if total_return > 0 else 0.0
        return total_return / max_drawdown
        
    def _calculate_value_at_risk(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, (1 - self.confidence_level) * 100)
        
    def _calculate_conditional_value_at_risk(self, returns: np.ndarray) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var = self._calculate_value_at_risk(returns)
        return np.mean(returns[returns <= var])
        
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 0.0
        return right_tail / left_tail
        
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if np.sum(losses) == 0:
            return np.inf if np.sum(gains) > 0 else 0.0
        return np.sum(gains) / np.sum(losses)
        
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze trade statistics"""
        if 'pnl' not in trades.columns:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
            
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf if total_wins > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss)
        }
        
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion for position sizing"""
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap at 25% for safety
        return min(max(kelly, 0), 0.25)
