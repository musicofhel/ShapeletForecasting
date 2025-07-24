"""
Advanced risk management system with dynamic position sizing,
portfolio heat mapping, and multi-factor risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position or portfolio."""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float


@dataclass
class PositionSizing:
    """Position sizing recommendation."""
    symbol: str
    base_size: float
    risk_adjusted_size: float
    max_position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    kelly_fraction: float


class AdvancedRiskManager:
    """
    Advanced risk management system for financial trading.
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,
                 max_position_risk: float = 0.01,
                 confidence_level: float = 0.95,
                 lookback_period: int = 252):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per period
            max_position_risk: Maximum risk per position
            confidence_level: Confidence level for VaR calculations
            lookback_period: Historical period for risk calculations
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        
        # Risk limits
        self.risk_limits = {
            'max_leverage': 3.0,
            'max_concentration': 0.3,
            'min_liquidity_ratio': 0.2,
            'max_correlation': 0.8,
            'max_var_pct': 0.05
        }
        
        # Historical data storage
        self.returns_history = {}
        self.volatility_history = {}
        self.correlation_matrix = None
        
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            
        Returns:
            VaR and CVaR values
        """
        # Parametric VaR (assuming normal distribution)
        mean = np.mean(returns)
        std = np.std(returns)
        var_param = mean - std * stats.norm.ppf(confidence)
        
        # Historical VaR
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        
        # Use more conservative estimate
        var = min(var_param, var_hist)
        
        # Conditional VaR (Expected Shortfall)
        cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
        
        return float(var), float(cvar)
    
    def calculate_risk_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Asset returns
            benchmark_returns: Benchmark returns for beta calculation
            
        Returns:
            Risk metrics
        """
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # VaR and CVaR
        var_95, cvar_95 = self.calculate_var(returns.values, 0.95)
        var_99, cvar_99 = self.calculate_var(returns.values, 0.99)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = mean_return * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Beta (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            beta = 1.0
            
        # Placeholder for other risks (would need portfolio context)
        correlation_risk = 0.0
        concentration_risk = 0.0
        liquidity_risk = 0.0
        
        return RiskMetrics(
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            cvar_99=abs(cvar_99),
            max_drawdown=abs(max_drawdown),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=beta,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk
        )
    
    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_loss_price: float,
                              portfolio_value: float,
                              confidence: float = 0.8,
                              volatility: Optional[float] = None) -> PositionSizing:
        """
        Calculate optimal position size using multiple methods.
        
        Args:
            symbol: Asset symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            portfolio_value: Total portfolio value
            confidence: Model confidence
            volatility: Asset volatility
            
        Returns:
            Position sizing recommendation
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_percentage = risk_per_share / entry_price
        
        # Fixed fractional position sizing
        max_risk_amount = portfolio_value * self.max_position_risk
        fixed_fraction_size = max_risk_amount / risk_per_share
        
        # Volatility-based sizing
        if volatility is not None and volatility > 0:
            target_volatility = 0.02  # 2% daily volatility target
            volatility_size = (target_volatility / volatility) * portfolio_value / entry_price
        else:
            volatility_size = fixed_fraction_size
            
        # Kelly criterion (simplified)
        win_rate = 0.5 + confidence * 0.3  # Convert confidence to win rate
        avg_win = 0.02  # Assumed average win
        avg_loss = risk_percentage
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1
            
        kelly_size = kelly_fraction * portfolio_value / entry_price
        
        # Combine methods (weighted average)
        weights = [0.4, 0.3, 0.3]  # Fixed, volatility, Kelly
        base_size = (
            weights[0] * fixed_fraction_size +
            weights[1] * volatility_size +
            weights[2] * kelly_size
        )
        
        # Risk adjustments
        risk_multiplier = confidence  # Reduce size for lower confidence
        risk_adjusted_size = base_size * risk_multiplier
        
        # Apply maximum position limit
        max_position_size = portfolio_value * self.risk_limits['max_concentration'] / entry_price
        final_size = min(risk_adjusted_size, max_position_size)
        
        # Calculate take profit
        risk_reward_ratio = 2.0  # Default 2:1 risk/reward
        take_profit_price = entry_price + (risk_per_share * risk_reward_ratio)
        
        return PositionSizing(
            symbol=symbol,
            base_size=float(base_size),
            risk_adjusted_size=float(final_size),
            max_position_size=float(max_position_size),
            stop_loss=float(stop_loss_price),
            take_profit=float(take_profit_price),
            risk_reward_ratio=float(risk_reward_ratio),
            kelly_fraction=float(kelly_fraction)
        )
    
    def assess_portfolio_risk(self, 
                            positions: Dict[str, Dict[str, float]],
                            returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall portfolio risk.
        
        Args:
            positions: Current positions {symbol: {'size': x, 'entry': y}}
            returns_data: Historical returns for all assets
            
        Returns:
            Portfolio risk assessment
        """
        # Calculate portfolio weights
        total_value = sum(pos['size'] * pos['entry'] for pos in positions.values())
        weights = {symbol: (pos['size'] * pos['entry']) / total_value 
                  for symbol, pos in positions.items()}
        
        # Calculate portfolio returns
        portfolio_returns = sum(returns_data[symbol] * weight 
                              for symbol, weight in weights.items() 
                              if symbol in returns_data.columns)
        
        # Calculate correlation matrix
        correlation_matrix = returns_data[list(positions.keys())].corr()
        
        # Portfolio risk metrics
        portfolio_metrics = self.calculate_risk_metrics(portfolio_returns)
        
        # Concentration risk
        concentration_risk = max(weights.values()) if weights else 0
        
        # Correlation risk
        correlation_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        avg_correlation = np.mean(correlation_values) if len(correlation_values) > 0 else 0
        max_correlation = np.max(correlation_values) if len(correlation_values) > 0 else 0
        
        # Diversification ratio
        individual_vols = [returns_data[symbol].std() for symbol in positions.keys() 
                          if symbol in returns_data.columns]
        weighted_avg_vol = sum(vol * weights[symbol] for symbol, vol in zip(positions.keys(), individual_vols))
        portfolio_vol = portfolio_returns.std()
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Risk budget utilization
        current_var = portfolio_metrics.var_95
        var_limit = self.risk_limits['max_var_pct']
        risk_utilization = current_var / var_limit
        
        assessment = {
            'portfolio_metrics': portfolio_metrics,
            'concentration_risk': concentration_risk,
            'correlation_risk': {
                'average': avg_correlation,
                'maximum': max_correlation
            },
            'diversification_ratio': diversification_ratio,
            'risk_utilization': risk_utilization,
            'risk_warnings': []
        }
        
        # Generate warnings
        if concentration_risk > self.risk_limits['max_concentration']:
            assessment['risk_warnings'].append(f"High concentration risk: {concentration_risk:.2%}")
            
        if max_correlation > self.risk_limits['max_correlation']:
            assessment['risk_warnings'].append(f"High correlation risk: {max_correlation:.2f}")
            
        if risk_utilization > 0.8:
            assessment['risk_warnings'].append(f"High risk utilization: {risk_utilization:.2%}")
            
        return assessment
    
    def calculate_risk_parity_weights(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk parity portfolio weights.
        
        Args:
            returns_data: Historical returns for assets
            
        Returns:
            Risk parity weights
        """
        # Calculate covariance matrix
        cov_matrix = returns_data.cov() * 252  # Annualized
        
        # Number of assets
        n_assets = len(returns_data.columns)
        
        # Risk parity optimization
        def risk_contribution(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights, cov_matrix):
            contrib = risk_contribution(weights, cov_matrix)
            # Minimize squared differences from equal contribution
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Return as dictionary
        if result.success:
            return dict(zip(returns_data.columns, result.x))
        else:
            logger.warning("Risk parity optimization failed, returning equal weights")
            return dict(zip(returns_data.columns, initial_weights))
    
    def generate_risk_report(self, 
                           portfolio_assessment: Dict[str, Any],
                           market_regime: Optional[str] = None) -> str:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_assessment: Portfolio risk assessment
            market_regime: Current market regime
            
        Returns:
            Formatted risk report
        """
        metrics = portfolio_assessment['portfolio_metrics']
        
        report = f"""
=== RISK MANAGEMENT REPORT ===
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO RISK METRICS:
- VaR (95%): {metrics.var_95:.2%}
- VaR (99%): {metrics.var_99:.2%}
- CVaR (95%): {metrics.cvar_95:.2%}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}

RISK ASSESSMENT:
- Concentration Risk: {portfolio_assessment['concentration_risk']:.2%}
- Correlation Risk: {portfolio_assessment['correlation_risk']['average']:.2f} (avg), {portfolio_assessment['correlation_risk']['maximum']:.2f} (max)
- Diversification Ratio: {portfolio_assessment['diversification_ratio']:.2f}
- Risk Budget Utilization: {portfolio_assessment['risk_utilization']:.2%}
"""
        
        if market_regime:
            report += f"\nMARKET REGIME: {market_regime}\n"
            
        if portfolio_assessment['risk_warnings']:
            report += "\nRISK WARNINGS:\n"
            for warning in portfolio_assessment['risk_warnings']:
                report += f"⚠️  {warning}\n"
                
        return report
    
    def calculate_stress_scenarios(self, 
                                 positions: Dict[str, Dict[str, float]],
                                 returns_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio impact under stress scenarios.
        
        Args:
            positions: Current positions
            returns_data: Historical returns
            
        Returns:
            Stress test results
        """
        scenarios = {}
        
        # Historical worst days
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        worst_days = portfolio_returns.nsmallest(5)
        scenarios['historical_worst'] = float(worst_days.mean())
        
        # Market crash scenario (-20% equity, -10% bonds)
        crash_impact = 0
        for symbol, pos in positions.items():
            weight = (pos['size'] * pos['entry']) / sum(p['size'] * p['entry'] for p in positions.values())
            # Simplified: assume all positions are equity
            crash_impact += weight * (-0.20)
        scenarios['market_crash'] = crash_impact
        
        # High volatility scenario (2x normal volatility)
        normal_vol = portfolio_returns.std()
        scenarios['high_volatility'] = -2 * normal_vol * 2  # 2 std dev move with 2x vol
        
        # Correlation breakdown (all correlations go to 1)
        individual_losses = [returns_data[symbol].min() for symbol in positions.keys() 
                           if symbol in returns_data.columns]
        scenarios['correlation_breakdown'] = np.mean(individual_losses) if individual_losses else 0
        
        return scenarios
    
    def _calculate_portfolio_returns(self, 
                                   positions: Dict[str, Dict[str, float]],
                                   returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from positions and returns data."""
        total_value = sum(pos['size'] * pos['entry'] for pos in positions.values())
        weights = {symbol: (pos['size'] * pos['entry']) / total_value 
                  for symbol, pos in positions.items()}
        
        portfolio_returns = sum(returns_data[symbol] * weight 
                              for symbol, weight in weights.items() 
                              if symbol in returns_data.columns)
        
        return portfolio_returns
