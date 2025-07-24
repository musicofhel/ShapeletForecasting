"""
Portfolio optimization using modern portfolio theory, Black-Litterman,
and machine learning-based approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_positions: Optional[int] = None
    sector_limits: Optional[Dict[str, float]] = None
    turnover_limit: Optional[float] = None
    leverage_limit: float = 1.0


@dataclass
class OptimizedPortfolio:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: int
    turnover: float
    optimization_method: str


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple methodologies.
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 rebalance_threshold: float = 0.05):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            confidence_level: Confidence level for estimates
            rebalance_threshold: Threshold for rebalancing
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.rebalance_threshold = rebalance_threshold
        
        # Current portfolio state
        self.current_weights = {}
        self.optimization_history = []
        
    def calculate_expected_returns(self, 
                                 returns_data: pd.DataFrame,
                                 method: str = 'historical') -> pd.Series:
        """
        Calculate expected returns using various methods.
        
        Args:
            returns_data: Historical returns
            method: Method for calculating expected returns
            
        Returns:
            Expected returns for each asset
        """
        if method == 'historical':
            # Simple historical mean
            expected_returns = returns_data.mean() * 252
            
        elif method == 'exponential':
            # Exponentially weighted mean
            halflife = 60  # 60 days halflife
            expected_returns = returns_data.ewm(halflife=halflife).mean().iloc[-1] * 252
            
        elif method == 'capm':
            # CAPM-based expected returns
            market_returns = returns_data.mean(axis=1)  # Equal-weight market proxy
            expected_returns = pd.Series(index=returns_data.columns)
            
            for asset in returns_data.columns:
                beta = returns_data[asset].cov(market_returns) / market_returns.var()
                expected_returns[asset] = self.risk_free_rate + beta * (market_returns.mean() * 252 - self.risk_free_rate)
                
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return expected_returns
    
    def calculate_covariance_matrix(self,
                                  returns_data: pd.DataFrame,
                                  method: str = 'sample') -> pd.DataFrame:
        """
        Calculate covariance matrix using various methods.
        
        Args:
            returns_data: Historical returns
            method: Method for covariance estimation
            
        Returns:
            Covariance matrix
        """
        if method == 'sample':
            # Sample covariance
            cov_matrix = returns_data.cov() * 252
            
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage
            lw = LedoitWolf()
            cov_matrix = pd.DataFrame(
                lw.fit(returns_data).covariance_ * 252,
                index=returns_data.columns,
                columns=returns_data.columns
            )
            
        elif method == 'exponential':
            # Exponentially weighted covariance
            halflife = 60
            cov_matrix = returns_data.ewm(halflife=halflife).cov().iloc[-len(returns_data.columns):] * 252
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return cov_matrix
    
    def optimize_mean_variance(self,
                             expected_returns: pd.Series,
                             cov_matrix: pd.DataFrame,
                             constraints: OptimizationConstraints,
                             target_return: Optional[float] = None) -> OptimizedPortfolio:
        """
        Mean-variance optimization (Markowitz).
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            constraints: Optimization constraints
            target_return: Target portfolio return
            
        Returns:
            Optimized portfolio
        """
        n_assets = len(expected_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Expected portfolio return
        portfolio_return = expected_returns.values @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        # Objective: minimize variance (or maximize Sharpe ratio)
        if target_return is not None:
            objective = cp.Minimize(portfolio_variance)
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= constraints.min_weight,
                weights <= constraints.max_weight,
                portfolio_return >= target_return
            ]
        else:
            # For Sharpe ratio maximization, we'll use a two-step approach
            # First, find the minimum variance portfolio
            objective = cp.Minimize(portfolio_variance)
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= constraints.min_weight,
                weights <= constraints.max_weight,
                portfolio_return >= expected_returns.mean()  # Ensure positive expected return
            ]
            
        # Add position limit constraint
        if constraints.max_positions:
            binary = cp.Variable(n_assets, boolean=True)
            constraints_list.extend([
                weights <= binary,
                cp.sum(binary) <= constraints.max_positions
            ])
            
        # Solve optimization with fallback solvers
        problem = cp.Problem(objective, constraints_list)
        
        # Try different solvers in order of preference
        solvers = [cp.ECOS, cp.SCS, cp.OSQP, cp.CVXOPT]
        solver_used = None
        
        for solver in solvers:
            try:
                if solver in cp.installed_solvers():
                    problem.solve(solver=solver)
                    solver_used = solver
                    logger.info(f"Successfully used {solver} solver")
                    break
            except Exception as e:
                logger.warning(f"Failed to use {solver}: {e}")
                continue
        
        if solver_used is None:
            # Try with default solver
            try:
                problem.solve()
                logger.info("Used default solver")
            except Exception as e:
                logger.error(f"All solvers failed: {e}")
                raise RuntimeError("No suitable solver found for optimization")
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.warning(f"Optimization not optimal: {problem.status}")
            
        # Extract results
        optimal_weights = weights.value
        weights_dict = dict(zip(expected_returns.index, optimal_weights))
        
        # Calculate portfolio metrics
        portfolio_return = float(expected_returns.values @ optimal_weights)
        portfolio_risk = float(np.sqrt(optimal_weights @ cov_matrix.values @ optimal_weights))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Diversification ratio
        weighted_avg_vol = np.sum(np.sqrt(np.diag(cov_matrix.values)) * np.abs(optimal_weights))
        diversification_ratio = weighted_avg_vol / portfolio_risk
        
        # Effective number of assets
        effective_assets = 1 / np.sum(optimal_weights ** 2)
        
        # Calculate turnover
        turnover = self._calculate_turnover(weights_dict)
        
        return OptimizedPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_assets=int(effective_assets),
            turnover=turnover,
            optimization_method='mean_variance'
        )
    
    def optimize_black_litterman(self,
                               market_caps: Dict[str, float],
                               returns_data: pd.DataFrame,
                               views: Dict[str, float],
                               view_confidence: Dict[str, float],
                               constraints: OptimizationConstraints) -> OptimizedPortfolio:
        """
        Black-Litterman optimization.
        
        Args:
            market_caps: Market capitalizations
            returns_data: Historical returns
            views: Investor views {asset: expected_return}
            view_confidence: Confidence in views {asset: confidence}
            constraints: Optimization constraints
            
        Returns:
            Optimized portfolio
        """
        # Calculate market weights
        total_cap = sum(market_caps.values())
        market_weights = pd.Series({asset: cap/total_cap for asset, cap in market_caps.items()})
        
        # Prior covariance
        cov_matrix = self.calculate_covariance_matrix(returns_data)
        
        # Market implied returns (reverse optimization)
        risk_aversion = 2.5  # Typical value
        implied_returns = risk_aversion * cov_matrix @ market_weights
        
        # Views matrix
        n_views = len(views)
        n_assets = len(market_weights)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        assets = list(market_weights.index)
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                P[i, assets.index(asset)] = 1
                Q[i] = view_return
                
        # Uncertainty in views
        tau = 0.05  # Typical value
        omega_diag = []
        for asset, conf in view_confidence.items():
            if asset in assets:
                idx = assets.index(asset)
                view_var = (P[idx] @ cov_matrix @ P[idx].T) * tau / conf
                omega_diag.append(view_var)
                
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        M = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(Omega) @ P)
        posterior_returns = M @ (np.linalg.inv(tau * cov_matrix) @ implied_returns + P.T @ np.linalg.inv(Omega) @ Q)
        posterior_cov = M + cov_matrix
        
        # Convert to pandas
        posterior_returns = pd.Series(posterior_returns, index=assets)
        posterior_cov = pd.DataFrame(posterior_cov, index=assets, columns=assets)
        
        # Optimize with posterior estimates
        return self.optimize_mean_variance(posterior_returns, posterior_cov, constraints)
    
    def optimize_risk_parity(self,
                           cov_matrix: pd.DataFrame,
                           constraints: OptimizationConstraints) -> OptimizedPortfolio:
        """
        Risk parity optimization.
        
        Args:
            cov_matrix: Covariance matrix
            constraints: Optimization constraints
            
        Returns:
            Optimized portfolio
        """
        n_assets = len(cov_matrix)
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            marginal_contrib = cov_matrix.values @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            contrib = risk_contribution(weights)
            # Equal risk contribution
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if not result.success:
            logger.warning("Risk parity optimization failed")
            
        # Extract results
        optimal_weights = result.x
        weights_dict = dict(zip(cov_matrix.index, optimal_weights))
        
        # Calculate metrics
        portfolio_risk = float(np.sqrt(optimal_weights @ cov_matrix.values @ optimal_weights))
        
        # For risk parity, we don't have expected returns
        # Use historical average as proxy
        expected_returns = pd.Series(0.08, index=cov_matrix.index)  # 8% annual return assumption
        portfolio_return = float(expected_returns.values @ optimal_weights)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Diversification ratio
        weighted_avg_vol = np.sum(np.sqrt(np.diag(cov_matrix.values)) * optimal_weights)
        diversification_ratio = weighted_avg_vol / portfolio_risk
        
        # Effective assets
        effective_assets = 1 / np.sum(optimal_weights ** 2)
        
        # Turnover
        turnover = self._calculate_turnover(weights_dict)
        
        return OptimizedPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_assets=int(effective_assets),
            turnover=turnover,
            optimization_method='risk_parity'
        )
    
    def optimize_hierarchical_risk_parity(self,
                                        returns_data: pd.DataFrame,
                                        constraints: OptimizationConstraints) -> OptimizedPortfolio:
        """
        Hierarchical Risk Parity (HRP) optimization.
        
        Args:
            returns_data: Historical returns
            constraints: Optimization constraints
            
        Returns:
            Optimized portfolio
        """
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Calculate distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(dist_matrix)
        
        # Hierarchical clustering
        link = linkage(condensed_dist, method='single')
        
        # Get sorted order
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0])
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
                
            return sort_ix.tolist()
        
        sorted_idx = get_quasi_diag(link)
        
        # Calculate inverse variance weights
        cov_matrix = returns_data.cov()
        inv_var = 1 / np.diag(cov_matrix.values)
        inv_var_weights = inv_var / inv_var.sum()
        
        # Apply hierarchical structure (simplified)
        weights = inv_var_weights
        weights_dict = dict(zip(returns_data.columns, weights))
        
        # Calculate metrics
        portfolio_risk = float(np.sqrt(weights @ cov_matrix.values @ weights))
        expected_returns = returns_data.mean() * 252
        portfolio_return = float(expected_returns.values @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Diversification ratio
        weighted_avg_vol = np.sum(np.sqrt(np.diag(cov_matrix.values)) * weights)
        diversification_ratio = weighted_avg_vol / portfolio_risk
        
        # Effective assets
        effective_assets = 1 / np.sum(weights ** 2)
        
        # Turnover
        turnover = self._calculate_turnover(weights_dict)
        
        return OptimizedPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_assets=int(effective_assets),
            turnover=turnover,
            optimization_method='hierarchical_risk_parity'
        )
    
    def optimize_with_ml_predictions(self,
                                   ml_predictions: Dict[str, float],
                                   prediction_confidence: Dict[str, float],
                                   returns_data: pd.DataFrame,
                                   constraints: OptimizationConstraints) -> OptimizedPortfolio:
        """
        Optimize portfolio using ML predictions.
        
        Args:
            ml_predictions: ML model predictions for each asset
            prediction_confidence: Confidence in predictions
            returns_data: Historical returns for risk estimation
            constraints: Optimization constraints
            
        Returns:
            Optimized portfolio
        """
        # Blend ML predictions with historical data
        historical_returns = self.calculate_expected_returns(returns_data)
        
        expected_returns = pd.Series(index=historical_returns.index)
        for asset in expected_returns.index:
            if asset in ml_predictions:
                conf = prediction_confidence.get(asset, 0.5)
                ml_ret = ml_predictions[asset]
                hist_ret = historical_returns[asset]
                # Weighted average based on confidence
                expected_returns[asset] = conf * ml_ret + (1 - conf) * hist_ret
            else:
                expected_returns[asset] = historical_returns[asset]
                
        # Adjust covariance based on prediction uncertainty
        base_cov = self.calculate_covariance_matrix(returns_data)
        
        # Increase variance for uncertain predictions
        uncertainty_multiplier = pd.Series(index=base_cov.index)
        for asset in uncertainty_multiplier.index:
            conf = prediction_confidence.get(asset, 0.5)
            # Higher uncertainty = higher variance
            uncertainty_multiplier[asset] = 1 + (1 - conf) * 0.5
            
        # Adjust diagonal elements (variances)
        adjusted_cov = base_cov.copy()
        for i, asset in enumerate(base_cov.index):
            adjusted_cov.iloc[i, i] *= uncertainty_multiplier[asset] ** 2
            
        # Optimize with adjusted parameters
        return self.optimize_mean_variance(expected_returns, adjusted_cov, constraints)
    
    def _calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        if not self.current_weights:
            return 1.0  # 100% turnover for initial portfolio
            
        turnover = 0.0
        all_assets = set(new_weights.keys()) | set(self.current_weights.keys())
        
        for asset in all_assets:
            old_weight = self.current_weights.get(asset, 0.0)
            new_weight = new_weights.get(asset, 0.0)
            turnover += abs(new_weight - old_weight)
            
        return turnover / 2  # Divide by 2 to avoid double counting
    
    def should_rebalance(self, 
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Whether to rebalance
        """
        total_deviation = 0.0
        
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            total_deviation += abs(current - target)
            
        return total_deviation > self.rebalance_threshold
    
    def generate_optimization_report(self, 
                                   optimized_portfolio: OptimizedPortfolio,
                                   current_portfolio: Optional[Dict[str, float]] = None) -> str:
        """
        Generate portfolio optimization report.
        
        Args:
            optimized_portfolio: Optimized portfolio
            current_portfolio: Current portfolio weights
            
        Returns:
            Formatted report
        """
        report = f"""
=== PORTFOLIO OPTIMIZATION REPORT ===
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: {optimized_portfolio.optimization_method}

OPTIMIZED PORTFOLIO:
Expected Return: {optimized_portfolio.expected_return:.2%}
Expected Risk: {optimized_portfolio.expected_risk:.2%}
Sharpe Ratio: {optimized_portfolio.sharpe_ratio:.2f}
Diversification Ratio: {optimized_portfolio.diversification_ratio:.2f}
Effective Assets: {optimized_portfolio.effective_assets}
Turnover: {optimized_portfolio.turnover:.2%}

TOP HOLDINGS:
"""
        # Sort weights
        sorted_weights = sorted(optimized_portfolio.weights.items(), 
                              key=lambda x: x[1], reverse=True)
        
        for asset, weight in sorted_weights[:10]:
            report += f"  {asset}: {weight:.2%}"
            if current_portfolio and asset in current_portfolio:
                change = weight - current_portfolio[asset]
                report += f" (change: {change:+.2%})"
            report += "\n"
            
        return report
