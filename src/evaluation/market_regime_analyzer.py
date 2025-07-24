"""
Market Regime Analysis Module

This module analyzes strategy performance across different market conditions
and regimes to understand when strategies work best.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Container for market regime information"""
    regime_id: int
    name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    characteristics: Dict[str, float]
    
    
class MarketRegimeAnalyzer:
    """
    Analyze market regimes and strategy performance in different conditions
    """
    
    def __init__(self, n_regimes: int = 4):
        """
        Initialize market regime analyzer
        
        Args:
            n_regimes: Number of market regimes to identify
        """
        self.n_regimes = n_regimes
        self.gmm = None
        self.scaler = StandardScaler()
        self.regime_labels = {
            0: "Low Volatility Bull",
            1: "High Volatility Bear",
            2: "Sideways Market",
            3: "Volatile Transition"
        }
        
    def identify_regimes(self, 
                        market_data: pd.DataFrame,
                        lookback: int = 20) -> pd.DataFrame:
        """
        Identify market regimes using regime switching model
        
        Args:
            market_data: DataFrame with OHLCV data
            lookback: Lookback period for feature calculation
            
        Returns:
            DataFrame with regime labels and probabilities
        """
        # Calculate regime features
        features = self._calculate_regime_features(market_data, lookback)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        
        # Predict regimes
        regime_labels = self.gmm.fit_predict(features_scaled)
        regime_probs = self.gmm.predict_proba(features_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame(index=market_data.index[lookback:])
        results['regime'] = regime_labels
        
        for i in range(self.n_regimes):
            results[f'regime_{i}_prob'] = regime_probs[:, i]
            
        # Add regime characteristics
        results = self._add_regime_characteristics(results, features, regime_labels)
        
        return results
        
    def analyze_performance_by_regime(self,
                                    backtest_results: Dict[str, any],
                                    regime_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze strategy performance in each market regime
        
        Args:
            backtest_results: Backtest results with equity curve and trades
            regime_data: DataFrame with regime labels
            
        Returns:
            Dictionary with performance metrics by regime
        """
        equity_curve = backtest_results['equity_curve']
        trades = backtest_results['trades']
        
        # Ensure equity_curve has datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        # Merge with regime data
        combined = equity_curve.join(regime_data[['regime']], how='inner')
        
        # Analyze by regime
        regime_performance = {}
        
        for regime in range(self.n_regimes):
            regime_data_subset = combined[combined['regime'] == regime]
            
            if len(regime_data_subset) > 0:
                # Calculate metrics
                returns = regime_data_subset['returns'].dropna()
                
                metrics = {
                    'days_in_regime': len(regime_data_subset),
                    'pct_of_time': len(regime_data_subset) / len(combined),
                    'total_return': (1 + returns).prod() - 1,
                    'avg_daily_return': returns.mean(),
                    'volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': self._calculate_sharpe(returns),
                    'max_drawdown': self._calculate_max_drawdown(regime_data_subset['equity']),
                    'win_rate': self._calculate_win_rate(returns),
                    'avg_trade_return': self._analyze_regime_trades(trades, regime_data_subset.index)
                }
                
                regime_performance[self.regime_labels.get(regime, f"Regime {regime}")] = metrics
                
        return regime_performance
        
    def detect_regime_changes(self,
                            regime_data: pd.DataFrame,
                            min_duration: int = 20) -> List[Dict[str, any]]:
        """
        Detect regime change points
        
        Args:
            regime_data: DataFrame with regime labels
            min_duration: Minimum regime duration to consider
            
        Returns:
            List of regime change events
        """
        regime_changes = []
        current_regime = regime_data['regime'].iloc[0]
        regime_start = regime_data.index[0]
        
        for date, row in regime_data.iterrows():
            if row['regime'] != current_regime:
                # Record regime change
                duration = (date - regime_start).days
                
                if duration >= min_duration:
                    regime_changes.append({
                        'date': date,
                        'from_regime': self.regime_labels.get(current_regime, f"Regime {current_regime}"),
                        'to_regime': self.regime_labels.get(row['regime'], f"Regime {row['regime']}"),
                        'duration_days': duration,
                        'confidence': row[f'regime_{row["regime"]}_prob']
                    })
                    
                current_regime = row['regime']
                regime_start = date
                
        return regime_changes
        
    def calculate_regime_transition_matrix(self, 
                                         regime_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix
        
        Args:
            regime_data: DataFrame with regime labels
            
        Returns:
            Transition probability matrix
        """
        # Count transitions
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        
        regimes = regime_data['regime'].values
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transitions[from_regime, to_regime] += 1
            
        # Normalize to get probabilities
        row_sums = transitions.sum(axis=1)
        transition_probs = transitions / row_sums[:, np.newaxis]
        
        # Create DataFrame
        regime_names = [self.regime_labels.get(i, f"Regime {i}") for i in range(self.n_regimes)]
        transition_matrix = pd.DataFrame(
            transition_probs,
            index=regime_names,
            columns=regime_names
        )
        
        return transition_matrix
        
    def analyze_regime_persistence(self, 
                                 regime_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze how long each regime typically persists
        
        Args:
            regime_data: DataFrame with regime labels
            
        Returns:
            Dictionary with persistence statistics for each regime
        """
        persistence_stats = {}
        
        for regime in range(self.n_regimes):
            # Find all periods of this regime
            regime_periods = []
            in_regime = False
            start_date = None
            
            for date, row in regime_data.iterrows():
                if row['regime'] == regime and not in_regime:
                    # Regime started
                    in_regime = True
                    start_date = date
                elif row['regime'] != regime and in_regime:
                    # Regime ended
                    duration = (date - start_date).days
                    regime_periods.append(duration)
                    in_regime = False
                    
            if regime_periods:
                persistence_stats[self.regime_labels.get(regime, f"Regime {regime}")] = {
                    'avg_duration_days': np.mean(regime_periods),
                    'median_duration_days': np.median(regime_periods),
                    'max_duration_days': np.max(regime_periods),
                    'min_duration_days': np.min(regime_periods),
                    'std_duration_days': np.std(regime_periods),
                    'num_occurrences': len(regime_periods)
                }
                
        return persistence_stats
        
    def generate_regime_report(self,
                             market_data: pd.DataFrame,
                             backtest_results: Dict[str, any],
                             lookback: int = 20) -> Dict[str, any]:
        """
        Generate comprehensive regime analysis report
        
        Args:
            market_data: Historical market data
            backtest_results: Backtest results
            lookback: Lookback period for regime identification
            
        Returns:
            Dictionary with complete regime analysis
        """
        # Identify regimes
        regime_data = self.identify_regimes(market_data, lookback)
        
        # Analyze performance by regime
        performance_by_regime = self.analyze_performance_by_regime(
            backtest_results, regime_data
        )
        
        # Detect regime changes
        regime_changes = self.detect_regime_changes(regime_data)
        
        # Calculate transition matrix
        transition_matrix = self.calculate_regime_transition_matrix(regime_data)
        
        # Analyze persistence
        persistence_stats = self.analyze_regime_persistence(regime_data)
        
        # Compile report
        report = {
            'regime_data': regime_data,
            'performance_by_regime': performance_by_regime,
            'regime_changes': regime_changes,
            'transition_matrix': transition_matrix,
            'persistence_stats': persistence_stats,
            'current_regime': {
                'regime': self.regime_labels.get(regime_data['regime'].iloc[-1]),
                'confidence': regime_data[f'regime_{regime_data["regime"].iloc[-1]}_prob'].iloc[-1],
                'days_in_regime': self._days_in_current_regime(regime_data)
            }
        }
        
        return report
        
    def _calculate_regime_features(self, 
                                 market_data: pd.DataFrame,
                                 lookback: int) -> np.ndarray:
        """Calculate features for regime identification"""
        features = []
        
        # Returns
        returns = market_data['close'].pct_change()
        
        # Rolling statistics
        features.append(returns.rolling(lookback).mean())  # Trend
        features.append(returns.rolling(lookback).std())   # Volatility
        features.append(returns.rolling(lookback).skew())  # Skewness
        features.append(returns.rolling(lookback).apply(lambda x: ((x < 0).sum() / len(x))))  # Down days ratio
        
        # Volume features
        if 'volume' in market_data.columns:
            volume_ma = market_data['volume'].rolling(lookback).mean()
            features.append(market_data['volume'] / volume_ma)  # Relative volume
            
        # Price features
        features.append((market_data['close'] - market_data['close'].rolling(lookback).min()) / 
                       (market_data['close'].rolling(lookback).max() - market_data['close'].rolling(lookback).min()))  # Price position
        
        # High-Low spread
        features.append((market_data['high'] - market_data['low']) / market_data['close'])  # Daily range
        
        # Combine features
        feature_matrix = np.column_stack(features)[lookback:]
        
        # Remove NaN values
        feature_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
        
        return feature_matrix
        
    def _add_regime_characteristics(self,
                                  results: pd.DataFrame,
                                  features: np.ndarray,
                                  regime_labels: np.ndarray) -> pd.DataFrame:
        """Add regime characteristics to results"""
        feature_names = ['trend', 'volatility', 'skewness', 'down_ratio', 
                        'rel_volume', 'price_position', 'daily_range']
        
        # Calculate average features for each regime
        for regime in range(self.n_regimes):
            regime_mask = regime_labels == regime
            if np.any(regime_mask):
                regime_features = features[regime_mask].mean(axis=0)
                
                for i, feature_name in enumerate(feature_names[:features.shape[1]]):
                    results.loc[results['regime'] == regime, f'regime_{feature_name}'] = regime_features[i]
                    
        return results
        
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
        
    def _analyze_regime_trades(self, 
                             trades: pd.DataFrame,
                             regime_dates: pd.DatetimeIndex) -> float:
        """Analyze trades within a specific regime"""
        if 'date' not in trades.columns or 'pnl' not in trades.columns:
            return 0.0
            
        # Filter trades within regime dates
        trades['date'] = pd.to_datetime(trades['date'])
        regime_trades = trades[trades['date'].isin(regime_dates)]
        
        if len(regime_trades) == 0:
            return 0.0
            
        return regime_trades['pnl'].mean()
        
    def _days_in_current_regime(self, regime_data: pd.DataFrame) -> int:
        """Calculate days in current regime"""
        current_regime = regime_data['regime'].iloc[-1]
        days = 0
        
        for i in range(len(regime_data) - 1, -1, -1):
            if regime_data['regime'].iloc[i] == current_regime:
                days += 1
            else:
                break
                
        return days
        
    def plot_regime_analysis(self,
                           market_data: pd.DataFrame,
                           regime_data: pd.DataFrame,
                           save_path: Optional[str] = None):
        """
        Plot regime analysis visualization
        
        Args:
            market_data: Market price data
            regime_data: Regime identification results
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price with regime coloring
        ax1 = axes[0]
        
        # Define colors for each regime
        colors = ['green', 'red', 'blue', 'orange']
        
        for regime in range(self.n_regimes):
            regime_mask = regime_data['regime'] == regime
            regime_dates = regime_data[regime_mask].index
            
            if len(regime_dates) > 0:
                # Plot price for this regime
                price_data = market_data.loc[regime_dates, 'close']
                ax1.plot(price_data.index, price_data.values, 
                        color=colors[regime], linewidth=2, alpha=0.8)
                        
        ax1.set_title('Market Price by Regime')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Create legend
        patches = [mpatches.Patch(color=colors[i], 
                                 label=self.regime_labels.get(i, f'Regime {i}')) 
                  for i in range(self.n_regimes)]
        ax1.legend(handles=patches)
        
        # Plot 2: Regime probabilities
        ax2 = axes[1]
        
        for regime in range(self.n_regimes):
            ax2.plot(regime_data.index, 
                    regime_data[f'regime_{regime}_prob'],
                    label=self.regime_labels.get(regime, f'Regime {regime}'),
                    linewidth=2)
                    
        ax2.set_title('Regime Probabilities')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime indicator
        ax3 = axes[2]
        ax3.plot(regime_data.index, regime_data['regime'], 
                drawstyle='steps-post', linewidth=2)
        ax3.set_title('Current Regime')
        ax3.set_ylabel('Regime ID')
        ax3.set_ylim(-0.5, self.n_regimes - 0.5)
        ax3.set_yticks(range(self.n_regimes))
        ax3.set_yticklabels([self.regime_labels.get(i, f'Regime {i}') 
                            for i in range(self.n_regimes)])
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
