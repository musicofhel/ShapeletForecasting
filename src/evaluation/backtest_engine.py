"""
Backtesting Engine for Financial Trading Strategies

This module implements comprehensive backtesting functionality including
walk-forward analysis and out-of-sample testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # Fraction of capital per trade
    max_positions: int = 5
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    stop_loss: Optional[float] = 0.02  # 2% stop loss
    take_profit: Optional[float] = 0.05  # 5% take profit
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    use_leverage: bool = False
    max_leverage: float = 2.0
    margin_call_level: float = 0.25  # 25% of initial capital


class BacktestEngine:
    """
    Main backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset backtesting state"""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.current_date = None
        self.peak_equity = self.config.initial_capital
        
    def run_backtest(self,
                     data: pd.DataFrame,
                     predictions: np.ndarray,
                     signal_generator: Callable,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data with columns ['open', 'high', 'low', 'close', 'volume']
            predictions: Model predictions aligned with data
            signal_generator: Function to generate trading signals from predictions
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Ensure predictions align with data
        if len(predictions) != len(data):
            raise ValueError(f"Predictions length {len(predictions)} doesn't match data length {len(data)}")
            
        # Generate trading signals
        signals = signal_generator(predictions, data)
        
        # Run backtest loop
        for i, (date, row) in enumerate(data.iterrows()):
            self.current_date = date
            signal = signals[i]
            
            # Update positions with current prices
            self._update_positions(row)
            
            # Check stop loss and take profit
            self._check_exit_conditions(row)
            
            # Process new signals
            if signal != 0:
                self._process_signal(signal, row, i)
                
            # Record equity
            current_equity = self._calculate_equity(row)
            self.equity_curve.append({
                'date': date,
                'equity': current_equity,
                'cash': self.capital,
                'positions_value': current_equity - self.capital
            })
            
            # Update drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.drawdown_curve.append({
                'date': date,
                'drawdown': drawdown
            })
            
            # Check margin call
            if self.config.use_leverage and current_equity < self.config.initial_capital * self.config.margin_call_level:
                logger.warning(f"Margin call at {date}. Closing all positions.")
                self._close_all_positions(row)
                
        # Close remaining positions
        if len(data) > 0:
            self._close_all_positions(data.iloc[-1])
            
        # Compile results
        results = self._compile_results(data)
        return results
        
    def _update_positions(self, current_prices: pd.Series):
        """Update position values with current prices"""
        for symbol, position in self.positions.items():
            position['current_price'] = current_prices['close']
            position['current_value'] = position['quantity'] * position['current_price']
            position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
            
    def _check_exit_conditions(self, current_prices: pd.Series):
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices['close']
            entry_price = position['entry_price']
            
            # Calculate return
            if position['side'] == 'long':
                returns = (current_price - entry_price) / entry_price
            else:  # short
                returns = (entry_price - current_price) / entry_price
                
            # Check stop loss
            if self.config.stop_loss and returns <= -self.config.stop_loss:
                positions_to_close.append((symbol, 'stop_loss'))
                
            # Check take profit
            elif self.config.take_profit and returns >= self.config.take_profit:
                positions_to_close.append((symbol, 'take_profit'))
                
        # Close positions
        for symbol, reason in positions_to_close:
            self._close_position(symbol, current_prices, reason)
            
    def _process_signal(self, signal: float, current_prices: pd.Series, index: int):
        """Process trading signal"""
        # Check if we can open new positions
        if len(self.positions) >= self.config.max_positions:
            return
            
        # Calculate position size
        current_equity = self._calculate_equity(current_prices)
        position_value = current_equity * self.config.position_size
        
        # Apply leverage if enabled
        if self.config.use_leverage:
            position_value *= min(abs(signal), self.config.max_leverage)
            
        # Open position
        if signal > 0:  # Long signal
            self._open_position('SPY', 'long', position_value, current_prices)
        elif signal < 0:  # Short signal
            self._open_position('SPY', 'short', position_value, current_prices)
            
    def _open_position(self, symbol: str, side: str, value: float, current_prices: pd.Series):
        """Open a new position"""
        price = current_prices['close']
        
        # Apply slippage
        if side == 'long':
            price *= (1 + self.config.slippage)
        else:
            price *= (1 - self.config.slippage)
            
        # Calculate quantity
        quantity = value / price
        
        # Apply commission
        commission = value * self.config.commission
        self.capital -= commission
        
        # Create position
        position = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': price,
            'entry_date': self.current_date,
            'cost_basis': value,
            'current_price': price,
            'current_value': value,
            'unrealized_pnl': 0
        }
        
        # Store position
        position_id = f"{symbol}_{self.current_date}_{side}"
        self.positions[position_id] = position
        
        # Deduct cash
        self.capital -= value
        
        # Record trade
        self.trades.append({
            'date': self.current_date,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': value,
            'commission': commission,
            'type': 'open'
        })
        
    def _close_position(self, position_id: str, current_prices: pd.Series, reason: str = 'signal'):
        """Close an existing position"""
        if position_id not in self.positions:
            return
            
        position = self.positions[position_id]
        price = current_prices['close']
        
        # Apply slippage (opposite direction)
        if position['side'] == 'long':
            price *= (1 - self.config.slippage)
        else:
            price *= (1 + self.config.slippage)
            
        # Calculate proceeds
        proceeds = position['quantity'] * price
        
        # Apply commission
        commission = proceeds * self.config.commission
        proceeds -= commission
        
        # Calculate P&L
        if position['side'] == 'long':
            pnl = proceeds - position['cost_basis']
        else:  # short
            pnl = position['cost_basis'] - proceeds
            
        # Update capital
        self.capital += proceeds
        
        # Record trade
        self.trades.append({
            'date': self.current_date,
            'symbol': position['symbol'],
            'side': 'sell' if position['side'] == 'long' else 'cover',
            'quantity': position['quantity'],
            'price': price,
            'value': proceeds,
            'commission': commission,
            'pnl': pnl,
            'type': 'close',
            'reason': reason,
            'holding_period': (self.current_date - position['entry_date']).days
        })
        
        # Remove position
        del self.positions[position_id]
        
    def _close_all_positions(self, current_prices: pd.Series):
        """Close all open positions"""
        position_ids = list(self.positions.keys())
        for position_id in position_ids:
            self._close_position(position_id, current_prices, 'end_of_backtest')
            
    def _calculate_equity(self, current_prices: pd.Series) -> float:
        """Calculate current total equity"""
        positions_value = sum(pos['current_value'] for pos in self.positions.values())
        return self.capital + positions_value
        
    def _compile_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compile backtest results"""
        equity_df = pd.DataFrame(self.equity_curve)
        drawdown_df = pd.DataFrame(self.drawdown_curve)
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate trade statistics
        if len(trades_df) > 0:
            closed_trades = trades_df[trades_df['type'] == 'close']
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            losing_trades = closed_trades[closed_trades['pnl'] < 0]
            
            trade_stats = {
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0,
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf,
                'avg_holding_period': closed_trades['holding_period'].mean() if 'holding_period' in closed_trades else 0
            }
        else:
            trade_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_period': 0
            }
            
        # Compile results
        results = {
            'equity_curve': equity_df,
            'drawdown_curve': drawdown_df,
            'trades': trades_df,
            'trade_statistics': trade_stats,
            'final_equity': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else self.config.initial_capital,
            'total_return': (equity_df['equity'].iloc[-1] - self.config.initial_capital) / self.config.initial_capital if len(equity_df) > 0 else 0,
            'max_drawdown': drawdown_df['drawdown'].max() if len(drawdown_df) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_df['returns']) if len(equity_df) > 1 else 0,
            'calmar_ratio': self._calculate_calmar_ratio(equity_df, drawdown_df) if len(equity_df) > 0 and len(drawdown_df) > 0 else 0
        }
        
        return results
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        # Annualize returns and volatility
        periods_per_year = 252  # Daily data
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        
        if std_return == 0:
            return 0.0
            
        return (mean_return - risk_free_rate) / std_return
        
    def _calculate_calmar_ratio(self, equity_df: pd.DataFrame, drawdown_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(equity_df) < 2 or len(drawdown_df) == 0:
            return 0.0
            
        # Calculate annualized return
        total_days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        if total_days == 0:
            return 0.0
            
        years = total_days / 365.25
        total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Get max drawdown
        max_dd = drawdown_df['drawdown'].max()
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
            
        return annual_return / max_dd


class WalkForwardBacktest:
    """
    Walk-forward backtesting for robust out-of-sample evaluation
    """
    
    def __init__(self,
                 backtest_engine: BacktestEngine,
                 train_period: int = 252,  # 1 year
                 test_period: int = 63,    # 3 months
                 step_size: int = 21):     # 1 month
        """
        Initialize walk-forward backtest
        
        Args:
            backtest_engine: Backtesting engine to use
            train_period: Number of periods for training
            test_period: Number of periods for testing
            step_size: Number of periods to step forward
        """
        self.engine = backtest_engine
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        
    def run(self,
            data: pd.DataFrame,
            model_trainer: Callable,
            signal_generator: Callable,
            feature_extractor: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run walk-forward backtest
        
        Args:
            data: Historical data
            model_trainer: Function to train model on data window
            signal_generator: Function to generate signals from predictions
            feature_extractor: Optional function to extract features from raw data
            
        Returns:
            Dictionary containing walk-forward results
        """
        results = []
        equity_curves = []
        
        # Calculate number of windows
        total_periods = len(data)
        num_windows = (total_periods - self.train_period - self.test_period) // self.step_size + 1
        
        logger.info(f"Running walk-forward backtest with {num_windows} windows")
        
        for i in tqdm(range(num_windows), desc="Walk-forward windows"):
            # Define window boundaries
            train_start = i * self.step_size
            train_end = train_start + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            
            # Skip if we don't have enough data
            if test_end > len(data):
                break
                
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Extract features if needed
            if feature_extractor:
                train_features = feature_extractor(train_data)
                test_features = feature_extractor(test_data)
            else:
                train_features = train_data
                test_features = test_data
                
            # Train model
            model = model_trainer(train_features)
            
            # Generate predictions
            predictions = model.predict(test_features)
            
            # Run backtest on test period
            window_results = self.engine.run_backtest(
                test_data,
                predictions,
                signal_generator
            )
            
            # Store results
            window_results['window'] = i
            window_results['train_start'] = train_data.index[0]
            window_results['train_end'] = train_data.index[-1]
            window_results['test_start'] = test_data.index[0]
            window_results['test_end'] = test_data.index[-1]
            
            results.append(window_results)
            equity_curves.append(window_results['equity_curve'])
            
        # Combine results
        combined_results = self._combine_results(results, equity_curves)
        return combined_results
        
    def _combine_results(self, results: List[Dict], equity_curves: List[pd.DataFrame]) -> Dict[str, Any]:
        """Combine results from all windows"""
        # Concatenate equity curves
        combined_equity = pd.concat(equity_curves, ignore_index=True)
        
        # Calculate overall statistics
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for result in results:
            returns.append(result['total_return'])
            sharpe_ratios.append(result['sharpe_ratio'])
            max_drawdowns.append(result['max_drawdown'])
            win_rates.append(result['trade_statistics']['win_rate'])
            
        # Compile combined results
        combined = {
            'num_windows': len(results),
            'combined_equity_curve': combined_equity,
            'window_results': results,
            'overall_statistics': {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_drawdown': np.max(max_drawdowns),
                'mean_win_rate': np.mean(win_rates),
                'consistency': np.sum([r > 0 for r in returns]) / len(returns)  # % of profitable windows
            }
        }
        
        return combined
