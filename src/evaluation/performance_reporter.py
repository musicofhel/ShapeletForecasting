"""
Performance Reporting Module for Trading Strategies

This module generates comprehensive performance reports with
visualizations and analytics for backtesting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

# Import components
from .risk_analyzer import RiskMetrics, RiskAnalyzer
from .trading_simulator import Trade, Position

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class BacktestReport:
    """Container for backtest report data"""
    summary: Dict[str, Any]
    risk_metrics: RiskMetrics
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trades: pd.DataFrame
    monthly_returns: pd.DataFrame
    rolling_metrics: pd.DataFrame
    market_regime_analysis: Optional[Dict[str, Any]] = None
    stress_test_results: Optional[Dict[str, Any]] = None


class PerformanceReporter:
    """
    Generate comprehensive performance reports for trading strategies
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize performance reporter
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.risk_analyzer = RiskAnalyzer()
        
    def generate_report(self,
                       backtest_results: Dict[str, Any],
                       strategy_name: str = "Strategy",
                       benchmark_data: Optional[pd.DataFrame] = None,
                       save_report: bool = True) -> BacktestReport:
        """
        Generate comprehensive performance report
        
        Args:
            backtest_results: Results from backtesting engine
            strategy_name: Name of the strategy
            benchmark_data: Optional benchmark data for comparison
            save_report: Whether to save report to disk
            
        Returns:
            BacktestReport object
        """
        # Extract data from backtest results
        equity_curve = backtest_results['equity_curve']
        drawdown_curve = backtest_results['drawdown_curve']
        trades = backtest_results['trades']
        
        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Analyze risk metrics
        risk_metrics = self.risk_analyzer.analyze(
            returns,
            benchmark_returns=benchmark_data['returns'] if benchmark_data is not None else None,
            trades=trades
        )
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        # Calculate rolling metrics
        rolling_metrics = self.risk_analyzer.calculate_rolling_metrics(returns)
        
        # Create summary
        summary = self._create_summary(backtest_results, risk_metrics)
        
        # Create report
        report = BacktestReport(
            summary=summary,
            risk_metrics=risk_metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            monthly_returns=monthly_returns,
            rolling_metrics=rolling_metrics
        )
        
        # Generate visualizations
        if save_report:
            self._save_report(report, strategy_name)
            self._generate_visualizations(report, strategy_name)
            
        return report
        
    def compare_strategies(self,
                          reports: Dict[str, BacktestReport],
                          save_comparison: bool = True) -> pd.DataFrame:
        """
        Compare multiple strategy reports
        
        Args:
            reports: Dictionary of strategy_name -> BacktestReport
            save_comparison: Whether to save comparison
            
        Returns:
            DataFrame with strategy comparison
        """
        comparison_data = []
        
        for strategy_name, report in reports.items():
            metrics = report.risk_metrics
            summary = report.summary
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'VaR (95%)': f"{metrics.value_at_risk:.2%}",
                'Total Trades': summary.get('total_trades', 0)
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_comparison:
            # Save comparison table
            comparison_path = self.output_dir / "strategy_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            
            # Generate comparison visualizations
            self._generate_comparison_plots(reports)
            
        return comparison_df
        
    def generate_tearsheet(self,
                          report: BacktestReport,
                          strategy_name: str = "Strategy") -> None:
        """
        Generate a comprehensive tearsheet with all visualizations
        
        Args:
            report: BacktestReport to visualize
            strategy_name: Name of the strategy
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Equity Curve
        ax1 = plt.subplot(6, 2, 1)
        self._plot_equity_curve(report.equity_curve, ax1)
        ax1.set_title(f"{strategy_name} - Equity Curve")
        
        # 2. Drawdown
        ax2 = plt.subplot(6, 2, 2)
        self._plot_drawdown(report.drawdown_curve, ax2)
        ax2.set_title("Drawdown")
        
        # 3. Monthly Returns Heatmap
        ax3 = plt.subplot(6, 2, (3, 4))
        self._plot_monthly_returns_heatmap(report.monthly_returns, ax3)
        ax3.set_title("Monthly Returns Heatmap")
        
        # 4. Returns Distribution
        ax4 = plt.subplot(6, 2, 5)
        self._plot_returns_distribution(report.equity_curve, ax4)
        ax4.set_title("Returns Distribution")
        
        # 5. Rolling Sharpe
        ax5 = plt.subplot(6, 2, 6)
        self._plot_rolling_sharpe(report.rolling_metrics, ax5)
        ax5.set_title("Rolling Sharpe Ratio (252-day)")
        
        # 6. Trade Analysis
        if len(report.trades) > 0:
            ax6 = plt.subplot(6, 2, 7)
            self._plot_trade_analysis(report.trades, ax6)
            ax6.set_title("Trade P&L Distribution")
            
            ax7 = plt.subplot(6, 2, 8)
            self._plot_win_loss_analysis(report.trades, ax7)
            ax7.set_title("Win/Loss Analysis")
        
        # 7. Risk Metrics Table
        ax8 = plt.subplot(6, 2, (9, 10))
        self._plot_metrics_table(report.risk_metrics, ax8)
        ax8.set_title("Risk Metrics Summary")
        
        # 8. Rolling Volatility
        ax9 = plt.subplot(6, 2, 11)
        self._plot_rolling_volatility(report.rolling_metrics, ax9)
        ax9.set_title("Rolling Volatility (252-day)")
        
        # 9. Underwater Plot
        ax10 = plt.subplot(6, 2, 12)
        self._plot_underwater(report.drawdown_curve, ax10)
        ax10.set_title("Underwater Plot")
        
        plt.tight_layout()
        
        # Save tearsheet
        tearsheet_path = self.output_dir / f"{strategy_name}_tearsheet.png"
        plt.savefig(tearsheet_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_summary(self, 
                       backtest_results: Dict[str, Any],
                       risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Create summary statistics"""
        summary = {
            'initial_capital': backtest_results.get('initial_capital', 100000),
            'final_equity': backtest_results.get('final_equity', 100000),
            'total_return': risk_metrics.total_return,
            'annual_return': risk_metrics.annualized_return,
            'volatility': risk_metrics.volatility,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'max_drawdown': risk_metrics.max_drawdown,
            'total_trades': backtest_results['trade_statistics'].get('total_trades', 0),
            'win_rate': risk_metrics.win_rate,
            'profit_factor': risk_metrics.profit_factor,
            'avg_trade_pnl': backtest_results['trade_statistics'].get('avg_trade_pnl', 0),
            'best_trade': backtest_results['trade_statistics'].get('best_trade', 0),
            'worst_trade': backtest_results['trade_statistics'].get('worst_trade', 0),
            'avg_holding_period': backtest_results['trade_statistics'].get('avg_holding_period', 0),
            'exposure_time': self._calculate_exposure_time(backtest_results.get('equity_curve', pd.DataFrame()))
        }
        
        return summary
        
    def _calculate_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns matrix"""
        if 'date' not in equity_curve.columns:
            return pd.DataFrame()
            
        # Set date as index
        equity_curve = equity_curve.set_index('date')
        
        # Calculate daily returns
        daily_returns = equity_curve['equity'].pct_change()
        
        # Resample to monthly
        monthly_returns = (1 + daily_returns).resample('M').prod() - 1
        
        # Create matrix format
        years = monthly_returns.index.year
        months = monthly_returns.index.month
        
        # Pivot to create year x month matrix
        returns_matrix = pd.DataFrame(index=sorted(set(years)), 
                                    columns=range(1, 13))
        
        for date, ret in monthly_returns.items():
            returns_matrix.loc[date.year, date.month] = ret
            
        return returns_matrix
        
    def _calculate_exposure_time(self, equity_curve: pd.DataFrame) -> float:
        """Calculate percentage of time with open positions"""
        if 'positions_value' not in equity_curve.columns:
            return 0.0
            
        exposure_days = (equity_curve['positions_value'] != 0).sum()
        total_days = len(equity_curve)
        
        return exposure_days / total_days if total_days > 0 else 0.0
        
    def _save_report(self, report: BacktestReport, strategy_name: str):
        """Save report data to files"""
        # Create strategy directory
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # Save summary as JSON
        summary_path = strategy_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(report.summary, f, indent=4, default=str)
            
        # Save risk metrics
        metrics_path = strategy_dir / "risk_metrics.json"
        metrics_dict = {k: v for k, v in report.risk_metrics.__dict__.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4, default=str)
            
        # Save data as CSV
        report.equity_curve.to_csv(strategy_dir / "equity_curve.csv", index=False)
        report.drawdown_curve.to_csv(strategy_dir / "drawdown_curve.csv", index=False)
        report.trades.to_csv(strategy_dir / "trades.csv", index=False)
        report.monthly_returns.to_csv(strategy_dir / "monthly_returns.csv")
        report.rolling_metrics.to_csv(strategy_dir / "rolling_metrics.csv")
        
    def _generate_visualizations(self, report: BacktestReport, strategy_name: str):
        """Generate all visualization plots"""
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # Generate individual plots
        self._save_equity_curve_plot(report.equity_curve, strategy_dir / "equity_curve.png")
        self._save_drawdown_plot(report.drawdown_curve, strategy_dir / "drawdown.png")
        self._save_monthly_returns_heatmap(report.monthly_returns, strategy_dir / "monthly_returns.png")
        self._save_returns_distribution(report.equity_curve, strategy_dir / "returns_dist.png")
        self._save_rolling_metrics_plot(report.rolling_metrics, strategy_dir / "rolling_metrics.png")
        
        if len(report.trades) > 0:
            self._save_trade_analysis_plots(report.trades, strategy_dir)
            
        # Generate tearsheet
        self.generate_tearsheet(report, strategy_name)
        
    # Plotting methods
    def _plot_equity_curve(self, equity_curve: pd.DataFrame, ax: plt.Axes):
        """Plot equity curve"""
        ax.plot(equity_curve.index, equity_curve['equity'], label='Strategy', linewidth=2)
        ax.fill_between(equity_curve.index, equity_curve['cash'], equity_curve['equity'], 
                       alpha=0.3, label='Positions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_drawdown(self, drawdown_curve: pd.DataFrame, ax: plt.Axes):
        """Plot drawdown curve"""
        ax.fill_between(drawdown_curve.index, 0, -drawdown_curve['drawdown'] * 100, 
                       color='red', alpha=0.3)
        ax.plot(drawdown_curve.index, -drawdown_curve['drawdown'] * 100, 
               color='red', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)
        
    def _plot_monthly_returns_heatmap(self, monthly_returns: pd.DataFrame, ax: plt.Axes):
        """Plot monthly returns heatmap"""
        # Convert to percentage
        monthly_pct = monthly_returns * 100
        
        # Create heatmap
        sns.heatmap(monthly_pct, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
    def _plot_returns_distribution(self, equity_curve: pd.DataFrame, ax: plt.Axes):
        """Plot returns distribution"""
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Plot histogram
        ax.hist(returns * 100, bins=50, alpha=0.7, density=True, label='Returns')
        
        # Fit normal distribution
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_rolling_sharpe(self, rolling_metrics: pd.DataFrame, ax: plt.Axes):
        """Plot rolling Sharpe ratio"""
        if 'rolling_sharpe' in rolling_metrics.columns:
            ax.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], linewidth=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='g', linestyle='--', alpha=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Sharpe Ratio')
            ax.grid(True, alpha=0.3)
            
    def _plot_rolling_volatility(self, rolling_metrics: pd.DataFrame, ax: plt.Axes):
        """Plot rolling volatility"""
        if 'rolling_volatility' in rolling_metrics.columns:
            ax.plot(rolling_metrics.index, rolling_metrics['rolling_volatility'] * 100, linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Annualized Volatility (%)')
            ax.grid(True, alpha=0.3)
            
    def _plot_underwater(self, drawdown_curve: pd.DataFrame, ax: plt.Axes):
        """Plot underwater chart (time in drawdown)"""
        underwater = drawdown_curve['drawdown'].values
        
        # Find periods underwater
        underwater_periods = []
        start_idx = None
        
        for i, dd in enumerate(underwater):
            if dd > 0 and start_idx is None:
                start_idx = i
            elif dd == 0 and start_idx is not None:
                underwater_periods.append((start_idx, i))
                start_idx = None
                
        # Plot
        ax.fill_between(range(len(underwater)), 0, -underwater * 100, 
                       where=underwater > 0, color='red', alpha=0.3)
        ax.plot(-underwater * 100, color='red', linewidth=1)
        ax.set_xlabel('Days')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)
        
    def _plot_trade_analysis(self, trades: pd.DataFrame, ax: plt.Axes):
        """Plot trade P&L distribution"""
        if 'pnl' in trades.columns:
            trades['pnl'].hist(bins=30, ax=ax, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trade P&L')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
    def _plot_win_loss_analysis(self, trades: pd.DataFrame, ax: plt.Axes):
        """Plot win/loss analysis"""
        if 'pnl' in trades.columns:
            wins = trades[trades['pnl'] > 0]['pnl']
            losses = trades[trades['pnl'] < 0]['pnl']
            
            # Create box plot
            data = [wins, losses]
            labels = [f'Wins (n={len(wins)})', f'Losses (n={len(losses)})']
            
            ax.boxplot(data, labels=labels)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('P&L')
            ax.grid(True, alpha=0.3)
            
    def _plot_metrics_table(self, metrics: RiskMetrics, ax: plt.Axes):
        """Plot metrics table"""
        # Create metrics data
        metrics_data = [
            ['Total Return', f'{metrics.total_return:.2%}'],
            ['Annual Return', f'{metrics.annualized_return:.2%}'],
            ['Volatility', f'{metrics.volatility:.2%}'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Sortino Ratio', f'{metrics.sortino_ratio:.2f}'],
            ['Max Drawdown', f'{metrics.max_drawdown:.2%}'],
            ['Calmar Ratio', f'{metrics.calmar_ratio:.2f}'],
            ['Win Rate', f'{metrics.win_rate:.2%}'],
            ['Profit Factor', f'{metrics.profit_factor:.2f}'],
            ['VaR (95%)', f'{metrics.value_at_risk:.2%}'],
            ['CVaR (95%)', f'{metrics.conditional_value_at_risk:.2%}'],
            ['Skewness', f'{metrics.skewness:.2f}'],
            ['Kurtosis', f'{metrics.kurtosis:.2f}'],
            ['Kelly Criterion', f'{metrics.kelly_criterion:.2%}']
        ]
        
        # Create table
        table = ax.table(cellText=metrics_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Remove axes
        ax.axis('off')
        
    def _generate_comparison_plots(self, reports: Dict[str, BacktestReport]):
        """Generate comparison plots for multiple strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curves comparison
        ax1 = axes[0, 0]
        for name, report in reports.items():
            normalized_equity = report.equity_curve['equity'] / report.equity_curve['equity'].iloc[0]
            ax1.plot(report.equity_curve.index, normalized_equity, label=name, linewidth=2)
        ax1.set_title('Normalized Equity Curves')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk-Return scatter
        ax2 = axes[0, 1]
        for name, report in reports.items():
            metrics = report.risk_metrics
            ax2.scatter(metrics.volatility * 100, metrics.annualized_return * 100, 
                       s=100, label=name)
            ax2.annotate(name, (metrics.volatility * 100, metrics.annualized_return * 100))
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Annual Return (%)')
        ax2.set_title('Risk-Return Profile')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown comparison
        ax3 = axes[1, 0]
        for name, report in reports.items():
            ax3.plot(report.drawdown_curve.index, -report.drawdown_curve['drawdown'] * 100, 
                    label=name, linewidth=1)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Drawdown Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(top=0)
        
        # 4. Metrics comparison bar chart
        ax4 = axes[1, 1]
        metrics_names = ['Sharpe', 'Sortino', 'Calmar']
        x = np.arange(len(metrics_names))
        width = 0.8 / len(reports)
        
        for i, (name, report) in enumerate(reports.items()):
            metrics = report.risk_metrics
            values = [metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio]
            ax4.bar(x + i * width, values, width, label=name)
            
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Value')
        ax4.set_title('Risk-Adjusted Returns Comparison')
        ax4.set_xticks(x + width * (len(reports) - 1) / 2)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = self.output_dir / "strategy_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    # Individual plot saving methods
    def _save_equity_curve_plot(self, equity_curve: pd.DataFrame, path: Path):
        """Save equity curve plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_equity_curve(equity_curve, ax)
        ax.set_title('Portfolio Equity Curve')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_drawdown_plot(self, drawdown_curve: pd.DataFrame, path: Path):
        """Save drawdown plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_drawdown(drawdown_curve, ax)
        ax.set_title('Portfolio Drawdown')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_monthly_returns_heatmap(self, monthly_returns: pd.DataFrame, path: Path):
        """Save monthly returns heatmap"""
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_monthly_returns_heatmap(monthly_returns, ax)
        ax.set_title('Monthly Returns Heatmap')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_returns_distribution(self, equity_curve: pd.DataFrame, path: Path):
        """Save returns distribution plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_returns_distribution(equity_curve, ax)
        ax.set_title('Daily Returns Distribution')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_rolling_metrics_plot(self, rolling_metrics: pd.DataFrame, path: Path):
        """Save rolling metrics plot"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        self._plot_rolling_sharpe(rolling_metrics, ax1)
        ax1.set_title('Rolling Sharpe Ratio (252-day)')
        
        self._plot_rolling_volatility(rolling_metrics, ax2)
        ax2.set_title('Rolling Volatility (252-day)')
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_trade_analysis_plots(self, trades: pd.DataFrame, output_dir: Path):
        """Save trade analysis plots"""
        if 'pnl' not in trades.columns:
            return
            
        # P&L distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_trade_analysis(trades, ax)
        ax.set_title('Trade P&L Distribution')
        plt.savefig(output_dir / 'trade_pnl_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Win/Loss analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_win_loss_analysis(trades, ax)
        ax.set_title('Win/Loss Analysis')
        plt.savefig(output_dir / 'win_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cumulative P&L
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative_pnl = trades['pnl'].cumsum()
        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.3)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L')
        ax.set_title('Cumulative Trade P&L')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'cumulative_pnl.png', dpi=300, bbox_inches='tight')
        plt.close()
