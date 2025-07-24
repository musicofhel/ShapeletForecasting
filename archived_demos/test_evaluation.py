"""
Test script for evaluation components
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig
from src.evaluation.trading_simulator import TradingSimulator, TradingCosts, Order, OrderType, OrderSide
from src.evaluation.risk_analyzer import RiskAnalyzer
from src.evaluation.performance_reporter import PerformanceReporter
from src.evaluation.market_regime_analyzer import MarketRegimeAnalyzer


def generate_test_data():
    """Generate test market data"""
    print("Generating test data...")
    
    # Create synthetic price data
    dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
    n_days = len(dates)
    
    # Generate price with trend and noise
    trend = np.linspace(100, 120, n_days)
    noise = np.random.normal(0, 2, n_days)
    price = trend + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': price * 0.99,
        'high': price * 1.01,
        'low': price * 0.98,
        'close': price,
        'volume': np.random.randint(1000000, 5000000, n_days),
        'returns': np.concatenate([[0], np.diff(price) / price[:-1]])
    }, index=dates)
    
    return data


def test_backtest_engine():
    """Test BacktestEngine"""
    print("\n" + "="*50)
    print("Testing BacktestEngine...")
    
    # Generate test data
    data = generate_test_data()
    
    # Generate mock predictions
    predictions = np.random.normal(0, 0.001, len(data))
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.1,
        max_positions=5,
        commission=0.001,
        slippage=0.0005
    )
    
    # Initialize engine
    engine = BacktestEngine(config)
    
    # Define simple signal generator
    def signal_generator(preds, data, threshold=0.001):
        signals = np.zeros(len(preds))
        signals[preds > threshold] = 1
        signals[preds < -threshold] = -1
        return signals
    
    # Run backtest
    results = engine.run_backtest(
        data=data,
        predictions=predictions,
        signal_generator=signal_generator
    )
    
    # Display results
    print(f"✓ Backtest completed")
    print(f"  Final equity: ${results['final_equity']:,.2f}")
    print(f"  Total return: {results['total_return']:.2%}")
    print(f"  Max drawdown: {results['max_drawdown']:.2%}")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Total trades: {results['trade_statistics']['total_trades']}")
    
    return results


def test_risk_analyzer():
    """Test RiskAnalyzer"""
    print("\n" + "="*50)
    print("Testing RiskAnalyzer...")
    
    # Generate test returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    # Create mock trades
    trades = pd.DataFrame({
        'entry_date': pd.date_range('2023-01-01', periods=10, freq='10D'),
        'exit_date': pd.date_range('2023-01-05', periods=10, freq='10D'),
        'pnl': np.random.normal(100, 500, 10),
        'return': np.random.normal(0.01, 0.05, 10)
    })
    
    # Initialize analyzer
    analyzer = RiskAnalyzer()
    
    # Analyze metrics
    metrics = analyzer.analyze(returns, trades=trades)
    
    # Display metrics
    print(f"✓ Risk analysis completed")
    print(f"  Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Max drawdown: {metrics.max_drawdown:.2%}")
    print(f"  VaR (95%): {metrics.value_at_risk:.2%}")
    print(f"  Win rate: {metrics.win_rate:.2%}")
    
    # Test stress testing
    stress_results = analyzer.stress_test(returns)
    print(f"\n✓ Stress tests completed")
    print(f"  Scenarios tested: {len(stress_results)}")
    
    return metrics


def test_trading_simulator():
    """Test TradingSimulator"""
    print("\n" + "="*50)
    print("Testing TradingSimulator...")
    
    # Configure costs
    costs = TradingCosts(
        commission_rate=0.001,
        commission_minimum=1.0,
        spread_cost=0.0001,
        market_impact=0.0005,
        borrowing_cost=0.02
    )
    
    # Initialize simulator
    simulator = TradingSimulator(
        initial_capital=100000,
        costs=costs,
        max_positions=10,
        allow_short=True
    )
    
    # Generate test data
    data = generate_test_data()
    
    # Simulate trading
    for i, (date, row) in enumerate(data.head(20).iterrows()):
        simulator.record_state(date)
        simulator.process_orders(row)
        
        # Place some test orders
        if i % 5 == 0:
            order = Order(
                symbol='TEST',
                side=OrderSide.BUY if i % 10 == 0 else OrderSide.SELL,
                quantity=100,
                order_type=OrderType.MARKET
            )
            simulator.place_order(order)
    
    # Generate report
    report = simulator.generate_report()
    
    print(f"✓ Trading simulation completed")
    print(f"  Total trades: {report['summary']['total_trades']}")
    print(f"  Total commission: ${report['summary']['total_commission']:.2f}")
    print(f"  Final value: ${report['summary']['final_portfolio_value']:,.2f}")
    
    return report


def test_market_regime_analyzer():
    """Test MarketRegimeAnalyzer"""
    print("\n" + "="*50)
    print("Testing MarketRegimeAnalyzer...")
    
    # Generate test data with different regimes
    dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
    n_days = len(dates)
    
    # Create data with distinct regimes
    price = np.zeros(n_days)
    
    # Bull market (days 0-90)
    price[:90] = 100 + np.linspace(0, 20, 90) + np.random.normal(0, 1, 90)
    
    # Bear market (days 90-180)
    price[90:180] = 120 - np.linspace(0, 30, 90) + np.random.normal(0, 2, 90)
    
    # Sideways (days 180-270)
    price[180:270] = 90 + np.random.normal(0, 0.5, 90)
    
    # Volatile (days 270+)
    price[270:] = 90 + np.random.normal(0, 5, n_days-270)
    
    data = pd.DataFrame({
        'open': price * 0.99,
        'high': price * 1.01,
        'low': price * 0.98,
        'close': price,
        'returns': np.concatenate([[0], np.diff(price) / price[:-1]]),
        'volume': np.random.randint(1000000, 5000000, n_days)
    }, index=dates)
    
    # Initialize analyzer
    analyzer = MarketRegimeAnalyzer(n_regimes=4)
    
    # Identify regimes
    regime_data = analyzer.identify_regimes(data, lookback=20)
    
    print(f"✓ Market regime analysis completed")
    print(f"  Regimes identified: {analyzer.n_regimes}")
    print(f"  Current regime: {regime_data['regime'].iloc[-1]}")
    
    # Create mock backtest results
    mock_results = {
        'equity_curve': pd.DataFrame({
            'date': data.index,
            'equity': 100000 + np.cumsum(np.random.normal(50, 200, n_days))
        }).set_index('date'),
        'trades': pd.DataFrame({
            'entry_date': pd.date_range('2022-01-01', periods=50, freq='W'),
            'exit_date': pd.date_range('2022-01-08', periods=50, freq='W'),
            'pnl': np.random.normal(100, 500, 50)
        })
    }
    
    # Analyze performance by regime
    performance = analyzer.analyze_performance_by_regime(mock_results, regime_data)
    
    print(f"\nPerformance by regime:")
    for regime, metrics in performance.items():
        print(f"  {regime}: {metrics['days_in_regime']} days, "
              f"return: {metrics['total_return']:.2%}")
    
    return regime_data


def test_performance_reporter():
    """Test PerformanceReporter"""
    print("\n" + "="*50)
    print("Testing PerformanceReporter...")
    
    # Create mock backtest results
    dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
    equity = 100000 + np.cumsum(np.random.normal(50, 200, len(dates)))
    
    # Calculate drawdown
    cummax = pd.Series(equity).cummax()
    drawdown = (equity - cummax) / cummax
    
    mock_results = {
        'equity_curve': pd.DataFrame({
            'date': dates,
            'equity': equity,
            'cash': equity * 0.3,
            'positions_value': equity * 0.7
        }).set_index('date'),
        'drawdown_curve': pd.DataFrame({
            'date': dates,
            'drawdown': drawdown
        }).set_index('date'),
        'trades': pd.DataFrame({
            'entry_date': pd.date_range('2022-01-01', periods=100, freq='3D'),
            'exit_date': pd.date_range('2022-01-04', periods=100, freq='3D'),
            'symbol': 'TEST',
            'side': np.random.choice(['long', 'short'], 100),
            'quantity': np.random.randint(10, 100, 100),
            'entry_price': np.random.uniform(90, 110, 100),
            'exit_price': np.random.uniform(90, 110, 100),
            'pnl': np.random.normal(50, 200, 100),
            'return': np.random.normal(0.005, 0.02, 100),
            'commission': np.random.uniform(1, 10, 100)
        }),
        'final_equity': equity[-1],
        'total_return': (equity[-1] - 100000) / 100000,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.15,
        'trade_statistics': {
            'total_trades': 100,
            'winning_trades': 60,
            'losing_trades': 40,
            'win_rate': 0.6,
            'avg_win': 150,
            'avg_loss': -75,
            'profit_factor': 2.0
        }
    }
    
    # Initialize reporter
    reporter = PerformanceReporter(output_dir="test_reports")
    
    # Generate report
    report = reporter.generate_report(
        backtest_results=mock_results,
        strategy_name="test_strategy",
        save_report=False
    )
    
    print(f"✓ Performance report generated")
    print(f"  Total return: {report.summary['total_return']:.2%}")
    print(f"  Sharpe ratio: {report.summary['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {report.summary['max_drawdown']:.2%}")
    print(f"  Win rate: {report.summary['win_rate']:.2%}")
    
    # Test strategy comparison
    reports = {
        'strategy1': report,
        'strategy2': report  # Using same report for demo
    }
    
    comparison = reporter.compare_strategies(reports)
    print(f"\n✓ Strategy comparison completed")
    print(f"  Strategies compared: {len(comparison)}")
    
    return report


def run_all_tests():
    """Run all evaluation tests"""
    print("Starting Evaluation Component Tests")
    print("="*50)
    
    try:
        # Test each component
        backtest_results = test_backtest_engine()
        risk_metrics = test_risk_analyzer()
        trading_report = test_trading_simulator()
        regime_data = test_market_regime_analyzer()
        performance_report = test_performance_reporter()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        print("\nSummary:")
        print("- BacktestEngine: ✓")
        print("- RiskAnalyzer: ✓")
        print("- TradingSimulator: ✓")
        print("- MarketRegimeAnalyzer: ✓")
        print("- PerformanceReporter: ✓")
        
        print("\nThe evaluation framework is ready for use!")
        print("Run 'python run_backtest.py' for comprehensive backtesting.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
