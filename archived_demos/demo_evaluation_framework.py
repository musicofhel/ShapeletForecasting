"""
Demonstration of the Financial Wavelet Prediction Evaluation Framework
This script showcases all evaluation components with synthetic data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig, WalkForwardBacktest
from src.evaluation.trading_simulator import TradingSimulator, TradingCosts, Order, OrderType, OrderSide
from src.evaluation.risk_analyzer import RiskAnalyzer
from src.evaluation.performance_reporter import PerformanceReporter
from src.evaluation.market_regime_analyzer import MarketRegimeAnalyzer


def generate_synthetic_market_data(n_days=1000, ticker='SYNTH'):
    """Generate synthetic market data with realistic patterns"""
    print(f"Generating {n_days} days of synthetic market data...")
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate price with trend, seasonality, and noise
    trend = np.linspace(100, 150, n_days)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 252)
    noise = np.random.normal(0, 2, n_days)
    
    # Add some volatility clusters
    volatility = np.ones(n_days)
    for i in range(5):
        start = np.random.randint(0, n_days - 50)
        volatility[start:start+50] *= 2
    
    price = trend + seasonality + noise * volatility
    
    # Ensure positive prices
    price = np.maximum(price, 10)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': price * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': price * (1 + np.random.uniform(-0.02, 0, n_days)),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, n_days),
        'returns': np.concatenate([[0], np.diff(price) / price[:-1]])
    }, index=dates)
    
    return data


def generate_synthetic_predictions(data, strategy='momentum'):
    """Generate synthetic predictions based on simple strategies"""
    n = len(data)
    
    if strategy == 'momentum':
        # Momentum strategy: predict based on recent returns
        returns = data['returns'].values
        predictions = np.zeros(n)
        for i in range(20, n):
            recent_return = np.mean(returns[i-20:i])
            predictions[i] = recent_return * 0.5 + np.random.normal(0, 0.001)
            
    elif strategy == 'mean_reversion':
        # Mean reversion strategy
        prices = data['close'].values
        predictions = np.zeros(n)
        for i in range(50, n):
            mean_price = np.mean(prices[i-50:i])
            deviation = (prices[i] - mean_price) / mean_price
            predictions[i] = -deviation * 0.1 + np.random.normal(0, 0.001)
            
    elif strategy == 'random':
        # Random predictions
        predictions = np.random.normal(0, 0.002, n)
        
    else:
        # Combined strategy
        mom_preds = generate_synthetic_predictions(data, 'momentum')
        mr_preds = generate_synthetic_predictions(data, 'mean_reversion')
        predictions = 0.6 * mom_preds + 0.4 * mr_preds
        
    return predictions


def demonstrate_backtest_engine():
    """Demonstrate the BacktestEngine functionality"""
    print("\n" + "="*60)
    print("DEMONSTRATING BACKTEST ENGINE")
    print("="*60)
    
    # Generate data
    data = generate_synthetic_market_data(500)
    predictions = generate_synthetic_predictions(data, 'momentum')
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.1,
        max_positions=5,
        commission=0.001,
        slippage=0.0005,
        stop_loss=0.02,
        take_profit=0.05
    )
    
    # Initialize engine
    engine = BacktestEngine(config)
    
    # Define signal generator
    def signal_generator(preds, data, threshold=0.001):
        signals = np.zeros(len(preds))
        signals[preds > threshold] = 1
        signals[preds < -threshold] = -1
        return signals
    
    # Run backtest
    print("\nRunning backtest...")
    results = engine.run_backtest(
        data=data,
        predictions=predictions,
        signal_generator=signal_generator
    )
    
    # Display results
    print("\nBacktest Results:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    if 'sortino_ratio' in results:
        print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
    if 'calmar_ratio' in results:
        print(f"  Calmar Ratio: {results['calmar_ratio']:.2f}")
    
    print("\nTrade Statistics:")
    stats = results['trade_statistics']
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2%}")
    if 'avg_win' in stats:
        print(f"  Average Win: ${stats['avg_win']:.2f}")
    if 'avg_loss' in stats:
        print(f"  Average Loss: ${stats['avg_loss']:.2f}")
    if 'profit_factor' in stats:
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    return results


def demonstrate_walk_forward():
    """Demonstrate walk-forward backtesting"""
    print("\n" + "="*60)
    print("DEMONSTRATING WALK-FORWARD ANALYSIS")
    print("="*60)
    
    # Generate longer dataset
    data = generate_synthetic_market_data(1000)
    
    # Configure backtest engine
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.1,
        max_positions=5,
        commission=0.001
    )
    engine = BacktestEngine(config)
    
    # Create walk-forward instance
    walk_forward = WalkForwardBacktest(
        backtest_engine=engine,
        train_period=200,
        test_period=50,
        step_size=50
    )
    
    # Define model trainer function
    def model_trainer(train_data):
        # In real scenario, this would train a model
        # Here we create a simple predictor
        class SimpleModel:
            def __init__(self, data):
                self.data = data
                
            def predict(self, test_data):
                return generate_synthetic_predictions(test_data, 'combined')
                
        return SimpleModel(train_data)
    
    # Define signal generator
    def signal_generator(preds, data):
        signals = np.zeros(len(preds))
        signals[preds > 0.001] = 1
        signals[preds < -0.001] = -1
        return signals
    
    # Run walk-forward
    print("\nRunning walk-forward analysis...")
    print(f"  Train window: {walk_forward.train_period} days")
    print(f"  Test window: {walk_forward.test_period} days")
    print(f"  Step size: {walk_forward.step_size} days")
    
    results = walk_forward.run(
        data=data,
        model_trainer=model_trainer,
        signal_generator=signal_generator
    )
    
    print(f"\nWalk-Forward Results:")
    print(f"  Windows tested: {results['num_windows']}")
    stats = results['overall_statistics']
    print(f"  Average return per window: {stats['mean_return']:.2%}")
    print(f"  Return consistency: {stats['consistency']:.1%}")
    print(f"  Average Sharpe ratio: {stats['mean_sharpe']:.2f}")
    print(f"  Average max drawdown: {stats['mean_max_drawdown']:.2%}")
    print(f"  Worst drawdown: {stats['worst_drawdown']:.2%}")
    
    return results


def demonstrate_risk_analysis():
    """Demonstrate risk analysis capabilities"""
    print("\n" + "="*60)
    print("DEMONSTRATING RISK ANALYSIS")
    print("="*60)
    
    # Generate returns and trades
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    trades = pd.DataFrame({
        'entry_date': pd.date_range('2023-01-01', periods=50, freq='5D'),
        'exit_date': pd.date_range('2023-01-03', periods=50, freq='5D'),
        'pnl': np.random.normal(100, 500, 50),
        'return': np.random.normal(0.01, 0.05, 50)
    })
    
    # Initialize analyzer
    analyzer = RiskAnalyzer()
    
    # Analyze metrics
    print("\nCalculating risk metrics...")
    metrics = analyzer.analyze(returns, trades=trades)
    
    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Max Drawdown Duration: {metrics.max_drawdown_duration} days")
    print(f"  Value at Risk (95%): {metrics.value_at_risk:.2%}")
    if hasattr(metrics, 'conditional_var'):
        print(f"  Conditional VaR (95%): {metrics.conditional_var:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Recovery Factor: {metrics.recovery_factor:.2f}")
    
    # Stress testing
    print("\nRunning stress tests...")
    stress_results = analyzer.stress_test(returns)
    
    print("\nStress Test Results:")
    for scenario, result in stress_results.items():
        print(f"  {scenario}:")
        if isinstance(result, dict):
            if 'total_return' in result:
                print(f"    Return: {result['total_return']:.2%}")
            if 'max_drawdown' in result:
                print(f"    Max Drawdown: {result['max_drawdown']:.2%}")
            if 'sharpe_ratio' in result:
                print(f"    Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        else:
            print(f"    Result: {result}")
        
    return metrics


def demonstrate_market_regimes():
    """Demonstrate market regime analysis"""
    print("\n" + "="*60)
    print("DEMONSTRATING MARKET REGIME ANALYSIS")
    print("="*60)
    
    # Generate data with distinct regimes
    n_days = 800
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Create different market regimes
    price = np.zeros(n_days)
    
    # Bull market (days 0-200)
    price[:200] = 100 + np.linspace(0, 30, 200) + np.random.normal(0, 1, 200)
    
    # Bear market (days 200-400)
    price[200:400] = 130 - np.linspace(0, 40, 200) + np.random.normal(0, 2, 200)
    
    # Sideways (days 400-600)
    price[400:600] = 90 + np.random.normal(0, 0.5, 200)
    
    # Volatile (days 600-800)
    price[600:] = 90 + np.random.normal(0, 5, 200)
    
    data = pd.DataFrame({
        'open': price * 0.99,
        'high': price * 1.01,
        'low': price * 0.98,
        'close': price,
        'volume': np.random.randint(1000000, 5000000, n_days),
        'returns': np.concatenate([[0], np.diff(price) / price[:-1]])
    }, index=dates)
    
    # Initialize analyzer
    analyzer = MarketRegimeAnalyzer(n_regimes=4)
    
    # Identify regimes
    print("\nIdentifying market regimes...")
    regime_data = analyzer.identify_regimes(data, lookback=20)
    
    # Analyze regime characteristics
    print("\nRegime Analysis:")
    for regime in range(4):
        regime_days = (regime_data['regime'] == regime).sum()
        if regime_days > 0:
            regime_name = analyzer.regime_labels.get(regime, f"Regime {regime}")
            print(f"\n  {regime_name}:")
            print(f"    Days in regime: {regime_days}")
            print(f"    Percentage of time: {regime_days/len(regime_data):.1%}")
    
    # Calculate transition matrix
    print("\n\nRegime Transition Matrix:")
    transition_matrix = analyzer.calculate_regime_transition_matrix(regime_data)
    print(transition_matrix.round(2))
    
    # Analyze persistence
    print("\n\nRegime Persistence Analysis:")
    persistence = analyzer.analyze_regime_persistence(regime_data)
    for regime, stats in persistence.items():
        print(f"\n  {regime}:")
        print(f"    Average duration: {stats['avg_duration_days']:.1f} days")
        print(f"    Max duration: {stats['max_duration_days']} days")
        print(f"    Occurrences: {stats['num_occurrences']}")
    
    return regime_data


def demonstrate_performance_reporting():
    """Demonstrate performance reporting"""
    print("\n" + "="*60)
    print("DEMONSTRATING PERFORMANCE REPORTING")
    print("="*60)
    
    # Run multiple strategies
    strategies = ['momentum', 'mean_reversion', 'combined']
    all_results = {}
    
    for strategy in strategies:
        print(f"\nRunning backtest for {strategy} strategy...")
        
        # Generate data and predictions
        data = generate_synthetic_market_data(500)
        predictions = generate_synthetic_predictions(data, strategy)
        
        # Run backtest
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        
        def signal_generator(preds, data):
            signals = np.zeros(len(preds))
            signals[preds > 0.001] = 1
            signals[preds < -0.001] = -1
            return signals
        
        results = engine.run_backtest(data, predictions, signal_generator)
        all_results[strategy] = results
    
    # Initialize reporter
    reporter = PerformanceReporter(output_dir="evaluation_reports")
    
    # Generate individual reports
    print("\nGenerating performance reports...")
    reports = {}
    for strategy, results in all_results.items():
        report = reporter.generate_report(
            backtest_results=results,
            strategy_name=strategy,
            save_report=False  # Don't save to disk for demo
        )
        reports[strategy] = report
        
        print(f"\n{strategy.upper()} Strategy Performance:")
        print(f"  Total Return: {report.summary['total_return']:.2%}")
        print(f"  Sharpe Ratio: {report.summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {report.summary['max_drawdown']:.2%}")
        print(f"  Win Rate: {report.summary['win_rate']:.2%}")
    
    # Compare strategies
    print("\n\nStrategy Comparison:")
    comparison = reporter.compare_strategies(reports)
    
    # Create comparison table
    if comparison and all(strategy in comparison for strategy in strategies):
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        print("\n{:<20} {:>15} {:>15} {:>15}".format("Metric", *strategies))
        print("-" * 65)
        
        for metric in metrics:
            values = []
            for strategy in strategies:
                if strategy in comparison and metric in comparison[strategy]:
                    values.append(comparison[strategy][metric])
                else:
                    values.append(0.0)
                    
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                print("{:<20} {:>14.2%} {:>14.2%} {:>14.2%}".format(metric, *values))
            else:
                print("{:<20} {:>14.2f} {:>14.2f} {:>14.2f}".format(metric, *values))
    
    return comparison


def main():
    """Run all demonstrations"""
    print("="*60)
    print("FINANCIAL WAVELET PREDICTION - EVALUATION FRAMEWORK DEMO")
    print("="*60)
    print("\nThis demo showcases all evaluation components:")
    print("1. Backtest Engine - Single backtest simulation")
    print("2. Walk-Forward Analysis - Rolling window backtesting")
    print("3. Risk Analysis - Comprehensive risk metrics")
    print("4. Market Regime Analysis - Regime identification")
    print("5. Performance Reporting - Strategy comparison")
    
    # Run demonstrations
    backtest_results = demonstrate_backtest_engine()
    walk_forward_results = demonstrate_walk_forward()
    risk_metrics = demonstrate_risk_analysis()
    regime_data = demonstrate_market_regimes()
    comparison = demonstrate_performance_reporting()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe evaluation framework provides:")
    print("✓ Realistic backtesting with transaction costs")
    print("✓ Walk-forward analysis for robustness testing")
    print("✓ Comprehensive risk metrics and stress testing")
    print("✓ Market regime identification and analysis")
    print("✓ Professional performance reporting")
    print("\nAll components are ready for production use!")


if __name__ == "__main__":
    main()
