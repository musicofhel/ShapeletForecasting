"""
Comprehensive Backtesting Script for Financial Wavelet Prediction Models

This script runs full backtesting with walk-forward analysis, trading simulation,
risk analysis, and performance reporting.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
import json
from pathlib import Path

# Import all components
from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig, WalkForwardBacktest
from src.evaluation.trading_simulator import TradingSimulator, TradingCosts, Order, OrderType, OrderSide
from src.evaluation.risk_analyzer import RiskAnalyzer
from src.evaluation.performance_reporter import PerformanceReporter
from src.evaluation.market_regime_analyzer import MarketRegimeAnalyzer

# Import models and features
from src.models.model_trainer import ModelTrainer
from src.features.feature_pipeline import FeaturePipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_market_data(symbol: str = "SPY", 
                    start_date: str = "2018-01-01",
                    end_date: str = "2024-01-01") -> pd.DataFrame:
    """Load historical market data"""
    logger.info(f"Loading market data for {symbol} from {start_date} to {end_date}")
    
    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Add returns
    data['returns'] = data['close'].pct_change()
    
    return data


def generate_trading_signals(predictions: np.ndarray, 
                           data: pd.DataFrame,
                           threshold: float = 0.001) -> np.ndarray:
    """
    Generate trading signals from model predictions
    
    Args:
        predictions: Model predictions (expected returns)
        data: Market data
        threshold: Minimum predicted return to generate signal
        
    Returns:
        Array of signals (-1, 0, 1)
    """
    signals = np.zeros(len(predictions))
    
    # Generate signals based on predicted returns
    signals[predictions > threshold] = 1  # Long signal
    signals[predictions < -threshold] = -1  # Short signal
    
    # Apply additional filters
    # 1. Trend filter - only trade in direction of trend
    sma_20 = data['close'].rolling(20).mean()
    sma_50 = data['close'].rolling(50).mean()
    trend = (sma_20 > sma_50).astype(int) * 2 - 1  # 1 for uptrend, -1 for downtrend
    
    # Only allow longs in uptrend, shorts in downtrend
    for i in range(len(signals)):
        if signals[i] * trend.iloc[i] < 0:
            signals[i] = 0
            
    # 2. Volatility filter - reduce position size in high volatility
    volatility = data['returns'].rolling(20).std()
    vol_percentile = volatility.rank(pct=True)
    
    # Reduce signals in high volatility (top 20%)
    high_vol_mask = vol_percentile > 0.8
    signals[high_vol_mask] *= 0.5
    
    return signals


def run_single_backtest(model_name: str = "ensemble"):
    """Run backtest for a single model"""
    logger.info(f"Running backtest for {model_name} model")
    
    # Load data
    data = load_market_data()
    
    # Split data
    train_end = "2022-01-01"
    test_start = "2022-01-01"
    
    train_data = data[data.index < train_end]
    test_data = data[data.index >= test_start]
    
    # Load trained model
    model_path = Path(f"models/{model_name}_model.pkl")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
        
    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline()
    
    # Extract features
    logger.info("Extracting features...")
    train_features = feature_pipeline.fit_transform(train_data)
    test_features = feature_pipeline.transform(test_data)
    
    # Load model and generate predictions
    import joblib
    model = joblib.load(model_path)
    
    # Generate predictions
    if hasattr(model, 'predict_proba'):
        # For classification models
        predictions = model.predict_proba(test_features)[:, 1]
        predictions = (predictions - 0.5) * 0.02  # Convert to return predictions
    else:
        # For regression models
        predictions = model.predict(test_features)
        
    # Ensure predictions align with test data
    predictions = predictions[:len(test_data)]
    
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
    
    # Initialize backtesting engine
    engine = BacktestEngine(config)
    
    # Run backtest
    logger.info("Running backtest...")
    results = engine.run_backtest(
        data=test_data,
        predictions=predictions,
        signal_generator=generate_trading_signals
    )
    
    return results


def run_walk_forward_backtest(model_name: str = "ensemble"):
    """Run walk-forward backtest"""
    logger.info(f"Running walk-forward backtest for {model_name} model")
    
    # Load data
    data = load_market_data()
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.1,
        max_positions=5,
        commission=0.001,
        slippage=0.0005
    )
    
    # Initialize engines
    backtest_engine = BacktestEngine(config)
    walk_forward = WalkForwardBacktest(
        backtest_engine=backtest_engine,
        train_period=252,  # 1 year training
        test_period=63,    # 3 months testing
        step_size=21       # 1 month step
    )
    
    # Define model trainer function
    def model_trainer(train_data):
        # Initialize and train model
        from src.models.ensemble_model import EnsembleModel
        model = EnsembleModel()
        
        # Prepare features and targets
        feature_pipeline = FeaturePipeline()
        features = feature_pipeline.fit_transform(train_data)
        
        # Create targets (next day returns)
        targets = train_data['returns'].shift(-1).fillna(0)
        targets = targets.iloc[:len(features)]
        
        # Train model
        model.fit(features, targets)
        return model
        
    # Define feature extractor
    def feature_extractor(data):
        feature_pipeline = FeaturePipeline()
        return feature_pipeline.transform(data)
        
    # Run walk-forward backtest
    results = walk_forward.run(
        data=data,
        model_trainer=model_trainer,
        signal_generator=generate_trading_signals,
        feature_extractor=feature_extractor
    )
    
    return results


def run_trading_simulation():
    """Run realistic trading simulation"""
    logger.info("Running trading simulation...")
    
    # Load data
    data = load_market_data()
    test_data = data[data.index >= "2022-01-01"]
    
    # Configure trading costs
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
        allow_short=True,
        use_leverage=False
    )
    
    # Load predictions (using ensemble model)
    results = run_single_backtest("ensemble")
    if results is None:
        return None
        
    # Simulate trading
    for i, (date, row) in enumerate(test_data.iterrows()):
        # Record state
        simulator.record_state(date)
        
        # Process orders
        simulator.process_orders(row)
        
        # Update positions
        simulator.update_positions(pd.DataFrame([row]))
        
        # Generate new orders based on signals
        if i < len(results['trades']) and results['trades'].iloc[i]['type'] == 'open':
            trade = results['trades'].iloc[i]
            
            order = Order(
                symbol='SPY',
                side=OrderSide.BUY if trade['side'] == 'long' else OrderSide.SELL,
                quantity=trade['quantity'],
                order_type=OrderType.MARKET
            )
            
            simulator.place_order(order)
            
    # Generate report
    report = simulator.generate_report()
    return report


def analyze_market_regimes():
    """Analyze performance across market regimes"""
    logger.info("Analyzing market regimes...")
    
    # Load data
    data = load_market_data()
    
    # Run backtest
    results = run_single_backtest("ensemble")
    if results is None:
        return None
        
    # Initialize regime analyzer
    regime_analyzer = MarketRegimeAnalyzer(n_regimes=4)
    
    # Generate regime report
    regime_report = regime_analyzer.generate_regime_report(
        market_data=data,
        backtest_results=results,
        lookback=20
    )
    
    # Plot regime analysis
    regime_analyzer.plot_regime_analysis(
        market_data=data,
        regime_data=regime_report['regime_data'],
        save_path="reports/regime_analysis.png"
    )
    
    return regime_report


def compare_all_models():
    """Compare performance of all models"""
    logger.info("Comparing all models...")
    
    # List of models to compare
    models = ["lstm", "gru", "transformer", "xgboost", "ensemble"]
    
    # Initialize reporter
    reporter = PerformanceReporter(output_dir="reports")
    
    # Run backtests for each model
    reports = {}
    
    for model_name in models:
        logger.info(f"Backtesting {model_name} model...")
        
        # Run backtest
        results = run_single_backtest(model_name)
        
        if results is not None:
            # Generate report
            report = reporter.generate_report(
                backtest_results=results,
                strategy_name=model_name,
                save_report=True
            )
            
            reports[model_name] = report
            
    # Compare strategies
    if reports:
        comparison_df = reporter.compare_strategies(reports)
        logger.info("\nStrategy Comparison:")
        logger.info(comparison_df.to_string())
        
    return reports


def run_comprehensive_evaluation():
    """Run complete evaluation pipeline"""
    logger.info("Starting comprehensive evaluation...")
    
    # 1. Run single backtest for ensemble model
    logger.info("\n1. Running single backtest...")
    single_results = run_single_backtest("ensemble")
    
    if single_results:
        # 2. Analyze risk metrics
        logger.info("\n2. Analyzing risk metrics...")
        risk_analyzer = RiskAnalyzer()
        
        equity_curve = single_results['equity_curve']
        returns = equity_curve['equity'].pct_change().dropna()
        
        risk_metrics = risk_analyzer.analyze(
            returns=returns,
            trades=single_results['trades']
        )
        
        logger.info(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        logger.info(f"Win Rate: {risk_metrics.win_rate:.2%}")
        
        # 3. Run stress tests
        logger.info("\n3. Running stress tests...")
        stress_results = risk_analyzer.stress_test(returns)
        
        for scenario, metrics in stress_results.items():
            logger.info(f"\n{scenario}:")
            logger.info(f"  Return: {metrics['total_return']:.2%}")
            logger.info(f"  Max DD: {metrics['max_drawdown']:.2%}")
            
    # 4. Run walk-forward analysis
    logger.info("\n4. Running walk-forward analysis...")
    wf_results = run_walk_forward_backtest("ensemble")
    
    if wf_results:
        logger.info(f"Walk-Forward Results:")
        logger.info(f"  Mean Return: {wf_results['overall_statistics']['mean_return']:.2%}")
        logger.info(f"  Consistency: {wf_results['overall_statistics']['consistency']:.2%}")
        
    # 5. Run trading simulation
    logger.info("\n5. Running trading simulation...")
    sim_results = run_trading_simulation()
    
    if sim_results:
        logger.info(f"Trading Simulation Results:")
        logger.info(f"  Total Return: {sim_results['summary']['total_return']:.2%}")
        logger.info(f"  Total Commission: ${sim_results['summary']['total_commission']:.2f}")
        
    # 6. Analyze market regimes
    logger.info("\n6. Analyzing market regimes...")
    regime_results = analyze_market_regimes()
    
    if regime_results:
        logger.info("\nPerformance by Market Regime:")
        for regime, metrics in regime_results['performance_by_regime'].items():
            logger.info(f"\n{regime}:")
            logger.info(f"  Days: {metrics['days_in_regime']}")
            logger.info(f"  Return: {metrics['total_return']:.2%}")
            logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            
    # 7. Compare all models
    logger.info("\n7. Comparing all models...")
    model_comparison = compare_all_models()
    
    logger.info("\nEvaluation complete! Check the 'reports' directory for detailed results.")


if __name__ == "__main__":
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Run comprehensive evaluation
    run_comprehensive_evaluation()
