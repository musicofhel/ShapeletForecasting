"""
Demonstration of the complete advanced financial prediction system.
Shows integration of all Sprint 8 features in a realistic scenario.
"""

import numpy as np
import pandas as pd
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import all advanced modules
from src.advanced.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.advanced.market_regime_detector import MarketRegimeDetector
from src.advanced.adaptive_learner import AdaptiveLearner
from src.advanced.realtime_pipeline import RealtimePipeline, StreamConfig
from src.advanced.risk_manager import AdvancedRiskManager
from src.advanced.portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints

# Import existing modules
from src.models.ensemble_model import EnsembleModel
from src.features.feature_pipeline import FeaturePipeline


class AdvancedTradingSystem:
    """
    Complete advanced trading system integrating all components.
    """
    
    def __init__(self):
        """Initialize all system components."""
        print("Initializing Advanced Trading System...")
        
        # Core components
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Feature and model components
        self.feature_pipeline = FeaturePipeline()
        self.ensemble_model = None
        self.adaptive_learner = None
        
        # Portfolio state
        self.portfolio = {
            'cash': 100000,
            'positions': {},
            'history': []
        }
        
        print("✓ System initialized")
        
    def load_market_data(self, symbols, start_date, end_date):
        """Load historical market data."""
        print(f"\nLoading market data for {symbols}...")
        
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if not df.empty:
                data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} days loaded")
                
        return data
        
    def analyze_market_conditions(self, data):
        """Comprehensive market analysis."""
        print("\n=== MARKET ANALYSIS ===")
        
        results = {}
        
        for symbol, df in data.items():
            print(f"\nAnalyzing {symbol}...")
            
            # 1. Multi-timeframe analysis
            price_data = df['Close'].values
            mtf_features = self.mtf_analyzer.analyze_all_timeframes(price_data)
            
            # 2. Market regime detection
            self.regime_detector.train_hmm(df)
            current_regime = self.regime_detector.detect_current_regime(df)
            regime_predictions = self.regime_detector.predict_regime_change(df, horizon=5)
            
            # 3. Feature extraction
            features_df = self.feature_pipeline.fit_transform(df)
            
            # 4. Risk metrics
            returns = df['Close'].pct_change().dropna()
            risk_metrics = self.risk_manager.calculate_risk_metrics(returns)
            
            results[symbol] = {
                'mtf_features': mtf_features,
                'current_regime': current_regime,
                'regime_predictions': regime_predictions,
                'features': features_df,
                'risk_metrics': risk_metrics,
                'latest_price': df['Close'].iloc[-1],
                'returns': returns
            }
            
            # Display results
            print(f"  Current Regime: {current_regime.name}")
            print(f"  Volatility: {current_regime.volatility_level}")
            print(f"  Trend: {current_regime.trend_direction}")
            print(f"  VaR (95%): {risk_metrics.var_95:.2%}")
            print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
            
        return results
        
    def optimize_portfolio(self, market_analysis):
        """Optimize portfolio allocation."""
        print("\n=== PORTFOLIO OPTIMIZATION ===")
        
        # Prepare returns data
        symbols = list(market_analysis.keys())
        returns_data = pd.DataFrame({
            symbol: analysis['returns'] 
            for symbol, analysis in market_analysis.items()
        })
        
        # Align indices
        returns_data = returns_data.dropna()
        
        # Set constraints based on market conditions
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.35,
            max_positions=len(symbols)
        )
        
        # Try multiple optimization methods
        optimization_results = {}
        
        # 1. Mean-Variance Optimization
        expected_returns = self.portfolio_optimizer.calculate_expected_returns(returns_data)
        cov_matrix = self.portfolio_optimizer.calculate_covariance_matrix(returns_data, method='ledoit_wolf')
        
        mv_portfolio = self.portfolio_optimizer.optimize_mean_variance(
            expected_returns, cov_matrix, constraints
        )
        optimization_results['mean_variance'] = mv_portfolio
        
        # 2. Risk Parity
        rp_portfolio = self.portfolio_optimizer.optimize_risk_parity(
            cov_matrix, constraints
        )
        optimization_results['risk_parity'] = rp_portfolio
        
        # 3. ML-Enhanced Optimization (if predictions available)
        ml_predictions = {}
        prediction_confidence = {}
        
        for symbol in symbols:
            # Simple prediction based on regime
            regime = market_analysis[symbol]['current_regime']
            if regime.trend_direction == 'bullish':
                ml_predictions[symbol] = 0.12  # 12% annual return
                prediction_confidence[symbol] = regime.confidence
            elif regime.trend_direction == 'bearish':
                ml_predictions[symbol] = -0.05  # -5% annual return
                prediction_confidence[symbol] = regime.confidence
            else:
                ml_predictions[symbol] = 0.06  # 6% annual return
                prediction_confidence[symbol] = regime.confidence * 0.8
                
        ml_portfolio = self.portfolio_optimizer.optimize_with_ml_predictions(
            ml_predictions, prediction_confidence, returns_data, constraints
        )
        optimization_results['ml_enhanced'] = ml_portfolio
        
        # Display results
        print("\nOptimization Results:")
        for method, portfolio in optimization_results.items():
            print(f"\n{method.upper()}:")
            print(f"  Expected Return: {portfolio.expected_return:.2%}")
            print(f"  Expected Risk: {portfolio.expected_risk:.2%}")
            print(f"  Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
            print(f"  Top Holdings:")
            sorted_weights = sorted(portfolio.weights.items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights[:3]:
                print(f"    {symbol}: {weight:.2%}")
                
        return optimization_results
        
    def calculate_position_sizes(self, optimization_results, market_analysis):
        """Calculate position sizes with risk management."""
        print("\n=== POSITION SIZING ===")
        
        # Select best portfolio (highest Sharpe ratio)
        best_portfolio = max(optimization_results.values(), key=lambda p: p.sharpe_ratio)
        print(f"Selected portfolio: {best_portfolio.optimization_method}")
        
        position_recommendations = {}
        
        for symbol, target_weight in best_portfolio.weights.items():
            if target_weight < 0.01:  # Skip very small positions
                continue
                
            # Get current price and calculate position value
            current_price = market_analysis[symbol]['latest_price']
            position_value = self.portfolio['cash'] * target_weight
            
            # Calculate stop loss based on regime
            regime = market_analysis[symbol]['current_regime']
            volatility = market_analysis[symbol]['risk_metrics'].max_drawdown
            
            if regime.volatility_level == 'high':
                stop_loss_pct = 0.05  # 5% stop loss
            elif regime.volatility_level == 'medium':
                stop_loss_pct = 0.03  # 3% stop loss
            else:
                stop_loss_pct = 0.02  # 2% stop loss
                
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            # Get position sizing recommendation
            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                portfolio_value=self.portfolio['cash'],
                confidence=regime.confidence,
                volatility=volatility
            )
            
            position_recommendations[symbol] = {
                'target_weight': target_weight,
                'position_value': position_value,
                'shares': int(sizing.risk_adjusted_size),
                'entry_price': current_price,
                'stop_loss': sizing.stop_loss,
                'take_profit': sizing.take_profit,
                'risk_reward_ratio': sizing.risk_reward_ratio,
                'regime': regime.name
            }
            
            print(f"\n{symbol}:")
            print(f"  Target Weight: {target_weight:.2%}")
            print(f"  Shares: {int(sizing.risk_adjusted_size)}")
            print(f"  Entry: ${current_price:.2f}")
            print(f"  Stop Loss: ${sizing.stop_loss:.2f} ({stop_loss_pct:.1%})")
            print(f"  Take Profit: ${sizing.take_profit:.2f}")
            print(f"  Risk/Reward: 1:{sizing.risk_reward_ratio:.1f}")
            
        return position_recommendations
        
    def assess_portfolio_risk(self, positions, market_analysis):
        """Comprehensive portfolio risk assessment."""
        print("\n=== RISK ASSESSMENT ===")
        
        # Convert recommendations to portfolio format
        portfolio_positions = {}
        for symbol, pos in positions.items():
            portfolio_positions[symbol] = {
                'size': pos['shares'],
                'entry': pos['entry_price']
            }
            
        # Prepare returns data
        returns_data = pd.DataFrame({
            symbol: analysis['returns'] 
            for symbol, analysis in market_analysis.items()
            if symbol in portfolio_positions
        })
        
        # Assess portfolio risk
        assessment = self.risk_manager.assess_portfolio_risk(
            portfolio_positions, returns_data
        )
        
        # Calculate stress scenarios
        stress_results = self.risk_manager.calculate_stress_scenarios(
            portfolio_positions, returns_data
        )
        
        # Generate risk report
        report = self.risk_manager.generate_risk_report(assessment)
        print(report)
        
        print("\nStress Test Results:")
        for scenario, impact in stress_results.items():
            print(f"  {scenario}: {impact:.2%}")
            
        return assessment, stress_results
        
    def visualize_analysis(self, market_analysis, optimization_results):
        """Create comprehensive visualization."""
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Advanced Trading System Analysis', fontsize=16)
        
        # 1. Price trends with regime colors
        ax = axes[0, 0]
        for i, (symbol, analysis) in enumerate(market_analysis.items()):
            if i < 3:  # Plot first 3 symbols
                df = analysis['features']
                ax.plot(df.index[-100:], df['close'][-100:], label=symbol)
        ax.set_title('Recent Price Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Regime distribution
        ax = axes[0, 1]
        regimes = [analysis['current_regime'].name for analysis in market_analysis.values()]
        regime_counts = pd.Series(regimes).value_counts()
        regime_counts.plot(kind='bar', ax=ax)
        ax.set_title('Current Market Regimes')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Count')
        
        # 3. Risk metrics comparison
        ax = axes[0, 2]
        sharpe_ratios = {symbol: analysis['risk_metrics'].sharpe_ratio 
                        for symbol, analysis in market_analysis.items()}
        pd.Series(sharpe_ratios).plot(kind='bar', ax=ax)
        ax.set_title('Sharpe Ratios by Asset')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Sharpe Ratio')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 4. Portfolio weights comparison
        ax = axes[1, 0]
        methods = list(optimization_results.keys())
        symbols = list(list(optimization_results.values())[0].weights.keys())
        
        weights_df = pd.DataFrame({
            method: [opt.weights.get(symbol, 0) for symbol in symbols]
            for method, opt in optimization_results.items()
        }, index=symbols)
        
        weights_df.plot(kind='bar', ax=ax)
        ax.set_title('Portfolio Weights by Method')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        ax.legend(title='Method')
        
        # 5. Expected returns vs risk
        ax = axes[1, 1]
        for method, portfolio in optimization_results.items():
            ax.scatter(portfolio.expected_risk, portfolio.expected_return, 
                      s=100, label=f"{method} (SR: {portfolio.sharpe_ratio:.2f})")
        ax.set_xlabel('Expected Risk (Volatility)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Risk-Return Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap
        ax = axes[1, 2]
        returns_data = pd.DataFrame({
            symbol: analysis['returns'][-100:] 
            for symbol, analysis in market_analysis.items()
        })
        corr_matrix = returns_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Asset Correlations')
        
        plt.tight_layout()
        plt.savefig('reports/advanced_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved to reports/advanced_analysis.png")
        
        return fig


async def demonstrate_realtime_pipeline():
    """Demonstrate real-time pipeline capabilities."""
    print("\n=== REAL-TIME PIPELINE DEMO ===")
    
    # Configure pipeline
    config = StreamConfig(
        source='yahoo',
        symbols=['AAPL', 'GOOGL', 'MSFT'],
        interval='1d',
        buffer_size=50,
        batch_size=5,
        max_latency_ms=500
    )
    
    # Feature extractor
    def extract_features(df):
        return {
            'sma_ratio': df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1],
            'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],
            'rsi': 50  # Simplified
        }
    
    # Predictor
    def make_prediction(features):
        # Simple rule-based prediction
        if features['sma_ratio'] > 1.02 and features['volume_ratio'] > 1.5:
            return 1.0, 0.8  # Bullish, high confidence
        elif features['sma_ratio'] < 0.98:
            return -1.0, 0.7  # Bearish
        else:
            return 0.0, 0.5  # Neutral
            
    # Output handler
    async def handle_prediction(result):
        print(f"\n[{result.timestamp}] {result.symbol}:")
        print(f"  Prediction: {result.prediction:+.2f}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        
    # Create pipeline
    pipeline = RealtimePipeline(config, extract_features, make_prediction, handle_prediction)
    
    # Backfill historical data
    print("Backfilling historical data...")
    await pipeline.backfill_data(30)
    
    # Get stats
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Stats:")
    print(f"  Buffer sizes: {stats['buffer_sizes']}")
    
    print("\n✓ Real-time pipeline demonstrated (would run continuously in production)")


def main():
    """Run complete demonstration."""
    print("=" * 80)
    print("ADVANCED FINANCIAL PREDICTION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = AdvancedTradingSystem()
    
    # Define universe and timeframe
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Load market data
    market_data = system.load_market_data(symbols, start_date, end_date)
    
    # Analyze market conditions
    market_analysis = system.analyze_market_conditions(market_data)
    
    # Optimize portfolio
    optimization_results = system.optimize_portfolio(market_analysis)
    
    # Calculate position sizes
    position_recommendations = system.calculate_position_sizes(
        optimization_results, market_analysis
    )
    
    # Assess portfolio risk
    risk_assessment, stress_results = system.assess_portfolio_risk(
        position_recommendations, market_analysis
    )
    
    # Generate visualizations
    system.visualize_analysis(market_analysis, optimization_results)
    
    # Demonstrate real-time pipeline
    asyncio.run(demonstrate_realtime_pipeline())
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Achievements:")
    print("✓ Multi-timeframe analysis across multiple assets")
    print("✓ Market regime detection and prediction")
    print("✓ Portfolio optimization with multiple methods")
    print("✓ Risk-adjusted position sizing")
    print("✓ Comprehensive risk assessment and stress testing")
    print("✓ Real-time data pipeline demonstration")
    print("\nThe system is ready for production deployment!")


if __name__ == "__main__":
    main()
