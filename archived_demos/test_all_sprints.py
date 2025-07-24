"""
Comprehensive test script for all sprint deliverables
Tests each sprint's components systematically
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SprintTester:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def test_sprint(self, sprint_num: int, test_func):
        """Test a specific sprint"""
        self.log(f"\n{'='*60}")
        self.log(f"Testing Sprint {sprint_num}")
        self.log(f"{'='*60}")
        
        try:
            start = time.time()
            results = test_func()
            duration = time.time() - start
            
            self.results[f"Sprint_{sprint_num}"] = {
                "status": "PASSED",
                "duration": duration,
                "details": results
            }
            self.log(f"Sprint {sprint_num} PASSED in {duration:.2f}s", "SUCCESS")
            
        except Exception as e:
            self.results[f"Sprint_{sprint_num}"] = {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.log(f"Sprint {sprint_num} FAILED: {str(e)}", "ERROR")
            
    def test_sprint_3_dtw(self) -> Dict:
        """Test Sprint 3: DTW Implementation"""
        self.log("Testing DTW components...")
        results = {}
        
        # Test DTW Calculator
        from src.dtw import DTWCalculator
        dtw = DTWCalculator()
        
        # Generate test data
        x = np.sin(np.linspace(0, 2*np.pi, 100))
        y = np.sin(np.linspace(0, 2*np.pi, 100) + 0.1)
        
        # Test standard DTW
        result = dtw.compute(x, y)
        results["dtw_distance"] = result.distance
        results["dtw_normalized"] = result.normalized_distance
        self.log(f"DTW Distance: {result.distance:.4f}")
        
        # Test FastDTW
        dtw_fast = DTWCalculator(dtw_type='fast')
        result_fast = dtw_fast.compute(x, y)
        results["fast_dtw_distance"] = result_fast.distance
        self.log(f"FastDTW Distance: {result_fast.distance:.4f}")
        
        # Test Similarity Engine
        from src.dtw import SimilarityEngine
        engine = SimilarityEngine(n_jobs=2)
        
        patterns = [np.sin(np.linspace(0, 2*np.pi, 100) + i*0.1) for i in range(5)]
        sim_matrix = engine.compute_similarity_matrix(patterns)
        results["similarity_matrix_shape"] = sim_matrix.shape
        self.log(f"Similarity matrix shape: {sim_matrix.shape}")
        
        # Test Pattern Clusterer
        from src.dtw import PatternClusterer
        clusterer = PatternClusterer(n_clusters=3)
        clusters = clusterer.fit_predict(patterns)
        results["n_clusters"] = len(np.unique(clusters))
        self.log(f"Found {len(np.unique(clusters))} clusters")
        
        return results
        
    def test_sprint_4_features(self) -> Dict:
        """Test Sprint 4: Feature Engineering"""
        self.log("Testing feature engineering components...")
        results = {}
        
        # Generate test data
        dates = pd.date_range('2024-01-01', periods=1000, freq='D')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        volumes = np.random.randint(1000000, 5000000, 1000)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(1000) * 0.1,
            'high': prices + np.abs(np.random.randn(1000) * 0.2),
            'low': prices - np.abs(np.random.randn(1000) * 0.2),
            'close': prices,
            'volume': volumes
        })
        
        # Test Technical Indicators
        from src.features import TechnicalIndicators
        ti = TechnicalIndicators()
        indicators = ti.calculate_all(df)
        results["n_technical_indicators"] = len(indicators.columns)
        self.log(f"Generated {len(indicators.columns)} technical indicators")
        
        # Test Pattern Feature Extractor
        from src.features import PatternFeatureExtractor
        pfe = PatternFeatureExtractor(pattern_length=20)
        pattern_features = pfe.extract_features(prices[-100:])
        results["n_pattern_features"] = len(pattern_features)
        self.log(f"Extracted {len(pattern_features)} pattern features")
        
        # Test Feature Pipeline
        from src.features import FeaturePipeline
        pipeline = FeaturePipeline(
            use_technical_indicators=True,
            use_pattern_features=True,
            scaler_type='standard'
        )
        
        # Create a smaller dataset for pipeline testing
        small_df = df.iloc[-200:].copy()
        features = pipeline.fit_transform(small_df)
        results["total_features"] = features.shape[1]
        self.log(f"Total features from pipeline: {features.shape[1]}")
        
        # Test Feature Selector
        from src.features import FeatureSelector
        selector = FeatureSelector(n_features=20)
        
        # Create target
        target = (small_df['close'].shift(-1) > small_df['close']).astype(int)[:-1]
        selected = selector.fit_transform(features[:-1], target)
        results["selected_features"] = selected.shape[1]
        self.log(f"Selected {selected.shape[1]} features")
        
        return results
        
    def test_sprint_5_models(self) -> Dict:
        """Test Sprint 5: Model Development"""
        self.log("Testing model components...")
        results = {}
        
        # Generate synthetic data
        n_samples = 500
        n_features = 20
        sequence_length = 10
        
        # Tabular data for XGBoost
        X_tab = np.random.randn(n_samples, n_features)
        y = (np.sum(X_tab[:, :5], axis=1) > 0).astype(float) + np.random.randn(n_samples) * 0.1
        
        # Sequential data for neural networks
        X_seq = np.random.randn(n_samples, sequence_length, n_features)
        
        # Test XGBoost
        from src.models import XGBoostPredictor
        xgb_model = XGBoostPredictor(n_estimators=50, max_depth=3)
        xgb_model.fit(X_tab[:400], y[:400])
        xgb_pred = xgb_model.predict(X_tab[400:])
        results["xgboost_tested"] = True
        self.log("XGBoost model tested successfully")
        
        # Test Sequence Predictor (LSTM)
        from src.models import SequencePredictor
        lstm_model = SequencePredictor(
            input_size=n_features,
            hidden_size=32,
            num_layers=1,
            output_size=1,
            model_type='lstm'
        )
        
        # Quick training
        lstm_model.compile(learning_rate=0.001)
        history = lstm_model.fit(
            X_seq[:400], y[:400],
            X_seq[400:], y[400:],
            epochs=2,
            batch_size=32,
            verbose=False
        )
        results["lstm_tested"] = True
        self.log("LSTM model tested successfully")
        
        # Test Transformer
        from src.models import TransformerPredictor
        transformer = TransformerPredictor(
            input_size=n_features,
            d_model=32,
            nhead=2,
            num_layers=1,
            output_size=1
        )
        transformer.compile(learning_rate=0.001)
        results["transformer_tested"] = True
        self.log("Transformer model tested successfully")
        
        # Test Ensemble
        from src.models import EnsembleModel
        ensemble = EnsembleModel(
            models={'xgboost': xgb_model},
            strategy='average'
        )
        ensemble_pred = ensemble.predict(X_tab[400:])
        results["ensemble_tested"] = True
        self.log("Ensemble model tested successfully")
        
        # Test Model Evaluator
        from src.models import ModelEvaluator
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y[400:], xgb_pred)
        results["metrics_calculated"] = len(metrics)
        self.log(f"Calculated {len(metrics)} evaluation metrics")
        
        return results
        
    def test_sprint_6_evaluation(self) -> Dict:
        """Test Sprint 6: Evaluation & Backtesting"""
        self.log("Testing evaluation components...")
        results = {}
        
        # Generate test data
        n_days = 500
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'returns': np.log(prices / np.roll(prices, 1))
        })
        data = data.iloc[1:]  # Remove first row with NaN return
        
        # Generate predictions (random for testing)
        predictions = np.random.randn(len(data)) * 0.01
        
        # Test Backtest Engine
        from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig
        
        config = BacktestConfig(
            initial_capital=100000,
            position_size=0.1,
            commission=0.001,
            slippage=0.0005
        )
        
        engine = BacktestEngine(config)
        
        # Simple signal generator
        def signal_generator(pred, threshold=0):
            return 1 if pred > threshold else -1 if pred < -threshold else 0
            
        backtest_results = engine.run_backtest(data, predictions, signal_generator)
        results["total_return"] = backtest_results.total_return
        results["sharpe_ratio"] = backtest_results.sharpe_ratio
        results["max_drawdown"] = backtest_results.max_drawdown
        self.log(f"Backtest - Return: {backtest_results.total_return:.2%}, Sharpe: {backtest_results.sharpe_ratio:.2f}")
        
        # Test Risk Analyzer
        from src.evaluation import RiskAnalyzer
        risk_analyzer = RiskAnalyzer()
        
        returns = data['returns'].values
        risk_metrics = risk_analyzer.calculate_risk_metrics(returns)
        results["risk_metrics_calculated"] = len(risk_metrics)
        self.log(f"Calculated {len(risk_metrics)} risk metrics")
        
        # Test Market Regime Analyzer
        from src.evaluation import MarketRegimeAnalyzer
        regime_analyzer = MarketRegimeAnalyzer(n_regimes=3)
        
        features = np.column_stack([
            returns,
            pd.Series(returns).rolling(20).std().fillna(0).values
        ])
        
        regime_analyzer.fit(features)
        current_regime = regime_analyzer.predict_regime(features[-1:])
        results["regime_detected"] = current_regime[0]
        self.log(f"Detected market regime: {current_regime[0]}")
        
        # Test Performance Reporter
        from src.evaluation import PerformanceReporter
        reporter = PerformanceReporter()
        
        # Create dummy results for multiple strategies
        strategy_results = {
            'Strategy_A': backtest_results,
            'Strategy_B': backtest_results  # Using same results for simplicity
        }
        
        comparison = reporter.compare_strategies(strategy_results)
        results["strategies_compared"] = len(comparison)
        self.log(f"Compared {len(comparison)} strategies")
        
        return results
        
    def test_sprint_7_deployment(self) -> Dict:
        """Test Sprint 7: Optimization & Deployment"""
        self.log("Testing deployment components...")
        results = {}
        
        # Test Model Compressor
        from src.optimization.model_compressor import XGBoostCompressor
        compressor = XGBoostCompressor()
        
        # Create a dummy model for compression testing
        from src.models import XGBoostPredictor
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        
        # Generate dummy data
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        model.fit(X, y)
        
        # Test compression
        compressed_model = compressor.compress(model.model, compression_ratio=0.5)
        metrics = compressor.get_compression_stats()
        results["compression_ratio"] = metrics.get('compression_ratio', 0)
        self.log(f"Model compression ratio: {metrics.get('compression_ratio', 0):.2f}")
        
        # Test API Models
        from src.api.models import PredictionRequest, PredictionResponse
        
        # Test request validation
        request = PredictionRequest(
            features=[1.0, 2.0, 3.0],
            model_type="xgboost"
        )
        results["api_models_tested"] = True
        self.log("API models validated successfully")
        
        # Test Predictor Service
        from src.api.predictor_service import PredictorService
        
        # Note: We're just testing initialization, not actual prediction
        # as that would require loading actual models
        service = PredictorService()
        results["predictor_service_initialized"] = True
        self.log("Predictor service initialized successfully")
        
        # Test Monitoring
        from src.api.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_prediction(model_type="xgboost", latency=0.05)
        metrics = collector.get_metrics()
        results["monitoring_metrics"] = len(metrics)
        self.log(f"Collected {len(metrics)} monitoring metrics")
        
        # Check Docker files exist
        docker_files = ['Dockerfile', 'docker-compose.yml']
        for file in docker_files:
            if os.path.exists(file):
                results[f"{file}_exists"] = True
                self.log(f"{file} exists")
                
        return results
        
    def test_sprint_8_advanced(self) -> Dict:
        """Test Sprint 8: Advanced Features"""
        self.log("Testing advanced features...")
        results = {}
        
        # Generate test data
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.001))
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Test Multi-Timeframe Analyzer
        from src.advanced import MultiTimeframeAnalyzer
        
        mtf_analyzer = MultiTimeframeAnalyzer(
            timeframes=['5min', '15min', '1h'],
            feature_extractors=None  # Use default
        )
        
        # Note: Using smaller dataset for testing
        mtf_features = mtf_analyzer.analyze_all_timeframes(price_data.iloc[-100:])
        results["mtf_features_shape"] = mtf_features.shape
        self.log(f"Multi-timeframe features shape: {mtf_features.shape}")
        
        # Test Market Regime Detector
        from src.advanced import MarketRegimeDetector
        
        regime_detector = MarketRegimeDetector(n_regimes=3)
        
        # Prepare features
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(20).std().fillna(0).values[1:]
        features = np.column_stack([returns, volatility])
        
        regime_detector.train_hmm(features[-500:])
        current_regime = regime_detector.detect_current_regime(features[-50:])
        results["current_regime"] = current_regime
        self.log(f"Detected regime: {current_regime}")
        
        # Test Adaptive Learner
        from src.advanced import AdaptiveLearner
        
        # Create a simple base model
        from sklearn.linear_model import SGDRegressor
        base_model = SGDRegressor()
        
        learner = AdaptiveLearner(
            base_model=base_model,
            buffer_size=1000,
            drift_threshold=0.05,
            update_frequency=10
        )
        
        # Test with small dataset
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        predictions = []
        for i in range(0, len(X), 10):
            pred, conf = learner.predict_adaptive(X[i:i+10])
            predictions.extend(pred)
            learner.incremental_update(X[i:i+10], y[i:i+10])
            
        results["adaptive_predictions_made"] = len(predictions)
        self.log(f"Made {len(predictions)} adaptive predictions")
        
        # Test Risk Manager
        from src.advanced import AdvancedRiskManager
        
        risk_manager = AdvancedRiskManager(
            confidence_level=0.95,
            lookback_period=252
        )
        
        portfolio_returns = np.random.randn(252) * 0.01
        var = risk_manager.calculate_var(portfolio_returns)
        results["var_95"] = var
        self.log(f"95% VaR: {var:.4f}")
        
        # Test Portfolio Optimizer
        from src.advanced import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(method='mean_variance')
        
        # Generate returns for 5 assets
        n_assets = 5
        asset_returns = np.random.randn(252, n_assets) * 0.01
        
        weights = optimizer.optimize(asset_returns)
        results["portfolio_weights"] = len(weights)
        self.log(f"Optimized weights for {len(weights)} assets")
        
        return results
        
    def generate_report(self):
        """Generate final test report"""
        self.log("\n" + "="*60)
        self.log("FINAL TEST REPORT")
        self.log("="*60)
        
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        
        self.log(f"\nTotal Sprints Tested: {len(self.results)}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Total Duration: {total_duration:.2f}s")
        
        # Save detailed results
        with open('sprint_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        self.log("\nDetailed results saved to: sprint_test_results.json")
        
        # Print summary for each sprint
        self.log("\nSprint Summary:")
        for sprint, result in self.results.items():
            status = result['status']
            symbol = "✓" if status == "PASSED" else "✗"
            self.log(f"{sprint}: {status} {symbol}")
            
            if status == "FAILED":
                self.log(f"  Error: {result.get('error', 'Unknown error')}")
                
        return passed == len(self.results)


def main():
    """Main test execution"""
    tester = SprintTester()
    
    # Test each sprint
    tester.test_sprint(3, tester.test_sprint_3_dtw)
    tester.test_sprint(4, tester.test_sprint_4_features)
    tester.test_sprint(5, tester.test_sprint_5_models)
    tester.test_sprint(6, tester.test_sprint_6_evaluation)
    tester.test_sprint(7, tester.test_sprint_7_deployment)
    tester.test_sprint(8, tester.test_sprint_8_advanced)
    
    # Generate report
    all_passed = tester.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
