"""
Test Dashboard Integration with Trained Models

This script tests the integration of trained models with the dashboard.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if models can be loaded successfully"""
    logger.info("Testing model loading...")
    
    try:
        from src.dashboard.model_loader import PatternPredictionModels
        
        # Initialize model loader
        models = PatternPredictionModels()
        
        # Get model status
        status = models.get_model_status()
        
        logger.info(f"Model Status:")
        logger.info(f"  - Models loaded: {status['models_loaded']}")
        logger.info(f"  - Config loaded: {status['config_loaded']}")
        logger.info(f"  - Label encoder: {status['label_encoder_loaded']}")
        logger.info(f"  - Feature scaler: {status['feature_scaler_loaded']}")
        logger.info(f"  - Markov model: {status['markov_model_loaded']}")
        logger.info(f"  - Device: {status['device']}")
        
        return len(status['models_loaded']) > 0
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def test_pattern_prediction():
    """Test pattern prediction functionality"""
    logger.info("\nTesting pattern prediction...")
    
    try:
        from src.dashboard.model_loader import PatternPredictionModels
        
        # Initialize model loader
        models = PatternPredictionModels()
        
        # Test pattern sequences
        test_sequences = [
            ['double_bottom', 'flag_bull', 'triangle_ascending'],
            ['double_top', 'flag_bear', 'triangle_descending'],
            ['head_shoulders', 'wedge_falling', 'double_bottom']
        ]
        
        for seq in test_sequences:
            prediction = models.predict_next_pattern(seq)
            logger.info(f"\nSequence: {' -> '.join(seq)}")
            logger.info(f"  Predicted: {prediction['next_pattern']} (confidence: {prediction['confidence']:.2%})")
            
            if prediction['alternatives']:
                logger.info("  Alternatives:")
                for alt in prediction['alternatives'][:2]:
                    logger.info(f"    - {alt['pattern']} ({alt['confidence']:.2%})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in pattern prediction: {e}")
        return False

def test_price_prediction():
    """Test price prediction functionality"""
    logger.info("\nTesting price prediction...")
    
    try:
        from src.dashboard.model_loader import PatternPredictionModels
        
        # Initialize model loader
        models = PatternPredictionModels()
        
        # Test price predictions
        test_cases = [
            (100.0, 'flag_bull', 1),
            (100.0, 'flag_bear', 1),
            (100.0, 'double_bottom', 5),
            (100.0, 'double_top', 5)
        ]
        
        for price, pattern, horizon in test_cases:
            predictions = models.predict_price_movement(price, pattern, horizon)
            logger.info(f"\nPrice: ${price}, Pattern: {pattern}, Horizon: {horizon}")
            logger.info(f"  LSTM: ${predictions['lstm']:.2f}")
            logger.info(f"  GRU: ${predictions['gru']:.2f}")
            logger.info(f"  Transformer: ${predictions['transformer']:.2f}")
            logger.info(f"  Ensemble: ${predictions['ensemble']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in price prediction: {e}")
        return False

def test_dashboard_startup():
    """Test if dashboard can start without errors"""
    logger.info("\nTesting dashboard startup...")
    
    try:
        # Import dashboard modules
        from src.dashboard.forecast_app_fixed import app, data_manager, pattern_models
        
        # Check if app is created
        if app is None:
            logger.error("Dashboard app not created")
            return False
        
        # Check if data manager is initialized
        if data_manager is None:
            logger.error("Data manager not initialized")
            return False
        
        # Check if pattern models are loaded
        if pattern_models is None:
            logger.error("Pattern models not initialized")
            return False
        
        logger.info("Dashboard components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("=" * 60)
    logger.info("Dashboard Integration Tests")
    logger.info("=" * 60)
    
    # Check if model files exist
    model_dir = "models/pattern_predictor"
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Please run Window 2 training script first")
        return
    
    # List model files
    logger.info(f"\nModel files in {model_dir}:")
    for file in os.listdir(model_dir):
        logger.info(f"  - {file}")
    
    # Run tests
    tests = [
        ("Model Loading", test_model_loading),
        ("Pattern Prediction", test_pattern_prediction),
        ("Price Prediction", test_price_prediction),
        ("Dashboard Startup", test_dashboard_startup)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✅ All integration tests passed! Dashboard is ready to use.")
        logger.info("\nTo run the dashboard:")
        logger.info("  python run_dashboard_fixed.py")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
