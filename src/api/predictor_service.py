"""
Predictor service for handling model predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
import json
import os
from datetime import datetime
import torch
from collections import deque

logger = logging.getLogger(__name__)


class PredictorService:
    """Service for managing and serving predictions"""
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize predictor service
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.models = {}
        self.model_info = {}
        self.feature_names = []
        self.prediction_history = deque(maxlen=1000)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        logger.info(f"Loading models from {self.model_dir}")
        
        # Load XGBoost model
        try:
            xgb_path = os.path.join(self.model_dir, "xgboost_optimized.pkl")
            if os.path.exists(xgb_path):
                self.models['xgboost'] = joblib.load(xgb_path)
                self.model_info['xgboost'] = self._load_model_info('xgboost')
                logger.info("XGBoost model loaded")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
        
        # Load LSTM model
        try:
            lstm_path = os.path.join(self.model_dir, "lstm_optimized.pth")
            if os.path.exists(lstm_path):
                # Import and create model
                from ..models.sequence_predictor import LSTMPredictor
                
                # Load model config
                config_path = lstm_path.replace('.pth', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    model = LSTMPredictor(**config)
                    model.load_state_dict(torch.load(lstm_path))
                    model.eval()
                    self.models['lstm'] = model
                    self.model_info['lstm'] = self._load_model_info('lstm')
                    logger.info("LSTM model loaded")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
        
        # Load Transformer model
        try:
            transformer_path = os.path.join(self.model_dir, "transformer_optimized.pth")
            if os.path.exists(transformer_path):
                from ..models.transformer_predictor import TransformerPredictor
                
                config_path = transformer_path.replace('.pth', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    model = TransformerPredictor(**config)
                    model.load_state_dict(torch.load(transformer_path))
                    model.eval()
                    self.models['transformer'] = model
                    self.model_info['transformer'] = self._load_model_info('transformer')
                    logger.info("Transformer model loaded")
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
        
        # Load Ensemble model
        try:
            ensemble_path = os.path.join(self.model_dir, "ensemble_optimized.pkl")
            if os.path.exists(ensemble_path):
                from ..models.ensemble_model import EnsembleModel
                self.models['ensemble'] = EnsembleModel.load(ensemble_path)
                self.model_info['ensemble'] = self._load_model_info('ensemble')
                logger.info("Ensemble model loaded")
        except Exception as e:
            logger.error(f"Failed to load Ensemble model: {e}")
        
        # Load feature names
        try:
            feature_path = os.path.join(self.model_dir, "feature_names.json")
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            logger.error(f"Failed to load feature names: {e}")
    
    def _load_model_info(self, model_type: str) -> Dict[str, Any]:
        """Load model information"""
        info_path = os.path.join(self.model_dir, f"{model_type}_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        # Default info
        return {
            'version': '1.0.0',
            'trained_date': datetime.now().isoformat(),
            'performance_metrics': {}
        }
    
    def predict(self, features: Union[List[float], np.ndarray], 
                model_type: str = 'ensemble') -> Tuple[float, Optional[float]]:
        """
        Make a single prediction
        
        Args:
            features: Input features
            model_type: Model to use
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        # Convert to numpy array
        if isinstance(features, list):
            features = np.array(features)
        
        # Reshape for single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        model = self.models[model_type]
        
        # Make prediction
        if model_type in ['lstm', 'transformer']:
            # PyTorch models
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                prediction = model(features_tensor).numpy()[0, 0]
        else:
            # Sklearn-style models
            prediction = model.predict(features)[0]
        
        # Calculate confidence (simplified - could be improved)
        confidence = self._calculate_confidence(features, prediction, model_type)
        
        # Store in history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'model_type': model_type,
            'prediction': prediction,
            'confidence': confidence
        })
        
        return float(prediction), confidence
    
    def predict_batch(self, samples: List[List[float]], 
                     model_type: str = 'ensemble',
                     return_confidence: bool = False) -> Dict[str, Any]:
        """
        Make batch predictions
        
        Args:
            samples: List of feature vectors
            model_type: Model to use
            return_confidence: Whether to calculate confidence
            
        Returns:
            Dictionary with predictions and metadata
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        # Convert to numpy array
        features = np.array(samples)
        
        model = self.models[model_type]
        
        # Make predictions
        if model_type in ['lstm', 'transformer']:
            # PyTorch models
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                predictions = model(features_tensor).numpy().flatten()
        else:
            # Sklearn-style models
            predictions = model.predict(features)
        
        # Calculate confidences if requested
        confidences = None
        if return_confidence:
            confidences = [
                self._calculate_confidence(feat, pred, model_type)
                for feat, pred in zip(features, predictions)
            ]
        
        return {
            'predictions': predictions.tolist(),
            'confidences': confidences,
            'model_type': model_type,
            'total_samples': len(samples)
        }
    
    def _calculate_confidence(self, features: np.ndarray, 
                            prediction: float,
                            model_type: str) -> float:
        """
        Calculate prediction confidence
        
        Args:
            features: Input features
            prediction: Model prediction
            model_type: Type of model
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence based on prediction history
        if len(self.prediction_history) < 10:
            return 0.5
        
        # Get recent predictions
        recent_predictions = [h['prediction'] for h in list(self.prediction_history)[-10:]]
        
        # Calculate standard deviation
        std = np.std(recent_predictions)
        mean = np.mean(recent_predictions)
        
        # Distance from mean
        distance = abs(prediction - mean)
        
        # Convert to confidence (closer to mean = higher confidence)
        if std > 0:
            z_score = distance / std
            confidence = max(0, 1 - (z_score / 3))  # 3 sigma rule
        else:
            confidence = 1.0
        
        return float(confidence)
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        info = self.model_info.get(model_type, {}).copy()
        info['model_type'] = model_type
        info['features'] = self.feature_names
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models"""
        return {
            model_type: self.get_model_info(model_type)
            for model_type in self.models.keys()
        }
    
    def update_model(self, model_type: str, model_path: str, 
                    version: str) -> bool:
        """
        Update a model
        
        Args:
            model_type: Type of model to update
            model_path: Path to new model file
            version: New version number
            
        Returns:
            Success status
        """
        try:
            # Load new model based on type
            if model_type == 'xgboost':
                new_model = joblib.load(model_path)
            elif model_type in ['lstm', 'transformer']:
                # Load PyTorch model
                if model_type == 'lstm':
                    from ..models.sequence_predictor import LSTMPredictor
                    ModelClass = LSTMPredictor
                else:
                    from ..models.transformer_predictor import TransformerPredictor
                    ModelClass = TransformerPredictor
                
                # Load config
                config_path = model_path.replace('.pth', '_config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                new_model = ModelClass(**config)
                new_model.load_state_dict(torch.load(model_path))
                new_model.eval()
            elif model_type == 'ensemble':
                from ..models.ensemble_model import EnsembleModel
                new_model = EnsembleModel.load(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Update model
            self.models[model_type] = new_model
            
            # Update model info
            self.model_info[model_type]['version'] = version
            self.model_info[model_type]['updated_date'] = datetime.now().isoformat()
            
            logger.info(f"Updated {model_type} model to version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update {model_type} model: {e}")
            return False
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about recent predictions"""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'model_usage': {},
                'avg_confidence': 0,
                'recent_predictions': []
            }
        
        # Calculate stats
        history = list(self.prediction_history)
        
        # Model usage
        model_usage = {}
        for h in history:
            model_type = h['model_type']
            model_usage[model_type] = model_usage.get(model_type, 0) + 1
        
        # Average confidence
        confidences = [h['confidence'] for h in history if h['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Recent predictions
        recent = history[-10:]
        
        return {
            'total_predictions': len(history),
            'model_usage': model_usage,
            'avg_confidence': float(avg_confidence),
            'recent_predictions': [
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'model_type': h['model_type'],
                    'prediction': h['prediction'],
                    'confidence': h['confidence']
                }
                for h in recent
            ]
        }


class StreamingPredictor:
    """Handler for streaming predictions"""
    
    def __init__(self, predictor_service: PredictorService, 
                 window_size: int = 10):
        """
        Initialize streaming predictor
        
        Args:
            predictor_service: Main predictor service
            window_size: Size of sliding window
        """
        self.predictor = predictor_service
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)
        self.previous_prediction = None
    
    def predict_stream(self, features: List[float], 
                      model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Make streaming prediction
        
        Args:
            features: Current features
            model_type: Model to use
            
        Returns:
            Streaming prediction response
        """
        # Make prediction
        prediction, confidence = self.predictor.predict(features, model_type)
        
        # Add to window
        self.prediction_window.append(prediction)
        
        # Calculate trend
        if len(self.prediction_window) >= 3:
            recent = list(self.prediction_window)[-3:]
            if recent[-1] > recent[-2] > recent[-3]:
                trend = 'up'
            elif recent[-1] < recent[-2] < recent[-3]:
                trend = 'down'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate change
        change = 0
        if self.previous_prediction is not None:
            change = prediction - self.previous_prediction
        
        # Moving average
        moving_avg = np.mean(list(self.prediction_window))
        
        # Update previous
        self.previous_prediction = prediction
        
        return {
            'prediction': prediction,
            'trend': trend,
            'change_from_previous': float(change),
            'moving_average': float(moving_avg),
            'confidence': confidence
        }


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create dummy models for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy XGBoost model
        from sklearn.linear_model import LinearRegression
        dummy_model = LinearRegression()
        dummy_model.fit([[1, 2], [3, 4]], [1, 2])
        
        model_path = os.path.join(tmpdir, "xgboost_optimized.pkl")
        joblib.dump(dummy_model, model_path)
        
        # Create feature names
        feature_names = ["feature_0", "feature_1"]
        with open(os.path.join(tmpdir, "feature_names.json"), 'w') as f:
            json.dump(feature_names, f)
        
        # Initialize service
        service = PredictorService(model_dir=tmpdir)
        
        # Test single prediction
        print("Testing single prediction...")
        prediction, confidence = service.predict([1.5, 2.5], model_type='xgboost')
        print(f"Prediction: {prediction}, Confidence: {confidence}")
        
        # Test batch prediction
        print("\nTesting batch prediction...")
        samples = [[1, 2], [2, 3], [3, 4]]
        results = service.predict_batch(samples, model_type='xgboost', return_confidence=True)
        print(f"Batch results: {results}")
        
        # Test streaming
        print("\nTesting streaming prediction...")
        streamer = StreamingPredictor(service)
        for i in range(5):
            stream_result = streamer.predict_stream([i, i+1], model_type='xgboost')
            print(f"Stream {i}: {stream_result}")
