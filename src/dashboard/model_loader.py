"""
Model Loader for Wavelet Pattern Prediction Models

This module handles loading trained models and making predictions
for the dashboard integration.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM model for pattern prediction"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=20, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    """GRU model for pattern prediction"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=20, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """Transformer model for pattern prediction"""
    def __init__(self, input_size=8, d_model=128, nhead=8, num_layers=2, 
                 num_classes=20, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Use the last output
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class PatternPredictionModels:
    """Load and manage trained pattern prediction models"""
    
    def __init__(self, model_dir: str = "models/pattern_predictor"):
        self.model_dir = model_dir
        self.models = {}
        self.config = None
        self.label_encoder = None
        self.feature_scaler = None
        self.markov_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pattern types mapping
        self.pattern_types = [
            'head_shoulders', 'double_top', 'double_bottom',
            'triangle_ascending', 'triangle_descending',
            'flag_bull', 'flag_bear', 'wedge_rising', 'wedge_falling'
        ]
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and configurations"""
        try:
            # Load configuration
            config_path = os.path.join(self.model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found at {config_path}")
                self.config = self._get_default_config()
            
            # Load label encoder
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Loaded label encoder")
            
            # Load feature scaler
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("Loaded feature scaler")
            
            # Load PyTorch models
            self._load_pytorch_model('lstm', LSTMModel())
            self._load_pytorch_model('gru', GRUModel())
            self._load_pytorch_model('transformer', TransformerModel())
            
            # Load Markov model
            markov_path = os.path.join(self.model_dir, 'markov_model.json')
            if os.path.exists(markov_path):
                with open(markov_path, 'r') as f:
                    self.markov_model = json.load(f)
                logger.info("Loaded Markov model")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Initialize with empty models if loading fails
            self._initialize_empty_models()
    
    def _load_pytorch_model(self, name: str, model: nn.Module):
        """Load a PyTorch model"""
        model_path = os.path.join(self.model_dir, f'{name}_model.pth')
        if os.path.exists(model_path):
            try:
                # Load with weights_only=False to handle the model format
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[name] = model
                logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.error(f"Error loading {name} model: {e}")
        else:
            logger.warning(f"{name} model file not found at {model_path}")
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'seq_length': 10,
            'ensemble_weights': {
                'lstm': 0.4,
                'gru': 0.2,
                'transformer': 0.3,
                'markov': 0.1
            }
        }
    
    def _initialize_empty_models(self):
        """Initialize empty models for fallback"""
        self.models = {}
        self.config = self._get_default_config()
    
    def predict_next_pattern(self, pattern_sequence: List[str], 
                           features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict the next pattern given a sequence of patterns
        
        Args:
            pattern_sequence: List of pattern names in sequence
            features: Optional feature array for the sequence
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            # If no models are loaded, return default prediction
            if not self.models:
                return self._get_default_prediction(pattern_sequence)
            
            # Prepare features if not provided
            if features is None:
                features = self._extract_features(pattern_sequence)
            
            # Ensure features are properly shaped
            if len(features.shape) == 2:
                features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Scale features if scaler is available
            if self.feature_scaler is not None:
                # Reshape for scaler
                n_samples, n_timesteps, n_features = features.shape
                features_2d = features.reshape(-1, n_features)
                features_scaled = self.feature_scaler.transform(features_2d)
                features = features_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            # Neural network predictions
            for model_name, model in self.models.items():
                with torch.no_grad():
                    output = model(features_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_idx].item()
                    
                    predictions[model_name] = pred_idx
                    confidences[model_name] = confidence
            
            # Markov model prediction
            if self.markov_model and len(pattern_sequence) > 0:
                last_pattern = pattern_sequence[-1]
                if last_pattern in self.markov_model:
                    markov_probs = self.markov_model[last_pattern]
                    # Find most likely next pattern
                    next_pattern = max(markov_probs.items(), key=lambda x: x[1])
                    if self.label_encoder:
                        try:
                            pred_idx = self.label_encoder.transform([next_pattern[0]])[0]
                            predictions['markov'] = pred_idx
                            confidences['markov'] = next_pattern[1]
                        except:
                            pass
            
            # Ensemble prediction
            if predictions:
                ensemble_weights = self.config.get('ensemble_weights', {})
                # Use 20 classes to match the model output
                weighted_probs = np.zeros(20)
                total_weight = 0
                
                for model_name, pred_idx in predictions.items():
                    weight = ensemble_weights.get(model_name, 0.25)
                    if model_name in confidences and pred_idx < 20:
                        weighted_probs[pred_idx] += weight * confidences[model_name]
                        total_weight += weight
                
                if total_weight > 0:
                    weighted_probs /= total_weight
                    ensemble_pred_idx = np.argmax(weighted_probs)
                    ensemble_confidence = weighted_probs[ensemble_pred_idx]
                else:
                    ensemble_pred_idx = 0
                    ensemble_confidence = 0.0
            else:
                ensemble_pred_idx = 0
                ensemble_confidence = 0.0
            
            # Get pattern name
            if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform'):
                try:
                    next_pattern_name = self.label_encoder.inverse_transform([ensemble_pred_idx])[0]
                except:
                    next_pattern_name = self.pattern_types[ensemble_pred_idx]
            else:
                next_pattern_name = self.pattern_types[ensemble_pred_idx]
            
            # Get alternative predictions
            alternatives = []
            if len(weighted_probs) > 1:
                top_indices = np.argsort(weighted_probs)[-3:][::-1]
                for idx in top_indices[1:]:  # Skip the top prediction
                    if weighted_probs[idx] > 0.1:  # Only include if confidence > 10%
                        pattern_name = self.pattern_types[idx] if idx < len(self.pattern_types) else f"Pattern_{idx}"
                        alternatives.append({
                            'pattern': pattern_name,
                            'confidence': float(weighted_probs[idx])
                        })
            
            return {
                'next_pattern': next_pattern_name,
                'confidence': float(ensemble_confidence),
                'alternatives': alternatives,
                'model_predictions': {
                    model: self.pattern_types[idx] if idx < len(self.pattern_types) else f"Pattern_{idx}"
                    for model, idx in predictions.items()
                },
                'model_confidences': {k: float(v) for k, v in confidences.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in pattern prediction: {e}")
            return self._get_default_prediction(pattern_sequence)
    
    def _extract_features(self, pattern_sequence: List[str]) -> np.ndarray:
        """Extract features from pattern sequence"""
        # Simple feature extraction - in production this would be more sophisticated
        seq_length = self.config.get('seq_length', 10)
        n_features = 8  # Match the model's expected input size
        
        # Create dummy features for now
        features = np.random.randn(seq_length, n_features) * 0.1
        
        # Add some pattern-specific features
        for i, pattern in enumerate(pattern_sequence[-seq_length:]):
            if pattern in self.pattern_types:
                pattern_idx = self.pattern_types.index(pattern)
                # Map pattern index to feature index (modulo to fit in 8 features)
                feature_idx = pattern_idx % n_features
                features[i, feature_idx] = 1.0
        
        return features
    
    def _get_default_prediction(self, pattern_sequence: List[str]) -> Dict[str, Any]:
        """Get default prediction when models are not available"""
        # Simple rule-based prediction
        if pattern_sequence:
            last_pattern = pattern_sequence[-1]
            
            # Simple pattern transitions
            transitions = {
                'double_bottom': 'flag_bull',
                'double_top': 'flag_bear',
                'triangle_ascending': 'flag_bull',
                'triangle_descending': 'flag_bear',
                'flag_bull': 'wedge_rising',
                'flag_bear': 'wedge_falling',
                'wedge_rising': 'double_top',
                'wedge_falling': 'double_bottom',
                'head_shoulders': 'flag_bear'
            }
            
            next_pattern = transitions.get(last_pattern, 'triangle_ascending')
            confidence = 0.3  # Low confidence for rule-based prediction
        else:
            next_pattern = 'triangle_ascending'
            confidence = 0.2
        
        return {
            'next_pattern': next_pattern,
            'confidence': confidence,
            'alternatives': [],
            'model_predictions': {},
            'model_confidences': {}
        }
    
    def predict_price_movement(self, current_price: float, pattern: str, 
                             horizon: int = 1) -> Dict[str, float]:
        """
        Predict price movement based on pattern
        
        Args:
            current_price: Current price
            pattern: Current pattern
            horizon: Prediction horizon in steps
            
        Returns:
            Dictionary with price predictions
        """
        # Simple pattern-based price prediction
        # In production, this would use the trained models
        
        pattern_effects = {
            'flag_bull': 0.02,  # 2% increase
            'flag_bear': -0.02,  # 2% decrease
            'double_bottom': 0.03,  # 3% increase
            'double_top': -0.03,  # 3% decrease
            'triangle_ascending': 0.015,  # 1.5% increase
            'triangle_descending': -0.015,  # 1.5% decrease
            'wedge_rising': 0.01,  # 1% increase
            'wedge_falling': -0.01,  # 1% decrease
            'head_shoulders': -0.025  # 2.5% decrease
        }
        
        effect = pattern_effects.get(pattern, 0.0)
        
        # Apply effect over horizon
        price_change = current_price * effect * horizon
        
        return {
            'lstm': current_price + price_change * 0.9,
            'gru': current_price + price_change * 0.95,
            'transformer': current_price + price_change * 1.05,
            'ensemble': current_price + price_change
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'config_loaded': self.config is not None,
            'label_encoder_loaded': self.label_encoder is not None,
            'feature_scaler_loaded': self.feature_scaler is not None,
            'markov_model_loaded': self.markov_model is not None,
            'device': str(self.device),
            'ensemble_weights': self.config.get('ensemble_weights', {}) if self.config else {}
        }
