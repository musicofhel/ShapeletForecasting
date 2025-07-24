"""
Adaptive learning system for online model updates.
Implements incremental learning and concept drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from river import drift
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Results from drift detection."""
    drift_detected: bool
    drift_score: float
    drift_type: str  # 'gradual', 'sudden', 'incremental', 'none'
    confidence: float
    timestamp: pd.Timestamp


class AdaptiveLearner:
    """
    Implements adaptive learning with concept drift detection
    and online model updates.
    """
    
    def __init__(self, 
                 base_model: Any,
                 buffer_size: int = 1000,
                 drift_threshold: float = 0.05,
                 update_frequency: int = 100):
        """
        Initialize adaptive learner.
        
        Args:
            base_model: Base model to adapt
            buffer_size: Size of data buffer for retraining
            drift_threshold: Threshold for drift detection
            update_frequency: How often to check for updates
        """
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.drift_threshold = drift_threshold
        self.update_frequency = update_frequency
        
        # Data buffers
        self.feature_buffer = deque(maxlen=buffer_size)
        self.target_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Drift detectors
        self.adwin = drift.ADWIN()  # Adaptive windowing
        self.page_hinkley = drift.PageHinkley()  # Page-Hinkley test
        self.kswin = drift.KSWIN()  # Kolmogorov-Smirnov Windowing method (replacement for DDM)
        
        # Performance tracking
        self.performance_history = []
        self.drift_history = []
        self.update_history = []
        
        # Model versioning
        self.model_version = 0
        self.model_checkpoints = {}
        
        # Statistics for normalization
        self.feature_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        
        self.update_counter = 0
        
    def update_statistics(self, features: np.ndarray):
        """
        Update feature statistics for normalization.
        
        Args:
            features: New feature data
        """
        if self.feature_stats['mean'] is None:
            self.feature_stats['mean'] = np.mean(features, axis=0)
            self.feature_stats['std'] = np.std(features, axis=0)
            self.feature_stats['min'] = np.min(features, axis=0)
            self.feature_stats['max'] = np.max(features, axis=0)
        else:
            # Exponential moving average update
            alpha = 0.01
            self.feature_stats['mean'] = (1 - alpha) * self.feature_stats['mean'] + alpha * np.mean(features, axis=0)
            self.feature_stats['std'] = (1 - alpha) * self.feature_stats['std'] + alpha * np.std(features, axis=0)
            self.feature_stats['min'] = np.minimum(self.feature_stats['min'], np.min(features, axis=0))
            self.feature_stats['max'] = np.maximum(self.feature_stats['max'], np.max(features, axis=0))
    
    def detect_drift(self, error: float) -> DriftDetectionResult:
        """
        Detect concept drift using multiple methods.
        
        Args:
            error: Prediction error
            
        Returns:
            Drift detection result
        """
        # Update drift detectors
        self.adwin.update(error)
        self.page_hinkley.update(error)
        self.kswin.update(error)
        
        # Check for drift
        drift_detected = False
        drift_type = 'none'
        confidence = 0.0
        
        # ADWIN detection (adaptive windowing)
        if self.adwin.drift_detected:
            drift_detected = True
            drift_type = 'sudden'
            confidence = max(confidence, 0.9)
            
        # Page-Hinkley detection
        if self.page_hinkley.drift_detected:
            drift_detected = True
            if drift_type == 'none':
                drift_type = 'gradual'
            confidence = max(confidence, 0.8)
            
        # KSWIN detection
        if self.kswin.drift_detected:
            drift_detected = True
            if drift_type == 'none':
                drift_type = 'incremental'
            confidence = max(confidence, 0.7)
            
        # Calculate drift score
        recent_errors = list(self.prediction_buffer)[-100:] if len(self.prediction_buffer) >= 100 else list(self.prediction_buffer)
        if recent_errors:
            drift_score = np.mean(recent_errors) / (np.std(recent_errors) + 1e-8)
        else:
            drift_score = 0.0
            
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            drift_type=drift_type,
            confidence=float(confidence),
            timestamp=pd.Timestamp.now()
        )
        
        if drift_detected:
            self.drift_history.append(result)
            logger.warning(f"Drift detected: {drift_type} with confidence {confidence:.2f}")
            
        return result
    
    def incremental_update(self, features: np.ndarray, targets: np.ndarray):
        """
        Perform incremental update of the model.
        
        Args:
            features: New feature data
            targets: New target values
        """
        # Update statistics
        self.update_statistics(features)
        
        # Normalize features
        features_norm = (features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8)
        
        # Model-specific incremental update
        if hasattr(self.base_model, 'partial_fit'):
            # Scikit-learn style incremental learning
            self.base_model.partial_fit(features_norm, targets)
            
        elif isinstance(self.base_model, nn.Module):
            # PyTorch model incremental update
            self._pytorch_incremental_update(features_norm, targets)
            
        else:
            # Fallback: retrain on buffer
            if len(self.feature_buffer) >= self.buffer_size // 2:
                self._retrain_on_buffer()
                
        self.model_version += 1
        logger.info(f"Model updated to version {self.model_version}")
    
    def _pytorch_incremental_update(self, features: np.ndarray, targets: np.ndarray):
        """
        Incremental update for PyTorch models.
        
        Args:
            features: Normalized features
            targets: Target values
        """
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(targets)
        
        # Set model to training mode
        self.base_model.train()
        
        # Define optimizer if not exists
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.001)
            
        # Mini-batch training
        batch_size = min(32, len(features))
        for i in range(0, len(features), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.base_model(batch_X)
            loss = nn.MSELoss()(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
        # Set back to eval mode
        self.base_model.eval()
    
    def _retrain_on_buffer(self):
        """Retrain model on the entire buffer."""
        if len(self.feature_buffer) < 100:
            return
            
        # Prepare data
        X = np.array(self.feature_buffer)
        y = np.array(self.target_buffer)
        
        # Normalize
        X_norm = (X - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8)
        
        # Retrain model
        if hasattr(self.base_model, 'fit'):
            self.base_model.fit(X_norm, y)
            logger.info("Model retrained on buffer")
    
    def predict_adaptive(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Make predictions with confidence estimation.
        
        Args:
            features: Input features
            
        Returns:
            Predictions and confidence score
        """
        # Normalize features
        if self.feature_stats['mean'] is not None:
            features_norm = (features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8)
        else:
            features_norm = features
            
        # Make prediction
        if hasattr(self.base_model, 'predict'):
            predictions = self.base_model.predict(features_norm)
        elif isinstance(self.base_model, nn.Module):
            self.base_model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(features_norm)
                predictions = self.base_model(X).numpy()
        else:
            predictions = np.zeros(len(features))
            
        # Estimate confidence based on recent performance
        if self.performance_history:
            recent_perf = self.performance_history[-10:]
            avg_error = np.mean([p['mae'] for p in recent_perf])
            confidence = 1.0 / (1.0 + avg_error)
        else:
            confidence = 0.5
            
        return predictions, float(confidence)
    
    def update(self, features: np.ndarray, targets: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None):
        """
        Update the adaptive learner with new data.
        
        Args:
            features: New features
            targets: True target values
            timestamps: Timestamps for the data
        """
        # Make predictions before update
        predictions, _ = self.predict_adaptive(features)
        
        # Calculate errors
        errors = np.abs(predictions - targets)
        
        # Add to buffers
        for i in range(len(features)):
            self.feature_buffer.append(features[i])
            self.target_buffer.append(targets[i])
            self.prediction_buffer.append(errors[i])
            if timestamps is not None:
                self.timestamp_buffer.append(timestamps[i])
                
        # Check for drift
        avg_error = np.mean(errors)
        drift_result = self.detect_drift(avg_error)
        
        # Update counter
        self.update_counter += len(features)
        
        # Perform update if needed
        if drift_result.drift_detected or self.update_counter >= self.update_frequency:
            self.incremental_update(features, targets)
            self.update_counter = 0
            
            # Save checkpoint if significant drift
            if drift_result.drift_detected and drift_result.confidence > 0.8:
                self.save_checkpoint()
                
        # Track performance
        performance = {
            'timestamp': pd.Timestamp.now(),
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'max_error': float(np.max(errors)),
            'drift_detected': drift_result.drift_detected,
            'model_version': self.model_version
        }
        self.performance_history.append(performance)
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"v{self.model_version}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
        checkpoint = {
            'model_state': self._get_model_state(),
            'feature_stats': self.feature_stats,
            'model_version': self.model_version,
            'performance_history': self.performance_history[-100:],  # Last 100 entries
            'drift_history': self.drift_history
        }
        
        self.model_checkpoints[checkpoint_name] = checkpoint
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def _get_model_state(self):
        """Get model state for checkpointing."""
        if isinstance(self.base_model, nn.Module):
            return self.base_model.state_dict()
        elif hasattr(self.base_model, '__getstate__'):
            return self.base_model.__getstate__()
        else:
            return pickle.dumps(self.base_model)
    
    def load_checkpoint(self, checkpoint_name: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load
        """
        if checkpoint_name not in self.model_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_name} not found")
            
        checkpoint = self.model_checkpoints[checkpoint_name]
        
        # Restore model state
        if isinstance(self.base_model, nn.Module):
            self.base_model.load_state_dict(checkpoint['model_state'])
        elif hasattr(self.base_model, '__setstate__'):
            self.base_model.__setstate__(checkpoint['model_state'])
        else:
            self.base_model = pickle.loads(checkpoint['model_state'])
            
        # Restore other states
        self.feature_stats = checkpoint['feature_stats']
        self.model_version = checkpoint['model_version']
        self.performance_history = checkpoint['performance_history']
        self.drift_history = checkpoint['drift_history']
        
        logger.info(f"Checkpoint loaded: {checkpoint_name}")
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """
        Generate report on model adaptation.
        
        Returns:
            Adaptation statistics and metrics
        """
        report = {
            'model_version': self.model_version,
            'total_updates': len(self.update_history),
            'total_drifts_detected': len(self.drift_history),
            'buffer_size': len(self.feature_buffer),
            'checkpoints_saved': len(self.model_checkpoints)
        }
        
        # Performance trends
        if self.performance_history:
            recent_perf = self.performance_history[-100:]
            report['recent_mae'] = np.mean([p['mae'] for p in recent_perf])
            report['recent_rmse'] = np.mean([p['rmse'] for p in recent_perf])
            report['performance_trend'] = 'improving' if recent_perf[-1]['mae'] < recent_perf[0]['mae'] else 'degrading'
            
        # Drift analysis
        if self.drift_history:
            drift_types = [d.drift_type for d in self.drift_history]
            report['drift_types'] = {t: drift_types.count(t) for t in set(drift_types)}
            report['avg_drift_confidence'] = np.mean([d.confidence for d in self.drift_history])
            
        return report
    
    def reset_drift_detectors(self):
        """Reset all drift detectors."""
        self.adwin = drift.ADWIN()
        self.page_hinkley = drift.PageHinkley()
        self.kswin = drift.KSWIN()
        logger.info("Drift detectors reset")
