"""
Pattern Predictor Module

Predicts next patterns in sequences using various ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class PatternPredictor:
    """
    Predicts next pattern in a sequence using ensemble methods.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the pattern predictor.
        
        Args:
            model_type: Type of model ('rf', 'gb', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize models based on type
        if model_type == 'rf':
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gb':
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'ensemble':
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
        
        logger.info(f"Initialized PatternPredictor with {model_type} model")
    
    def train(self, sequences: List[List[str]], features: Optional[np.ndarray] = None):
        """
        Train the predictor on pattern sequences.
        
        Args:
            sequences: List of pattern sequences
            features: Optional additional features for each sequence
        """
        if len(sequences) < 10:
            raise ValueError("Need at least 10 sequences for training")
        
        # Prepare training data
        X, y = self._prepare_training_data(sequences, features)
        
        # Fit label encoder
        all_patterns = []
        for seq in sequences:
            all_patterns.extend(seq)
        self.label_encoder.fit(list(set(all_patterns)))
        
        # Encode targets
        y_encoded = self.label_encoder.transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_scaled, y_encoded)
        
        self.is_trained = True
        logger.info("Pattern predictor training complete")
    
    def predict(self, sequence: List[str], features: Optional[np.ndarray] = None) -> Dict:
        """
        Predict the next pattern in a sequence.
        
        Args:
            sequence: Pattern sequence
            features: Optional additional features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Prepare features
        X = self._extract_sequence_features(sequence)
        if features is not None:
            X = np.concatenate([X, features])
        
        X_scaled = self.scaler.transform([X])
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            predictions[name] = self.label_encoder.inverse_transform([pred])[0]
            probabilities[name] = dict(zip(
                self.label_encoder.classes_,
                self.label_encoder.inverse_transform(range(len(proba))),
                proba
            ))
        
        # Ensemble prediction
        if self.model_type == 'ensemble':
            # Average probabilities
            ensemble_proba = np.zeros(len(self.label_encoder.classes_))
            for proba in probabilities.values():
                for i, pattern in enumerate(self.label_encoder.classes_):
                    pattern_name = self.label_encoder.inverse_transform([pattern])[0]
                    ensemble_proba[i] += proba.get(pattern_name, 0)
            
            ensemble_proba /= len(self.models)
            
            # Get ensemble prediction
            ensemble_pred_idx = np.argmax(ensemble_proba)
            ensemble_pred = self.label_encoder.inverse_transform([ensemble_pred_idx])[0]
            
            result = {
                'prediction': ensemble_pred,
                'confidence': float(np.max(ensemble_proba)),
                'probabilities': dict(zip(
                    self.label_encoder.inverse_transform(range(len(ensemble_proba))),
                    ensemble_proba
                )),
                'model_predictions': predictions
            }
        else:
            # Single model prediction
            model_name = list(self.models.keys())[0]
            result = {
                'prediction': predictions[model_name],
                'confidence': float(max(probabilities[model_name].values())),
                'probabilities': probabilities[model_name],
                'model_predictions': predictions
            }
        
        return result
    
    def _prepare_training_data(self, sequences: List[List[str]], 
                              features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from sequences.
        
        Args:
            sequences: List of pattern sequences
            features: Optional additional features
            
        Returns:
            Tuple of (X, y) for training
        """
        X = []
        y = []
        
        for i, seq in enumerate(sequences[:-1]):
            # Extract features from sequence
            seq_features = self._extract_sequence_features(seq)
            
            # Add additional features if provided
            if features is not None and i < len(features):
                seq_features = np.concatenate([seq_features, features[i]])
            
            X.append(seq_features)
            
            # Next pattern as target
            # Assuming sequences are overlapping, so next sequence starts one step ahead
            if i + 1 < len(sequences):
                y.append(sequences[i + 1][-1])
        
        return np.array(X), np.array(y)
    
    def _extract_sequence_features(self, sequence: List[str]) -> np.ndarray:
        """
        Extract features from a pattern sequence.
        
        Args:
            sequence: Pattern sequence
            
        Returns:
            Feature vector
        """
        features = []
        
        # Pattern type counts
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        
        for pattern_type in pattern_types:
            count = sequence.count(pattern_type)
            features.append(count)
        
        # Pattern transitions
        transitions = 0
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                transitions += 1
        features.append(transitions)
        
        # Last pattern encoding
        last_pattern = sequence[-1]
        for pattern_type in pattern_types:
            features.append(1.0 if last_pattern == pattern_type else 0.0)
        
        # Sequence length
        features.append(len(sequence))
        
        # Pattern diversity
        unique_patterns = len(set(sequence))
        features.append(unique_patterns)
        
        return np.array(features)
    
    def evaluate(self, test_sequences: List[List[str]], 
                test_features: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate predictor performance on test sequences.
        
        Args:
            test_sequences: Test sequences
            test_features: Optional test features
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        predictions = []
        actuals = []
        confidences = []
        
        for i in range(len(test_sequences) - 1):
            # Predict next pattern
            result = self.predict(test_sequences[i], 
                                test_features[i] if test_features is not None else None)
            
            predictions.append(result['prediction'])
            actuals.append(test_sequences[i + 1][-1])
            confidences.append(result['confidence'])
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        accuracy = np.mean(predictions == actuals)
        
        # Per-class accuracy
        class_accuracies = {}
        for pattern_type in set(actuals):
            mask = actuals == pattern_type
            if np.any(mask):
                class_accuracies[pattern_type] = np.mean(
                    predictions[mask] == actuals[mask]
                )
        
        # Confidence calibration
        confidence_bins = np.linspace(0, 1, 11)
        calibration_data = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (
                (confidences >= confidence_bins[i]) & 
                (confidences < confidence_bins[i + 1])
            )
            if np.any(bin_mask):
                bin_accuracy = np.mean(predictions[bin_mask] == actuals[bin_mask])
                bin_confidence = np.mean(np.array(confidences)[bin_mask])
                calibration_data.append({
                    'confidence': bin_confidence,
                    'accuracy': bin_accuracy,
                    'count': np.sum(bin_mask)
                })
        
        return {
            'overall_accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'mean_confidence': float(np.mean(confidences)),
            'calibration_data': calibration_data
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        import joblib
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
