"""
Ensemble model framework for combining multiple predictors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import logging
import joblib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEnsembleStrategy(ABC):
    """Base class for ensemble strategies"""
    
    @abstractmethod
    def combine_predictions(self, predictions: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Combine predictions from multiple models"""
        pass


class AveragingStrategy(BaseEnsembleStrategy):
    """Simple averaging of predictions"""
    
    def combine_predictions(self, predictions: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Average predictions
        
        Args:
            predictions: Array of predictions (n_models, n_samples)
            weights: Optional weights for each model
            
        Returns:
            Combined predictions
        """
        if weights is not None:
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        return np.mean(predictions, axis=0)


class MedianStrategy(BaseEnsembleStrategy):
    """Median of predictions"""
    
    def combine_predictions(self, predictions: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Take median of predictions
        
        Args:
            predictions: Array of predictions (n_models, n_samples)
            weights: Not used for median
            
        Returns:
            Combined predictions
        """
        return np.median(predictions, axis=0)


class StackingStrategy(BaseEnsembleStrategy):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, meta_learner: BaseEstimator):
        """
        Initialize stacking strategy
        
        Args:
            meta_learner: Model to learn combination weights
        """
        self.meta_learner = meta_learner
        self.is_fitted = False
    
    def fit(self, predictions: np.ndarray, y_true: np.ndarray):
        """
        Train meta-learner
        
        Args:
            predictions: Array of predictions (n_models, n_samples)
            y_true: True target values
        """
        # Transpose to (n_samples, n_models)
        X_meta = predictions.T
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True
    
    def combine_predictions(self, predictions: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine predictions using meta-learner
        
        Args:
            predictions: Array of predictions (n_models, n_samples)
            weights: Not used in stacking
            
        Returns:
            Combined predictions
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted. Call fit() first.")
        
        X_meta = predictions.T
        return self.meta_learner.predict(X_meta)


class EnsembleModel(BaseEstimator, RegressorMixin):
    """Ensemble model combining multiple predictors"""
    
    def __init__(self, 
                 models: Dict[str, Any],
                 strategy: str = 'averaging',
                 weights: Optional[Dict[str, float]] = None,
                 meta_learner: Optional[BaseEstimator] = None,
                 use_cv_predictions: bool = True,
                 cv_folds: int = 5):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of models {name: model}
            strategy: Ensemble strategy ('averaging', 'median', 'stacking')
            weights: Model weights for averaging
            meta_learner: Meta-learner for stacking
            use_cv_predictions: Whether to use CV predictions for training meta-learner
            cv_folds: Number of CV folds
        """
        self.models = models
        self.strategy_name = strategy
        self.weights = weights
        self.meta_learner = meta_learner
        self.use_cv_predictions = use_cv_predictions
        self.cv_folds = cv_folds
        
        # Initialize strategy
        if strategy == 'averaging':
            self.strategy = AveragingStrategy()
        elif strategy == 'median':
            self.strategy = MedianStrategy()
        elif strategy == 'stacking':
            if meta_learner is None:
                raise ValueError("Meta-learner required for stacking strategy")
            self.strategy = StackingStrategy(meta_learner)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Convert weights to array
        if weights is not None:
            self.weight_array = np.array([weights.get(name, 1.0) for name in sorted(models.keys())])
        else:
            self.weight_array = None
        
        self.is_fitted = False
        self.model_predictions_ = {}
        
        logger.info(f"Initialized ensemble with {len(models)} models using {strategy} strategy")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'EnsembleModel':
        """
        Fit all models in the ensemble
        
        Args:
            X: Training features
            y: Training targets
            **fit_params: Additional parameters for fit
            
        Returns:
            Self
        """
        logger.info("Training ensemble models...")
        
        # Fit each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Handle different model types
            if hasattr(model, 'fit'):
                # Sklearn-style model
                model.fit(X, y, **fit_params.get(name, {}))
            elif isinstance(model, torch.nn.Module):
                # PyTorch model - assume it has a custom training method
                if hasattr(model, 'train_model'):
                    model.train_model(X, y, **fit_params.get(name, {}))
                else:
                    logger.warning(f"PyTorch model {name} has no train_model method")
            else:
                raise ValueError(f"Unknown model type for {name}")
        
        # For stacking, train meta-learner
        if self.strategy_name == 'stacking':
            if self.use_cv_predictions:
                # Get out-of-fold predictions
                predictions = self._get_cv_predictions(X, y)
            else:
                # Get in-sample predictions
                predictions = self._get_predictions(X)
            
            # Train meta-learner
            self.strategy.fit(predictions, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble
        
        Args:
            X: Features to predict
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all models
        predictions = self._get_predictions(X)
        
        # Combine predictions
        return self.strategy.combine_predictions(predictions, self.weight_array)
    
    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all models
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions (n_models, n_samples)
        """
        predictions = []
        
        for name in sorted(self.models.keys()):
            model = self.models[name]
            
            # Handle different model types
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif isinstance(model, torch.nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy()
            else:
                raise ValueError(f"Model {name} has no predict method")
            
            predictions.append(pred.flatten())
        
        return np.array(predictions)
    
    def _get_cv_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get cross-validated predictions for stacking
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Array of CV predictions (n_models, n_samples)
        """
        cv_predictions = []
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name in sorted(self.models.keys()):
            model = self.models[name]
            
            # Get CV predictions
            if hasattr(model, 'predict'):
                # Sklearn-style model
                pred = cross_val_predict(model, X, y, cv=kf)
            else:
                # Manual CV for non-sklearn models
                pred = np.zeros(len(y))
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                    
                    # Clone and train model
                    if hasattr(model, 'clone'):
                        model_fold = model.clone()
                    else:
                        # Simple copy for basic models
                        import copy
                        model_fold = copy.deepcopy(model)
                    
                    if hasattr(model_fold, 'fit'):
                        model_fold.fit(X_train, y_train)
                        pred[val_idx] = model_fold.predict(X_val)
            
            cv_predictions.append(pred)
        
        return np.array(cv_predictions)
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get model weights (for stacking, returns meta-learner coefficients)
        
        Returns:
            Dictionary of model weights
        """
        if self.strategy_name == 'stacking' and hasattr(self.strategy.meta_learner, 'coef_'):
            coefs = self.strategy.meta_learner.coef_
            return {name: coef for name, coef in zip(sorted(self.models.keys()), coefs)}
        elif self.weights is not None:
            return self.weights
        else:
            # Equal weights
            n_models = len(self.models)
            return {name: 1.0 / n_models for name in self.models.keys()}
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble and individual models
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Ensemble predictions
        y_pred_ensemble = self.predict(X)
        results['ensemble'] = {
            'mse': mean_squared_error(y, y_pred_ensemble),
            'mae': mean_absolute_error(y, y_pred_ensemble),
            'r2': r2_score(y, y_pred_ensemble)
        }
        
        # Individual model predictions
        predictions = self._get_predictions(X)
        for i, name in enumerate(sorted(self.models.keys())):
            y_pred = predictions[i]
            results[name] = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return results
    
    def save(self, filepath: str):
        """
        Save ensemble model
        
        Args:
            filepath: Path to save model
        """
        # Save configuration
        config = {
            'strategy': self.strategy_name,
            'weights': self.weights,
            'cv_folds': self.cv_folds,
            'use_cv_predictions': self.use_cv_predictions,
            'model_names': list(self.models.keys())
        }
        
        # Save models separately
        model_paths = {}
        for name, model in self.models.items():
            model_path = filepath.replace('.pkl', f'_{name}.pkl')
            
            if hasattr(model, 'save_model'):
                model.save_model(model_path)
            elif hasattr(model, 'save'):
                model.save(model_path)
            else:
                joblib.dump(model, model_path)
            
            model_paths[name] = model_path
        
        config['model_paths'] = model_paths
        
        # Save meta-learner if stacking
        if self.strategy_name == 'stacking':
            meta_path = filepath.replace('.pkl', '_meta.pkl')
            joblib.dump(self.strategy.meta_learner, meta_path)
            config['meta_learner_path'] = meta_path
        
        # Save configuration
        config_path = filepath.replace('.pkl', '_config.pkl')
        joblib.dump(config, config_path)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load ensemble model
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded ensemble model
        """
        # Load configuration
        config_path = filepath.replace('.pkl', '_config.pkl')
        config = joblib.load(config_path)
        
        # Load individual models
        models = {}
        for name, model_path in config['model_paths'].items():
            # Try different loading methods
            try:
                models[name] = joblib.load(model_path)
            except:
                # Custom loading for specific model types
                logger.warning(f"Could not load {name} with joblib, trying custom loader")
        
        # Load meta-learner if stacking
        meta_learner = None
        if config['strategy'] == 'stacking':
            meta_learner = joblib.load(config['meta_learner_path'])
        
        # Create ensemble
        ensemble = cls(
            models=models,
            strategy=config['strategy'],
            weights=config.get('weights'),
            meta_learner=meta_learner,
            use_cv_predictions=config.get('use_cv_predictions', True),
            cv_folds=config.get('cv_folds', 5)
        )
        
        ensemble.is_fitted = True
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble


class DynamicEnsemble(EnsembleModel):
    """Dynamic ensemble that adjusts weights based on recent performance"""
    
    def __init__(self, models: Dict[str, Any], window_size: int = 100,
                 update_frequency: int = 10, **kwargs):
        """
        Initialize dynamic ensemble
        
        Args:
            models: Dictionary of models
            window_size: Size of performance window
            update_frequency: How often to update weights
            **kwargs: Additional arguments for EnsembleModel
        """
        super().__init__(models, strategy='averaging', **kwargs)
        
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.performance_history = {name: [] for name in models.keys()}
        self.prediction_count = 0
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions and optionally update weights
        
        Args:
            X: Features to predict
            y_true: True values for weight update
            
        Returns:
            Predictions
        """
        predictions = self._get_predictions(X)
        
        # Update performance history if true values provided
        if y_true is not None:
            for i, name in enumerate(sorted(self.models.keys())):
                error = mean_squared_error(y_true, predictions[i])
                self.performance_history[name].append(error)
                
                # Keep only recent history
                if len(self.performance_history[name]) > self.window_size:
                    self.performance_history[name].pop(0)
            
            self.prediction_count += 1
            
            # Update weights periodically
            if self.prediction_count % self.update_frequency == 0:
                self._update_weights()
        
        # Combine predictions with current weights
        return self.strategy.combine_predictions(predictions, self.weight_array)
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        if all(len(hist) > 0 for hist in self.performance_history.values()):
            # Calculate average recent performance
            avg_errors = {}
            for name, history in self.performance_history.items():
                avg_errors[name] = np.mean(history)
            
            # Convert to weights (inverse of error)
            total_inv_error = sum(1.0 / (err + 1e-6) for err in avg_errors.values())
            
            new_weights = {}
            for name, err in avg_errors.items():
                new_weights[name] = (1.0 / (err + 1e-6)) / total_inv_error
            
            self.weights = new_weights
            self.weight_array = np.array([new_weights[name] for name in sorted(self.models.keys())])
            
            logger.info(f"Updated weights: {new_weights}")


if __name__ == "__main__":
    # Example usage
    print("Ensemble model module loaded successfully")
    print("Use EnsembleModel class to create ensemble predictors")
