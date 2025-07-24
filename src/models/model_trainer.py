"""
Model training pipeline with cross-validation and hyperparameter optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import logging
import json
import os
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.pytorch

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified training pipeline for all model types"""
    
    def __init__(self,
                 model_type: str,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 experiment_name: str = "financial_prediction",
                 use_mlflow: bool = True):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'transformer', 'xgboost', 'ensemble')
            model_config: Model configuration parameters
            training_config: Training configuration
            experiment_name: MLflow experiment name
            use_mlflow: Whether to use MLflow for tracking
        """
        self.model_type = model_type
        self.model_config = model_config
        self.training_config = training_config
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        
        # Initialize MLflow
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
        
        # Training history
        self.training_history = {}
        self.best_model = None
        self.best_params = None
        self.best_score = None
        
        logger.info(f"Initialized trainer for {model_type} model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              optimize_hyperparams: bool = False) -> Any:
        """
        Train model with optional hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Trained model
        """
        if optimize_hyperparams:
            logger.info("Starting hyperparameter optimization...")
            self.best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            self.model_config.update(self.best_params)
        
        # Train final model
        logger.info("Training final model...")
        
        if self.use_mlflow:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.model_config)
                mlflow.log_params(self.training_config)
                
                # Train model
                model = self._train_model(X_train, y_train, X_val, y_val)
                
                # Log metrics
                if X_val is not None and y_val is not None:
                    metrics = self._evaluate_model(model, X_val, y_val)
                    mlflow.log_metrics(metrics)
                
                # Log model
                if self.model_type in ['lstm', 'gru', 'transformer']:
                    mlflow.pytorch.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
        else:
            model = self._train_model(X_train, y_train, X_val, y_val)
        
        self.best_model = model
        return model
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """
        Train a single model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained model
        """
        if self.model_type in ['lstm', 'gru']:
            return self._train_sequence_model(X_train, y_train, X_val, y_val)
        elif self.model_type == 'transformer':
            return self._train_transformer_model(X_train, y_train, X_val, y_val)
        elif self.model_type == 'xgboost':
            return self._train_xgboost_model(X_train, y_train, X_val, y_val)
        elif self.model_type == 'ensemble':
            return self._train_ensemble_model(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_sequence_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> nn.Module:
        """Train LSTM or GRU model"""
        from .sequence_predictor import LSTMModel, GRUModel, TimeSeriesDataset, train_sequence_model
        
        # Create model
        if self.model_type == 'lstm':
            model = LSTMModel(**self.model_config)
        else:
            model = GRUModel(**self.model_config)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            X_train, y_train,
            sequence_length=self.training_config.get('sequence_length', 20),
            prediction_horizon=self.training_config.get('prediction_horizon', 1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(
                X_val, y_val,
                sequence_length=self.training_config.get('sequence_length', 20),
                prediction_horizon=self.training_config.get('prediction_horizon', 1)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False
            )
        
        # Train model
        history = train_sequence_model(
            model, train_loader, val_loader,
            epochs=self.training_config.get('epochs', 100),
            learning_rate=self.training_config.get('learning_rate', 0.001),
            early_stopping_patience=self.training_config.get('early_stopping_patience', 10),
            checkpoint_path=self.training_config.get('checkpoint_path')
        )
        
        self.training_history = history
        return model
    
    def _train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> nn.Module:
        """Train Transformer model"""
        from .transformer_predictor import TransformerPredictor
        from .sequence_predictor import TimeSeriesDataset, train_sequence_model
        
        # Create model
        model = TransformerPredictor(**self.model_config)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            X_train, y_train,
            sequence_length=self.training_config.get('sequence_length', 20),
            prediction_horizon=self.training_config.get('prediction_horizon', 1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(
                X_val, y_val,
                sequence_length=self.training_config.get('sequence_length', 20),
                prediction_horizon=self.training_config.get('prediction_horizon', 1)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False
            )
        
        # Train model
        history = train_sequence_model(
            model, train_loader, val_loader,
            epochs=self.training_config.get('epochs', 100),
            learning_rate=self.training_config.get('learning_rate', 0.001),
            early_stopping_patience=self.training_config.get('early_stopping_patience', 10),
            checkpoint_path=self.training_config.get('checkpoint_path')
        )
        
        self.training_history = history
        return model
    
    def _train_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Train XGBoost model"""
        from .xgboost_predictor import XGBoostPredictor
        
        # Create model
        model = XGBoostPredictor(**self.model_config)
        
        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=self.training_config.get('verbose', True)
        )
        
        return model
    
    def _train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Train ensemble model"""
        from .ensemble_model import EnsembleModel
        
        # Create individual models based on config
        models = {}
        for model_name, model_config in self.model_config['models'].items():
            model_type = model_config['type']
            model_params = model_config['params']
            
            if model_type == 'xgboost':
                from .xgboost_predictor import XGBoostPredictor
                models[model_name] = XGBoostPredictor(**model_params)
            elif model_type == 'ridge':
                from sklearn.linear_model import Ridge
                models[model_name] = Ridge(**model_params)
            elif model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                models[model_name] = RandomForestRegressor(**model_params)
            # Add more model types as needed
        
        # Create ensemble
        ensemble = EnsembleModel(
            models=models,
            strategy=self.model_config.get('strategy', 'averaging'),
            weights=self.model_config.get('weights'),
            meta_learner=self.model_config.get('meta_learner')
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Sample hyperparameters based on model type
            if self.model_type in ['lstm', 'gru']:
                params = {
                    'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=32),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                }
            elif self.model_type == 'transformer':
                params = {
                    'd_model': trial.suggest_int('d_model', 32, 256, step=32),
                    'n_heads': trial.suggest_int('n_heads', 2, 8),
                    'n_layers': trial.suggest_int('n_layers', 1, 6),
                    'd_ff': trial.suggest_int('d_ff', 128, 1024, step=128),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                }
            elif self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
                }
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {self.model_type}")
            
            # Update config with trial parameters
            trial_config = self.model_config.copy()
            trial_config.update(params)
            
            # Train model with trial parameters
            if self.model_type in ['lstm', 'gru', 'transformer']:
                # For neural networks, use validation loss
                model = self._create_model(self.model_type, trial_config)
                val_loss = self._train_and_evaluate_nn(model, X_train, y_train, X_val, y_val, params)
                return val_loss
            else:
                # For other models, use cross-validation
                model = self._create_model(self.model_type, trial_config)
                cv_score = self._cross_validate(model, X_train, y_train)
                return -cv_score  # Minimize negative score
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.training_config.get('n_trials', 50))
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value}")
        
        return study.best_params
    
    def _create_model(self, model_type: str, config: Dict[str, Any]) -> Any:
        """Create model instance"""
        if model_type == 'lstm':
            from .sequence_predictor import LSTMModel
            return LSTMModel(**config)
        elif model_type == 'gru':
            from .sequence_predictor import GRUModel
            return GRUModel(**config)
        elif model_type == 'transformer':
            from .transformer_predictor import TransformerPredictor
            return TransformerPredictor(**config)
        elif model_type == 'xgboost':
            from .xgboost_predictor import XGBoostPredictor
            return XGBoostPredictor(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_and_evaluate_nn(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray, params: Dict[str, Any]) -> float:
        """Train and evaluate neural network for hyperparameter optimization"""
        from .sequence_predictor import TimeSeriesDataset, train_sequence_model
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train with early stopping
        history = train_sequence_model(
            model, train_loader, val_loader,
            epochs=50,  # Reduced for hyperparameter search
            learning_rate=params.get('learning_rate', 0.001),
            early_stopping_patience=5
        )
        
        # Return best validation loss
        return min(history['val_loss'])
    
    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Perform time series cross-validation"""
        from sklearn.metrics import mean_squared_error
        
        tscv = TimeSeriesSplit(n_splits=self.training_config.get('cv_splits', 5))
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model
            import copy
            model_fold = copy.deepcopy(model)
            
            # Train
            if hasattr(model_fold, 'fit'):
                model_fold.fit(X_train, y_train)
                y_pred = model_fold.predict(X_val)
                score = mean_squared_error(y_val, y_pred)
                scores.append(score)
        
        return np.mean(scores)
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_pred = model(X_tensor).numpy()
        else:
            raise ValueError("Model has no predict method")
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics
    
    def save_training_results(self, save_dir: str):
        """
        Save training results and configuration
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        config = {
            'model_type': self.model_type,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        if self.training_history:
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        # Save model
        if self.best_model is not None:
            model_path = os.path.join(save_dir, 'model.pkl')
            
            if isinstance(self.best_model, nn.Module):
                torch.save(self.best_model.state_dict(), model_path)
            elif hasattr(self.best_model, 'save_model'):
                self.best_model.save_model(model_path)
            else:
                import joblib
                joblib.dump(self.best_model, model_path)
        
        logger.info(f"Training results saved to {save_dir}")


class WalkForwardValidator:
    """Walk-forward validation for time series"""
    
    def __init__(self, 
                 train_size: int,
                 test_size: int,
                 step_size: int,
                 retrain_frequency: int = 1):
        """
        Initialize walk-forward validator
        
        Args:
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for moving window
            retrain_frequency: How often to retrain model
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
    
    def validate(self, trainer: ModelTrainer, X: np.ndarray, y: np.ndarray,
                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform walk-forward validation
        
        Args:
            trainer: Model trainer
            X: Features
            y: Targets
            feature_names: Feature names
            
        Returns:
            Validation results
        """
        results = {
            'predictions': [],
            'actuals': [],
            'metrics': [],
            'models': []
        }
        
        n_samples = len(X)
        current_model = None
        retrain_counter = 0
        
        # Walk forward through time
        for i in range(self.train_size, n_samples - self.test_size + 1, self.step_size):
            # Define windows
            train_start = max(0, i - self.train_size)
            train_end = i
            test_start = i
            test_end = min(i + self.test_size, n_samples)
            
            # Get data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train or reuse model
            if current_model is None or retrain_counter % self.retrain_frequency == 0:
                logger.info(f"Training model for window {i}...")
                current_model = trainer.train(X_train, y_train)
                results['models'].append(current_model)
                retrain_counter = 0
            
            retrain_counter += 1
            
            # Make predictions
            if hasattr(current_model, 'predict'):
                y_pred = current_model.predict(X_test)
            else:
                # Handle neural network models
                current_model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    y_pred = current_model(X_tensor).numpy()
            
            # Store results
            results['predictions'].extend(y_pred)
            results['actuals'].extend(y_test)
            
            # Calculate metrics for this window
            window_metrics = trainer._evaluate_model(current_model, X_test, y_test)
            window_metrics['window_start'] = test_start
            window_metrics['window_end'] = test_end
            results['metrics'].append(window_metrics)
        
        # Calculate overall metrics
        y_true = np.array(results['actuals'])
        y_pred = np.array(results['predictions'])
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        results['overall_metrics'] = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Model trainer module loaded successfully")
    print("Use ModelTrainer class to train models with cross-validation and hyperparameter optimization")
