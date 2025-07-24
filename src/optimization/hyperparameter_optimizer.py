"""
Hyperparameter optimization for financial prediction models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import optuna
from optuna import Trial
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import logging
import json
import joblib
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: pd.DataFrame
    best_model: Any
    study: Optional[optuna.Study] = None
    
    def save(self, filepath: str):
        """Save optimization results"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model separately
        model_path = filepath.replace('.json', '_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        logger.info(f"Optimization results saved to {filepath}")


class HyperparameterOptimizer(ABC):
    """Base class for hyperparameter optimization"""
    
    @abstractmethod
    def optimize(self, objective_func: Callable, n_trials: int = 100) -> OptimizationResult:
        """Run optimization"""
        pass


class OptunaOptimizer(HyperparameterOptimizer):
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, 
                 direction: str = 'minimize',
                 sampler: str = 'tpe',
                 seed: int = 42,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None):
        """
        Initialize Optuna optimizer
        
        Args:
            direction: Optimization direction ('minimize' or 'maximize')
            sampler: Sampling algorithm ('tpe', 'random')
            seed: Random seed
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
        """
        self.direction = direction
        self.seed = seed
        self.n_jobs = n_jobs
        self.timeout = timeout
        
        # Create sampler
        if sampler == 'tpe':
            self.sampler = TPESampler(seed=seed)
        elif sampler == 'random':
            self.sampler = RandomSampler(seed=seed)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
    
    def optimize(self, objective_func: Callable, n_trials: int = 100,
                callbacks: Optional[List[Callable]] = None) -> OptimizationResult:
        """
        Run Optuna optimization
        
        Args:
            objective_func: Objective function to optimize
            n_trials: Number of trials
            callbacks: Optional callbacks
            
        Returns:
            Optimization results
        """
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler
        )
        
        # Run optimization
        study.optimize(
            objective_func,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            callbacks=callbacks
        )
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        # Create history dataframe
        history_data = []
        for trial in study.trials:
            trial_data = trial.params.copy()
            trial_data['value'] = trial.value
            trial_data['trial'] = trial.number
            history_data.append(trial_data)
        
        optimization_history = pd.DataFrame(history_data)
        
        # Get best model (if stored in user attributes)
        best_model = study.best_trial.user_attrs.get('model', None)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            best_model=best_model,
            study=study
        )


class XGBoostOptimizer:
    """Hyperparameter optimization for XGBoost models"""
    
    def __init__(self, 
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 cv_folds: int = 5,
                 early_stopping_rounds: int = 50):
        """
        Initialize XGBoost optimizer
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping patience
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        
        # Time series split for CV
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def create_objective(self, metric: str = 'mse') -> Callable:
        """
        Create objective function for Optuna
        
        Args:
            metric: Evaluation metric
            
        Returns:
            Objective function
        """
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': 42
            }
            
            # Cross-validation
            cv_scores = []
            
            for train_idx, val_idx in self.tscv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]
                
                # Train model
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
                
                # Evaluate
                y_pred = model.predict(X_fold_val)
                
                if metric == 'mse':
                    score = mean_squared_error(y_fold_val, y_pred)
                elif metric == 'mae':
                    score = mean_absolute_error(y_fold_val, y_pred)
                elif metric == 'r2':
                    score = -r2_score(y_fold_val, y_pred)  # Negative for minimization
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                cv_scores.append(score)
            
            # Average CV score
            avg_score = np.mean(cv_scores)
            
            # Train final model on full training data
            final_model = xgb.XGBRegressor(**params)
            final_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
            
            # Store model in trial
            trial.set_user_attr('model', final_model)
            
            return avg_score
        
        return objective
    
    def optimize(self, n_trials: int = 100, metric: str = 'mse') -> OptimizationResult:
        """
        Run optimization
        
        Args:
            n_trials: Number of trials
            metric: Evaluation metric
            
        Returns:
            Optimization results
        """
        optimizer = OptunaOptimizer(direction='minimize')
        objective = self.create_objective(metric)
        
        return optimizer.optimize(objective, n_trials)


class TransformerOptimizer:
    """Hyperparameter optimization for Transformer models"""
    
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 sequence_length: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Transformer optimizer
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            sequence_length: Sequence length for transformer
            device: Device to use
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sequence_length = sequence_length
        self.device = device
    
    def create_objective(self) -> Callable:
        """Create objective function for Optuna"""
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = {
                'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'd_ff': trial.suggest_categorical('d_ff', [256, 512, 1024]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            }
            
            # Import transformer model
            from ..models.transformer_predictor import TransformerPredictor
            
            # Create model
            model = TransformerPredictor(
                input_dim=self.X_train.shape[1],
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                n_layers=params['n_layers'],
                d_ff=params['d_ff'],
                dropout=params['dropout'],
                max_seq_length=self.sequence_length
            )
            
            # Train model
            try:
                model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=50,
                    batch_size=params['batch_size'],
                    learning_rate=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    early_stopping_patience=10,
                    verbose=False
                )
                
                # Evaluate
                y_pred = model.predict(self.X_val)
                score = mean_squared_error(self.y_val, y_pred)
                
                # Store model
                trial.set_user_attr('model', model)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                score = float('inf')
            
            return score
        
        return objective
    
    def optimize(self, n_trials: int = 50) -> OptimizationResult:
        """Run optimization"""
        optimizer = OptunaOptimizer(direction='minimize', n_jobs=1)  # Single job for GPU
        objective = self.create_objective()
        
        return optimizer.optimize(objective, n_trials)


class EnsembleOptimizer:
    """Hyperparameter optimization for ensemble models"""
    
    def __init__(self,
                 models: Dict[str, Any],
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray):
        """
        Initialize ensemble optimizer
        
        Args:
            models: Dictionary of base models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def create_objective(self) -> Callable:
        """Create objective function for ensemble weights"""
        def objective(trial: Trial) -> float:
            # Suggest weights for each model
            weights = {}
            remaining = 1.0
            
            model_names = sorted(self.models.keys())
            for i, name in enumerate(model_names[:-1]):
                weight = trial.suggest_float(f'weight_{name}', 0, remaining)
                weights[name] = weight
                remaining -= weight
            
            # Last weight is the remainder
            weights[model_names[-1]] = remaining
            
            # Import ensemble model
            from ..models.ensemble_model import EnsembleModel
            
            # Create ensemble
            ensemble = EnsembleModel(
                models=self.models,
                strategy='averaging',
                weights=weights
            )
            
            # Fit ensemble
            ensemble.fit(self.X_train, self.y_train)
            
            # Evaluate
            y_pred = ensemble.predict(self.X_val)
            score = mean_squared_error(self.y_val, y_pred)
            
            # Store ensemble
            trial.set_user_attr('model', ensemble)
            trial.set_user_attr('weights', weights)
            
            return score
        
        return objective
    
    def optimize(self, n_trials: int = 100) -> OptimizationResult:
        """Run optimization"""
        optimizer = OptunaOptimizer(direction='minimize')
        objective = self.create_objective()
        
        result = optimizer.optimize(objective, n_trials)
        
        # Add weights to result
        if result.study:
            result.best_params['weights'] = result.study.best_trial.user_attrs.get('weights', {})
        
        return result


class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Gaussian Processes"""
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]], 
                 random_state: int = 42):
        """
        Initialize Bayesian optimizer
        
        Args:
            bounds: Parameter bounds
            random_state: Random seed
        """
        self.bounds = bounds
        self.random_state = random_state
        
        # Note: This is a simplified implementation
        # For production, consider using skopt or other libraries
    
    def optimize(self, objective_func: Callable, n_trials: int = 100) -> OptimizationResult:
        """
        Run Bayesian optimization
        
        Args:
            objective_func: Objective function
            n_trials: Number of trials
            
        Returns:
            Optimization results
        """
        # For now, delegate to Optuna with TPE sampler (which is Bayesian)
        optuna_optimizer = OptunaOptimizer(sampler='tpe')
        return optuna_optimizer.optimize(objective_func, n_trials)


def visualize_optimization_history(result: OptimizationResult, save_path: Optional[str] = None):
    """
    Visualize optimization history
    
    Args:
        result: Optimization results
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Optimization history
    history = result.optimization_history
    axes[0, 0].plot(history['trial'], history['value'])
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].set_title('Optimization History')
    axes[0, 0].axhline(y=result.best_score, color='r', linestyle='--', label='Best')
    axes[0, 0].legend()
    
    # Parameter importance (if available)
    if result.study and hasattr(result.study, 'get_param_importances'):
        try:
            importances = optuna.importance.get_param_importances(result.study)
            params = list(importances.keys())
            values = list(importances.values())
            
            axes[0, 1].barh(params, values)
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Parameter Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance not available',
                           ha='center', va='center')
    
    # Parameter distributions
    param_cols = [col for col in history.columns if col not in ['value', 'trial']]
    if len(param_cols) > 0:
        param = param_cols[0]  # Show first parameter
        axes[1, 0].scatter(history[param], history['value'], alpha=0.5)
        axes[1, 0].set_xlabel(param)
        axes[1, 0].set_ylabel('Objective Value')
        axes[1, 0].set_title(f'{param} vs Objective')
    
    # Convergence plot
    best_values = []
    for i in range(len(history)):
        best_values.append(history['value'].iloc[:i+1].min())
    
    axes[1, 1].plot(history['trial'], best_values)
    axes[1, 1].set_xlabel('Trial')
    axes[1, 1].set_ylabel('Best Objective Value')
    axes[1, 1].set_title('Convergence Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Optimization visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * np.random.randn(n_samples)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Optimize XGBoost
    print("Optimizing XGBoost hyperparameters...")
    xgb_optimizer = XGBoostOptimizer(X_train, y_train, X_val, y_val)
    xgb_result = xgb_optimizer.optimize(n_trials=20)
    
    print(f"\nBest XGBoost parameters: {xgb_result.best_params}")
    print(f"Best validation score: {xgb_result.best_score:.4f}")
    
    # Test best model
    if xgb_result.best_model:
        y_pred = xgb_result.best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {test_mse:.4f}")
    
    # Visualize results
    visualize_optimization_history(xgb_result, 'optimization_history.png')
