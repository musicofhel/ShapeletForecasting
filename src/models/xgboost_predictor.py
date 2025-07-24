"""
XGBoost baseline model for financial time series prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
import json

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost model for time series prediction"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 gamma: float = 0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 objective: str = 'reg:squarederror',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 early_stopping_rounds: int = 10,
                 eval_metric: str = 'rmse'):
        """
        Initialize XGBoost predictor
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            objective: Learning objective
            random_state: Random seed
            n_jobs: Number of parallel threads
            early_stopping_rounds: Early stopping patience
            eval_metric: Evaluation metric
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': objective,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'tree_method': 'hist',  # Use histogram-based algorithm
            'predictor': 'cpu_predictor'
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.model = None
        self.feature_importance_ = None
        self.best_iteration_ = None
        
        # Remove eval_metric from params as it's not a valid XGBRegressor parameter
        # It will be handled separately during training
        self.eval_metric_value = eval_metric
        
        logger.info(f"Initialized XGBoost model with params: {self.params}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
            sample_weight: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'XGBoostPredictor':
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Validation sets for early stopping
            sample_weight: Sample weights
            verbose: Whether to print training progress
            
        Returns:
            Self
        """
        # Create XGBoost model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Prepare eval_set for XGBoost format
        if eval_set is not None:
            eval_set_xgb = [(X, y)] + eval_set
        else:
            eval_set_xgb = [(X, y)]
        
        # Train model
        # Set early_stopping_rounds as a parameter if eval_set is provided
        fit_params = {
            'eval_set': eval_set_xgb,
            'verbose': verbose
        }
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        # Only use early stopping if we have a validation set
        if eval_set is not None and self.early_stopping_rounds > 0:
            # For newer versions of XGBoost, early_stopping_rounds is set differently
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        
        self.model.fit(X, y, **fit_params)
        
        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_
        self.best_iteration_ = self.model.best_iteration
        
        logger.info(f"Training complete. Best iteration: {self.best_iteration_}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, ntree_limit: Optional[int] = None) -> np.ndarray:
        """
        Predict with all trees up to ntree_limit
        
        Args:
            X: Features to predict
            ntree_limit: Limit number of trees used
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X, ntree_limit=ntree_limit)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if importance_type == 'gain':
            importance = self.model.feature_importances_
        else:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            # Convert to array format
            importance = np.array([importance.get(f'f{i}', 0) for i in range(len(self.feature_importance_))])
        
        return {f'feature_{i}': score for i, score in enumerate(importance)}
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Save model
        self.model.save_model(filepath)
        
        # Save additional attributes
        metadata = {
            'params': self.params,
            'feature_importance': self.feature_importance_.tolist() if self.feature_importance_ is not None else None,
            'best_iteration': self.best_iteration_
        }
        
        metadata_path = filepath.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        # Load model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.json', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_importance_ = np.array(metadata.get('feature_importance'))
            self.best_iteration_ = metadata.get('best_iteration')
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")
        
        logger.info(f"Model loaded from {filepath}")


class XGBoostTimeSeriesCV:
    """Cross-validation for XGBoost with time series data"""
    
    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    def cross_validate(self, model: XGBoostPredictor, X: np.ndarray, y: np.ndarray,
                      scoring: Dict[str, callable] = None, return_train_score: bool = False) -> Dict[str, np.ndarray]:
        """
        Perform time series cross-validation
        
        Args:
            model: XGBoost model
            X: Features
            y: Targets
            scoring: Scoring functions
            return_train_score: Whether to return training scores
            
        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = {
                'mse': lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
                'mae': lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
                'r2': r2_score
            }
        
        results = {f'test_{name}': [] for name in scoring.keys()}
        if return_train_score:
            results.update({f'train_{name}': [] for name in scoring.keys()})
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model_fold = XGBoostPredictor(**model.params)
            model_fold.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            # Evaluate
            y_pred_test = model_fold.predict(X_test)
            
            for name, scorer in scoring.items():
                results[f'test_{name}'].append(scorer(y_test, y_pred_test))
            
            if return_train_score:
                y_pred_train = model_fold.predict(X_train)
                for name, scorer in scoring.items():
                    results[f'train_{name}'].append(scorer(y_train, y_pred_train))
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def grid_search(self, param_grid: Dict[str, List], X: np.ndarray, y: np.ndarray,
                   scoring: str = 'neg_mean_squared_error', n_jobs: int = -1) -> Dict:
        """
        Perform grid search with time series cross-validation
        
        Args:
            param_grid: Parameter grid
            X: Features
            y: Targets
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and results
        """
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1  # Set to 1 for each individual model
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2
        )
        
        grid_search.fit(X, y)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best score: {results['best_score']:.4f}")
        
        return results


class XGBoostFeatureEngineering:
    """Feature engineering utilities for XGBoost"""
    
    @staticmethod
    def create_lag_features(data: np.ndarray, lags: List[int]) -> np.ndarray:
        """
        Create lag features
        
        Args:
            data: Time series data
            lags: List of lag values
            
        Returns:
            Lag features
        """
        features = []
        
        for lag in lags:
            if lag > 0:
                lagged = np.roll(data, lag)
                lagged[:lag] = np.nan
                features.append(lagged)
        
        return np.column_stack(features)
    
    @staticmethod
    def create_rolling_features(data: np.ndarray, windows: List[int], 
                              functions: List[str] = ['mean', 'std', 'min', 'max']) -> np.ndarray:
        """
        Create rolling window features
        
        Args:
            data: Time series data
            windows: List of window sizes
            functions: List of functions to apply
            
        Returns:
            Rolling features
        """
        features = []
        df = pd.DataFrame(data, columns=['value'])
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    feature = df['value'].rolling(window).mean()
                elif func == 'std':
                    feature = df['value'].rolling(window).std()
                elif func == 'min':
                    feature = df['value'].rolling(window).min()
                elif func == 'max':
                    feature = df['value'].rolling(window).max()
                else:
                    raise ValueError(f"Unknown function: {func}")
                
                features.append(feature.values)
        
        return np.column_stack(features)
    
    @staticmethod
    def create_time_features(timestamps: np.ndarray) -> np.ndarray:
        """
        Create time-based features
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Time features
        """
        df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})
        
        features = pd.DataFrame({
            'hour': df['timestamp'].dt.hour,
            'day_of_week': df['timestamp'].dt.dayofweek,
            'day_of_month': df['timestamp'].dt.day,
            'month': df['timestamp'].dt.month,
            'quarter': df['timestamp'].dt.quarter,
            'is_weekend': (df['timestamp'].dt.dayofweek >= 5).astype(int),
            'is_month_start': df['timestamp'].dt.is_month_start.astype(int),
            'is_month_end': df['timestamp'].dt.is_month_end.astype(int)
        })
        
        return features.values


if __name__ == "__main__":
    # Example usage
    print("XGBoost predictor module loaded successfully")
    print("Use XGBoostPredictor class for time series prediction")
