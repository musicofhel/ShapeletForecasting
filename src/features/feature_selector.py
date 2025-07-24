"""
Feature Selection and Importance Analysis

This module provides methods for selecting the most relevant features
and analyzing feature importance for financial prediction tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection and importance analysis for financial features.
    
    Methods include:
    - Statistical tests (mutual information, ANOVA F-test)
    - Model-based selection (Lasso, Random Forest)
    - Recursive feature elimination
    - Correlation-based filtering
    - SHAP value analysis
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 n_features: Optional[int] = None,
                 selection_method: str = 'mutual_info',
                 correlation_threshold: float = 0.95,
                 importance_threshold: float = 0.01):
        """
        Initialize feature selector.
        
        Parameters:
        -----------
        task_type : str
            Type of task ('classification' or 'regression')
        n_features : int, optional
            Number of features to select (if None, use importance_threshold)
        selection_method : str
            Feature selection method ('mutual_info', 'f_test', 'lasso', 'rf', 'rfe')
        correlation_threshold : float
            Threshold for removing highly correlated features
        importance_threshold : float
            Minimum importance score to keep feature
        """
        self.task_type = task_type
        self.n_features = n_features
        self.selection_method = selection_method
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        
        # Storage for results
        self.feature_scores_ = None
        self.selected_features_ = None
        self.feature_ranking_ = None
        self.correlation_matrix_ = None
        self.removed_correlated_ = None
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'FeatureSelector':
        """
        Fit feature selector on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target variable
            
        Returns:
        --------
        self : FeatureSelector
        """
        logger.info(f"Fitting feature selector with {self.selection_method} method...")
        
        # Remove highly correlated features first
        X_filtered, removed_features = self._remove_correlated_features(X)
        self.removed_correlated_ = removed_features
        
        # Apply feature selection method
        if self.selection_method == 'mutual_info':
            scores = self._mutual_info_selection(X_filtered, y)
        elif self.selection_method == 'f_test':
            scores = self._f_test_selection(X_filtered, y)
        elif self.selection_method == 'lasso':
            scores = self._lasso_selection(X_filtered, y)
        elif self.selection_method == 'rf':
            scores = self._random_forest_selection(X_filtered, y)
        elif self.selection_method == 'rfe':
            scores = self._rfe_selection(X_filtered, y)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
            
        # Store scores
        self.feature_scores_ = pd.Series(scores, index=X_filtered.columns)
        
        # Select features
        if self.n_features is not None:
            # Select top n features
            self.selected_features_ = self.feature_scores_.nlargest(self.n_features).index.tolist()
        else:
            # Select features above threshold
            self.selected_features_ = self.feature_scores_[
                self.feature_scores_ >= self.importance_threshold
            ].index.tolist()
            
        # Create ranking
        self.feature_ranking_ = self.feature_scores_.rank(ascending=False, method='min')
        
        logger.info(f"Selected {len(self.selected_features_)} features from "
                   f"{len(X.columns)} original features")
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting important features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to transform
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Transformed feature matrix with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector must be fitted before transform")
            
        # Select only the important features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        return X[available_features]
        
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
        
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        self.correlation_matrix_ = corr_matrix
        
        # Find features to remove
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_remove = []
        for column in upper_tri.columns:
            if column in to_remove:
                continue
            correlated = upper_tri[column][upper_tri[column] > self.correlation_threshold].index.tolist()
            to_remove.extend(correlated)
            
        # Remove duplicates
        to_remove = list(set(to_remove))
        
        # Keep features not in removal list
        to_keep = [col for col in X.columns if col not in to_remove]
        
        logger.info(f"Removed {len(to_remove)} highly correlated features")
        
        return X[to_keep], to_remove
        
    def _mutual_info_selection(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Select features using mutual information."""
        if self.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
        # Normalize scores
        mi_scores = mi_scores / mi_scores.max()
        
        return mi_scores
        
    def _f_test_selection(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Select features using F-test."""
        if self.task_type == 'classification':
            f_scores, _ = f_classif(X, y)
        else:
            f_scores, _ = f_regression(X, y)
            
        # Handle infinite values
        f_scores = np.nan_to_num(f_scores, nan=0, posinf=0)
        
        # Normalize scores
        if f_scores.max() > 0:
            f_scores = f_scores / f_scores.max()
            
        return f_scores
        
    def _lasso_selection(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Select features using Lasso regularization."""
        if self.task_type == 'classification':
            # Use Lasso with logistic regression
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(penalty='l1', solver='liblinear', 
                                      C=1.0, random_state=42)
            model.fit(X, y)
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            # Use LassoCV for regression
            model = LassoCV(cv=5, random_state=42, max_iter=5000)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            
        # Normalize scores
        if importance.max() > 0:
            importance = importance / importance.max()
            
        return importance
        
    def _random_forest_selection(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Select features using Random Forest importance."""
        if self.task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            
        model.fit(X, y)
        importance = model.feature_importances_
        
        # Normalize scores
        importance = importance / importance.max()
        
        return importance
        
    def _rfe_selection(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Select features using Recursive Feature Elimination."""
        # Use Random Forest as base estimator
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1
            )
            
        # Determine number of features to select
        n_features_to_select = self.n_features or max(10, int(0.5 * X.shape[1]))
        
        # Apply RFE
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        # Convert ranking to importance scores
        max_rank = rfe.ranking_.max()
        importance = (max_rank - rfe.ranking_ + 1) / max_rank
        
        return importance
        
    def compute_shap_values(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                           model: Optional[Any] = None) -> pd.DataFrame:
        """
        Compute SHAP values for feature importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target variable
        model : sklearn model, optional
            Model to use for SHAP analysis (if None, use Random Forest)
            
        Returns:
        --------
        shap_importance : pd.DataFrame
            SHAP-based feature importance
        """
        # Use provided model or create default
        if model is None:
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            model.fit(X, y)
            
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class classification
        if isinstance(shap_values, list):
            # Average across classes
            shap_values = np.abs(shap_values).mean(axis=0)
        else:
            shap_values = np.abs(shap_values)
            
        # Create importance dataframe
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': shap_values.mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        return shap_importance
        
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance scores."""
        if self.feature_scores_ is None:
            raise ValueError("FeatureSelector must be fitted before plotting")
            
        # Get top features
        top_features = self.feature_scores_.nlargest(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features.values)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance Scores ({self.selection_method})')
        
        # Add value labels
        for i, v in enumerate(top_features.values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
        plt.tight_layout()
        return fig
        
    def plot_correlation_matrix(self, selected_only: bool = True, 
                              figsize: Tuple[int, int] = (12, 10)):
        """Plot correlation matrix of features."""
        if self.correlation_matrix_ is None:
            raise ValueError("No correlation matrix available. Run fit() first.")
            
        # Select features to plot
        if selected_only and self.selected_features_:
            features = self.selected_features_
            corr_matrix = self.correlation_matrix_.loc[features, features]
        else:
            corr_matrix = self.correlation_matrix_
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=False, fmt='.2f', ax=ax)
        
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        return fig
        
    def get_feature_report(self) -> pd.DataFrame:
        """Generate comprehensive feature importance report."""
        if self.feature_scores_ is None:
            raise ValueError("FeatureSelector must be fitted first")
            
        report = pd.DataFrame({
            'feature': self.feature_scores_.index,
            'importance_score': self.feature_scores_.values,
            'rank': self.feature_ranking_.values,
            'selected': self.feature_scores_.index.isin(self.selected_features_)
        })
        
        # Add correlation info if available
        if self.correlation_matrix_ is not None:
            max_corr = []
            for feature in report['feature']:
                if feature in self.correlation_matrix_.columns:
                    corr_values = self.correlation_matrix_[feature].drop(feature)
                    max_corr.append(corr_values.abs().max())
                else:
                    max_corr.append(np.nan)
            report['max_correlation'] = max_corr
            
        # Sort by importance
        report = report.sort_values('importance_score', ascending=False)
        
        return report
        
    def save(self, filepath: str):
        """Save feature selector to file."""
        data = {
            'config': {
                'task_type': self.task_type,
                'n_features': self.n_features,
                'selection_method': self.selection_method,
                'correlation_threshold': self.correlation_threshold,
                'importance_threshold': self.importance_threshold
            },
            'feature_scores': self.feature_scores_,
            'selected_features': self.selected_features_,
            'feature_ranking': self.feature_ranking_,
            'correlation_matrix': self.correlation_matrix_,
            'removed_correlated': self.removed_correlated_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved feature selector to {filepath}")
        
    def load(self, filepath: str):
        """Load feature selector from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Restore configuration
        config = data['config']
        self.task_type = config['task_type']
        self.n_features = config['n_features']
        self.selection_method = config['selection_method']
        self.correlation_threshold = config['correlation_threshold']
        self.importance_threshold = config['importance_threshold']
        
        # Restore results
        self.feature_scores_ = data['feature_scores']
        self.selected_features_ = data['selected_features']
        self.feature_ranking_ = data['feature_ranking']
        self.correlation_matrix_ = data['correlation_matrix']
        self.removed_correlated_ = data['removed_correlated']
        
        logger.info(f"Loaded feature selector from {filepath}")
