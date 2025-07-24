"""
Model Comparison Framework for Wavelet Pattern Forecasting Dashboard

This module provides comprehensive model comparison capabilities including:
1. Compare different prediction methods
2. Ensemble model performance analysis
3. A/B testing framework
4. Model selection assistant
5. Performance reports generation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')


class ModelComparison:
    """Framework for comparing different pattern prediction models"""
    
    def __init__(self):
        """Initialize the model comparison framework"""
        self.models = {}
        self.results = {}
        self.ab_test_results = {}
        self.ensemble_weights = {}
        
    def add_model(self, model_name: str, model_instance: Any, 
                  model_type: str = 'classifier'):
        """
        Add a model to the comparison framework
        
        Args:
            model_name: Unique identifier for the model
            model_instance: The model instance with predict method
            model_type: Type of model ('classifier' or 'regressor')
        """
        self.models[model_name] = {
            'instance': model_instance,
            'type': model_type,
            'performance': {},
            'predictions': {}
        }
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray,
                      pattern_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare all models on the same test set
        
        Args:
            X_test: Test features
            y_test: True labels/values
            pattern_names: Optional pattern names for classification
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['instance']
            model_type = model_info['type']
            
            # Get predictions
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue
            
            # Calculate metrics based on model type
            if model_type == 'classifier':
                metrics = self._calculate_classification_metrics(
                    y_test, y_pred, y_pred_proba
                )
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Store results
            comparison_results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model_type': model_type
            }
            
            # Update model performance history
            self.models[model_name]['performance'] = metrics
            self.models[model_name]['predictions'] = {
                'y_pred': y_pred,
                'y_true': y_test,
                'timestamp': datetime.now()
            }
        
        self.results = comparison_results
        return comparison_results
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add per-class metrics if multi-class
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 2:
            metrics['per_class_precision'] = precision_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
            metrics['per_class_recall'] = recall_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
            metrics['per_class_f1'] = f1_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            # Calculate log loss
            from sklearn.metrics import log_loss
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # Calculate AUC for binary classification
            if len(unique_classes) == 2:
                from sklearn.metrics import roc_auc_score
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def create_ensemble_model(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create an ensemble model from the added models
        
        Args:
            weights: Optional weights for each model (defaults to equal weights)
            
        Returns:
            Ensemble performance metrics
        """
        if not self.results:
            raise ValueError("No model results available. Run compare_models first.")
        
        # Set default weights if not provided
        if weights is None:
            n_models = len(self.models)
            weights = {name: 1/n_models for name in self.models.keys()}
        
        self.ensemble_weights = weights
        
        # Combine predictions
        ensemble_predictions = self._combine_predictions(weights)
        
        # Calculate ensemble metrics
        y_true = list(self.models.values())[0]['predictions']['y_true']
        
        if list(self.models.values())[0]['type'] == 'classifier':
            ensemble_metrics = self._calculate_classification_metrics(
                y_true, ensemble_predictions, None
            )
        else:
            ensemble_metrics = self._calculate_regression_metrics(
                y_true, ensemble_predictions
            )
        
        ensemble_results = {
            'metrics': ensemble_metrics,
            'weights': weights,
            'predictions': ensemble_predictions
        }
        
        return ensemble_results
    
    def _combine_predictions(self, weights: Dict[str, float]) -> np.ndarray:
        """Combine predictions from multiple models"""
        model_type = list(self.models.values())[0]['type']
        
        if model_type == 'classifier':
            # For classification, use weighted voting
            weighted_votes = None
            
            for model_name, weight in weights.items():
                predictions = self.models[model_name]['predictions']['y_pred']
                
                if weighted_votes is None:
                    n_samples = len(predictions)
                    n_classes = len(np.unique(predictions))
                    weighted_votes = np.zeros((n_samples, n_classes))
                
                for i, pred in enumerate(predictions):
                    weighted_votes[i, int(pred)] += weight
            
            return np.argmax(weighted_votes, axis=1)
        
        else:
            # For regression, use weighted average
            weighted_sum = None
            
            for model_name, weight in weights.items():
                predictions = self.models[model_name]['predictions']['y_pred']
                
                if weighted_sum is None:
                    weighted_sum = weight * predictions
                else:
                    weighted_sum += weight * predictions
            
            return weighted_sum
    
    def run_ab_test(self, model_a: str, model_b: str, 
                    test_size: int = 1000,
                    confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Run A/B test between two models
        
        Args:
            model_a: Name of first model
            model_b: Name of second model
            test_size: Number of samples for testing
            confidence_level: Confidence level for statistical test
            
        Returns:
            A/B test results
        """
        if model_a not in self.models or model_b not in self.models:
            raise ValueError("Both models must be added to the framework")
        
        # Get predictions
        pred_a = self.models[model_a]['predictions']['y_pred']
        pred_b = self.models[model_b]['predictions']['y_pred']
        y_true = self.models[model_a]['predictions']['y_true']
        
        # Limit to test size
        if len(pred_a) > test_size:
            indices = np.random.choice(len(pred_a), test_size, replace=False)
            pred_a = pred_a[indices]
            pred_b = pred_b[indices]
            y_true = y_true[indices]
        
        # Calculate performance
        if self.models[model_a]['type'] == 'classifier':
            perf_a = (pred_a == y_true).astype(float)
            perf_b = (pred_b == y_true).astype(float)
        else:
            # For regression, use negative absolute error
            perf_a = -np.abs(pred_a - y_true)
            perf_b = -np.abs(pred_b - y_true)
        
        # Perform statistical test
        t_stat, p_value = stats.ttest_rel(perf_a, perf_b)
        
        # Calculate effect size (Cohen's d)
        diff = perf_a - perf_b
        effect_size = np.mean(diff) / np.std(diff)
        
        # Determine winner
        if p_value < (1 - confidence_level):
            if np.mean(perf_a) > np.mean(perf_b):
                winner = model_a
                confidence = f"{confidence_level*100:.0f}% confident"
            else:
                winner = model_b
                confidence = f"{confidence_level*100:.0f}% confident"
        else:
            winner = "No significant difference"
            confidence = f"p-value: {p_value:.4f}"
        
        ab_results = {
            'model_a': {
                'name': model_a,
                'mean_performance': np.mean(perf_a),
                'std_performance': np.std(perf_a)
            },
            'model_b': {
                'name': model_b,
                'mean_performance': np.mean(perf_b),
                'std_performance': np.std(perf_b)
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size
            },
            'conclusion': {
                'winner': winner,
                'confidence': confidence,
                'recommendation': self._generate_ab_recommendation(winner, effect_size)
            }
        }
        
        self.ab_test_results[f"{model_a}_vs_{model_b}"] = ab_results
        return ab_results
    
    def _generate_ab_recommendation(self, winner: str, effect_size: float) -> str:
        """Generate recommendation based on A/B test results"""
        if winner == "No significant difference":
            return "Continue testing with more data or consider both models equivalent"
        
        if abs(effect_size) < 0.2:
            return f"Small effect size. {winner} is slightly better but difference may not be practical"
        elif abs(effect_size) < 0.5:
            return f"Medium effect size. {winner} shows meaningful improvement"
        else:
            return f"Large effect size. {winner} shows substantial improvement and should be preferred"
    
    def create_model_selection_assistant(self) -> Dict[str, Any]:
        """
        Create an intelligent model selection assistant
        
        Returns:
            Model selection recommendations
        """
        if not self.results:
            raise ValueError("No model results available. Run compare_models first.")
        
        recommendations = {
            'best_overall': None,
            'best_by_metric': {},
            'ensemble_recommendation': None,
            'use_cases': {},
            'warnings': []
        }
        
        # Find best overall model
        model_scores = {}
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            if results['model_type'] == 'classifier':
                # Use F1 score as primary metric
                model_scores[model_name] = metrics['f1_score']
            else:
                # Use negative RMSE for regression (higher is better)
                model_scores[model_name] = -metrics['rmse']
        
        best_model = max(model_scores, key=model_scores.get)
        recommendations['best_overall'] = {
            'model': best_model,
            'score': model_scores[best_model],
            'reason': "Highest overall performance score"
        }
        
        # Find best model for each metric
        all_metrics = list(self.results[best_model]['metrics'].keys())
        for metric in all_metrics:
            if isinstance(self.results[best_model]['metrics'][metric], (int, float)):
                if metric in ['mse', 'rmse', 'mae', 'log_loss']:
                    # Lower is better
                    best_for_metric = min(
                        self.results.keys(),
                        key=lambda x: self.results[x]['metrics'].get(metric, float('inf'))
                    )
                else:
                    # Higher is better
                    best_for_metric = max(
                        self.results.keys(),
                        key=lambda x: self.results[x]['metrics'].get(metric, -float('inf'))
                    )
                
                recommendations['best_by_metric'][metric] = best_for_metric
        
        # Ensemble recommendation
        if len(self.models) > 2:
            recommendations['ensemble_recommendation'] = {
                'should_ensemble': True,
                'reason': "Multiple models available - ensemble likely to improve performance",
                'suggested_weights': self._calculate_optimal_weights()
            }
        else:
            recommendations['ensemble_recommendation'] = {
                'should_ensemble': False,
                'reason': "Too few models for meaningful ensemble"
            }
        
        # Use case recommendations
        recommendations['use_cases'] = {
            'high_accuracy_needed': recommendations['best_by_metric'].get('accuracy', best_model),
            'low_false_positives': recommendations['best_by_metric'].get('precision', best_model),
            'low_false_negatives': recommendations['best_by_metric'].get('recall', best_model),
            'balanced_performance': recommendations['best_by_metric'].get('f1_score', best_model)
        }
        
        # Add warnings
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            # Check for poor performance
            if results['model_type'] == 'classifier':
                if metrics['accuracy'] < 0.6:
                    recommendations['warnings'].append(
                        f"{model_name} has low accuracy ({metrics['accuracy']:.2f})"
                    )
                if metrics['f1_score'] < 0.5:
                    recommendations['warnings'].append(
                        f"{model_name} has poor F1 score ({metrics['f1_score']:.2f})"
                    )
        
        return recommendations
    
    def _calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on model performance"""
        weights = {}
        total_score = 0
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            # Use F1 score for classification, 1/RMSE for regression
            if results['model_type'] == 'classifier':
                score = metrics['f1_score']
            else:
                score = 1 / (metrics['rmse'] + 1e-6)
            
            weights[model_name] = score
            total_score += score
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_score
        
        return weights
    
    def create_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(self.models.keys()),
            'model_results': self.results,
            'ab_tests': self.ab_test_results,
            'recommendations': self.create_model_selection_assistant(),
            'visualizations': {
                'comparison_chart': self._create_comparison_chart_data(),
                'confusion_matrices': self._create_confusion_matrices_data(),
                'performance_radar': self._create_radar_chart_data()
            }
        }
        
        return report
    
    def _create_comparison_chart_data(self) -> Dict[str, Any]:
        """Create data for comparison bar chart"""
        models = []
        metrics_data = {}
        
        for model_name, results in self.results.items():
            models.append(model_name)
            
            for metric, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
        
        return {
            'models': models,
            'metrics': metrics_data
        }
    
    def _create_confusion_matrices_data(self) -> Dict[str, Any]:
        """Create confusion matrix data for classification models"""
        confusion_data = {}
        
        for model_name, model_info in self.models.items():
            if model_info['type'] == 'classifier' and 'predictions' in model_info:
                y_true = model_info['predictions']['y_true']
                y_pred = model_info['predictions']['y_pred']
                
                # Calculate confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                
                confusion_data[model_name] = cm.tolist()
        
        return confusion_data
    
    def _create_radar_chart_data(self) -> Dict[str, Any]:
        """Create data for radar chart visualization"""
        radar_data = {
            'categories': [],
            'models': {}
        }
        
        # Get common metrics
        if self.results:
            first_model = list(self.results.values())[0]
            for metric in first_model['metrics']:
                if isinstance(first_model['metrics'][metric], (int, float)):
                    radar_data['categories'].append(metric)
        
        # Get values for each model
        for model_name, results in self.results.items():
            values = []
            for metric in radar_data['categories']:
                value = results['metrics'].get(metric, 0)
                
                # Normalize to 0-1 scale
                if metric in ['mse', 'rmse', 'mae', 'log_loss']:
                    # Lower is better - invert
                    value = 1 / (1 + value)
                
                values.append(value)
            
            radar_data['models'][model_name] = values
        
        return radar_data
    
    def create_comparison_visualizations(self) -> Dict[str, go.Figure]:
        """
        Create all comparison visualizations
        
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        # 1. Performance comparison bar chart
        figures['performance_bars'] = self._create_performance_bar_chart()
        
        # 2. Confusion matrices (for classifiers)
        figures['confusion_matrices'] = self._create_confusion_matrix_plot()
        
        # 3. Radar chart
        figures['radar_chart'] = self._create_radar_chart()
        
        # 4. A/B test results
        if self.ab_test_results:
            figures['ab_tests'] = self._create_ab_test_plot()
        
        # 5. Time series of predictions
        figures['prediction_timeline'] = self._create_prediction_timeline()
        
        return figures
    
    def _create_performance_bar_chart(self) -> go.Figure:
        """Create performance comparison bar chart"""
        data = self._create_comparison_chart_data()
        
        fig = go.Figure()
        
        for metric, values in data['metrics'].items():
            fig.add_trace(go.Bar(
                name=metric,
                x=data['models'],
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Metric Value',
            barmode='group',
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    def _create_confusion_matrix_plot(self) -> go.Figure:
        """Create confusion matrix subplots"""
        cm_data = self._create_confusion_matrices_data()
        
        if not cm_data:
            return go.Figure().add_annotation(
                text="No classification models to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        n_models = len(cm_data)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(cm_data.keys()),
            horizontal_spacing=0.1
        )
        
        for i, (model_name, cm) in enumerate(cm_data.items(), 1):
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    colorscale='Blues',
                    showscale=(i == 1),
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12}
                ),
                row=1, col=i
            )
            
            fig.update_xaxes(title_text="Predicted", row=1, col=i)
            fig.update_yaxes(title_text="Actual", row=1, col=i)
        
        fig.update_layout(
            title='Confusion Matrices',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def _create_radar_chart(self) -> go.Figure:
        """Create radar chart for multi-metric comparison"""
        radar_data = self._create_radar_chart_data()
        
        fig = go.Figure()
        
        for model_name, values in radar_data['models'].items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_data['categories'],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart",
            template='plotly_dark'
        )
        
        return fig
    
    def _create_ab_test_plot(self) -> go.Figure:
        """Create A/B test results visualization"""
        fig = make_subplots(
            rows=len(self.ab_test_results), cols=2,
            subplot_titles=[f"{test} - Performance", f"{test} - Statistical Test" 
                          for test in self.ab_test_results.keys()],
            vertical_spacing=0.1
        )
        
        for i, (test_name, results) in enumerate(self.ab_test_results.items(), 1):
            # Performance comparison
            models = [results['model_a']['name'], results['model_b']['name']]
            means = [results['model_a']['mean_performance'], 
                    results['model_b']['mean_performance']]
            stds = [results['model_a']['std_performance'], 
                   results['model_b']['std_performance']]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    name='Performance',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Statistical significance
            p_value = results['statistical_test']['p_value']
            effect_size = results['statistical_test']['effect_size']
            
            fig.add_trace(
                go.Scatter(
                    x=['p-value', 'effect size'],
                    y=[p_value, abs(effect_size)],
                    mode='markers+text',
                    marker=dict(size=20),
                    text=[f'{p_value:.4f}', f'{effect_size:.3f}'],
                    textposition='top center',
                    name='Statistics',
                    showlegend=(i == 1)
                ),
                row=i, col=2
            )
            
            # Add significance threshold
            fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                         annotation_text="α=0.05", row=i, col=2)
        
        fig.update_layout(
            title='A/B Test Results',
            height=300 * len(self.ab_test_results),
            template='plotly_dark'
        )
        
        return fig
    
    def _create_prediction_timeline(self) -> go.Figure:
        """Create timeline of model predictions"""
        fig = go.Figure()
        
        # Get a sample of predictions to visualize
        sample_size = min(100, len(list(self.models.values())[0]['predictions']['y_true']))
        x = list(range(sample_size))
        
        # Add true values
        y_true = list(self.models.values())[0]['predictions']['y_true'][:sample_size]
        fig.add_trace(go.Scatter(
            x=x,
            y=y_true,
            mode='lines',
            name='True Values',
            line=dict(color='white', width=2)
        ))
        
        # Add predictions for each model
        colors = px.colors.qualitative.Set3
        for i, (model_name, model_info) in enumerate(self.models.items()):
            y_pred = model_info['predictions']['y_pred'][:sample_size]
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y_pred,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.7
            ))
        
        fig.update_layout(
            title='Model Predictions Timeline',
            xaxis_title='Sample Index',
            yaxis_title='Predicted Value',
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create comparison framework
    comparison = ModelComparison()
    
    # Example: Add dummy models for testing
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 3, 1000)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randint(0, 3, 200)
    
    # Create and train models
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    # Add models to comparison
    comparison.add_model('dummy', dummy, 'classifier')
    comparison.add_model('random_forest', rf, 'classifier')
    comparison.add_model('logistic_regression', lr, 'classifier')
    
    # Compare models
    results = comparison.compare_models(X_test, y_test)
    
    # Create ensemble
    ensemble_results = comparison.create_ensemble_model()
    
    # Run A/B test
    ab_results = comparison.run_ab_test('random_forest', 'logistic_regression')
    
    # Get recommendations
    recommendations = comparison.create_model_selection_assistant()
    
    # Generate report
    report = comparison.create_performance_report()
    
    print("Model Comparison Report:")
    print(json.dumps(report['recommendations'], indent=2))
    
    # Create visualizations
    figures = comparison.create_comparison_visualizations()
    for name, fig in figures.items():
        fig.show()
