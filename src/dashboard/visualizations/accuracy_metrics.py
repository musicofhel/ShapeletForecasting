"""
Accuracy Metrics Dashboard for Pattern Prediction System.

This module provides comprehensive accuracy metrics and visualizations including:
- Prediction accuracy over time
- Accuracy by pattern type
- Confidence calibration plots
- Error distribution analysis
- Model performance comparison
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class AccuracyMetricsDashboard:
    """Creates comprehensive accuracy metrics visualizations."""
    
    def __init__(self):
        """Initialize accuracy metrics dashboard."""
        # Pattern type colors (consistent with forecast_view)
        self.pattern_colors = {
            'Head and Shoulders': '#FF6B6B',
            'Double Top': '#4ECDC4',
            'Double Bottom': '#45B7D1',
            'Ascending Triangle': '#96CEB4',
            'Descending Triangle': '#FECA57',
            'Bull Flag': '#48C9B0',
            'Bear Flag': '#F97F51',
            'Cup and Handle': '#B983FF',
            'Inverse Head and Shoulders': '#FD79A8',
            'Unknown': '#95A5A6'
        }
        
        # Model colors for comparison
        self.model_colors = {
            'LSTM': '#3498db',
            'Transformer': '#e74c3c',
            'Markov': '#2ecc71',
            'Ensemble': '#f39c12',
            'Baseline': '#95a5a6'
        }
    
    def create_accuracy_over_time(self,
                                accuracy_data: pd.DataFrame,
                                window_sizes: List[int] = [7, 30, 90]) -> go.Figure:
        """
        Create visualization of prediction accuracy over time.
        
        Args:
            accuracy_data: DataFrame with columns ['date', 'accuracy', 'model']
            window_sizes: List of rolling window sizes for smoothing
            
        Returns:
            Plotly figure showing accuracy trends
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Prediction Accuracy Over Time', 'Daily Accuracy Distribution'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Check if data is empty
        if accuracy_data.empty or 'model' not in accuracy_data.columns:
            # Return empty figure with message
            fig.add_annotation(
                text="No accuracy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title='Model Accuracy Performance Over Time',
                height=800,
                template='plotly_white'
            )
            return fig
        
        # Plot raw accuracy and rolling averages
        for model in accuracy_data['model'].unique():
            model_data = accuracy_data[accuracy_data['model'] == model]
            
            # Raw daily accuracy
            fig.add_trace(
                go.Scatter(
                    x=model_data['date'],
                    y=model_data['accuracy'],
                    mode='markers',
                    name=f'{model} (daily)',
                    marker=dict(
                        size=4,
                        color=self.model_colors.get(model, '#95a5a6'),
                        opacity=0.3
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Rolling averages
            for window in window_sizes:
                rolling_acc = model_data.set_index('date')['accuracy'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_acc.index,
                        y=rolling_acc.values,
                        mode='lines',
                        name=f'{model} ({window}d avg)',
                        line=dict(
                            color=self.model_colors.get(model, '#95a5a6'),
                            width=2 if window == 30 else 1,
                            dash='solid' if window == 30 else 'dot'
                        )
                    ),
                    row=1, col=1
                )
        
        # Add target accuracy line
        fig.add_hline(
            y=0.7, line_dash="dash", line_color="red",
            annotation_text="Target: 70%",
            row=1, col=1
        )
        
        # Accuracy distribution histogram
        for model in accuracy_data['model'].unique():
            model_data = accuracy_data[accuracy_data['model'] == model]
            
            fig.add_trace(
                go.Histogram(
                    x=model_data['accuracy'],
                    name=model,
                    opacity=0.7,
                    nbinsx=20,
                    histnorm='probability',
                    marker_color=self.model_colors.get(model, '#95a5a6')
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Model Accuracy Performance Over Time',
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            barmode='overlay'
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Accuracy', row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text='Accuracy', row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        
        return fig
    
    def create_pattern_type_accuracy(self,
                                   pattern_accuracy: pd.DataFrame) -> go.Figure:
        """
        Create accuracy breakdown by pattern type.
        
        Args:
            pattern_accuracy: DataFrame with pattern types and accuracy metrics
            
        Returns:
            Plotly figure showing pattern-specific accuracy
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accuracy by Pattern Type',
                'Precision vs Recall',
                'Pattern Confusion Matrix',
                'F1 Score Comparison'
            ),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Check if data is empty
        if pattern_accuracy.empty:
            fig.add_annotation(
                text="No pattern accuracy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title='Pattern-Specific Accuracy Analysis',
                height=900,
                template='plotly_white'
            )
            return fig
        
        # Sort patterns by accuracy
        pattern_accuracy = pattern_accuracy.sort_values('accuracy', ascending=True)
        
        # 1. Accuracy by pattern type
        fig.add_trace(
            go.Bar(
                y=pattern_accuracy['pattern_type'],
                x=pattern_accuracy['accuracy'],
                orientation='h',
                marker_color=[self.pattern_colors.get(p, '#95a5a6') 
                            for p in pattern_accuracy['pattern_type']],
                text=[f'{acc:.1%}' for acc in pattern_accuracy['accuracy']],
                textposition='auto',
                name='Accuracy'
            ),
            row=1, col=1
        )
        
        # Add target line
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(
                x=pattern_accuracy['precision'],
                y=pattern_accuracy['recall'],
                mode='markers+text',
                marker=dict(
                    size=pattern_accuracy['support'] / pattern_accuracy['support'].max() * 50,
                    color=[self.pattern_colors.get(p, '#95a5a6') 
                          for p in pattern_accuracy['pattern_type']],
                    showscale=False
                ),
                text=pattern_accuracy['pattern_type'],
                textposition='top center',
                name='Patterns'
            ),
            row=1, col=2
        )
        
        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Confusion matrix
        if 'confusion_matrix' in pattern_accuracy.columns:
            # Aggregate confusion matrix
            pattern_types = pattern_accuracy['pattern_type'].tolist()
            cm = pattern_accuracy['confusion_matrix'].values
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=pattern_types,
                    y=pattern_types,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    showscale=True
                ),
                row=2, col=1
            )
        
        # 4. F1 Score comparison
        fig.add_trace(
            go.Bar(
                x=pattern_accuracy['pattern_type'],
                y=pattern_accuracy['f1_score'],
                marker_color=[self.pattern_colors.get(p, '#95a5a6') 
                            for p in pattern_accuracy['pattern_type']],
                text=[f'{f1:.2f}' for f1 in pattern_accuracy['f1_score']],
                textposition='auto',
                name='F1 Score'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Pattern-Specific Accuracy Analysis',
            height=900,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Accuracy', row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text='Precision', row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Recall', row=1, col=2, range=[0, 1])
        fig.update_xaxes(title_text='Predicted', row=2, col=1)
        fig.update_yaxes(title_text='Actual', row=2, col=1)
        fig.update_yaxes(title_text='F1 Score', row=2, col=2, range=[0, 1])
        
        return fig
    
    def create_confidence_calibration(self,
                                    predictions: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive confidence calibration plots.
        
        Args:
            predictions: DataFrame with predictions, probabilities, and outcomes
            
        Returns:
            Plotly figure showing calibration analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Calibration Plot',
                'Reliability Diagram',
                'Confidence Distribution',
                'Calibration by Model'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Main calibration plot
        # Bin predictions
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        predictions['confidence_bin'] = pd.cut(
            predictions['probability'],
            bins=bin_edges,
            labels=bin_centers
        )
        
        calibration = predictions.groupby('confidence_bin').agg({
            'correct': ['mean', 'count'],
            'probability': 'mean'
        }).reset_index()
        
        calibration.columns = ['bin_center', 'actual_prob', 'count', 'mean_prob']
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )
        
        # Actual calibration with confidence intervals
        fig.add_trace(
            go.Scatter(
                x=calibration['mean_prob'],
                y=calibration['actual_prob'],
                mode='lines+markers',
                name='Actual',
                marker=dict(
                    size=np.sqrt(calibration['count']) * 2,
                    color='blue'
                ),
                line=dict(width=3),
                error_y=dict(
                    type='data',
                    array=1.96 * np.sqrt(
                        calibration['actual_prob'] * (1 - calibration['actual_prob']) 
                        / calibration['count']
                    ),
                    visible=True
                )
            ),
            row=1, col=1
        )
        
        # 2. Reliability diagram
        gap = calibration['actual_prob'] - calibration['mean_prob']
        colors = ['red' if g < -0.1 else 'green' if g > 0.1 else 'yellow' for g in gap]
        
        fig.add_trace(
            go.Bar(
                x=calibration['bin_center'],
                y=calibration['count'],
                marker_color=colors,
                name='Bin Counts',
                text=[f'{c}' for c in calibration['count']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=predictions['probability'],
                nbinsx=30,
                name='All Predictions',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Overlay correct vs incorrect distributions
        fig.add_trace(
            go.Histogram(
                x=predictions[predictions['correct'] == True]['probability'],
                nbinsx=30,
                name='Correct',
                marker_color='green',
                opacity=0.5
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=predictions[predictions['correct'] == False]['probability'],
                nbinsx=30,
                name='Incorrect',
                marker_color='red',
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # 4. Calibration by model
        if 'model' in predictions.columns:
            for model in predictions['model'].unique():
                model_data = predictions[predictions['model'] == model]
                model_cal = model_data.groupby('confidence_bin').agg({
                    'correct': 'mean',
                    'probability': 'mean'
                }).reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=model_cal['probability'],
                        y=model_cal['correct'],
                        mode='lines+markers',
                        name=model,
                        line=dict(
                            color=self.model_colors.get(model, '#95a5a6'),
                            width=2
                        )
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Prediction Confidence Calibration Analysis',
            height=900,
            showlegend=True,
            template='plotly_white',
            barmode='overlay'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Mean Predicted Probability', row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text='Actual Probability', row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text='Confidence Bin', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_xaxes(title_text='Confidence', row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_xaxes(title_text='Mean Predicted', row=2, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Actual', row=2, col=2, range=[0, 1])
        
        return fig
    
    def create_error_distribution(self,
                                errors: pd.DataFrame) -> go.Figure:
        """
        Create error distribution analysis.
        
        Args:
            errors: DataFrame with error metrics
            
        Returns:
            Plotly figure showing error analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Error Distribution',
                'Error by Time of Day',
                'Error Autocorrelation',
                'Error vs Confidence'
            ),
            specs=[[{'type': 'histogram'}, {'type': 'box'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Error distribution with normal overlay
        fig.add_trace(
            go.Histogram(
                x=errors['error'],
                nbinsx=50,
                name='Error Distribution',
                histnorm='probability density',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(errors['error'])
        x_range = np.linspace(errors['error'].min(), errors['error'].max(), 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, mu, sigma),
                mode='lines',
                name=f'Normal(μ={mu:.2f}, σ={sigma:.2f})',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Error by time of day
        if 'hour' in errors.columns:
            fig.add_trace(
                go.Box(
                    x=errors['hour'],
                    y=errors['error'],
                    name='Hourly Errors',
                    boxpoints='outliers',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # 3. Error autocorrelation
        if len(errors) > 20:
            lags = range(1, min(21, len(errors) // 2))
            autocorr = [errors['error'].autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(
                go.Bar(
                    x=list(lags),
                    y=autocorr,
                    name='Autocorrelation',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            # Add significance bounds
            n = len(errors)
            sig_level = 1.96 / np.sqrt(n)
            fig.add_hline(y=sig_level, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-sig_level, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Error vs Confidence
        if 'confidence' in errors.columns:
            fig.add_trace(
                go.Scatter(
                    x=errors['confidence'],
                    y=np.abs(errors['error']),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=errors['error'],
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title='Error')
                    ),
                    name='Error vs Confidence'
                ),
                row=2, col=2
            )
            
            # Add trend line
            z = np.polyfit(errors['confidence'], np.abs(errors['error']), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Prediction Error Analysis',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Error', row=1, col=1)
        fig.update_yaxes(title_text='Density', row=1, col=1)
        fig.update_xaxes(title_text='Hour of Day', row=1, col=2)
        fig.update_yaxes(title_text='Error', row=1, col=2)
        fig.update_xaxes(title_text='Lag', row=2, col=1)
        fig.update_yaxes(title_text='Autocorrelation', row=2, col=1)
        fig.update_xaxes(title_text='Confidence', row=2, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Absolute Error', row=2, col=2)
        
        return fig
    
    def create_model_comparison(self,
                              model_metrics: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive model performance comparison.
        
        Args:
            model_metrics: DataFrame with model performance metrics
            
        Returns:
            Plotly figure comparing models
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Overall Model Performance',
                'Performance Radar Chart',
                'Model Learning Curves',
                'Computational Efficiency'
            ),
            specs=[[{'type': 'bar'}, {'type': 'scatterpolar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Overall performance bars
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            if metric in model_metrics.columns:
                fig.add_trace(
                    go.Bar(
                        x=model_metrics['model'],
                        y=model_metrics[metric],
                        name=metric.replace('_', ' ').title(),
                        text=[f'{v:.2f}' for v in model_metrics[metric]],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
        
        # 2. Radar chart
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Speed']
        
        for _, model_data in model_metrics.iterrows():
            model_name = model_data['model']
            
            values = [
                model_data.get('accuracy', 0),
                model_data.get('precision', 0),
                model_data.get('recall', 0),
                model_data.get('f1_score', 0),
                1 - model_data.get('inference_time', 0) / model_metrics['inference_time'].max()
            ]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model_name,
                    line=dict(
                        color=self.model_colors.get(model_name, '#95a5a6'),
                        width=2
                    ),
                    opacity=0.6
                ),
                row=1, col=2
            )
        
        # 3. Learning curves
        if 'learning_history' in model_metrics.columns:
            for _, model_data in model_metrics.iterrows():
                model_name = model_data['model']
                history = model_data['learning_history']
                
                if isinstance(history, dict):
                    epochs = history.get('epochs', [])
                    val_acc = history.get('val_accuracy', [])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=val_acc,
                            mode='lines+markers',
                            name=model_name,
                            line=dict(
                                color=self.model_colors.get(model_name, '#95a5a6'),
                                width=2
                            )
                        ),
                        row=2, col=1
                    )
        
        # 4. Computational efficiency
        fig.add_trace(
            go.Scatter(
                x=model_metrics['inference_time'],
                y=model_metrics['accuracy'],
                mode='markers+text',
                marker=dict(
                    size=model_metrics.get('model_size', 10),
                    color=[self.model_colors.get(m, '#95a5a6') 
                          for m in model_metrics['model']],
                    showscale=False
                ),
                text=model_metrics['model'],
                textposition='top center',
                name='Models'
            ),
            row=2, col=2
        )
        
        # Add efficiency frontier
        if len(model_metrics) > 2:
            # Simple pareto frontier
            sorted_models = model_metrics.sort_values('inference_time')
            pareto_models = []
            best_acc = 0
            
            for _, model in sorted_models.iterrows():
                if model['accuracy'] > best_acc:
                    pareto_models.append(model)
                    best_acc = model['accuracy']
            
            if len(pareto_models) > 1:
                pareto_df = pd.DataFrame(pareto_models)
                fig.add_trace(
                    go.Scatter(
                        x=pareto_df['inference_time'],
                        y=pareto_df['accuracy'],
                        mode='lines',
                        name='Efficiency Frontier',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Model Performance Comparison Dashboard',
            height=900,
            showlegend=True,
            template='plotly_white',
            barmode='group'
        )
        
        # Update axes
        fig.update_yaxes(title_text='Score', row=1, col=1, range=[0, 1])
        fig.update_xaxes(title_text='Epoch', row=2, col=1)
        fig.update_yaxes(title_text='Validation Accuracy', row=2, col=1)
        fig.update_xaxes(title_text='Inference Time (ms)', row=2, col=2)
        fig.update_yaxes(title_text='Accuracy', row=2, col=2)
        
        # Update polar subplot
        fig.update_polars(
            radialaxis=dict(range=[0, 1]),
            row=1, col=2
        )
        
        return fig
    
    def create_summary_metrics_card(self,
                                  metrics: Dict[str, Any]) -> go.Figure:
        """
        Create a summary metrics card visualization.
        
        Args:
            metrics: Dictionary of key metrics
            
        Returns:
            Plotly figure with metric cards
        """
        # Create a simple metric cards layout
        fig = go.Figure()
        
        # Define metric cards
        cards = [
            {'title': 'Overall Accuracy', 'value': metrics.get('overall_accuracy', 0), 'format': '.1%'},
            {'title': 'Best Model', 'value': metrics.get('best_model', 'N/A'), 'format': 's'},
            {'title': 'Avg Confidence', 'value': metrics.get('avg_confidence', 0), 'format': '.1%'},
            {'title': 'Total Predictions', 'value': metrics.get('total_predictions', 0), 'format': ',d'}
        ]
        
        # Create text annotations for each card
        annotations = []
        for i, card in enumerate(cards):
            x_pos = (i % 2) * 0.5 + 0.25
            y_pos = 0.7 if i < 2 else 0.3
            
            # Title
            annotations.append(dict(
                x=x_pos, y=y_pos + 0.1,
                text=card['title'],
                showarrow=False,
                font=dict(size=14, color='gray'),
                xanchor='center'
            ))
            
            # Value
            value_text = format(card['value'], card['format']) if card['format'] != 's' else card['value']
            annotations.append(dict(
                x=x_pos, y=y_pos,
                text=value_text,
                showarrow=False,
                font=dict(size=24, color='black', family='Arial Black'),
                xanchor='center'
            ))
        
        # Update layout
        fig.update_layout(
            title='Key Performance Metrics',
            annotations=annotations,
            showlegend=False,
            height=300,
            template='plotly_white',
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1])
        )
        
        return fig
