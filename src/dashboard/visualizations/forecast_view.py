"""
Forecast Visualization Component for Wavelet Pattern Prediction Dashboard.

This module provides interactive visualizations for pattern predictions including:
- Current pattern context visualization
- Predicted next pattern(s) with confidence bands
- Multiple prediction scenarios
- Historical accuracy overlay
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

from ..pattern_classifier import PatternClassifier
from ..pattern_predictor import PatternPredictor
from ..wavelet_sequence_analyzer import WaveletSequenceAnalyzer

logger = logging.getLogger(__name__)


class ForecastVisualization:
    """Creates interactive forecast visualizations for pattern predictions."""
    
    def __init__(self, 
                 pattern_classifier: Optional[PatternClassifier] = None,
                 pattern_predictor: Optional[PatternPredictor] = None):
        """
        Initialize forecast visualization component.
        
        Args:
            pattern_classifier: Pattern classifier instance
            pattern_predictor: Pattern predictor instance
        """
        self.pattern_classifier = pattern_classifier or PatternClassifier()
        self.pattern_predictor = pattern_predictor or PatternPredictor()
        
        # Color scheme for different pattern types
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
        
        # Confidence level colors
        self.confidence_colors = {
            'high': 'rgba(46, 204, 113, 0.3)',      # Green
            'medium': 'rgba(241, 196, 15, 0.3)',    # Yellow
            'low': 'rgba(231, 76, 60, 0.3)'         # Red
        }
    
    def create_current_context_view(self,
                                  current_pattern: Dict[str, Any],
                                  historical_patterns: List[Dict[str, Any]],
                                  price_data: pd.DataFrame) -> go.Figure:
        """
        Create visualization of current pattern context.
        
        Args:
            current_pattern: Current detected pattern
            historical_patterns: Recent historical patterns
            price_data: Price data DataFrame
            
        Returns:
            Plotly figure showing current context
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Current Pattern Context', 'Recent Pattern History'),
            vertical_spacing=0.1
        )
        
        # Plot price data with current pattern highlighted
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Highlight current pattern
        if current_pattern:
            start_idx = current_pattern.get('start_idx', 0)
            end_idx = current_pattern.get('end_idx', len(price_data) - 1)
            pattern_type = current_pattern.get('pattern_type', 'Unknown')
            
            # Add shaded region for current pattern
            fig.add_vrect(
                x0=price_data.index[start_idx],
                x1=price_data.index[end_idx],
                fillcolor=self.pattern_colors.get(pattern_type, '#95A5A6'),
                opacity=0.2,
                layer="below",
                line_width=2,
                annotation_text=f"Current: {pattern_type}",
                annotation_position="top left",
                row=1, col=1
            )
        
        # Plot recent pattern history timeline
        for i, pattern in enumerate(historical_patterns[-10:]):  # Last 10 patterns
            pattern_type = pattern.get('pattern_type', 'Unknown')
            start_time = pattern.get('start_time', i)
            duration = pattern.get('duration', 1)
            
            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[i],
                    orientation='h',
                    name=pattern_type,
                    marker_color=self.pattern_colors.get(pattern_type, '#95A5A6'),
                    text=pattern_type,
                    textposition='inside',
                    showlegend=False,
                    base=start_time
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Pattern Forecast Context',
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Pattern Index', row=2, col=1)
        
        return fig
    
    def create_prediction_visualization(self,
                                      predictions: List[Dict[str, Any]],
                                      confidence_bands: Dict[str, List[float]],
                                      time_horizon: int = 20) -> go.Figure:
        """
        Create visualization for predicted patterns with confidence bands.
        
        Args:
            predictions: List of predicted patterns with probabilities
            confidence_bands: Confidence intervals for predictions
            time_horizon: Number of time steps to show
            
        Returns:
            Plotly figure showing predictions
        """
        fig = go.Figure()
        
        # Sort predictions by probability
        predictions = sorted(predictions, key=lambda x: x.get('probability', 0), reverse=True)
        
        # Create time axis for predictions
        time_steps = list(range(time_horizon))
        
        # Plot top predictions with confidence bands
        for i, pred in enumerate(predictions[:3]):  # Top 3 predictions
            pattern_type = pred.get('pattern_type', 'Unknown')
            probability = pred.get('probability', 0)
            
            # Main prediction line
            y_values = [probability * (1 - t/time_horizon) for t in time_steps]
            
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=y_values,
                    mode='lines+markers',
                    name=f'{pattern_type} ({probability:.1%})',
                    line=dict(
                        color=self.pattern_colors.get(pattern_type, '#95A5A6'),
                        width=3 - i  # Thicker lines for higher probability
                    ),
                    marker=dict(size=8)
                )
            )
            
            # Add confidence bands
            if pattern_type in confidence_bands:
                upper_band = confidence_bands[pattern_type].get('upper', y_values)
                lower_band = confidence_bands[pattern_type].get('lower', y_values)
                
                # Determine confidence level based on probability
                if probability > 0.7:
                    conf_color = self.confidence_colors['high']
                elif probability > 0.4:
                    conf_color = self.confidence_colors['medium']
                else:
                    conf_color = self.confidence_colors['low']
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps + time_steps[::-1],
                        y=upper_band + lower_band[::-1],
                        fill='toself',
                        fillcolor=conf_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name=f'{pattern_type} confidence'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Pattern Predictions with Confidence Bands',
            xaxis_title='Time Steps Ahead',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_scenario_analysis(self,
                               scenarios: List[Dict[str, Any]],
                               current_price: float) -> go.Figure:
        """
        Create multiple prediction scenario visualization.
        
        Args:
            scenarios: List of prediction scenarios
            current_price: Current price level
            
        Returns:
            Plotly figure showing scenarios
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Prediction Scenarios', 'Probability Distribution'),
            column_widths=[0.7, 0.3]
        )
        
        # Plot each scenario
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Scenario')
            pattern_sequence = scenario.get('pattern_sequence', [])
            price_path = scenario.get('price_path', [current_price])
            probability = scenario.get('probability', 0)
            
            # Price path for this scenario
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(price_path))),
                    y=price_path,
                    mode='lines',
                    name=f'{scenario_name} (p={probability:.2f})',
                    line=dict(width=2 + probability * 3),  # Width based on probability
                    opacity=0.3 + probability * 0.7  # Opacity based on probability
                ),
                row=1, col=1
            )
        
        # Probability distribution
        scenario_probs = [s.get('probability', 0) for s in scenarios]
        scenario_names = [s.get('name', f'S{i}') for i, s in enumerate(scenarios)]
        
        fig.add_trace(
            go.Bar(
                y=scenario_names,
                x=scenario_probs,
                orientation='h',
                marker_color='lightblue',
                text=[f'{p:.1%}' for p in scenario_probs],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Multi-Scenario Pattern Predictions',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Time Steps', row=1, col=1)
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_xaxes(title_text='Probability', row=1, col=2)
        
        return fig
    
    def create_historical_accuracy_overlay(self,
                                         predictions_history: pd.DataFrame,
                                         actual_patterns: pd.DataFrame) -> go.Figure:
        """
        Create visualization showing historical prediction accuracy.
        
        Args:
            predictions_history: Historical predictions DataFrame
            actual_patterns: Actual patterns that occurred
            
        Returns:
            Plotly figure with accuracy overlay
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Predictions vs Actual', 'Accuracy Over Time'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Plot predictions vs actual patterns
        if not predictions_history.empty and not actual_patterns.empty:
            # Predicted patterns
            for pattern_type in predictions_history['pattern_type'].unique():
                pattern_data = predictions_history[predictions_history['pattern_type'] == pattern_type]
                
                fig.add_trace(
                    go.Scatter(
                        x=pattern_data.index,
                        y=pattern_data['probability'],
                        mode='markers',
                        name=f'Predicted: {pattern_type}',
                        marker=dict(
                            symbol='circle-open',
                            size=10,
                            color=self.pattern_colors.get(pattern_type, '#95A5A6')
                        )
                    ),
                    row=1, col=1
                )
            
            # Actual patterns
            for pattern_type in actual_patterns['pattern_type'].unique():
                pattern_data = actual_patterns[actual_patterns['pattern_type'] == pattern_type]
                
                fig.add_trace(
                    go.Scatter(
                        x=pattern_data.index,
                        y=[1.0] * len(pattern_data),  # Actual patterns at y=1
                        mode='markers',
                        name=f'Actual: {pattern_type}',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color=self.pattern_colors.get(pattern_type, '#95A5A6')
                        )
                    ),
                    row=1, col=1
                )
            
            # Calculate rolling accuracy
            window_size = 20
            accuracy_series = self._calculate_rolling_accuracy(
                predictions_history, actual_patterns, window_size
            )
            
            fig.add_trace(
                go.Scatter(
                    x=accuracy_series.index,
                    y=accuracy_series.values,
                    mode='lines',
                    name='Rolling Accuracy',
                    line=dict(color='green', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add accuracy threshold line
            fig.add_hline(
                y=0.7, line_dash="dash", line_color="red",
                annotation_text="Target Accuracy (70%)",
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Historical Prediction Accuracy',
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Prediction Probability', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy %', row=2, col=1, range=[0, 1])
        
        return fig
    
    def _calculate_rolling_accuracy(self,
                                  predictions: pd.DataFrame,
                                  actuals: pd.DataFrame,
                                  window: int) -> pd.Series:
        """Calculate rolling prediction accuracy."""
        # Merge predictions with actuals
        merged = pd.merge_asof(
            predictions.sort_index(),
            actuals.sort_index(),
            left_index=True,
            right_index=True,
            direction='forward',
            tolerance=pd.Timedelta('5D')
        )
        
        # Calculate accuracy
        merged['correct'] = merged['pattern_type_x'] == merged['pattern_type_y']
        
        # Rolling mean
        accuracy = merged['correct'].rolling(window=window, min_periods=1).mean()
        
        return accuracy
    
    def create_confidence_calibration_plot(self,
                                         predictions: pd.DataFrame) -> go.Figure:
        """
        Create confidence calibration plot.
        
        Args:
            predictions: DataFrame with predictions and outcomes
            
        Returns:
            Plotly figure showing calibration
        """
        # Bin predictions by confidence level
        bins = np.linspace(0, 1, 11)
        predictions['confidence_bin'] = pd.cut(
            predictions['probability'],
            bins=bins,
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)]
        )
        
        # Calculate actual accuracy per bin
        calibration = predictions.groupby('confidence_bin').agg({
            'correct': 'mean',
            'probability': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            )
        )
        
        # Actual calibration
        fig.add_trace(
            go.Scatter(
                x=calibration['probability'],
                y=calibration['correct'],
                mode='lines+markers',
                name='Actual Calibration',
                marker=dict(size=10),
                line=dict(width=3)
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Prediction Confidence Calibration',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Actual Accuracy',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=500
        )
        
        return fig
