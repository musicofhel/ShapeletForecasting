"""
Pattern Sequence Visualization Component

This module provides interactive visualizations for pattern sequences,
including timelines, transitions, and probability overlays.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PatternSequenceVisualizer:
    """Creates interactive visualizations for pattern sequences"""
    
    def __init__(self):
        """Initialize the pattern sequence visualizer"""
        self.color_palette = {
            'head_shoulders': '#FF6B6B',
            'double_top': '#4ECDC4',
            'double_bottom': '#45B7D1',
            'triangle_ascending': '#96CEB4',
            'triangle_descending': '#FECA57',
            'flag_bull': '#48C9B0',
            'flag_bear': '#F97F51',
            'wedge_rising': '#B983FF',
            'wedge_falling': '#FD79A8',
            'channel_up': '#A0E7E5',
            'channel_down': '#FFBE76',
            'fractal': '#6C5CE7',
            'shapelet': '#A29BFE',
            'unknown': '#95A5A6'
        }
        
        self.transition_colors = {
            'high_prob': 'rgba(46, 204, 113, 0.8)',  # Green
            'medium_prob': 'rgba(241, 196, 15, 0.8)',  # Yellow
            'low_prob': 'rgba(231, 76, 60, 0.8)'  # Red
        }
    
    def create_pattern_timeline(self,
                              patterns: List[Dict],
                              transitions: Optional[Dict] = None,
                              show_probabilities: bool = True,
                              height: int = 600) -> go.Figure:
        """
        Create a timeline visualization of detected patterns
        
        Args:
            patterns: List of pattern dictionaries with keys:
                     - 'type': Pattern type name
                     - 'start_time': Start timestamp
                     - 'end_time': End timestamp
                     - 'confidence': Confidence score
                     - 'id': Unique pattern ID
            transitions: Dictionary of pattern transitions with probabilities
            show_probabilities: Whether to show transition probabilities
            height: Figure height in pixels
            
        Returns:
            Plotly figure object
        """
        if not patterns:
            logger.warning("No patterns provided for timeline visualization")
            return self._create_empty_figure("No patterns to display")
        
        # Sort patterns by start time
        patterns = sorted(patterns, key=lambda x: x['start_time'])
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Pattern Timeline', 'Pattern Duration Distribution')
        )
        
        # Add pattern bars
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get('type', 'unknown')
            color = self.color_palette.get(pattern_type, self.color_palette['unknown'])
            
            # Calculate duration
            start = pd.to_datetime(pattern['start_time'])
            end = pd.to_datetime(pattern['end_time'])
            duration = (end - start).total_seconds() / 3600  # Hours
            
            # Add pattern bar
            fig.add_trace(
                go.Scatter(
                    x=[start, end, end, start, start],
                    y=[i-0.4, i-0.4, i+0.4, i+0.4, i-0.4],
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=2),
                    mode='lines',
                    name=pattern_type,
                    text=f"Pattern: {pattern_type}<br>"
                         f"Confidence: {pattern.get('confidence', 0):.2%}<br>"
                         f"Duration: {duration:.1f} hours",
                    hoverinfo='text',
                    showlegend=i == 0 or pattern_type not in [p['type'] for p in patterns[:i]]
                ),
                row=1, col=1
            )
            
            # Add pattern label
            fig.add_trace(
                go.Scatter(
                    x=[start + (end - start) / 2],
                    y=[i],
                    mode='text',
                    text=pattern.get('id', f'P{i+1}'),
                    textposition='middle center',
                    textfont=dict(color='white', size=10),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        # Add transitions if provided
        if transitions and show_probabilities:
            self._add_transitions(fig, patterns, transitions)
        
        # Add duration distribution
        self._add_duration_distribution(fig, patterns)
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(
            title_text="Pattern Sequence",
            tickmode='array',
            tickvals=list(range(len(patterns))),
            ticktext=[f"P{i+1}" for i in range(len(patterns))],
            row=1, col=1
        )
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_layout(
            height=height,
            title="Pattern Sequence Timeline",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def create_transition_matrix(self,
                               transition_probs: Dict[Tuple[str, str], float],
                               min_probability: float = 0.01) -> go.Figure:
        """
        Create a heatmap visualization of pattern transition probabilities
        
        Args:
            transition_probs: Dictionary mapping (from_pattern, to_pattern) to probability
            min_probability: Minimum probability to display
            
        Returns:
            Plotly figure object
        """
        # Extract unique pattern types
        pattern_types = sorted(set(
            [t[0] for t in transition_probs.keys()] +
            [t[1] for t in transition_probs.keys()]
        ))
        
        # Create transition matrix
        n = len(pattern_types)
        matrix = np.zeros((n, n))
        
        for i, from_pattern in enumerate(pattern_types):
            for j, to_pattern in enumerate(pattern_types):
                prob = transition_probs.get((from_pattern, to_pattern), 0)
                if prob >= min_probability:
                    matrix[i, j] = prob
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=pattern_types,
            y=pattern_types,
            colorscale='Viridis',
            text=[[f'{val:.2%}' if val > 0 else '' for val in row] for row in matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Pattern Transition Probability Matrix',
            xaxis_title='To Pattern',
            yaxis_title='From Pattern',
            height=600,
            width=800
        )
        
        return fig
    
    def create_pattern_flow_diagram(self,
                                  patterns: List[Dict],
                                  transitions: Dict[str, List[Dict]],
                                  max_depth: int = 3) -> go.Figure:
        """
        Create a flow diagram showing pattern sequences and transitions
        
        Args:
            patterns: List of detected patterns
            transitions: Dictionary of pattern transitions
            max_depth: Maximum depth of transitions to show
            
        Returns:
            Plotly figure object
        """
        if not patterns:
            return self._create_empty_figure("No patterns to display")
        
        # Create Sankey diagram
        labels = []
        sources = []
        targets = []
        values = []
        colors = []
        
        # Build node labels and links
        pattern_indices = {}
        current_idx = 0
        
        for i, pattern in enumerate(patterns[:max_depth]):
            pattern_id = f"{pattern['type']}_{i}"
            if pattern_id not in pattern_indices:
                pattern_indices[pattern_id] = current_idx
                labels.append(f"{pattern['type']}\n{pattern.get('id', f'P{i+1}')}")
                current_idx += 1
            
            # Add transitions from this pattern
            if pattern['id'] in transitions:
                for trans in transitions[pattern['id']][:3]:  # Top 3 transitions
                    next_pattern_id = f"{trans['to_pattern']}_{i+1}"
                    if next_pattern_id not in pattern_indices:
                        pattern_indices[next_pattern_id] = current_idx
                        labels.append(f"{trans['to_pattern']}\n(Next)")
                        current_idx += 1
                    
                    sources.append(pattern_indices[pattern_id])
                    targets.append(pattern_indices[next_pattern_id])
                    values.append(trans['probability'])
                    
                    # Color based on probability
                    if trans['probability'] > 0.7:
                        colors.append(self.transition_colors['high_prob'])
                    elif trans['probability'] > 0.4:
                        colors.append(self.transition_colors['medium_prob'])
                    else:
                        colors.append(self.transition_colors['low_prob'])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[self.color_palette.get(l.split('\n')[0], self.color_palette['unknown']) 
                       for l in labels]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])
        
        fig.update_layout(
            title="Pattern Flow Diagram",
            height=600,
            font_size=10
        )
        
        return fig
    
    def create_pattern_duration_analysis(self,
                                       patterns: List[Dict]) -> go.Figure:
        """
        Create detailed duration analysis visualization
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Plotly figure object
        """
        if not patterns:
            return self._create_empty_figure("No patterns to display")
        
        # Calculate durations by pattern type
        duration_data = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            start = pd.to_datetime(pattern['start_time'])
            end = pd.to_datetime(pattern['end_time'])
            duration = (end - start).total_seconds() / 3600  # Hours
            
            if pattern_type not in duration_data:
                duration_data[pattern_type] = []
            duration_data[pattern_type].append(duration)
        
        # Create box plots
        fig = go.Figure()
        
        for pattern_type, durations in duration_data.items():
            color = self.color_palette.get(pattern_type, self.color_palette['unknown'])
            fig.add_trace(go.Box(
                y=durations,
                name=pattern_type,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker_color=color,
                line_color=color
            ))
        
        fig.update_layout(
            title='Pattern Duration Distribution by Type',
            yaxis_title='Duration (hours)',
            xaxis_title='Pattern Type',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_confidence_timeline(self,
                                 patterns: List[Dict]) -> go.Figure:
        """
        Create a timeline showing pattern confidence scores
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Plotly figure object
        """
        if not patterns:
            return self._create_empty_figure("No patterns to display")
        
        # Sort patterns by start time
        patterns = sorted(patterns, key=lambda x: x['start_time'])
        
        fig = go.Figure()
        
        # Add confidence line
        times = []
        confidences = []
        pattern_types = []
        
        for pattern in patterns:
            start = pd.to_datetime(pattern['start_time'])
            end = pd.to_datetime(pattern['end_time'])
            confidence = pattern.get('confidence', 0)
            
            times.extend([start, end])
            confidences.extend([confidence, confidence])
            pattern_types.extend([pattern['type'], pattern['type']])
        
        fig.add_trace(go.Scatter(
            x=times,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Time: %{x}<br>Confidence: %{y:.2%}<extra></extra>'
        ))
        
        # Add confidence threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                      annotation_text="High Confidence (80%)")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                      annotation_text="Medium Confidence (60%)")
        fig.add_hline(y=0.4, line_dash="dash", line_color="red",
                      annotation_text="Low Confidence (40%)")
        
        fig.update_layout(
            title='Pattern Detection Confidence Timeline',
            xaxis_title='Time',
            yaxis_title='Confidence Score',
            yaxis_tickformat='.0%',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _add_transitions(self, fig: go.Figure, patterns: List[Dict], 
                        transitions: Dict) -> None:
        """Add transition arrows between patterns"""
        for i in range(len(patterns) - 1):
            current = patterns[i]
            next_pattern = patterns[i + 1]
            
            # Get transition probability
            trans_key = f"{current['id']}_to_{next_pattern['id']}"
            probability = transitions.get(trans_key, {}).get('probability', 0)
            
            if probability > 0:
                # Determine arrow color based on probability
                if probability > 0.7:
                    color = self.transition_colors['high_prob']
                elif probability > 0.4:
                    color = self.transition_colors['medium_prob']
                else:
                    color = self.transition_colors['low_prob']
                
                # Add arrow
                start_time = pd.to_datetime(current['end_time'])
                end_time = pd.to_datetime(next_pattern['start_time'])
                
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[i, i + 1],
                        mode='lines+markers',
                        line=dict(color=color, width=3),
                        marker=dict(symbol='arrow', size=15, angleref='previous'),
                        text=f"Transition Probability: {probability:.2%}",
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    def _add_duration_distribution(self, fig: go.Figure, patterns: List[Dict]) -> None:
        """Add duration distribution histogram"""
        durations = []
        for pattern in patterns:
            start = pd.to_datetime(pattern['start_time'])
            end = pd.to_datetime(pattern['end_time'])
            duration = (end - start).total_seconds() / 3600  # Hours
            durations.append(duration)
        
        fig.add_trace(
            go.Histogram(
                x=durations,
                nbinsx=20,
                name='Duration Distribution',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
