"""
Pattern Information Cards Component for Financial Wavelet Prediction Dashboard

This module provides detailed pattern information cards that display:
- Full pattern visualization
- Discovery timestamp and duration
- Statistical properties (mean, std, trend, energy)
- Quality metrics and confidence scores
- List of all occurrences with timestamps
- Associated predictions and their accuracy

Features:
- Expand/collapse for more details
- Link to main chart locations
- Support pattern comparison
- Export individual patterns
"""

import dash
from dash import dcc, html, Input, Output, State, callback, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import base64
import io
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternInfo:
    """Container for comprehensive pattern information"""
    pattern_id: str
    pattern_type: str
    ticker: str
    discovery_timestamp: datetime
    duration_hours: float
    start_time: datetime
    end_time: datetime
    
    # Pattern data
    pattern_data: List[float]
    normalized_data: List[float]
    timestamps: List[datetime]
    
    # Statistical properties
    mean: float
    std: float
    trend: float
    energy: float
    entropy: float
    skewness: float
    kurtosis: float
    
    # Quality metrics
    quality_score: float
    confidence_score: float
    significance_score: float
    stability_score: float
    robustness_score: float
    
    # Occurrences
    occurrences: List[Dict[str, Any]]
    total_occurrences: int
    
    # Predictions
    predictions: List[Dict[str, Any]]
    prediction_accuracy: float
    mean_absolute_error: float
    hit_rate: float
    
    # Additional metadata
    metadata: Dict[str, Any]


class PatternCards:
    """Component for displaying detailed pattern information cards"""
    
    def __init__(self):
        """Initialize pattern cards component"""
        self.expanded_cards = set()
        self.comparison_patterns = []
        self.export_format = 'json'
        
    def create_pattern_card(self, pattern_info: PatternInfo, 
                          card_index: int, is_expanded: bool = False) -> dbc.Card:
        """
        Create a single pattern information card
        
        Args:
            pattern_info: Pattern information data
            card_index: Index of the card for unique IDs
            is_expanded: Whether card is expanded
            
        Returns:
            Pattern card component
        """
        # Card header with key metrics
        header = dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        f"{pattern_info.ticker} - {pattern_info.pattern_type}",
                        dbc.Badge(
                            f"Q: {pattern_info.quality_score:.0%}",
                            color=self._get_quality_color(pattern_info.quality_score),
                            className="ms-2"
                        )
                    ], className="mb-0")
                ], width=8),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "📍",
                            id={"type": "locate-pattern", "index": card_index},
                            color="link",
                            size="sm",
                            title="Locate on main chart"
                        ),
                        dbc.Button(
                            "🔄",
                            id={"type": "compare-pattern", "index": card_index},
                            color="link",
                            size="sm",
                            title="Add to comparison"
                        ),
                        dbc.Button(
                            "💾",
                            id={"type": "export-pattern", "index": card_index},
                            color="link",
                            size="sm",
                            title="Export pattern"
                        ),
                        dbc.Button(
                            "▼" if not is_expanded else "▲",
                            id={"type": "expand-pattern", "index": card_index},
                            color="link",
                            size="sm",
                            title="Expand/Collapse"
                        )
                    ], size="sm")
                ], width=4, className="text-end")
            ])
        ])
        
        # Basic info section (always visible)
        basic_info = dbc.CardBody([
            # Pattern thumbnail and key metrics
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=self._create_pattern_thumbnail(pattern_info),
                        config={'displayModeBar': False},
                        style={'height': '150px'}
                    )
                ], width=4),
                dbc.Col([
                    self._create_key_metrics_display(pattern_info)
                ], width=8)
            ]),
            
            # Discovery info
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Small([
                        html.I(className="fas fa-clock me-1"),
                        f"Discovered: {pattern_info.discovery_timestamp.strftime('%Y-%m-%d %H:%M')}"
                    ], className="text-muted")
                ], width=6),
                dbc.Col([
                    html.Small([
                        html.I(className="fas fa-history me-1"),
                        f"Duration: {pattern_info.duration_hours:.1f}h"
                    ], className="text-muted")
                ], width=6, className="text-end")
            ])
        ])
        
        # Expanded details section
        expanded_details = dbc.Collapse([
            dbc.CardBody([
                # Detailed visualizations
                dbc.Tabs([
                    dbc.Tab(
                        self._create_statistical_tab(pattern_info),
                        label="Statistics",
                        tab_id=f"stats-{card_index}"
                    ),
                    dbc.Tab(
                        self._create_occurrences_tab(pattern_info),
                        label=f"Occurrences ({pattern_info.total_occurrences})",
                        tab_id=f"occur-{card_index}"
                    ),
                    dbc.Tab(
                        self._create_predictions_tab(pattern_info),
                        label="Predictions",
                        tab_id=f"pred-{card_index}"
                    ),
                    dbc.Tab(
                        self._create_analysis_tab(pattern_info),
                        label="Analysis",
                        tab_id=f"analysis-{card_index}"
                    )
                ], id=f"pattern-tabs-{card_index}", active_tab=f"stats-{card_index}")
            ])
        ], id={"type": "pattern-collapse", "index": card_index}, is_open=is_expanded)
        
        # Create complete card
        card = dbc.Card([
            header,
            basic_info,
            expanded_details
        ], className="mb-3 pattern-info-card", id=f"pattern-card-{card_index}")
        
        return card
    
    def _create_pattern_thumbnail(self, pattern_info: PatternInfo) -> go.Figure:
        """Create pattern thumbnail visualization"""
        fig = go.Figure()
        
        # Pattern line
        fig.add_trace(go.Scatter(
            x=list(range(len(pattern_info.pattern_data))),
            y=pattern_info.pattern_data,
            mode='lines',
            line=dict(color='#00D9FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)',
            showlegend=False,
            hovertemplate='Value: %{y:.4f}<extra></extra>'
        ))
        
        # Add trend line
        x = np.arange(len(pattern_info.pattern_data))
        z = np.polyfit(x, pattern_info.pattern_data, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=p(x),
            mode='lines',
            line=dict(color='#FF6B6B', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x'
        )
        
        return fig
    
    def _create_key_metrics_display(self, pattern_info: PatternInfo) -> html.Div:
        """Create key metrics display"""
        return html.Div([
            # Quality scores row
            dbc.Row([
                dbc.Col([
                    self._create_metric_badge("Confidence", pattern_info.confidence_score)
                ], width=3),
                dbc.Col([
                    self._create_metric_badge("Significance", pattern_info.significance_score)
                ], width=3),
                dbc.Col([
                    self._create_metric_badge("Stability", pattern_info.stability_score)
                ], width=3),
                dbc.Col([
                    self._create_metric_badge("Robustness", pattern_info.robustness_score)
                ], width=3)
            ], className="mb-2"),
            
            # Statistical metrics row
            dbc.Row([
                dbc.Col([
                    html.Small("Mean:", className="text-muted"),
                    html.Div(f"{pattern_info.mean:.4f}", className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Std Dev:", className="text-muted"),
                    html.Div(f"{pattern_info.std:.4f}", className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Trend:", className="text-muted"),
                    html.Div([
                        f"{pattern_info.trend:+.2%}",
                        html.I(className=f"fas fa-arrow-{'up' if pattern_info.trend > 0 else 'down'} ms-1")
                    ], className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Energy:", className="text-muted"),
                    html.Div(f"{pattern_info.energy:.2f}", className="fw-bold")
                ], width=3)
            ]),
            
            # Prediction accuracy
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col([
                    html.Small("Prediction Accuracy:", className="text-muted"),
                    dbc.Progress(
                        value=pattern_info.prediction_accuracy * 100,
                        color=self._get_accuracy_color(pattern_info.prediction_accuracy),
                        className="mb-1",
                        style={"height": "10px"}
                    ),
                    html.Small(f"{pattern_info.prediction_accuracy:.1%} | MAE: {pattern_info.mean_absolute_error:.4f}")
                ])
            ])
        ])
    
    def _create_metric_badge(self, label: str, value: float) -> html.Div:
        """Create a metric badge"""
        color = self._get_quality_color(value)
        return html.Div([
            html.Small(label, className="text-muted d-block"),
            dbc.Badge(f"{value:.0%}", color=color, className="w-100")
        ])
    
    def _create_statistical_tab(self, pattern_info: PatternInfo) -> html.Div:
        """Create statistical analysis tab"""
        # Create statistical plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pattern Distribution', 'Autocorrelation',
                          'Power Spectrum', 'Statistical Summary'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=pattern_info.pattern_data,
                nbinsx=30,
                marker_color='#00D9FF',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Autocorrelation
        autocorr = self._compute_autocorrelation(pattern_info.pattern_data)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(autocorr))),
                y=autocorr,
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='#FF6B6B'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Power spectrum
        freqs, power = self._compute_power_spectrum(pattern_info.pattern_data)
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode='lines',
                line=dict(color='#4ECDC4'),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.2)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Statistical summary table
        stats_data = [
            ['Metric', 'Value'],
            ['Mean', f'{pattern_info.mean:.6f}'],
            ['Std Dev', f'{pattern_info.std:.6f}'],
            ['Skewness', f'{pattern_info.skewness:.4f}'],
            ['Kurtosis', f'{pattern_info.kurtosis:.4f}'],
            ['Entropy', f'{pattern_info.entropy:.4f}'],
            ['Energy', f'{pattern_info.energy:.4f}'],
            ['Trend', f'{pattern_info.trend:+.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#1E1E1E',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=list(zip(*stats_data[1:])),
                    fill_color='#0E1117',
                    font=dict(color='white', size=11),
                    height=25
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white', size=10)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridcolor='#333')
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    
    def _create_occurrences_tab(self, pattern_info: PatternInfo) -> html.Div:
        """Create occurrences list tab"""
        # Create timeline visualization
        fig = go.Figure()
        
        # Sort occurrences by timestamp
        occurrences = sorted(pattern_info.occurrences, 
                           key=lambda x: x['timestamp'], reverse=True)
        
        # Extract data for visualization
        timestamps = [occ['timestamp'] for occ in occurrences]
        qualities = [occ['quality_score'] for occ in occurrences]
        outcomes = [occ.get('outcome', 0) for occ in occurrences]
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=qualities,
            mode='markers',
            marker=dict(
                size=10,
                color=outcomes,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Outcome")
            ),
            text=[f"Quality: {q:.2%}<br>Outcome: {o:+.2%}" 
                  for q, o in zip(qualities, outcomes)],
            hovertemplate='%{text}<br>%{x}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Pattern Occurrences Timeline",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            height=300,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white')
        )
        
        # Create occurrences table
        table_data = []
        for i, occ in enumerate(occurrences[:20]):  # Show top 20
            table_data.append(
                dbc.Row([
                    dbc.Col([
                        html.Small(f"#{i+1}", className="text-muted"),
                        html.Div(occ['timestamp'].strftime('%Y-%m-%d %H:%M'))
                    ], width=3),
                    dbc.Col([
                        dbc.Badge(
                            f"Q: {occ['quality_score']:.0%}",
                            color=self._get_quality_color(occ['quality_score'])
                        )
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            f"{occ.get('outcome', 0):+.2%}",
                            html.I(className=f"fas fa-arrow-{'up' if occ.get('outcome', 0) > 0 else 'down'} ms-1")
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Small(f"Duration: {occ.get('duration_hours', 0):.1f}h")
                    ], width=2),
                    dbc.Col([
                        dbc.Button(
                            "View",
                            id={"type": "view-occurrence", "index": f"{pattern_info.pattern_id}-{i}"},
                            size="sm",
                            color="info",
                            outline=True
                        )
                    ], width=3)
                ], className="mb-2 p-2 border-bottom")
            )
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.Hr(),
            html.H6(f"Recent Occurrences (showing {min(20, len(occurrences))} of {len(occurrences)})"),
            html.Div(table_data, style={"maxHeight": "400px", "overflowY": "auto"})
        ])
    
    def _create_predictions_tab(self, pattern_info: PatternInfo) -> html.Div:
        """Create predictions analysis tab"""
        # Create prediction performance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction vs Actual', 'Error Distribution',
                          'Accuracy Over Time', 'Performance Metrics'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Get prediction data
        predictions = pattern_info.predictions[:50]  # Last 50 predictions
        if predictions:
            pred_values = [p['predicted'] for p in predictions]
            actual_values = [p['actual'] for p in predictions]
            errors = [p['actual'] - p['predicted'] for p in predictions]
            timestamps = [p['timestamp'] for p in predictions]
            
            # Prediction vs Actual scatter
            fig.add_trace(
                go.Scatter(
                    x=pred_values,
                    y=actual_values,
                    mode='markers',
                    marker=dict(color='#00D9FF', size=8),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add perfect prediction line
            min_val = min(min(pred_values), min(actual_values))
            max_val = max(max(pred_values), max(actual_values))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Error distribution
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    nbinsx=20,
                    marker_color='#FF6B6B',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Accuracy over time
            rolling_accuracy = self._compute_rolling_accuracy(predictions, window=10)
            fig.add_trace(
                go.Scatter(
                    x=timestamps[-len(rolling_accuracy):],
                    y=rolling_accuracy,
                    mode='lines+markers',
                    line=dict(color='#4ECDC4'),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Performance metrics gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pattern_info.hit_rate * 100,
                title={'text': "Hit Rate (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self._get_accuracy_color(pattern_info.hit_rate)},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white', size=10)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridcolor='#333')
        
        # Create summary stats
        summary_stats = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("Prediction Summary", className="mb-3"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Total Predictions:", width=6),
                                dbc.Col(f"{len(pattern_info.predictions)}", width=6, className="text-end fw-bold")
                            ])
                        ], color="dark"),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Accuracy:", width=6),
                                dbc.Col(f"{pattern_info.prediction_accuracy:.1%}", width=6, className="text-end fw-bold")
                            ])
                        ], color="dark"),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Mean Absolute Error:", width=6),
                                dbc.Col(f"{pattern_info.mean_absolute_error:.4f}", width=6, className="text-end fw-bold")
                            ])
                        ], color="dark"),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Hit Rate:", width=6),
                                dbc.Col(f"{pattern_info.hit_rate:.1%}", width=6, className="text-end fw-bold")
                            ])
                        ], color="dark")
                    ])
                ])
            ])
        ], className="mt-3")
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            summary_stats
        ])
    
    def _create_analysis_tab(self, pattern_info: PatternInfo) -> html.Div:
        """Create detailed analysis tab"""
        return html.Div([
            # Pattern characteristics
            html.H6("Pattern Characteristics", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    self._create_characteristic_card(
                        "Pattern Type",
                        pattern_info.pattern_type,
                        "Identified pattern category",
                        "fas fa-shapes"
                    )
                ], width=6),
                dbc.Col([
                    self._create_characteristic_card(
                        "Complexity",
                        self._assess_complexity(pattern_info),
                        "Pattern complexity level",
                        "fas fa-brain"
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Time analysis
            html.H6("Temporal Analysis", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    self._create_characteristic_card(
                        "Avg Duration",
                        f"{pattern_info.duration_hours:.1f} hours",
                        "Average pattern duration",
                        "fas fa-clock"
                    )
                ], width=4),
                dbc.Col([
                    self._create_characteristic_card(
                        "Frequency",
                        f"{pattern_info.total_occurrences} times",
                        "Total occurrences found",
                        "fas fa-redo"
                    )
                ], width=4),
                dbc.Col([
                    self._create_characteristic_card(
                        "Last Seen",
                        self._format_last_seen(pattern_info.occurrences),
                        "Most recent occurrence",
                        "fas fa-calendar"
                    )
                ], width=4)
            ], className="mb-3"),
            
            # Market conditions
            html.H6("Market Conditions", className="mb-3"),
            self._create_market_conditions_analysis(pattern_info),
            
            # Trading implications
            html.H6("Trading Implications", className="mb-3"),
            self._create_trading_implications(pattern_info)
        ])
    
    def _create_characteristic_card(self, title: str, value: str, 
                                  description: str, icon: str) -> dbc.Card:
        """Create a characteristic display card"""
        return dbc.Card([
            dbc.CardBody([
                html.I(className=f"{icon} fa-2x mb-2", style={"color": "#00D9FF"}),
                html.H6(title, className="mb-1"),
                html.Div(value, className="h5 mb-1"),
                html.Small(description, className="text-muted")
            ], className="text-center")
        ], className="h-100")
    
    def _create_market_conditions_analysis(self, pattern_info: PatternInfo) -> html.Div:
        """Create market conditions analysis"""
        # Analyze market conditions when pattern occurs
        conditions = self._analyze_market_conditions(pattern_info)
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Volatility", className="mb-2"),
                        dbc.Progress(
                            value=conditions['volatility'] * 100,
                            color="warning" if conditions['volatility'] > 0.7 else "info",
                            className="mb-2"
                        ),
                        html.Small(f"{conditions['volatility']:.1%} average")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Trend Strength", className="mb-2"),
                        dbc.Progress(
                            value=conditions['trend_strength'] * 100,
                            color="success" if conditions['trend_strength'] > 0.6 else "warning",
                            className="mb-2"
                        ),
                        html.Small(f"{conditions['trend_strength']:.1%} average")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Volume", className="mb-2"),
                        dbc.Progress(
                            value=conditions['volume_ratio'] * 100,
                            color="info",
                            className="mb-2"
                        ),
                        html.Small(f"{conditions['volume_ratio']:.1x} avg ratio")
                    ])
                ])
            ], width=4)
        ])
    
    def _create_trading_implications(self, pattern_info: PatternInfo) -> html.Div:
        """Create trading implications section"""
        implications = self._analyze_trading_implications(pattern_info)
        
        return dbc.Alert([
            html.H6("Key Insights:", className="alert-heading"),
            html.Ul([
                html.Li(implication) for implication in implications['insights']
            ]),
            html.Hr(),
            html.P([
                html.Strong("Recommended Action: "),
                implications['recommendation']
            ], className="mb-0")
        ], color="info")
    
    def create_pattern_cards_layout(self, patterns: List[PatternInfo],
                                  max_cards: int = 10) -> html.Div:
        """
        Create layout with multiple pattern cards
        
        Args:
            patterns: List of pattern information
            max_cards: Maximum number of cards to display
            
        Returns:
            Layout component
        """
        # Sort patterns by quality score
        sorted_patterns = sorted(patterns, 
                               key=lambda x: x.quality_score, 
                               reverse=True)[:max_cards]
        
        # Create cards
        cards = []
        for i, pattern in enumerate(sorted_patterns):
            is_expanded = i in self.expanded_cards
            card = self.create_pattern_card(pattern, i, is_expanded)
            cards.append(card)
        
        # Create layout
        layout = html.Div([
            # Header with controls
            dbc.Row([
                dbc.Col([
                    html.H4("Pattern Information Cards", className="mb-0")
                ], width=6),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "Expand All",
                            id="expand-all-patterns",
                            color="secondary",
                            size="sm"
                        ),
                        dbc.Button(
                            "Collapse All",
                            id="collapse-all-patterns",
                            color="secondary",
                            size="sm"
                        ),
                        dbc.Button(
                            "Compare Selected",
                            id="compare-selected-patterns",
                            color="info",
                            size="sm"
                        ),
                        dbc.Button(
                            "Export All",
                            id="export-all-patterns",
                            color="success",
                            size="sm"
                        )
                    ])
                ], width=6, className="text-end")
            ], className="mb-3"),
            
            # Filter controls
            dbc.Row([
                dbc.Col([
                    dbc.Label("Sort by:"),
                    dbc.Select(
                        id="pattern-sort-select",
                        options=[
                            {"label": "Quality Score", "value": "quality"},
                            {"label": "Recency", "value": "recency"},
                            {"label": "Occurrences", "value": "occurrences"},
                            {"label": "Accuracy", "value": "accuracy"}
                        ],
                        value="quality"
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Filter by Type:"),
                    dbc.Select(
                        id="pattern-type-filter",
                        options=[
                            {"label": "All Types", "value": "all"}
                        ] + [
                            {"label": ptype, "value": ptype}
                            for ptype in set(p.pattern_type for p in patterns)
                        ],
                        value="all"
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Min Quality:"),
                    dbc.Input(
                        id="min-quality-filter",
                        type="number",
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Show:"),
                    dbc.Input(
                        id="max-cards-input",
                        type="number",
                        min=1,
                        max=50,
                        value=max_cards
                    )
                ], width=3)
            ], className="mb-3"),
            
            # Pattern cards container
            html.Div(cards, id="pattern-cards-container"),
            
            # Hidden stores
            dcc.Store(id="expanded-cards-store", data=list(self.expanded_cards)),
            dcc.Store(id="comparison-patterns-store", data=self.comparison_patterns),
            dcc.Store(id="pattern-data-store", data=[asdict(p) for p in patterns])
        ])
        
        return layout
    
    # Helper methods
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 0.8:
            return "success"
        elif score >= 0.6:
            return "warning"
        else:
            return "danger"
    
    def _get_accuracy_color(self, accuracy: float) -> str:
        """Get color based on accuracy"""
        if accuracy >= 0.8:
            return "success"
        elif accuracy >= 0.6:
            return "info"
        else:
            return "warning"
    
    def _compute_autocorrelation(self, data: List[float], max_lag: int = 20) -> List[float]:
        """Compute autocorrelation function"""
        data_array = np.array(data)
        mean = np.mean(data_array)
        c0 = np.sum((data_array - mean) ** 2) / len(data_array)
        
        acf = []
        for lag in range(max_lag):
            if lag == 0:
                acf.append(1.0)
            else:
                c_lag = np.sum((data_array[:-lag] - mean) * (data_array[lag:] - mean)) / len(data_array)
                acf.append(c_lag / c0 if c0 > 0 else 0)
        
        return acf
    
    def _compute_power_spectrum(self, data: List[float]) -> Tuple[List[float], List[float]]:
        """Compute power spectrum using FFT"""
        data_array = np.array(data)
        fft = np.fft.fft(data_array)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(data_array))
        
        # Return only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        return positive_freqs.tolist(), positive_power.tolist()
    
    def _compute_rolling_accuracy(self, predictions: List[Dict[str, Any]], 
                                window: int = 10) -> List[float]:
        """Compute rolling accuracy"""
        if len(predictions) < window:
            return []
        
        accuracies = []
        for i in range(window, len(predictions) + 1):
            window_preds = predictions[i-window:i]
            correct = sum(1 for p in window_preds 
                         if np.sign(p['predicted']) == np.sign(p['actual']))
            accuracies.append(correct / window)
        
        return accuracies
    
    def _assess_complexity(self, pattern_info: PatternInfo) -> str:
        """Assess pattern complexity"""
        # Simple heuristic based on statistical properties
        complexity_score = (
            abs(pattern_info.skewness) * 0.3 +
            abs(pattern_info.kurtosis - 3) * 0.3 +
            pattern_info.entropy * 0.4
        )
        
        if complexity_score < 0.3:
            return "Simple"
        elif complexity_score < 0.6:
            return "Moderate"
        else:
            return "Complex"
    
    def _format_last_seen(self, occurrences: List[Dict[str, Any]]) -> str:
        """Format last seen time"""
        if not occurrences:
            return "Never"
        
        latest = max(occurrences, key=lambda x: x['timestamp'])
        time_diff = datetime.now() - latest['timestamp']
        
        if time_diff.days > 0:
            return f"{time_diff.days}d ago"
        elif time_diff.seconds > 3600:
            return f"{time_diff.seconds // 3600}h ago"
        else:
            return f"{time_diff.seconds // 60}m ago"
    
    def _analyze_market_conditions(self, pattern_info: PatternInfo) -> Dict[str, float]:
        """Analyze market conditions during pattern occurrences"""
        # Placeholder analysis - would use actual market data
        return {
            'volatility': np.random.uniform(0.2, 0.8),
            'trend_strength': np.random.uniform(0.3, 0.9),
            'volume_ratio': np.random.uniform(0.8, 1.5)
        }
    
    def _analyze_trading_implications(self, pattern_info: PatternInfo) -> Dict[str, Any]:
        """Analyze trading implications of pattern"""
        insights = []
        
        # Based on pattern statistics
        if pattern_info.trend > 0.02:
            insights.append("Strong upward trend detected")
        elif pattern_info.trend < -0.02:
            insights.append("Strong downward trend detected")
        
        if pattern_info.prediction_accuracy > 0.8:
            insights.append("High prediction reliability")
        
        if pattern_info.total_occurrences > 20:
            insights.append("Frequently occurring pattern")
        
        # Recommendation based on analysis
        if pattern_info.prediction_accuracy > 0.7 and pattern_info.hit_rate > 0.6:
            recommendation = "Consider trading this pattern with appropriate risk management"
        else:
            recommendation = "Monitor pattern but wait for improved metrics before trading"
        
        return {
            'insights': insights,
            'recommendation': recommendation
        }
    
    @staticmethod
    def register_callbacks(app):
        """Register callbacks for pattern cards component"""
        
        # Expand/collapse individual cards
        @app.callback(
            Output({"type": "pattern-collapse", "index": MATCH}, "is_open"),
            Output("expanded-cards-store", "data"),
            Input({"type": "expand-pattern", "index": MATCH}, "n_clicks"),
            State({"type": "pattern-collapse", "index": MATCH}, "is_open"),
            State("expanded-cards-store", "data"),
            prevent_initial_call=True
        )
        def toggle_card_expansion(n_clicks, is_open, expanded_cards):
            """Toggle individual card expansion"""
            if n_clicks:
                ctx = dash.callback_context
                card_index = ctx.triggered[0]["prop_id"].split('"index":')[1].split('}')[0]
                
                expanded_cards = set(expanded_cards or [])
                if is_open:
                    expanded_cards.discard(int(card_index))
                else:
                    expanded_cards.add(int(card_index))
                
                return not is_open, list(expanded_cards)
            return is_open, expanded_cards
        
        # Expand/collapse all cards
        @app.callback(
            [Output({"type": "pattern-collapse", "index": ALL}, "is_open"),
             Output("expanded-cards-store", "data", allow_duplicate=True)],
            [Input("expand-all-patterns", "n_clicks"),
             Input("collapse-all-patterns", "n_clicks")],
            State("pattern-cards-container", "children"),
            prevent_initial_call=True
        )
        def toggle_all_cards(expand_clicks, collapse_clicks, cards):
            """Toggle all cards expansion"""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            num_cards = len(cards) if cards else 0
            
            if trigger == "expand-all-patterns":
                return [True] * num_cards, list(range(num_cards))
            else:  # collapse-all-patterns
                return [False] * num_cards, []
        
        # Export pattern data
        @app.callback(
            Output("download-pattern-data", "data"),
            Input({"type": "export-pattern", "index": ALL}, "n_clicks"),
            State("pattern-data-store", "data"),
            prevent_initial_call=True
        )
        def export_pattern(n_clicks_list, patterns_data):
            """Export individual pattern data"""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks_list):
                raise dash.exceptions.PreventUpdate
            
            # Find which button was clicked
            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks:
                    pattern_data = patterns_data[i]
                    
                    # Convert datetime strings back to datetime objects for formatting
                    export_data = {
                        'pattern_info': pattern_data,
                        'export_timestamp': datetime.now().isoformat(),
                        'format_version': '1.0'
                    }
                    
                    return dict(
                        content=json.dumps(export_data, indent=2),
                        filename=f"pattern_{pattern_data['pattern_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
        
        # Pattern comparison
        @app.callback(
            Output("comparison-patterns-store", "data"),
            Input({"type": "compare-pattern", "index": ALL}, "n_clicks"),
            State("comparison-patterns-store", "data"),
            State("pattern-data-store", "data"),
            prevent_initial_call=True
        )
        def add_to_comparison(n_clicks_list, comparison_patterns, patterns_data):
            """Add pattern to comparison list"""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks_list):
                raise dash.exceptions.PreventUpdate
            
            comparison_patterns = comparison_patterns or []
            
            # Find which button was clicked
            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks:
                    pattern_id = patterns_data[i]['pattern_id']
                    if pattern_id not in comparison_patterns:
                        comparison_patterns.append(pattern_id)
                        if len(comparison_patterns) > 4:  # Limit to 4 patterns
                            comparison_patterns.pop(0)
            
            return comparison_patterns
        
        # Locate pattern on main chart
        @app.callback(
            Output("main-chart-highlight", "data"),
            Input({"type": "locate-pattern", "index": ALL}, "n_clicks"),
            State("pattern-data-store", "data"),
            prevent_initial_call=True
        )
        def locate_on_chart(n_clicks_list, patterns_data):
            """Highlight pattern on main chart"""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks_list):
                raise dash.exceptions.PreventUpdate
            
            # Find which button was clicked
            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks:
                    pattern = patterns_data[i]
                    return {
                        'pattern_id': pattern['pattern_id'],
                        'start_time': pattern['start_time'],
                        'end_time': pattern['end_time'],
                        'ticker': pattern['ticker']
                    }
        
        # View occurrence details
        @app.callback(
            Output("occurrence-detail-modal", "is_open"),
            Output("occurrence-detail-content", "children"),
            Input({"type": "view-occurrence", "index": ALL}, "n_clicks"),
            State("pattern-data-store", "data"),
            prevent_initial_call=True
        )
        def view_occurrence_details(n_clicks_list, patterns_data):
            """Show occurrence details in modal"""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks_list):
                raise dash.exceptions.PreventUpdate
            
            # Parse the clicked occurrence
            trigger_id = ctx.triggered[0]["prop_id"].split('"index":"')[1].split('"')[0]
            pattern_id, occurrence_idx = trigger_id.split('-')
            
            # Find the pattern and occurrence
            pattern = next((p for p in patterns_data if p['pattern_id'] == pattern_id), None)
            if pattern and int(occurrence_idx) < len(pattern['occurrences']):
                occurrence = pattern['occurrences'][int(occurrence_idx)]
                
                # Create detailed view
                content = html.Div([
                    html.H5(f"{pattern['ticker']} - {pattern['pattern_type']}"),
                    html.Hr(),
                    html.P(f"Occurrence Time: {occurrence['timestamp']}"),
                    html.P(f"Quality Score: {occurrence['quality_score']:.2%}"),
                    html.P(f"Outcome: {occurrence.get('outcome', 0):+.2%}"),
                    html.P(f"Duration: {occurrence.get('duration_hours', 0):.1f} hours")
                ])
                
                return True, content
            
            return False, ""


def generate_demo_pattern_info(num_patterns: int = 10) -> List[PatternInfo]:
    """Generate demo pattern information for testing"""
    patterns = []
    
    pattern_types = ['Head and Shoulders', 'Double Top', 'Triangle', 'Flag', 'Wedge']
    tickers = ['BTC-USD', 'ETH-USD', 'SPY', 'AAPL', 'TSLA']
    
    for i in range(num_patterns):
        # Generate pattern data
        length = np.random.randint(30, 100)
        t = np.linspace(0, 2*np.pi, length)
        pattern_data = np.sin(t) * np.exp(-t/10) + np.random.normal(0, 0.05, length)
        pattern_data = (pattern_data - pattern_data.min()) / (pattern_data.max() - pattern_data.min())
        
        # Generate occurrences
        num_occurrences = np.random.randint(5, 30)
        occurrences = []
        for j in range(num_occurrences):
            occ_time = datetime.now() - timedelta(days=np.random.randint(1, 90))
            occurrences.append({
                'timestamp': occ_time,
                'quality_score': np.random.uniform(0.6, 0.95),
                'outcome': np.random.normal(0.02, 0.05),
                'duration_hours': length * np.random.uniform(0.8, 1.2)
            })
        
        # Generate predictions
        num_predictions = np.random.randint(20, 100)
        predictions = []
        for j in range(num_predictions):
            pred_value = np.random.normal(0.02, 0.03)
            actual_value = pred_value + np.random.normal(0, 0.02)
            predictions.append({
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'predicted': pred_value,
                'actual': actual_value
            })
        
        # Calculate statistics
        pattern_array = np.array(pattern_data)
        
        pattern_info = PatternInfo(
            pattern_id=f"pattern_{i}",
            pattern_type=np.random.choice(pattern_types),
            ticker=np.random.choice(tickers),
            discovery_timestamp=datetime.now() - timedelta(days=np.random.randint(1, 30)),
            duration_hours=length,
            start_time=datetime.now() - timedelta(hours=length),
            end_time=datetime.now(),
            pattern_data=pattern_data.tolist(),
            normalized_data=pattern_data.tolist(),
            timestamps=[datetime.now() - timedelta(hours=length-i) for i in range(length)],
            mean=float(np.mean(pattern_array)),
            std=float(np.std(pattern_array)),
            trend=float(np.polyfit(range(length), pattern_array, 1)[0]),
            energy=float(np.sum(pattern_array**2)),
            entropy=float(-np.sum(pattern_array * np.log(pattern_array + 1e-10))),
            skewness=float(np.random.normal(0, 1)),
            kurtosis=float(np.random.normal(3, 1)),
            quality_score=np.random.uniform(0.6, 0.95),
            confidence_score=np.random.uniform(0.7, 0.95),
            significance_score=np.random.uniform(0.6, 0.9),
            stability_score=np.random.uniform(0.5, 0.9),
            robustness_score=np.random.uniform(0.7, 0.95),
            occurrences=occurrences,
            total_occurrences=num_occurrences,
            predictions=predictions,
            prediction_accuracy=np.random.uniform(0.6, 0.9),
            mean_absolute_error=np.random.uniform(0.01, 0.05),
            hit_rate=np.random.uniform(0.5, 0.8),
            metadata={'source': 'demo', 'version': '1.0'}
        )
        
        patterns.append(pattern_info)
    
    return patterns


if __name__ == "__main__":
    # Create demo
    demo_patterns = generate_demo_pattern_info(5)
    cards_component = PatternCards()
    
    # Create a simple dash app for testing
    import dash
    import dash_bootstrap_components as dbc
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = html.Div([
        cards_component.create_pattern_cards_layout(demo_patterns),
        
        # Hidden components for callbacks
        dcc.Download(id="download-pattern-data"),
        html.Div(id="main-chart-highlight"),
        dbc.Modal([
            dbc.ModalHeader("Occurrence Details"),
            dbc.ModalBody(id="occurrence-detail-content"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ml-auto")
            )
        ], id="occurrence-detail-modal", is_open=False)
    ])
    
    # Register callbacks
    PatternCards.register_callbacks(app)
    
    # Run the app
    print("Running pattern cards demo on http://localhost:8050")
    app.run_server(debug=True)
