"""
Time Series Visualization Component for Pattern Dashboard

This module provides an interactive time series plot that:
- Shows price data for selected tickers
- Highlights discovered patterns with colored regions
- Overlays n+1 predictions with confidence bands
- Includes interactive tooltips with pattern details
- Supports zoom, pan, and time range selection
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import colorsys


class TimeSeriesVisualizer:
    """Interactive time series visualization with pattern highlighting and predictions"""
    
    def __init__(self):
        """Initialize the time series visualizer"""
        # Pattern type color mapping
        self.pattern_colors = {
            'head_shoulders': '#FF6B6B',      # Red
            'double_top': '#4ECDC4',          # Teal
            'double_bottom': '#45B7D1',       # Light Blue
            'triangle': '#96CEB4',            # Green
            'wedge': '#FECA57',               # Yellow
            'flag': '#DDA0DD',                # Plum
            'channel': '#98D8C8',             # Mint
            'support_resistance': '#F7DC6F',   # Light Yellow
            'trend_continuation': '#85C1E2',   # Sky Blue
            'trend_reversal': '#F1948A'       # Light Red
        }
        
        # Default plot configuration
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pattern_analysis',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    
    def create_timeseries_plot(
        self,
        price_data: pd.DataFrame,
        patterns: List[Dict[str, Any]],
        predictions: Optional[Dict[str, Any]] = None,
        selected_tickers: Optional[List[str]] = None,
        show_volume: bool = True,
        height: int = 800
    ) -> go.Figure:
        """
        Create interactive time series plot with patterns and predictions
        
        Args:
            price_data: DataFrame with columns ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            patterns: List of pattern dictionaries with keys:
                     ['pattern_type', 'start_time', 'end_time', 'quality', 'statistics']
            predictions: Dictionary with keys ['timestamps', 'values', 'confidence_lower', 'confidence_upper']
            selected_tickers: List of tickers to display (None for all)
            show_volume: Whether to show volume subplot
            height: Plot height in pixels
            
        Returns:
            Plotly figure object
        """
        # Filter data for selected tickers
        if selected_tickers:
            price_data = price_data[price_data['ticker'].isin(selected_tickers)]
        
        # Create subplots
        rows = 2 if show_volume else 1
        row_heights = [0.7, 0.3] if show_volume else [1.0]
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=('Price & Patterns', 'Volume') if show_volume else ('Price & Patterns',)
        )
        
        # Add price data for each ticker
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('timestamp')
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=ticker_data['timestamp'],
                    open=ticker_data['open'],
                    high=ticker_data['high'],
                    low=ticker_data['low'],
                    close=ticker_data['close'],
                    name=f'{ticker} Price',
                    increasing_line_color='#26A69A',
                    decreasing_line_color='#EF5350'
                ),
                row=1, col=1
            )
            
            # Add volume bars if requested
            if show_volume:
                colors = ['#26A69A' if close >= open else '#EF5350' 
                         for close, open in zip(ticker_data['close'], ticker_data['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=ticker_data['timestamp'],
                        y=ticker_data['volume'],
                        name=f'{ticker} Volume',
                        marker_color=colors,
                        opacity=0.7,
                        hovertext=[f'Volume: {v:,.0f}' for v in ticker_data['volume']]
                    ),
                    row=2, col=1
                )
        
        # Add pattern highlights
        self._add_pattern_highlights(fig, patterns, price_data)
        
        # Add predictions if provided
        if predictions:
            self._add_predictions(fig, predictions)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Pattern Analysis & Predictions',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=height,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='white',
                activecolor='lightblue'
            ),
            type="date"
        )
        
        fig.update_yaxes(
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            title_text="Price",
            row=1, col=1
        )
        
        if show_volume:
            fig.update_yaxes(
                gridcolor='lightgray',
                showgrid=True,
                zeroline=False,
                title_text="Volume",
                row=2, col=1
            )
        
        return fig
    
    def _add_pattern_highlights(
        self,
        fig: go.Figure,
        patterns: List[Dict[str, Any]],
        price_data: pd.DataFrame
    ) -> None:
        """Add pattern highlighting regions to the plot"""
        
        # Get y-axis range for pattern highlighting
        y_min = price_data[['low']].min().min()
        y_max = price_data[['high']].max().max()
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        
        # Sort patterns by quality for layering
        sorted_patterns = sorted(patterns, key=lambda p: p.get('quality', 0))
        
        for pattern in sorted_patterns:
            pattern_type = pattern['pattern_type']
            start_time = pd.to_datetime(pattern['start_time'])
            end_time = pd.to_datetime(pattern['end_time'])
            quality = pattern.get('quality', 0.5)
            stats = pattern.get('statistics', {})
            
            # Get color for pattern type
            base_color = self.pattern_colors.get(pattern_type, '#808080')
            
            # Adjust opacity based on quality
            opacity = 0.1 + (quality * 0.3)  # Range: 0.1 to 0.4
            
            # Create hover text with pattern details
            hover_text = self._create_pattern_hover_text(pattern_type, quality, stats)
            
            # Add shaded region for pattern
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=start_time,
                y0=y_min - y_padding,
                x1=end_time,
                y1=y_max + y_padding,
                fillcolor=base_color,
                opacity=opacity,
                layer="below",
                line=dict(width=0)
            )
            
            # Add pattern label
            mid_time = start_time + (end_time - start_time) / 2
            fig.add_annotation(
                x=mid_time,
                y=y_max + y_padding * 0.5,
                text=f"{pattern_type.replace('_', ' ').title()}",
                showarrow=False,
                font=dict(size=10, color=base_color),
                bgcolor='white',
                bordercolor=base_color,
                borderwidth=1,
                borderpad=2,
                opacity=0.8,
                hovertext=hover_text,
                hoverlabel=dict(
                    bgcolor=base_color,
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            # Add quality indicator
            if quality > 0.7:
                fig.add_annotation(
                    x=mid_time,
                    y=y_max + y_padding * 0.3,
                    text="★" * int(quality * 5),
                    showarrow=False,
                    font=dict(size=12, color='gold'),
                    opacity=0.8
                )
    
    def _add_predictions(
        self,
        fig: go.Figure,
        predictions: Dict[str, Any]
    ) -> None:
        """Add prediction line with confidence bands"""
        
        timestamps = predictions['timestamps']
        values = predictions['values']
        lower_bound = predictions.get('confidence_lower', values)
        upper_bound = predictions.get('confidence_upper', values)
        
        # Add confidence band
        # Convert timestamps to list if it's a pandas object
        if hasattr(timestamps, 'tolist'):
            x_values = timestamps.tolist() + timestamps[::-1].tolist()
        else:
            x_values = list(timestamps) + list(timestamps[::-1])
            
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Confidence Band',
                hoverinfo='skip',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add prediction line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='blue', width=3, dash='dash'),
                marker=dict(size=8, color='blue'),
                hovertemplate=(
                    '<b>Prediction</b><br>' +
                    'Time: %{x}<br>' +
                    'Value: %{y:.2f}<br>' +
                    'Upper: ' + ', '.join([f'{u:.2f}' for u in upper_bound]) + '<br>' +
                    'Lower: ' + ', '.join([f'{l:.2f}' for l in lower_bound]) + '<br>' +
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # Add prediction start marker
        if len(timestamps) > 0:
            fig.add_annotation(
                x=timestamps[0],
                y=values[0],
                text="Prediction Start",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="blue",
                ax=30,
                ay=-30,
                bgcolor="white",
                bordercolor="blue",
                borderwidth=2
            )
    
    def _create_pattern_hover_text(
        self,
        pattern_type: str,
        quality: float,
        statistics: Dict[str, Any]
    ) -> str:
        """Create detailed hover text for patterns"""
        
        lines = [
            f"<b>{pattern_type.replace('_', ' ').title()}</b>",
            f"Quality: {quality:.2%}",
            ""
        ]
        
        # Add statistics if available
        if statistics:
            if 'confidence' in statistics:
                lines.append(f"Confidence: {statistics['confidence']:.2%}")
            if 'strength' in statistics:
                lines.append(f"Strength: {statistics['strength']:.2f}")
            if 'duration' in statistics:
                lines.append(f"Duration: {statistics['duration']} periods")
            if 'price_change' in statistics:
                lines.append(f"Price Change: {statistics['price_change']:.2%}")
            if 'volume_ratio' in statistics:
                lines.append(f"Volume Ratio: {statistics['volume_ratio']:.2f}")
            if 'success_rate' in statistics:
                lines.append(f"Historical Success: {statistics['success_rate']:.1%}")
        
        return "<br>".join(lines)
    
    def create_pattern_focus_plot(
        self,
        price_data: pd.DataFrame,
        pattern: Dict[str, Any],
        context_periods: int = 20
    ) -> go.Figure:
        """
        Create a focused view of a specific pattern
        
        Args:
            price_data: Price data DataFrame
            pattern: Pattern dictionary
            context_periods: Number of periods before/after pattern to show
            
        Returns:
            Plotly figure focused on the pattern
        """
        # Extract pattern time range
        start_time = pd.to_datetime(pattern['start_time'])
        end_time = pd.to_datetime(pattern['end_time'])
        
        # Add context periods
        time_delta = price_data['timestamp'].diff().median()
        context_start = start_time - (time_delta * context_periods)
        context_end = end_time + (time_delta * context_periods)
        
        # Filter data for the focused range
        mask = (price_data['timestamp'] >= context_start) & (price_data['timestamp'] <= context_end)
        focused_data = price_data[mask].copy()
        
        # Create the plot
        fig = self.create_timeseries_plot(
            focused_data,
            [pattern],
            show_volume=True,
            height=600
        )
        
        # Update title
        pattern_type = pattern['pattern_type'].replace('_', ' ').title()
        fig.update_layout(
            title={
                'text': f'Pattern Focus: {pattern_type}',
                'x': 0.5,
                'xanchor': 'center'
            }
        )
        
        # Add pattern-specific annotations
        self._add_pattern_annotations(fig, pattern, focused_data)
        
        return fig
    
    def _add_pattern_annotations(
        self,
        fig: go.Figure,
        pattern: Dict[str, Any],
        price_data: pd.DataFrame
    ) -> None:
        """Add detailed annotations for a focused pattern view"""
        
        pattern_type = pattern['pattern_type']
        stats = pattern.get('statistics', {})
        
        # Pattern-specific annotations
        if pattern_type == 'head_shoulders':
            # Mark head and shoulders
            if 'key_points' in stats:
                points = stats['key_points']
                for point_name, point_data in points.items():
                    fig.add_annotation(
                        x=point_data['time'],
                        y=point_data['price'],
                        text=point_name.title(),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='red',
                        bgcolor='white',
                        bordercolor='red'
                    )
        
        elif pattern_type in ['double_top', 'double_bottom']:
            # Mark peaks/troughs
            if 'peaks' in stats:
                for i, peak in enumerate(stats['peaks']):
                    fig.add_annotation(
                        x=peak['time'],
                        y=peak['price'],
                        text=f"Peak {i+1}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='blue',
                        ay=-30 if pattern_type == 'double_top' else 30
                    )
        
        elif pattern_type in ['support_resistance']:
            # Draw support/resistance lines
            if 'levels' in stats:
                for level in stats['levels']:
                    fig.add_hline(
                        y=level['price'],
                        line_dash="dash",
                        line_color="purple",
                        annotation_text=f"{level['type']}: {level['price']:.2f}",
                        annotation_position="right"
                    )
    
    def create_pattern_comparison_plot(
        self,
        patterns: List[Dict[str, Any]],
        metric: str = 'quality'
    ) -> go.Figure:
        """
        Create a comparison plot of patterns by a specific metric
        
        Args:
            patterns: List of pattern dictionaries
            metric: Metric to compare ('quality', 'duration', 'price_change', etc.)
            
        Returns:
            Plotly figure with pattern comparison
        """
        # Extract data for comparison
        pattern_types = []
        values = []
        colors = []
        
        for pattern in patterns:
            pattern_type = pattern['pattern_type']
            pattern_types.append(pattern_type.replace('_', ' ').title())
            
            if metric == 'quality':
                values.append(pattern.get('quality', 0))
            elif metric in pattern.get('statistics', {}):
                values.append(pattern['statistics'][metric])
            else:
                values.append(0)
            
            colors.append(self.pattern_colors.get(pattern_type, '#808080'))
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=pattern_types,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
                hovertemplate='%{x}<br>%{text}<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f'Pattern Comparison by {metric.replace("_", " ").title()}',
            xaxis_title='Pattern Type',
            yaxis_title=metric.replace('_', ' ').title(),
            showlegend=False,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        
        return fig


def create_timeseries_component(
    price_data: pd.DataFrame,
    patterns: List[Dict[str, Any]],
    predictions: Optional[Dict[str, Any]] = None,
    selected_tickers: Optional[List[str]] = None,
    show_volume: bool = True,
    height: int = 800
) -> go.Figure:
    """
    Convenience function to create time series visualization
    
    Args:
        price_data: Price data DataFrame
        patterns: List of patterns to highlight
        predictions: Prediction data
        selected_tickers: Tickers to display
        show_volume: Whether to show volume subplot
        height: Plot height
        
    Returns:
        Plotly figure
    """
    visualizer = TimeSeriesVisualizer()
    return visualizer.create_timeseries_plot(
        price_data=price_data,
        patterns=patterns,
        predictions=predictions,
        selected_tickers=selected_tickers,
        show_volume=show_volume,
        height=height
    )


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    import random
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1H')
    price_data = []
    
    for ticker in ['BTC', 'ETH']:
        base_price = 40000 if ticker == 'BTC' else 2500
        prices = []
        
        for i, date in enumerate(dates):
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + random.gauss(0, 0.01))
            
            prices.append(price)
            
            price_data.append({
                'timestamp': date,
                'ticker': ticker,
                'open': price * (1 + random.uniform(-0.005, 0.005)),
                'high': price * (1 + random.uniform(0, 0.01)),
                'low': price * (1 - random.uniform(0, 0.01)),
                'close': price,
                'volume': random.randint(1000000, 5000000)
            })
    
    price_df = pd.DataFrame(price_data)
    
    # Create sample patterns
    patterns = [
        {
            'pattern_type': 'head_shoulders',
            'start_time': '2024-01-10',
            'end_time': '2024-01-20',
            'quality': 0.85,
            'statistics': {
                'confidence': 0.92,
                'strength': 2.5,
                'duration': 240,
                'price_change': -0.05,
                'success_rate': 0.75
            }
        },
        {
            'pattern_type': 'double_bottom',
            'start_time': '2024-02-01',
            'end_time': '2024-02-10',
            'quality': 0.72,
            'statistics': {
                'confidence': 0.80,
                'strength': 1.8,
                'duration': 216,
                'price_change': 0.08,
                'success_rate': 0.68
            }
        }
    ]
    
    # Create sample predictions
    pred_dates = pd.date_range(start='2024-03-01', periods=24, freq='1H')
    pred_values = [price_df[price_df['ticker'] == 'BTC']['close'].iloc[-1]]
    
    for i in range(1, 24):
        pred_values.append(pred_values[-1] * (1 + random.gauss(0.001, 0.005)))
    
    predictions = {
        'timestamps': pred_dates,
        'values': pred_values,
        'confidence_lower': [v * 0.98 for v in pred_values],
        'confidence_upper': [v * 1.02 for v in pred_values]
    }
    
    # Create visualizer and test plots
    visualizer = TimeSeriesVisualizer()
    
    # Test main plot
    fig = visualizer.create_timeseries_plot(
        price_data=price_df,
        patterns=patterns,
        predictions=predictions,
        selected_tickers=['BTC'],
        show_volume=True
    )
    
    print("Time series visualization created successfully!")
    print(f"Total traces: {len(fig.data)}")
    print(f"Pattern highlights: {len([s for s in fig.layout.shapes if s.type == 'rect'])}")
