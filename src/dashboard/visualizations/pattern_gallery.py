"""
Pattern Gallery Component for Financial Wavelet Prediction Dashboard

This module provides an interactive gallery view of discovered patterns with:
- Grid layout showing pattern thumbnails
- Key metrics display (quality, frequency, recency)
- Sorting and filtering capabilities
- Click-to-highlight functionality on main chart
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json


class PatternGallery:
    """Interactive pattern gallery with filtering and sorting capabilities"""
    
    def __init__(self, pattern_data: Optional[Dict[str, Any]] = None):
        """
        Initialize pattern gallery
        
        Args:
            pattern_data: Dictionary containing pattern information
        """
        self.pattern_data = pattern_data or self._generate_demo_patterns()
        self.selected_pattern_id = None
        
    def _generate_demo_patterns(self) -> Dict[str, Any]:
        """Generate demo pattern data for visualization"""
        np.random.seed(42)
        
        patterns = []
        pattern_types = ['Head and Shoulders', 'Double Top', 'Triangle', 
                        'Flag', 'Wedge', 'Channel']
        tickers = ['BTC/USD', 'ETH/USD', 'SPY', 'AAPL']
        
        # Generate 24 demo patterns
        for i in range(24):
            # Generate pattern shape
            length = np.random.randint(20, 50)
            t = np.linspace(0, 1, length)
            
            pattern_type = np.random.choice(pattern_types)
            
            # Create different pattern shapes
            if pattern_type == 'Head and Shoulders':
                y = np.sin(t * np.pi * 3) * np.exp(-t * 2)
                y[len(y)//3:2*len(y)//3] *= 1.5
            elif pattern_type == 'Double Top':
                y = np.sin(t * np.pi * 2) * (1 - t * 0.3)
            elif pattern_type == 'Triangle':
                y = np.sin(t * np.pi * 4) * (1 - t)
            elif pattern_type == 'Flag':
                y = np.sin(t * np.pi * 8) * 0.3 + t * 0.5
            elif pattern_type == 'Wedge':
                y = np.sin(t * np.pi * 6) * (1 - t * 0.7) + t * 0.3
            else:  # Channel
                y = np.sin(t * np.pi * 5) * 0.4 + np.sin(t * np.pi * 0.5)
            
            # Add noise
            y += np.random.normal(0, 0.05, length)
            
            # Normalize
            y = (y - y.min()) / (y.max() - y.min())
            
            # Generate metrics
            quality_score = np.random.uniform(0.6, 0.95)
            occurrence_count = np.random.randint(5, 50)
            
            # Generate time range
            end_time = datetime.now() - timedelta(hours=np.random.randint(1, 168))
            start_time = end_time - timedelta(hours=length)
            
            patterns.append({
                'id': f'pattern_{i}',
                'type': pattern_type,
                'ticker': np.random.choice(tickers),
                'data': y.tolist(),
                'quality_score': quality_score,
                'occurrence_count': occurrence_count,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': length,
                'confidence': np.random.uniform(0.7, 0.95),
                'profit_potential': np.random.uniform(-0.05, 0.15),
                'risk_reward_ratio': np.random.uniform(1.5, 3.5)
            })
        
        return {'patterns': patterns}
    
    def create_pattern_thumbnail(self, pattern: Dict[str, Any], 
                               width: int = 200, height: int = 150) -> go.Figure:
        """
        Create a thumbnail visualization for a single pattern
        
        Args:
            pattern: Pattern data dictionary
            width: Thumbnail width
            height: Thumbnail height
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Pattern line
        y_data = pattern['data']
        x_data = list(range(len(y_data)))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(
                color='#00D9FF',
                width=2
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add quality indicator as background
        quality = pattern['quality_score']
        if quality > 0.8:
            bg_color = 'rgba(0, 255, 0, 0.1)'
        elif quality > 0.6:
            bg_color = 'rgba(255, 255, 0, 0.1)'
        else:
            bg_color = 'rgba(255, 0, 0, 0.1)'
            
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            fillcolor=bg_color,
            layer="below",
            line_width=0,
        )
        
        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode=False
        )
        
        return fig
    
    def create_pattern_card(self, pattern: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Create pattern card data for display
        
        Args:
            pattern: Pattern data
            index: Card index in grid
            
        Returns:
            Card data dictionary
        """
        # Calculate recency
        end_time = datetime.fromisoformat(pattern['end_time'])
        hours_ago = (datetime.now() - end_time).total_seconds() / 3600
        
        if hours_ago < 1:
            recency = f"{int(hours_ago * 60)}m ago"
        elif hours_ago < 24:
            recency = f"{int(hours_ago)}h ago"
        else:
            recency = f"{int(hours_ago / 24)}d ago"
        
        # Format metrics
        quality_pct = f"{pattern['quality_score'] * 100:.0f}%"
        confidence_pct = f"{pattern['confidence'] * 100:.0f}%"
        profit_pct = f"{pattern['profit_potential'] * 100:+.1f}%"
        
        return {
            'id': pattern['id'],
            'index': index,
            'type': pattern['type'],
            'ticker': pattern['ticker'],
            'quality': quality_pct,
            'quality_raw': pattern['quality_score'],
            'occurrences': pattern['occurrence_count'],
            'recency': recency,
            'recency_hours': hours_ago,
            'confidence': confidence_pct,
            'profit': profit_pct,
            'risk_reward': f"{pattern['risk_reward_ratio']:.1f}:1",
            'duration': f"{pattern['duration_hours']}h",
            'thumbnail': self.create_pattern_thumbnail(pattern)
        }
    
    def create_gallery_layout(self, 
                            sort_by: str = 'quality',
                            filter_type: Optional[str] = None,
                            filter_ticker: Optional[str] = None,
                            cards_per_row: int = 4) -> go.Figure:
        """
        Create the main gallery layout with pattern cards
        
        Args:
            sort_by: Sort criterion ('quality', 'frequency', 'recency')
            filter_type: Pattern type filter
            filter_ticker: Ticker filter
            cards_per_row: Number of cards per row
            
        Returns:
            Plotly figure with gallery layout
        """
        # Filter patterns
        patterns = self.pattern_data['patterns']
        
        if filter_type:
            patterns = [p for p in patterns if p['type'] == filter_type]
        
        if filter_ticker:
            patterns = [p for p in patterns if p['ticker'] == filter_ticker]
        
        # Sort patterns
        if sort_by == 'quality':
            patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        elif sort_by == 'frequency':
            patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)
        elif sort_by == 'recency':
            patterns.sort(key=lambda x: x['end_time'], reverse=True)
        
        # Create cards
        cards = [self.create_pattern_card(p, i) for i, p in enumerate(patterns)]
        
        # Calculate grid dimensions
        num_cards = len(cards)
        num_rows = (num_cards + cards_per_row - 1) // cards_per_row
        
        # Create subplots for thumbnails
        fig = make_subplots(
            rows=num_rows,
            cols=cards_per_row,
            subplot_titles=[f"{c['ticker']} - {c['type']}" for c in cards],
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
            specs=[[{'type': 'xy'} for _ in range(cards_per_row)] 
                   for _ in range(num_rows)]
        )
        
        # Add pattern thumbnails
        for i, card in enumerate(cards):
            row = i // cards_per_row + 1
            col = i % cards_per_row + 1
            
            # Get thumbnail traces
            thumbnail = card['thumbnail']
            for trace in thumbnail.data:
                fig.add_trace(
                    trace,
                    row=row,
                    col=col
                )
            
            # Add metrics as annotations
            subplot_ref = f"x{i+1}y{i+1}" if i > 0 else "xy"
            
            # Quality badge
            fig.add_annotation(
                text=f"Q: {card['quality']}",
                xref=f"x{i+1} domain" if i > 0 else "x domain",
                yref=f"y{i+1} domain" if i > 0 else "y domain",
                x=0.05, y=0.95,
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor='green' if card['quality_raw'] > 0.8 else 'orange',
                borderpad=2
            )
            
            # Occurrence count
            fig.add_annotation(
                text=f"×{card['occurrences']}",
                xref=f"x{i+1} domain" if i > 0 else "x domain",
                yref=f"y{i+1} domain" if i > 0 else "y domain",
                x=0.95, y=0.95,
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor='#666',
                borderpad=2
            )
            
            # Recency
            fig.add_annotation(
                text=card['recency'],
                xref=f"x{i+1} domain" if i > 0 else "x domain",
                yref=f"y{i+1} domain" if i > 0 else "y domain",
                x=0.05, y=0.05,
                showarrow=False,
                font=dict(size=9, color='#AAA')
            )
            
            # Risk/Reward
            fig.add_annotation(
                text=card['risk_reward'],
                xref=f"x{i+1} domain" if i > 0 else "x domain",
                yref=f"y{i+1} domain" if i > 0 else "y domain",
                x=0.95, y=0.05,
                showarrow=False,
                font=dict(size=9, color='#AAA')
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Pattern Gallery",
                font=dict(size=20, color='white')
            ),
            showlegend=False,
            height=250 * num_rows,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white'),
            hovermode='closest'
        )
        
        # Update all axes
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        
        return fig
    
    def create_detailed_view(self, pattern_id: str) -> go.Figure:
        """
        Create detailed view of a selected pattern
        
        Args:
            pattern_id: ID of pattern to display
            
        Returns:
            Detailed pattern visualization
        """
        # Find pattern
        pattern = next((p for p in self.pattern_data['patterns'] 
                       if p['id'] == pattern_id), None)
        
        if not pattern:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pattern Shape', 'Quality Metrics', 
                          'Occurrence Timeline', 'Performance Stats'),
            specs=[[{'type': 'xy'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Pattern shape
        y_data = pattern['data']
        x_data = list(range(len(y_data)))
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                line=dict(color='#00D9FF', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.2)',
                name='Pattern'
            ),
            row=1, col=1
        )
        
        # Quality metrics
        metrics = ['Quality', 'Confidence', 'Frequency']
        values = [
            pattern['quality_score'] * 100,
            pattern['confidence'] * 100,
            min(pattern['occurrence_count'] / 50 * 100, 100)
        ]
        colors = ['#00D9FF', '#FF6B6B', '#4ECDC4']
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=[f'{v:.0f}%' for v in values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Occurrence timeline (simulated)
        num_occurrences = min(pattern['occurrence_count'], 20)
        occurrence_times = pd.date_range(
            end=datetime.now(),
            periods=num_occurrences,
            freq='D'
        )
        occurrence_values = np.random.uniform(0.6, 1.0, num_occurrences)
        
        fig.add_trace(
            go.Scatter(
                x=occurrence_times,
                y=occurrence_values,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=occurrence_values,
                    colorscale='Viridis',
                    showscale=False
                ),
                line=dict(color='#666', width=1),
                name='Occurrences'
            ),
            row=2, col=1
        )
        
        # Performance indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta+gauge",
                value=pattern['profit_potential'] * 100,
                delta={'reference': 0, 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [-10, 20]},
                    'bar': {'color': "darkgreen" if pattern['profit_potential'] > 0 else "darkred"},
                    'steps': [
                        {'range': [-10, 0], 'color': "lightgray"},
                        {'range': [0, 20], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                },
                title={'text': "Profit Potential (%)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{pattern['ticker']} - {pattern['type']} Pattern Details",
                font=dict(size=18, color='white')
            ),
            showlegend=False,
            height=600,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white')
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='#333', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#333', row=1, col=1)
        fig.update_xaxes(showgrid=False, row=1, col=2)
        fig.update_yaxes(showgrid=True, gridcolor='#333', row=1, col=2)
        fig.update_xaxes(showgrid=True, gridcolor='#333', row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#333', row=2, col=1)
        
        return fig
    
    def get_pattern_types(self) -> List[str]:
        """Get unique pattern types"""
        return list(set(p['type'] for p in self.pattern_data['patterns']))
    
    def get_tickers(self) -> List[str]:
        """Get unique tickers"""
        return list(set(p['ticker'] for p in self.pattern_data['patterns']))
    
    def get_pattern_for_highlight(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern data for highlighting on main chart
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern data or None
        """
        return next((p for p in self.pattern_data['patterns'] 
                    if p['id'] == pattern_id), None)


def create_pattern_gallery_demo():
    """Create a demo of the pattern gallery"""
    gallery = PatternGallery()
    
    # Create main gallery view
    fig = gallery.create_gallery_layout(sort_by='quality')
    
    # Save to HTML
    fig.write_html('pattern_gallery_demo.html')
    print("Pattern gallery demo saved to pattern_gallery_demo.html")
    
    # Create detailed view for first pattern
    pattern_id = gallery.pattern_data['patterns'][0]['id']
    detail_fig = gallery.create_detailed_view(pattern_id)
    detail_fig.write_html('pattern_detail_demo.html')
    print("Pattern detail demo saved to pattern_detail_demo.html")
    
    return gallery


if __name__ == "__main__":
    create_pattern_gallery_demo()
