"""
Shapelet Visualization Module

This module provides visualization components for discovered shapelets,
including color-coded overlays on time series data and shapelet library displays.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import colorsys

from ...shapelet_discovery import ShapeletDiscoverer, Shapelet


class ShapeletVisualizer:
    """
    Visualizes discovered shapelets with color-coded overlays on time series data.
    """
    
    def __init__(self):
        self.color_map = {}  # Maps SAX labels to colors
        self.shapelet_discoverer = ShapeletDiscoverer()
    
    def _get_stat(self, shapelet, key, default=0):
        """Safely get a statistic from shapelet"""
        if hasattr(shapelet, 'statistics'):
            stats = shapelet.statistics
            if isinstance(stats, dict):
                return stats.get(key, default)
            # Handle case where stats might be an Annotation or other object
            elif hasattr(stats, '__dict__'):
                # Try to access as attribute
                return getattr(stats, key, default)
            elif hasattr(stats, 'get') and callable(getattr(stats, 'get')):
                try:
                    return stats.get(key, default)
                except:
                    return default
        return default
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig
        
    def _get_color_for_sax(self, sax_label: str) -> str:
        """
        Get a consistent color for a SAX label.
        Uses HSV color space to generate distinct colors.
        """
        if sax_label not in self.color_map:
            # Generate color based on hash of SAX label
            hash_val = hash(sax_label)
            hue = (hash_val % 360) / 360.0
            saturation = 0.7 + (hash_val % 30) / 100.0  # 0.7-1.0
            value = 0.8 + (hash_val % 20) / 100.0  # 0.8-1.0
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
            self.color_map[sax_label] = color
            
        return self.color_map[sax_label]
    
    def create_shapelet_overlay_visualization(self,
                                            data: pd.DataFrame,
                                            discovered_shapelets: List[Shapelet],
                                            price_col: str = 'close',
                                            title: str = "Time Series with Shapelet Overlays") -> go.Figure:
        """
        Create a time series visualization with color-coded shapelet overlays.
        
        Args:
            data: Time series data
            discovered_shapelets: List of discovered shapelets
            price_col: Column name for price data
            title: Chart title
            
        Returns:
            Plotly figure with shapelet overlays
        """
        fig = go.Figure()
        
        # Plot base time series
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['timestamp'])
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=data[price_col],
            mode='lines',
            name='Price',
            line=dict(color='white', width=2),
            opacity=0.8
        ))
        
        # Find shapelet matches in the data
        prices = data[price_col].values
        matches = self.shapelet_discoverer.find_shapelet_matches(prices, threshold=0.85)
        
        # Group matches by SAX label for legend
        sax_groups = {}
        for shapelet, similarity, position in matches:
            if shapelet.sax_label not in sax_groups:
                sax_groups[shapelet.sax_label] = []
            sax_groups[shapelet.sax_label].append((shapelet, similarity, position))
        
        # Plot overlays for each SAX group
        for sax_label, group_matches in sax_groups.items():
            color = self._get_color_for_sax(sax_label)
            
            # Add first match to legend
            first_match = group_matches[0]
            shapelet, similarity, position = first_match
            
            # Get time range for this match
            start_idx = position
            end_idx = position + shapelet.length
            
            if end_idx <= len(timestamps):
                x_range = timestamps[start_idx:end_idx]
                y_range = data[price_col].iloc[start_idx:end_idx]
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'SAX: {sax_label}',
                    line=dict(color=color, width=3),
                    opacity=0.7,
                    legendgroup=sax_label,
                    hovertemplate=f'SAX: {sax_label}<br>Similarity: {similarity:.2f}<br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>'
                ))
                
                # Add remaining matches without legend entries
                for shapelet, similarity, position in group_matches[1:]:
                    start_idx = position
                    end_idx = position + shapelet.length
                    
                    if end_idx <= len(timestamps):
                        x_range = timestamps[start_idx:end_idx]
                        y_range = data[price_col].iloc[start_idx:end_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode='lines',
                            line=dict(color=color, width=3),
                            opacity=0.7,
                            showlegend=False,
                            legendgroup=sax_label,
                            hovertemplate=f'SAX: {sax_label}<br>Similarity: {similarity:.2f}<br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>'
                        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        return fig
    
    def create_shapelet_library_visualization(self,
                                            shapelets: List[Shapelet],
                                            max_display: int = 20) -> go.Figure:
        """
        Create a visualization of the shapelet library showing individual shapelets.
        
        Args:
            shapelets: List of shapelets to display
            max_display: Maximum number of shapelets to display
            
        Returns:
            Plotly figure showing shapelet library
        """
        if not shapelets:
            return self._create_empty_figure("No shapelets to display")
            
        # Sort shapelets alphabetically by SAX label
        sorted_shapelets = sorted(shapelets, 
                                key=lambda s: s.sax_label)[:max_display]
        
        # Create single column layout
        n_shapelets = len(sorted_shapelets)
        
        # Create subplots with custom titles
        subplot_titles = []
        for s in sorted_shapelets:
            # Create bold title with SAX label
            title = f"<b>{s.sax_label}</b>"
            subplot_titles.append(title)
        
        fig = make_subplots(
            rows=n_shapelets, 
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03,
            row_heights=[1.0/n_shapelets] * n_shapelets
        )
        
        # Plot each shapelet
        for idx, shapelet in enumerate(sorted_shapelets):
            row = idx + 1
            
            color = self._get_color_for_sax(shapelet.sax_label)
            
            # Normalize shapelet data for display
            shapelet_data = np.array(shapelet.raw_data)
            # Normalize to [0, 1] range for consistent display
            if len(shapelet_data) > 0:
                data_min = np.min(shapelet_data)
                data_max = np.max(shapelet_data)
                if data_max > data_min:
                    shapelet_data = (shapelet_data - data_min) / (data_max - data_min)
            
            x = np.arange(len(shapelet_data))
            
            # Safe statistics access
            freq = self._get_stat(shapelet, 'frequency', 0)
            avg_return = self._get_stat(shapelet, 'avg_return_after', 0)
            win_rate = self._get_stat(shapelet, 'win_rate', 0)
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=shapelet_data,
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>SAX: {shapelet.sax_label}</b><br>'
                        f'Ticker: {shapelet.ticker}<br>'
                        f'Timeframe: {shapelet.timeframe}<br>'
                        f'Frequency: {freq}<br>'
                        f'Avg Return: {avg_return:.2%}<br>'
                        f'Win Rate: {win_rate:.2%}<extra></extra>'
                    )
                ),
                row=row, col=1
            )
            
            # Add statistics annotation on the right side
            fig.add_annotation(
                text=f"n={freq} | μ={avg_return:.1%}",
                xref=f"x{idx+1}",
                yref=f"y{idx+1}",
                x=len(shapelet_data) * 1.05,
                y=0.5,
                showarrow=False,
                font=dict(size=9, color='gray'),
                xanchor='left'
            )
        
        # Update layout for clean appearance
        fig.update_layout(
            title=dict(
                text="<b>Shapelet Library</b>",
                font=dict(size=16)
            ),
            template="plotly_white",
            height=max(600, 80 * n_shapelets),  # Dynamic height based on number of shapelets
            showlegend=False,
            margin=dict(l=20, r=100, t=60, b=20),  # Right margin for annotations
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Update all axes to be minimal
        for i in range(1, n_shapelets + 1):
            fig.update_xaxes(
                showticklabels=False, 
                showgrid=False,
                zeroline=False,
                row=i, col=1
            )
            fig.update_yaxes(
                showticklabels=False, 
                showgrid=False,
                zeroline=False,
                row=i, col=1
            )
            
        # Update subplot title formatting
        for annotation in fig['layout']['annotations']:
            if annotation.get('text', '').startswith('<b>'):
                annotation['font'] = dict(size=12, color='black')
                annotation['xanchor'] = 'left'
                annotation['x'] = 0
        
        return fig
    
    def create_shapelet_distribution_visualization(self, shapelets: List[Shapelet]) -> go.Figure:
        """
        Create visualizations showing the distribution of shapelets.
        
        Args:
            shapelets: List of shapelets
            
        Returns:
            Plotly figure with distribution charts
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "SAX Label Distribution (Top 20)",
                "Shapelet Length Distribution",
                "Average Return Distribution",
                "Frequency vs Win Rate"
            ],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. SAX Label Distribution
        sax_counts = {}
        for s in shapelets:
            sax_counts[s.sax_label] = sax_counts.get(s.sax_label, 0) + 1
        
        top_sax = sorted(sax_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        sax_labels = [item[0] for item in top_sax]
        sax_frequencies = [item[1] for item in top_sax]
        
        # Color bars by SAX label
        colors = [self._get_color_for_sax(label) for label in sax_labels]
        
        fig.add_trace(
            go.Bar(
                x=sax_labels,
                y=sax_frequencies,
                marker_color=colors,
                hovertemplate='SAX: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Shapelet Length Distribution
        lengths = [s.length for s in shapelets]
        
        fig.add_trace(
            go.Histogram(
                x=lengths,
                nbinsx=20,
                marker_color='lightblue',
                hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Average Return Distribution
        returns = [self._get_stat(s, 'avg_return_after', 0) * 100 for s in shapelets]  # Convert to percentage
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                marker_color='lightgreen',
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Frequency vs Win Rate scatter
        frequencies = [self._get_stat(s, 'frequency', 0) for s in shapelets]
        win_rates = [self._get_stat(s, 'win_rate', 0) * 100 for s in shapelets]  # Convert to percentage
        sax_labels_all = [s.sax_label for s in shapelets]
        colors_all = [self._get_color_for_sax(label) for label in sax_labels_all]
        
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=win_rates,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors_all,
                    opacity=0.6
                ),
                text=sax_labels_all,
                hovertemplate='SAX: %{text}<br>Frequency: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add 50% win rate line
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                     annotation_text="50% Win Rate", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Shapelet Distribution Analysis",
            template="plotly_dark",
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="SAX Label", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Shapelet Length", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Average Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        return fig
    
    def create_shapelet_timeline_visualization(self,
                                             data: pd.DataFrame,
                                             shapelets: List[Shapelet],
                                             ticker: str,
                                             price_col: str = 'close') -> go.Figure:
        """
        Create a timeline visualization showing when different shapelets occur.
        
        Args:
            data: Time series data
            shapelets: List of shapelets
            ticker: Ticker to filter shapelets
            price_col: Column name for price data
            
        Returns:
            Plotly figure with timeline visualization
        """
        # Filter shapelets for this ticker
        ticker_shapelets = [s for s in shapelets if s.ticker == ticker]
        
        if not ticker_shapelets:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No shapelets found for {ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(template="plotly_dark", height=400)
            return fig
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Price with Shapelet Overlays", "Shapelet Timeline"]
        )
        
        # Plot price data
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['timestamp'])
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data[price_col],
                mode='lines',
                name='Price',
                line=dict(color='white', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Find and plot shapelet matches
        prices = data[price_col].values
        self.shapelet_discoverer.shapelet_library = {s.shapelet_id: s for s in ticker_shapelets}
        self.shapelet_discoverer.sax_to_shapelets = {}
        for s in ticker_shapelets:
            if s.sax_label not in self.shapelet_discoverer.sax_to_shapelets:
                self.shapelet_discoverer.sax_to_shapelets[s.sax_label] = []
            self.shapelet_discoverer.sax_to_shapelets[s.sax_label].append(s.shapelet_id)
        
        matches = self.shapelet_discoverer.find_shapelet_matches(prices, threshold=0.85)
        
        # Create timeline data
        timeline_data = []
        
        for shapelet, similarity, position in matches:
            if position + shapelet.length <= len(timestamps):
                start_time = timestamps[position]
                end_time = timestamps[position + shapelet.length - 1]
                color = self._get_color_for_sax(shapelet.sax_label)
                
                # Add overlay on price chart
                x_range = timestamps[position:position + shapelet.length]
                y_range = data[price_col].iloc[position:position + shapelet.length]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        line=dict(color=color, width=3),
                        opacity=0.7,
                        showlegend=False,
                        hovertemplate=f'SAX: {shapelet.sax_label}<br>Similarity: {similarity:.2f}<br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add to timeline
                timeline_data.append({
                    'start': start_time,
                    'end': end_time,
                    'sax': shapelet.sax_label,
                    'color': color,
                    'y_pos': hash(shapelet.sax_label) % 10  # Distribute on y-axis
                })
        
        # Plot timeline
        for event in timeline_data:
            fig.add_trace(
                go.Scatter(
                    x=[event['start'], event['end']],
                    y=[event['y_pos'], event['y_pos']],
                    mode='lines+markers',
                    line=dict(color=event['color'], width=4),
                    marker=dict(size=8),
                    showlegend=False,
                    hovertemplate=f"SAX: {event['sax']}<br>Start: %{{x[0]}}<br>End: %{{x[1]}}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Shapelet Timeline for {ticker}",
            template="plotly_dark",
            height=800,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="SAX Pattern", showticklabels=False, row=2, col=1)
        
        return fig
    
    def create_shapelet_comparison_matrix(self, shapelets: List[Shapelet], max_display: int = 10) -> go.Figure:
        """
        Create a matrix visualization comparing different shapelets.
        
        Args:
            shapelets: List of shapelets to compare
            max_display: Maximum number of shapelets to display
            
        Returns:
            Plotly figure with comparison matrix
        """
        # Select top shapelets by frequency
        sorted_shapelets = sorted(shapelets, 
                                key=lambda s: self._get_stat(s, 'frequency', 0), 
                                reverse=True)[:max_display]
        
        n = len(sorted_shapelets)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compare SAX strings (simple similarity)
                    sax1 = sorted_shapelets[i].sax_label
                    sax2 = sorted_shapelets[j].sax_label
                    
                    # Pad to same length
                    max_len = max(len(sax1), len(sax2))
                    sax1_padded = sax1.ljust(max_len, sax1[-1] if sax1 else 'a')
                    sax2_padded = sax2.ljust(max_len, sax2[-1] if sax2 else 'a')
                    
                    # Calculate similarity
                    matches = sum(1 for a, b in zip(sax1_padded, sax2_padded) if a == b)
                    similarity_matrix[i, j] = matches / max_len
        
        # Create heatmap
        labels = [s.sax_label for s in sorted_shapelets]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='SAX 1: %{y}<br>SAX 2: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Shapelet Similarity Matrix",
            xaxis_title="SAX Label",
            yaxis_title="SAX Label",
            template="plotly_dark",
            height=600,
            width=700
        )
        
        return fig
