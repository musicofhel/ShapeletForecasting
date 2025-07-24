"""
Interactive Wavelet Scalogram Visualization Module

This module provides an interactive scalogram visualization for wavelet analysis,
featuring:
- Heatmap of wavelet coefficients
- Time-scale representation
- Ridge detection overlay
- Synchronized highlighting with main time series
- Click interactions for time period selection
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import pywt
from scipy import signal
from scipy.ndimage import maximum_filter
import warnings
warnings.filterwarnings('ignore')


class ScalogramVisualizer:
    """
    Interactive wavelet scalogram visualization with ridge detection
    and time series synchronization.
    """
    
    def __init__(self, wavelet: str = 'morl', sampling_rate: float = 1.0):
        """
        Initialize the scalogram visualizer.
        
        Args:
            wavelet: Wavelet type for CWT (default: 'morl' - Morlet)
            sampling_rate: Sampling rate of the time series
        """
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.coefficients = None
        self.scales = None
        self.frequencies = None
        self.time_points = None
        self.ridges = None
        
    def compute_cwt(self, 
                    data: np.ndarray, 
                    scales: Optional[np.ndarray] = None,
                    min_scale: float = 1,
                    max_scale: float = 128,
                    num_scales: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute continuous wavelet transform.
        
        Args:
            data: Input time series data
            scales: Custom scales array (optional)
            min_scale: Minimum scale for automatic scale generation
            max_scale: Maximum scale for automatic scale generation
            num_scales: Number of scales for automatic generation
            
        Returns:
            Dictionary containing coefficients, scales, and frequencies
        """
        # Generate scales if not provided
        if scales is None:
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
        
        # Compute CWT
        coefficients, frequencies = pywt.cwt(data, scales, self.wavelet, 
                                           sampling_period=1/self.sampling_rate)
        
        # Store results
        self.coefficients = coefficients
        self.scales = scales
        self.frequencies = frequencies
        self.time_points = np.arange(len(data))
        
        return {
            'coefficients': coefficients,
            'scales': scales,
            'frequencies': frequencies,
            'time_points': self.time_points
        }
    
    def detect_ridges(self, 
                      min_length: int = 10,
                      min_snr: float = 1.0,
                      gap_threshold: int = 3) -> List[Dict[str, Any]]:
        """
        Detect ridges in the scalogram using local maxima tracking.
        
        Args:
            min_length: Minimum ridge length to consider
            min_snr: Minimum signal-to-noise ratio for ridge points
            gap_threshold: Maximum gap allowed in ridge continuity
            
        Returns:
            List of ridge dictionaries with properties
        """
        if self.coefficients is None:
            raise ValueError("Must compute CWT before detecting ridges")
        
        # Get magnitude of coefficients
        magnitude = np.abs(self.coefficients)
        
        # Find local maxima at each time point
        ridges = []
        
        for scale_idx in range(len(self.scales)):
            # Find peaks in this scale
            peaks, properties = signal.find_peaks(magnitude[scale_idx, :], 
                                                prominence=min_snr * np.std(magnitude[scale_idx, :]))
            
            if len(peaks) < min_length:
                continue
                
            # Group consecutive peaks into ridges
            current_ridge = []
            for i, peak in enumerate(peaks):
                if not current_ridge:
                    current_ridge = [peak]
                elif peak - current_ridge[-1] <= gap_threshold:
                    current_ridge.append(peak)
                else:
                    # Save current ridge if long enough
                    if len(current_ridge) >= min_length:
                        ridge_data = {
                            'scale_idx': scale_idx,
                            'scale': self.scales[scale_idx],
                            'frequency': self.frequencies[scale_idx],
                            'time_indices': current_ridge,
                            'strength': np.mean([magnitude[scale_idx, t] for t in current_ridge])
                        }
                        ridges.append(ridge_data)
                    current_ridge = [peak]
            
            # Don't forget the last ridge
            if len(current_ridge) >= min_length:
                ridge_data = {
                    'scale_idx': scale_idx,
                    'scale': self.scales[scale_idx],
                    'frequency': self.frequencies[scale_idx],
                    'time_indices': current_ridge,
                    'strength': np.mean([magnitude[scale_idx, t] for t in current_ridge])
                }
                ridges.append(ridge_data)
        
        # Sort ridges by strength
        ridges.sort(key=lambda x: x['strength'], reverse=True)
        self.ridges = ridges
        
        return ridges
    
    def create_scalogram_plot(self,
                            data: np.ndarray,
                            time_index: Optional[pd.DatetimeIndex] = None,
                            show_ridges: bool = True,
                            colorscale: str = 'Viridis',
                            height: int = 800) -> go.Figure:
        """
        Create interactive scalogram visualization with time series.
        
        Args:
            data: Original time series data
            time_index: DateTime index for x-axis (optional)
            show_ridges: Whether to overlay ridge detection
            colorscale: Plotly colorscale name
            height: Total figure height
            
        Returns:
            Plotly figure with scalogram and time series
        """
        # Compute CWT if not already done
        if self.coefficients is None:
            self.compute_cwt(data)
        
        # Detect ridges if requested and not already done
        if show_ridges and self.ridges is None:
            self.detect_ridges()
        
        # Create time axis
        if time_index is not None:
            x_axis = time_index
        else:
            x_axis = self.time_points
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Wavelet Scalogram', 'Time Series')
        )
        
        # Add scalogram heatmap
        magnitude = np.abs(self.coefficients)
        
        # Use log scale for better visualization
        z_data = np.log10(magnitude + 1e-10)
        
        heatmap = go.Heatmap(
            x=x_axis,
            y=self.frequencies,
            z=z_data,
            colorscale=colorscale,
            colorbar=dict(
                title="Log10(Magnitude)",
                titleside="right",
                tickmode="linear",
                tick0=-2,
                dtick=1,
                len=0.65,
                y=0.85
            ),
            hovertemplate='Time: %{x}<br>Frequency: %{y:.3f}<br>Magnitude: %{z:.2f}<extra></extra>',
            name='Scalogram'
        )
        
        fig.add_trace(heatmap, row=1, col=1)
        
        # Add ridge overlays if requested
        if show_ridges and self.ridges:
            for i, ridge in enumerate(self.ridges[:10]):  # Show top 10 ridges
                ridge_x = [x_axis[t] if time_index is not None else t 
                          for t in ridge['time_indices']]
                ridge_y = [ridge['frequency']] * len(ridge['time_indices'])
                
                fig.add_trace(
                    go.Scatter(
                        x=ridge_x,
                        y=ridge_y,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='red',
                            symbol='circle-open',
                            line=dict(width=2)
                        ),
                        name=f'Ridge {i+1}',
                        hovertemplate=f'Ridge {i+1}<br>Frequency: {ridge["frequency"]:.3f}<br>Strength: {ridge["strength"]:.2f}<extra></extra>',
                        showlegend=i < 3  # Only show legend for top 3
                    ),
                    row=1, col=1
                )
        
        # Add time series
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=data,
                mode='lines',
                name='Time Series',
                line=dict(color='blue', width=1),
                hovertemplate='Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(
            title_text="Frequency (Hz)",
            type="log",
            row=1, col=1
        )
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        fig.update_layout(
            height=height,
            title={
                'text': 'Interactive Wavelet Scalogram Analysis',
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='x unified',
            dragmode='pan',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.05),
            row=2, col=1
        )
        
        return fig
    
    def create_3d_scalogram(self,
                           data: np.ndarray,
                           time_index: Optional[pd.DatetimeIndex] = None,
                           colorscale: str = 'Viridis') -> go.Figure:
        """
        Create 3D surface plot of the scalogram.
        
        Args:
            data: Original time series data
            time_index: DateTime index for x-axis (optional)
            colorscale: Plotly colorscale name
            
        Returns:
            Plotly 3D figure
        """
        # Compute CWT if not already done
        if self.coefficients is None:
            self.compute_cwt(data)
        
        # Create meshgrid for 3D plot
        if time_index is not None:
            # Convert datetime to numeric for 3D plot
            x_numeric = np.arange(len(time_index))
            x_labels = [str(t) for t in time_index[::max(1, len(time_index)//10)]]
        else:
            x_numeric = self.time_points
            x_labels = None
        
        X, Y = np.meshgrid(x_numeric, np.log10(self.frequencies))
        Z = np.log10(np.abs(self.coefficients) + 1e-10)
        
        # Create 3D surface
        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                name='Scalogram',
                hovertemplate='Time: %{x}<br>Log(Frequency): %{y:.2f}<br>Log(Magnitude): %{z:.2f}<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title='3D Wavelet Scalogram',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Log(Frequency)',
                zaxis_title='Log(Magnitude)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        if x_labels:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=np.linspace(0, len(time_index)-1, min(10, len(time_index))),
                        ticktext=x_labels
                    )
                )
            )
        
        return fig
    
    def create_ridge_analysis_plot(self,
                                 data: np.ndarray,
                                 time_index: Optional[pd.DatetimeIndex] = None,
                                 top_n_ridges: int = 5) -> go.Figure:
        """
        Create detailed ridge analysis plot.
        
        Args:
            data: Original time series data
            time_index: DateTime index for x-axis (optional)
            top_n_ridges: Number of top ridges to analyze
            
        Returns:
            Plotly figure with ridge analysis
        """
        # Ensure ridges are detected
        if self.ridges is None:
            self.detect_ridges()
        
        # Create subplots for each ridge
        n_ridges = min(top_n_ridges, len(self.ridges))
        
        fig = make_subplots(
            rows=n_ridges + 1, cols=1,
            row_heights=[0.3] + [0.7/n_ridges]*n_ridges,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=['Original Signal'] + [f'Ridge {i+1} (f={r["frequency"]:.3f} Hz)' 
                                                 for i, r in enumerate(self.ridges[:n_ridges])]
        )
        
        # Create time axis
        if time_index is not None:
            x_axis = time_index
        else:
            x_axis = self.time_points
        
        # Add original signal
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=data,
                mode='lines',
                name='Original',
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
        
        # Add ridge components
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, ridge in enumerate(self.ridges[:n_ridges]):
            # Extract ridge component
            ridge_component = np.zeros_like(data)
            scale_idx = ridge['scale_idx']
            
            for t in ridge['time_indices']:
                if t < len(data):
                    # Reconstruct using inverse CWT approximation
                    ridge_component[t] = np.real(self.coefficients[scale_idx, t])
            
            # Smooth the component
            from scipy.ndimage import gaussian_filter1d
            ridge_component = gaussian_filter1d(ridge_component, sigma=2)
            
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=ridge_component,
                    mode='lines',
                    name=f'Ridge {i+1}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=i+2, col=1
            )
            
            # Highlight ridge regions
            for t_start in ridge['time_indices']:
                if t_start < len(x_axis) - 1:
                    fig.add_vrect(
                        x0=x_axis[t_start] if time_index is not None else t_start,
                        x1=x_axis[t_start+1] if time_index is not None else t_start+1,
                        fillcolor=colors[i % len(colors)],
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        row=i+2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=200 + 150*n_ridges,
            title='Wavelet Ridge Analysis',
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=n_ridges+1, col=1)
        
        for i in range(n_ridges + 1):
            fig.update_yaxes(title_text="Amplitude", row=i+1, col=1)
        
        return fig
    
    def get_time_scale_features(self, time_point: int) -> Dict[str, Any]:
        """
        Extract features at a specific time point across all scales.
        
        Args:
            time_point: Time index to analyze
            
        Returns:
            Dictionary of features at the time point
        """
        if self.coefficients is None:
            raise ValueError("Must compute CWT before extracting features")
        
        # Get coefficients at this time point
        time_coeffs = self.coefficients[:, time_point]
        magnitude = np.abs(time_coeffs)
        phase = np.angle(time_coeffs)
        
        # Find dominant scale
        dominant_scale_idx = np.argmax(magnitude)
        
        # Extract features
        features = {
            'time_index': time_point,
            'dominant_frequency': self.frequencies[dominant_scale_idx],
            'dominant_scale': self.scales[dominant_scale_idx],
            'max_magnitude': np.max(magnitude),
            'mean_magnitude': np.mean(magnitude),
            'energy': np.sum(magnitude**2),
            'entropy': -np.sum(magnitude * np.log(magnitude + 1e-10)),
            'dominant_phase': phase[dominant_scale_idx],
            'phase_coherence': np.abs(np.mean(np.exp(1j * phase))),
            'active_scales': np.sum(magnitude > np.mean(magnitude))
        }
        
        # Check if this point is part of any ridge
        ridge_memberships = []
        if self.ridges:
            for i, ridge in enumerate(self.ridges):
                if time_point in ridge['time_indices']:
                    ridge_memberships.append({
                        'ridge_id': i,
                        'frequency': ridge['frequency'],
                        'strength': ridge['strength']
                    })
        
        features['ridge_memberships'] = ridge_memberships
        
        return features


def create_demo_scalogram():
    """Create a demo scalogram visualization."""
    # Generate synthetic data with multiple frequency components
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Create signal with changing frequencies
    signal1 = np.sin(2 * np.pi * 2 * t) * np.exp(-t/5)  # Decaying 2 Hz
    signal2 = np.sin(2 * np.pi * 5 * t) * np.exp(-(t-5)**2/2)  # Gaussian windowed 5 Hz
    signal3 = np.sin(2 * np.pi * 10 * t) * (t > 7)  # Step function 10 Hz
    noise = 0.1 * np.random.randn(len(t))
    
    data = signal1 + signal2 + signal3 + noise
    
    # Create time index
    time_index = pd.date_range('2024-01-01', periods=len(t), freq='10min')
    
    # Initialize visualizer
    viz = ScalogramVisualizer(wavelet='morl', sampling_rate=100)
    
    # Create main scalogram plot
    fig1 = viz.create_scalogram_plot(data, time_index, show_ridges=True)
    
    # Create 3D scalogram
    fig2 = viz.create_3d_scalogram(data, time_index)
    
    # Create ridge analysis
    fig3 = viz.create_ridge_analysis_plot(data, time_index, top_n_ridges=3)
    
    return fig1, fig2, fig3, viz


if __name__ == "__main__":
    # Create demo visualizations
    fig1, fig2, fig3, viz = create_demo_scalogram()
    
    # Save as HTML files
    fig1.write_html("scalogram_main.html")
    fig2.write_html("scalogram_3d.html") 
    fig3.write_html("scalogram_ridges.html")
    
    print("Scalogram visualizations created successfully!")
    print("- scalogram_main.html: Main interactive scalogram with time series")
    print("- scalogram_3d.html: 3D surface plot of scalogram")
    print("- scalogram_ridges.html: Detailed ridge analysis")
    
    # Example of extracting features at a specific time point
    features = viz.get_time_scale_features(500)
    print(f"\nFeatures at time point 500:")
    for key, value in features.items():
        if key != 'ridge_memberships':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
