"""
Demo script for interactive wavelet scalogram visualization.

This script demonstrates the scalogram visualization capabilities including:
- Interactive heatmap with time-scale representation
- Ridge detection and overlay
- Synchronized time series plot
- Click interactions for detailed analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from src.dashboard.visualizations.scalogram import ScalogramVisualizer
import json


def generate_complex_signal(n_points=2000, sampling_rate=100):
    """Generate a complex signal with multiple frequency components."""
    t = np.linspace(0, n_points/sampling_rate, n_points)
    
    # Multiple frequency components
    # 1. Low frequency trend
    trend = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    
    # 2. Medium frequency with amplitude modulation
    medium = np.sin(2 * np.pi * 2 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
    
    # 3. High frequency burst
    burst_mask = (t > 5) & (t < 10)
    high_burst = 0.8 * np.sin(2 * np.pi * 10 * t) * burst_mask
    
    # 4. Chirp signal (increasing frequency)
    chirp_start = 15
    chirp_mask = t > chirp_start
    chirp_freq = 1 + 0.5 * (t - chirp_start)
    chirp = 0.6 * np.sin(2 * np.pi * np.cumsum(chirp_freq) / sampling_rate) * chirp_mask
    
    # 5. Noise
    noise = 0.1 * np.random.randn(n_points)
    
    # Combine all components
    signal = trend + medium + high_burst + chirp + noise
    
    return t, signal


def create_interactive_dashboard():
    """Create an interactive Dash application for scalogram visualization."""
    
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Generate initial data
    t, signal = generate_complex_signal()
    time_index = pd.date_range('2024-01-01', periods=len(signal), freq='1s')
    
    # Initialize visualizer
    viz = ScalogramVisualizer(wavelet='morl', sampling_rate=100)
    
    # Create initial plots
    main_fig = viz.create_scalogram_plot(signal, time_index, show_ridges=True)
    
    # Layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Interactive Wavelet Scalogram Visualization", 
                       className="text-center mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Control Panel"),
                    dbc.CardBody([
                        html.Label("Wavelet Type"),
                        dcc.Dropdown(
                            id='wavelet-dropdown',
                            options=[
                                {'label': 'Morlet', 'value': 'morl'},
                                {'label': 'Mexican Hat', 'value': 'mexh'},
                                {'label': 'Gaussian', 'value': 'gaus8'},
                                {'label': 'Complex Morlet', 'value': 'cmor1.5-1.0'}
                            ],
                            value='morl',
                            className="mb-3"
                        ),
                        
                        html.Label("Colorscale"),
                        dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=[
                                {'label': 'Viridis', 'value': 'Viridis'},
                                {'label': 'Plasma', 'value': 'Plasma'},
                                {'label': 'Inferno', 'value': 'Inferno'},
                                {'label': 'Jet', 'value': 'Jet'},
                                {'label': 'Hot', 'value': 'Hot'}
                            ],
                            value='Viridis',
                            className="mb-3"
                        ),
                        
                        html.Label("Ridge Detection"),
                        dbc.Switch(
                            id='ridge-switch',
                            label="Show Ridges",
                            value=True,
                            className="mb-3"
                        ),
                        
                        html.Label("View Mode"),
                        dbc.RadioItems(
                            id='view-mode',
                            options=[
                                {'label': '2D Scalogram', 'value': '2d'},
                                {'label': '3D Surface', 'value': '3d'},
                                {'label': 'Ridge Analysis', 'value': 'ridge'}
                            ],
                            value='2d',
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        html.H6("Click Information"),
                        html.Div(id='click-info', className="small")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(
                        id='scalogram-plot',
                        figure=main_fig,
                        style={'height': '800px'}
                    ),
                    type="default"
                )
            ], width=9)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Time-Scale Features"),
                    dbc.CardBody([
                        html.Div(id='feature-display')
                    ])
                ], className="mt-3")
            ], width=12)
        ])
    ], fluid=True)
    
    # Store data in browser
    app.layout.children.append(
        dcc.Store(id='signal-data', data={
            'signal': signal.tolist(),
            'time_index': time_index.astype(str).tolist()
        })
    )
    
    @app.callback(
        Output('scalogram-plot', 'figure'),
        [Input('wavelet-dropdown', 'value'),
         Input('colorscale-dropdown', 'value'),
         Input('ridge-switch', 'value'),
         Input('view-mode', 'value')],
        [State('signal-data', 'data')]
    )
    def update_scalogram(wavelet, colorscale, show_ridges, view_mode, data):
        """Update scalogram based on control inputs."""
        # Reconstruct signal and time index
        signal = np.array(data['signal'])
        time_index = pd.to_datetime(data['time_index'])
        
        # Create new visualizer with selected wavelet
        viz = ScalogramVisualizer(wavelet=wavelet, sampling_rate=100)
        
        # Generate appropriate plot based on view mode
        if view_mode == '2d':
            fig = viz.create_scalogram_plot(
                signal, time_index, 
                show_ridges=show_ridges, 
                colorscale=colorscale
            )
        elif view_mode == '3d':
            fig = viz.create_3d_scalogram(signal, time_index, colorscale=colorscale)
        else:  # ridge analysis
            fig = viz.create_ridge_analysis_plot(signal, time_index, top_n_ridges=5)
        
        return fig
    
    @app.callback(
        [Output('click-info', 'children'),
         Output('feature-display', 'children')],
        [Input('scalogram-plot', 'clickData')],
        [State('signal-data', 'data'),
         State('wavelet-dropdown', 'value')]
    )
    def display_click_info(clickData, data, wavelet):
        """Display information about clicked point."""
        if not clickData:
            return "Click on the scalogram to see details", ""
        
        # Get click coordinates
        point = clickData['points'][0]
        
        # Reconstruct signal
        signal = np.array(data['signal'])
        
        # Create visualizer
        viz = ScalogramVisualizer(wavelet=wavelet, sampling_rate=100)
        viz.compute_cwt(signal)
        viz.detect_ridges()
        
        # Get time index from click
        time_idx = int(point['x']) if isinstance(point['x'], (int, float)) else point['pointIndex']
        
        # Get features at clicked time
        features = viz.get_time_scale_features(time_idx)
        
        # Format click info
        click_info = [
            html.P(f"Time Index: {time_idx}"),
            html.P(f"Clicked at: {point.get('y', 'N/A'):.3f} Hz" if 'y' in point else "")
        ]
        
        # Format feature display
        feature_display = [
            html.H6("Features at Selected Time:"),
            html.Ul([
                html.Li(f"Dominant Frequency: {features['dominant_frequency']:.3f} Hz"),
                html.Li(f"Max Magnitude: {features['max_magnitude']:.4f}"),
                html.Li(f"Energy: {features['energy']:.4f}"),
                html.Li(f"Entropy: {features['entropy']:.4f}"),
                html.Li(f"Phase Coherence: {features['phase_coherence']:.4f}"),
                html.Li(f"Active Scales: {features['active_scales']}")
            ])
        ]
        
        if features['ridge_memberships']:
            feature_display.append(html.H6("Ridge Memberships:"))
            ridge_list = []
            for ridge in features['ridge_memberships']:
                ridge_list.append(
                    html.Li(f"Ridge {ridge['ridge_id']+1}: "
                           f"{ridge['frequency']:.3f} Hz "
                           f"(strength: {ridge['strength']:.3f})")
                )
            feature_display.append(html.Ul(ridge_list))
        
        return click_info, feature_display
    
    return app


def create_standalone_demo():
    """Create standalone HTML visualizations without Dash."""
    print("Creating standalone scalogram visualizations...")
    
    # Generate signal
    t, signal = generate_complex_signal()
    time_index = pd.date_range('2024-01-01', periods=len(signal), freq='1s')
    
    # Create visualizer
    viz = ScalogramVisualizer(wavelet='morl', sampling_rate=100)
    
    # Create different visualizations
    print("1. Creating main scalogram with ridges...")
    fig1 = viz.create_scalogram_plot(signal, time_index, show_ridges=True)
    fig1.write_html("scalogram_interactive.html")
    
    print("2. Creating 3D scalogram...")
    fig2 = viz.create_3d_scalogram(signal, time_index)
    fig2.write_html("scalogram_3d_view.html")
    
    print("3. Creating ridge analysis...")
    fig3 = viz.create_ridge_analysis_plot(signal, time_index, top_n_ridges=5)
    fig3.write_html("scalogram_ridge_analysis.html")
    
    # Create comparison with different wavelets
    print("4. Creating wavelet comparison...")
    wavelets = ['morl', 'mexh', 'gaus8']
    fig4 = make_subplots(
        rows=len(wavelets), cols=1,
        subplot_titles=[f'{w} Wavelet' for w in wavelets],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    for i, wavelet in enumerate(wavelets):
        viz_comp = ScalogramVisualizer(wavelet=wavelet, sampling_rate=100)
        viz_comp.compute_cwt(signal)
        
        magnitude = np.abs(viz_comp.coefficients)
        z_data = np.log10(magnitude + 1e-10)
        
        fig4.add_trace(
            go.Heatmap(
                x=time_index,
                y=viz_comp.frequencies,
                z=z_data,
                colorscale='Viridis',
                showscale=i == 0,  # Only show colorbar for first subplot
                hovertemplate='Time: %{x}<br>Frequency: %{y:.3f}<br>Magnitude: %{z:.2f}<extra></extra>'
            ),
            row=i+1, col=1
        )
        
        fig4.update_yaxes(title_text="Frequency (Hz)", type="log", row=i+1, col=1)
    
    fig4.update_xaxes(title_text="Time", row=len(wavelets), col=1)
    fig4.update_layout(height=250*len(wavelets), title="Wavelet Comparison")
    fig4.write_html("scalogram_wavelet_comparison.html")
    
    print("\nStandalone visualizations created:")
    print("- scalogram_interactive.html: Main interactive scalogram")
    print("- scalogram_3d_view.html: 3D surface visualization")
    print("- scalogram_ridge_analysis.html: Detailed ridge analysis")
    print("- scalogram_wavelet_comparison.html: Comparison of different wavelets")
    
    return viz


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dash':
        # Run interactive Dash app
        print("Starting interactive Dash application...")
        app = create_interactive_dashboard()
        app.run_server(debug=True, port=8052)
    else:
        # Create standalone demos
        viz = create_standalone_demo()
        
        # Also run the simple demo from the scalogram module
        print("\nRunning scalogram module demo...")
        from src.dashboard.visualizations.scalogram import create_demo_scalogram
        fig1, fig2, fig3, viz_demo = create_demo_scalogram()
        
        print("\nAll demos completed successfully!")
        print("\nTo run the interactive Dash app, use:")
        print("  python demo_scalogram_visualization.py --dash")
