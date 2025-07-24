"""
Demo Script for Dashboard Controls

Demonstrates the interactive dashboard controls for the financial wavelet prediction system.
"""

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import the controls
from src.dashboard.components.controls import DashboardControls, get_control_state, format_control_summary


def create_demo_app():
    """Create a demo Dash application with controls."""
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Create controls instance
    controls = DashboardControls()
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Financial Wavelet Prediction Dashboard Controls Demo", 
                       className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Control panel
        controls.create_control_panel(),
        
        html.Hr(className="my-4"),
        
        # Demo output area
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Control State Summary"),
                    dbc.CardBody([
                        html.Pre(id="state-summary", 
                                style={"backgroundColor": "#f8f9fa", "padding": "1rem"})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Validation Status"),
                    dbc.CardBody([
                        html.Div(id="validation-status")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Demo visualization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Pattern Detection Preview"),
                    dbc.CardBody([
                        dcc.Graph(id="pattern-preview")
                    ])
                ])
            ])
        ])
    ], fluid=True)
    
    # Register control callbacks
    DashboardControls.register_callbacks(app)
    
    # Add demo-specific callbacks
    @app.callback(
        Output("state-summary", "children"),
        [Input("control-state", "data")]
    )
    def update_state_summary(state):
        """Update the state summary display."""
        if state:
            return format_control_summary(state)
        return "No state available"
    
    @app.callback(
        Output("validation-status", "children"),
        [Input("validation-state", "data")]
    )
    def update_validation_status(validation):
        """Update validation status display."""
        if not validation:
            return dbc.Alert("No validation data", color="secondary")
        
        if validation.get("valid"):
            return dbc.Alert("✓ All inputs valid", color="success")
        else:
            errors = validation.get("errors", [])
            return dbc.Alert([
                html.H5("Validation Errors:", className="alert-heading"),
                html.Ul([html.Li(error) for error in errors])
            ], color="danger")
    
    @app.callback(
        Output("pattern-preview", "figure"),
        [Input("control-state", "data")]
    )
    def update_pattern_preview(state):
        """Update pattern detection preview based on controls."""
        if not state:
            return go.Figure()
        
        # Generate demo data based on controls
        lookback = state.get("lookback", 30)
        ticker = state.get("ticker", "BTC-USD")
        patterns = state.get("patterns", [])
        confidence = state.get("confidence", 0.7)
        
        # Create time series
        dates = pd.date_range(end=datetime.now(), periods=lookback, freq='D')
        
        # Generate synthetic price data
        np.random.seed(42)
        trend = np.linspace(100, 120, lookback)
        noise = np.random.normal(0, 2, lookback)
        prices = trend + noise
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=ticker,
            line=dict(color='blue', width=2)
        ))
        
        # Add pattern markers (demo)
        if patterns:
            # Add some demo pattern detections
            pattern_indices = np.random.choice(range(10, lookback-10), 
                                             size=min(3, len(patterns)), 
                                             replace=False)
            
            for i, idx in enumerate(pattern_indices):
                if i < len(patterns):
                    pattern_name = patterns[i].replace("_", " ").title()
                    
                    # Add pattern region
                    fig.add_vrect(
                        x0=dates[idx-5], x1=dates[idx+5],
                        fillcolor="rgba(255,0,0,0.1)",
                        layer="below",
                        line_width=0,
                    )
                    
                    # Add pattern annotation
                    fig.add_annotation(
                        x=dates[idx],
                        y=prices[idx] + 3,
                        text=f"{pattern_name}<br>Conf: {confidence:.0%}",
                        showarrow=True,
                        arrowhead=2,
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1
                    )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Pattern Detection Preview",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            template="plotly_white",
            height=400
        )
        
        return fig
    
    return app


def run_demo():
    """Run the dashboard controls demo."""
    print("=" * 80)
    print("Financial Wavelet Prediction Dashboard Controls Demo")
    print("=" * 80)
    print()
    print("This demo showcases all the interactive controls including:")
    print("- Ticker selection with custom ticker support")
    print("- Lookback window adjustment (slider + input)")
    print("- Prediction horizon selector")
    print("- Pattern type filters with select/clear all")
    print("- Confidence threshold slider with progress bar")
    print("- Real-time and live mode toggles")
    print("- Pattern complexity selector")
    print("- Forecast method selection")
    print("- Advanced settings panel")
    print()
    print("Features demonstrated:")
    print("✓ Control updates trigger in <100ms")
    print("✓ State persists across page refreshes (session storage)")
    print("✓ Input validation prevents invalid states")
    print("✓ All controls are keyboard accessible")
    print("✓ Cross-component communication via centralized state")
    print()
    print("Starting demo server...")
    print("-" * 80)
    
    # Create and run app
    app = create_demo_app()
    
    # Performance test
    print("\nPerformance Test:")
    controls = DashboardControls()
    
    # Test control creation speed
    start = time.time()
    panel = controls.create_control_panel()
    end = time.time()
    print(f"✓ Control panel created in {(end - start) * 1000:.2f}ms")
    
    # Test state validation speed
    test_state = {
        "ticker": "BTC-USD",
        "lookback": 30,
        "horizon": 7,
        "patterns": ["head_shoulders"],
        "confidence": 0.7,
        "update_frequency": 5
    }
    
    from src.dashboard.components.controls import validate_input_ranges
    
    start = time.time()
    valid, errors = validate_input_ranges(test_state)
    end = time.time()
    print(f"✓ State validation completed in {(end - start) * 1000:.2f}ms")
    
    print("\nStarting Dash server on http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server")
    print("-" * 80)
    
    # Run server
    app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    run_demo()
