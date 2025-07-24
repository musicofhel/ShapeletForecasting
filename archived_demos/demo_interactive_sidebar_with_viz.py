"""
Enhanced Demo script for the Interactive Sidebar Component with Data Visualization

This script demonstrates the sidebar with actual data visualizations that update
based on the sidebar selections.
"""

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

# Import the sidebar component
from src.dashboard.components.sidebar import InteractiveSidebar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(ticker, start_date, end_date, granularity):
    """Generate sample financial data for demonstration."""
    # For demo purposes, generate synthetic data
    # In production, this would fetch real data
    
    # Calculate number of periods
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_points = len(date_range)
    
    # Generate price data with trend and noise
    base_price = 100
    trend = np.linspace(0, 20, n_points)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.normal(0, 5, n_points)
    
    prices = base_price + trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_points),
        'High': prices + np.abs(np.random.normal(0, 2, n_points)),
        'Low': prices - np.abs(np.random.normal(0, 2, n_points))
    })
    
    return df


def create_price_chart(data_dict, selected_tickers):
    """Create price chart for selected tickers."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, ticker in enumerate(selected_tickers[:5]):  # Limit to 5 tickers for clarity
        if ticker in data_dict:
            df = data_dict[ticker]
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name=ticker,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title="Price Chart - Selected Tickers",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white",
        height=400
    )
    
    return fig


def create_pattern_analysis(data_dict, patterns, thresholds):
    """Create pattern analysis visualization."""
    fig = go.Figure()
    
    # Simulate pattern detection results
    pattern_types = []
    pattern_counts = []
    pattern_confidence = []
    
    for category, items in patterns.items():
        for pattern in items:
            pattern_types.append(f"{category}: {pattern}")
            # Simulate count based on threshold
            base_count = np.random.randint(5, 50)
            adjusted_count = int(base_count * (1 - thresholds.get('confidence', 0.5)))
            pattern_counts.append(adjusted_count)
            pattern_confidence.append(np.random.uniform(
                thresholds.get('confidence', 0.5), 
                1.0
            ))
    
    if pattern_types:
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=pattern_types,
                y=pattern_counts,
                marker_color=pattern_confidence,
                marker_colorscale='Viridis',
                text=[f"{c:.0%}" for c in pattern_confidence],
                textposition='auto',
                hovertemplate='%{x}<br>Count: %{y}<br>Confidence: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Pattern Detection Results",
            xaxis_title="Pattern Type",
            yaxis_title="Occurrences",
            template="plotly_white",
            height=350,
            showlegend=False
        )
    else:
        # No patterns selected
        fig.add_annotation(
            text="No patterns selected. Please select patterns from the sidebar.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
            height=350
        )
    
    return fig


def create_quality_metrics_gauge(thresholds):
    """Create gauge charts for quality metrics."""
    metrics = ['confidence', 'significance', 'stability', 'robustness']
    
    fig = go.Figure()
    
    # Create subplots for gauges
    for i, metric in enumerate(metrics):
        value = thresholds.get(metric, 0.5)
        
        # Determine color based on value
        if value >= 0.7:
            color = "green"
        elif value >= 0.5:
            color = "yellow"
        else:
            color = "red"
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            title={'text': metric.capitalize()},
            domain={'x': [i*0.25, (i+1)*0.25 - 0.05], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
    
    fig.update_layout(
        title="Quality Thresholds",
        height=200,
        template="plotly_white"
    )
    
    return fig


def create_data_summary_table(sidebar_state):
    """Create summary table of current selections."""
    tickers = sidebar_state.get("tickers", [])
    date_range = sidebar_state.get("date_range", {})
    patterns = sidebar_state.get("patterns", {})
    
    # Count patterns
    total_patterns = sum(len(patterns.get(cat, [])) for cat in ["wavelets", "shapelets", "motifs"])
    
    # Create summary data
    summary_data = {
        "Setting": ["Tickers", "Date Range", "Granularity", "Patterns", "Data Mode"],
        "Value": [
            f"{len(tickers)} selected",
            f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}",
            date_range.get('granularity', 'N/A'),
            f"{total_patterns} selected",
            "Real-time" if sidebar_state.get("realtime", {}).get("enabled") else "Historical"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(
        title="Current Configuration",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_demo_app():
    """Create a demo Dash app with sidebar and visualizations."""
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
        ],
        suppress_callback_exceptions=True
    )
    
    # Create sidebar instance
    sidebar = InteractiveSidebar()
    
    # Create main content area with visualizations
    main_content = dbc.Col([
        html.H2("Financial Analysis Dashboard", className="mb-4"),
        
        # Row 1: Price chart and pattern analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="price-chart")
                    ])
                ])
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="pattern-analysis")
                    ])
                ])
            ], width=5)
        ], className="mb-3"),
        
        # Row 2: Quality metrics and summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="quality-gauges")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="config-summary")
                    ])
                ])
            ], width=4)
        ]),
        
        # Hidden store for data
        dcc.Store(id="data-store")
    ], width=9)
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            html.H1("Interactive Financial Analysis System", className="text-center mb-4")
        ]),
        dbc.Row([
            sidebar.create_sidebar_layout(),
            main_content
        ])
    ], fluid=True, className="p-4")
    
    # Register sidebar callbacks
    sidebar.register_callbacks(app)
    
    # Data loading callback
    @app.callback(
        Output("data-store", "data"),
        [Input("sidebar-state", "data")]
    )
    def load_data(sidebar_state):
        """Load data based on sidebar selections."""
        if not sidebar_state:
            return {}
        
        tickers = sidebar_state.get("tickers", [])
        date_range = sidebar_state.get("date_range", {})
        
        if not tickers or not date_range.get("start") or not date_range.get("end"):
            return {}
        
        # Generate sample data for each ticker
        data_dict = {}
        for ticker in tickers[:10]:  # Limit to 10 tickers
            data_dict[ticker] = generate_sample_data(
                ticker,
                date_range["start"],
                date_range["end"],
                date_range.get("granularity", "1d")
            ).to_dict('records')
        
        return data_dict
    
    # Price chart update
    @app.callback(
        Output("price-chart", "figure"),
        [Input("data-store", "data")],
        [State("sidebar-state", "data")]
    )
    def update_price_chart(data_store, sidebar_state):
        """Update price chart based on data."""
        if not data_store or not sidebar_state:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Select tickers and click 'Apply Settings & Analyze' to view data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
                height=400
            )
            return fig
        
        # Convert data back to DataFrames
        data_dict = {}
        for ticker, records in data_store.items():
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[ticker] = df
        
        tickers = sidebar_state.get("tickers", [])
        return create_price_chart(data_dict, tickers)
    
    # Pattern analysis update
    @app.callback(
        Output("pattern-analysis", "figure"),
        [Input("sidebar-state", "data")]
    )
    def update_pattern_analysis(sidebar_state):
        """Update pattern analysis visualization."""
        if not sidebar_state:
            return go.Figure()
        
        patterns = sidebar_state.get("patterns", {})
        thresholds = sidebar_state.get("thresholds", {})
        
        return create_pattern_analysis({}, patterns, thresholds)
    
    # Quality gauges update
    @app.callback(
        Output("quality-gauges", "figure"),
        [Input("sidebar-state", "data")]
    )
    def update_quality_gauges(sidebar_state):
        """Update quality threshold gauges."""
        if not sidebar_state:
            return go.Figure()
        
        thresholds = sidebar_state.get("thresholds", {})
        return create_quality_metrics_gauge(thresholds)
    
    # Configuration summary update
    @app.callback(
        Output("config-summary", "figure"),
        [Input("sidebar-state", "data")]
    )
    def update_config_summary(sidebar_state):
        """Update configuration summary table."""
        if not sidebar_state:
            return go.Figure()
        
        return create_data_summary_table(sidebar_state)
    
    return app


def main():
    """Run the enhanced demo application."""
    print("\n" + "="*60)
    print("Interactive Sidebar Demo with Data Visualization")
    print("="*60)
    print("\nFeatures demonstrated:")
    print("- Interactive sidebar controls")
    print("- Real-time data visualization updates")
    print("- Price charts for selected tickers")
    print("- Pattern analysis visualization")
    print("- Quality threshold gauges")
    print("- Configuration summary")
    print("\n" + "="*60)
    
    # Create and run the app
    app = create_demo_app()
    
    print("\n✅ Starting enhanced demo server...")
    print("📍 Open http://localhost:8051 in your browser")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.run_server(debug=True, port=8051)


if __name__ == "__main__":
    main()
