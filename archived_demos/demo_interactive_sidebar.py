"""
Demo script for the Interactive Sidebar Component

This script demonstrates the comprehensive ticker selection system with:
- Multi-select dropdown for tickers
- Date range picker with presets
- Pattern type filters (wavelets, shapelets, motifs)
- Quality threshold sliders
- Real-time data toggle
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import the sidebar component
from src.dashboard.components.sidebar import InteractiveSidebar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_app():
    """Create a demo Dash app showcasing the interactive sidebar."""
    # Initialize Dash app with Bootstrap theme
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
    
    # Create main content area
    main_content = dbc.Col([
        html.H2("Interactive Sidebar Demo", className="mb-4"),
        
        # State display card
        dbc.Card([
            dbc.CardHeader("Current Sidebar State"),
            dbc.CardBody([
                html.Div(id="state-display", className="font-monospace small"),
                dcc.Interval(id="state-update-interval", interval=1000)
            ])
        ], className="mb-4"),
        
        # Feature highlights
        dbc.Card([
            dbc.CardHeader("Feature Highlights"),
            dbc.CardBody([
                html.H5("✨ Key Features:", className="mb-3"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("Multi-Select Ticker Dropdown: "),
                        "Select multiple tickers across categories (Crypto, Stocks, ETFs, etc.)"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Date Range Picker: "),
                        "Quick presets (1D to 5Y) and custom date selection with granularity options"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Pattern Type Filters: "),
                        "Organized by categories - Wavelets, Shapelets, and Motifs"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Quality Thresholds: "),
                        "Adjustable sliders for confidence, significance, stability, and robustness"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Real-time Data Toggle: "),
                        "Switch between historical and live data with connection status"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Advanced Settings: "),
                        "Performance options, cache management, and configuration export/import"
                    ])
                ], flush=True)
            ])
        ], className="mb-4"),
        
        # Instructions
        dbc.Card([
            dbc.CardHeader("Instructions"),
            dbc.CardBody([
                html.H5("🎯 Try These Actions:", className="mb-3"),
                html.Ol([
                    html.Li("Switch between ticker categories (Crypto, Stocks, ETFs, etc.)"),
                    html.Li("Use the multi-select dropdown to choose multiple tickers"),
                    html.Li("Try the 'Popular' button for quick selection"),
                    html.Li("Click date presets (1W, 1M, 3M, etc.) to quickly set date ranges"),
                    html.Li("Expand pattern categories and select different pattern types"),
                    html.Li("Adjust quality thresholds or use preset levels (Conservative, Balanced, Aggressive)"),
                    html.Li("Toggle real-time data and watch the connection status"),
                    html.Li("Explore advanced settings for performance options"),
                    html.Li("Try exporting your configuration using the Export Config button")
                ])
            ])
        ])
    ], width=9)
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            html.H1("Financial Wavelet Prediction - Interactive Sidebar", className="text-center mb-4")
        ]),
        dbc.Row([
            sidebar.create_sidebar_layout(),
            main_content
        ])
    ], fluid=True, className="p-4")
    
    # Register sidebar callbacks
    sidebar.register_callbacks(app)
    
    # Additional demo callbacks
    @app.callback(
        dash.Output("state-display", "children"),
        [dash.Input("state-update-interval", "n_intervals"),
         dash.Input("sidebar-state", "data")]
    )
    def update_state_display(n_intervals, state):
        """Display current sidebar state."""
        if not state:
            return "No state available yet. Interact with the sidebar to see changes."
        
        # Format state for display
        formatted_state = []
        
        # Tickers
        tickers = state.get("tickers", [])
        formatted_state.append(f"📊 Tickers ({len(tickers)}): {', '.join(tickers[:3])}{'...' if len(tickers) > 3 else ''}")
        
        # Date range
        date_range = state.get("date_range", {})
        formatted_state.append(f"📅 Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
        formatted_state.append(f"⏱️ Granularity: {date_range.get('granularity', 'N/A')}")
        
        # Patterns
        patterns = state.get("patterns", {})
        total_patterns = sum(len(patterns.get(cat, [])) for cat in ["wavelets", "shapelets", "motifs"])
        formatted_state.append(f"🔍 Patterns Selected: {total_patterns}")
        
        # Thresholds
        thresholds = state.get("thresholds", {})
        avg_threshold = sum(thresholds.values()) / len(thresholds) if thresholds else 0
        formatted_state.append(f"📈 Avg Quality Threshold: {avg_threshold:.0%}")
        
        # Real-time settings
        realtime = state.get("realtime", {})
        mode = "Real-time" if realtime.get("enabled") else "Historical"
        formatted_state.append(f"🔄 Data Mode: {mode}")
        
        if realtime.get("auto_refresh"):
            formatted_state.append(f"♻️ Auto-refresh: Every {realtime.get('interval', 30)}s")
        
        # Performance options
        performance = state.get("performance", [])
        if performance:
            formatted_state.append(f"⚡ Performance: {', '.join(performance)}")
        
        # Last update
        timestamp = state.get("timestamp", "")
        if timestamp:
            dt = datetime.fromisoformat(timestamp)
            formatted_state.append(f"🕐 Last Update: {dt.strftime('%H:%M:%S')}")
        
        return html.Pre("\n".join(formatted_state))
    
    return app


def main():
    """Run the demo application."""
    print("\n" + "="*60)
    print("Interactive Sidebar Demo")
    print("="*60)
    print("\nFeatures demonstrated:")
    print("- Multi-select ticker dropdown with categories")
    print("- Date range picker with presets and granularity")
    print("- Pattern type filters (wavelets, shapelets, motifs)")
    print("- Quality threshold sliders with presets")
    print("- Real-time data toggle with status indicators")
    print("- Advanced settings and configuration management")
    print("\n" + "="*60)
    
    # Create and run the app
    app = create_demo_app()
    
    print("\n✅ Starting demo server...")
    print("📍 Open http://localhost:8050 in your browser")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    main()
