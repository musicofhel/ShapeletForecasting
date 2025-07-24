"""
Dashboard Controls Component for Financial Wavelet Prediction System

This module provides interactive controls for configuring pattern detection,
prediction parameters, and visualization settings.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardControls:
    """Main dashboard controls component with all interactive elements."""
    
    def __init__(self):
        """Initialize dashboard controls with default values."""
        self.default_ticker = "BTC-USD"
        self.default_lookback = 30
        self.default_horizon = 7
        self.default_confidence = 0.7
        self.default_pattern_types = ["head_shoulders", "double_top", "triangle"]
        self.default_complexity = "medium"
        self.default_forecast_method = "ensemble"
        
        # Available options
        self.tickers = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "AAPL", "MSFT", "GOOGL"]
        self.pattern_types = [
            "head_shoulders", "double_top", "double_bottom", 
            "triangle", "wedge", "flag", "pennant", "channel"
        ]
        self.complexity_levels = ["simple", "medium", "complex"]
        self.forecast_methods = ["lstm", "gru", "transformer", "ensemble", "markov"]
        
    def create_ticker_selector(self) -> dbc.Card:
        """Create ticker selection control."""
        return dbc.Card([
            dbc.CardHeader("Ticker Selection"),
            dbc.CardBody([
                dbc.Label("Select Ticker:", html_for="ticker-dropdown"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": ticker, "value": ticker} for ticker in self.tickers],
                    value=self.default_ticker,
                    clearable=False,
                    searchable=True,
                    placeholder="Select a ticker...",
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("Choose the financial instrument to analyze"),
                html.Hr(),
                dbc.Button(
                    "Add Custom Ticker",
                    id="add-ticker-btn",
                    color="secondary",
                    size="sm",
                    className="mt-2"
                ),
                dbc.Input(
                    id="custom-ticker-input",
                    placeholder="Enter ticker symbol",
                    type="text",
                    className="mt-2",
                    style={"display": "none"}
                )
            ])
        ], className="mb-3")
    
    def create_lookback_control(self) -> dbc.Card:
        """Create lookback window adjustment control."""
        return dbc.Card([
            dbc.CardHeader("Lookback Window"),
            dbc.CardBody([
                dbc.Label("Historical Data Window (days):", html_for="lookback-slider"),
                dcc.Slider(
                    id="lookback-slider",
                    min=7,
                    max=365,
                    step=1,
                    value=self.default_lookback,
                    marks={
                        7: "1W",
                        30: "1M",
                        90: "3M",
                        180: "6M",
                        365: "1Y"
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("Amount of historical data to analyze"),
                html.Hr(),
                dbc.InputGroup([
                    dbc.InputGroupText("Custom:"),
                    dbc.Input(
                        id="lookback-input",
                        type="number",
                        min=7,
                        max=365,
                        value=self.default_lookback,
                        persistence=True
                    ),
                    dbc.InputGroupText("days")
                ], size="sm", className="mt-2")
            ])
        ], className="mb-3")
    
    def create_horizon_selector(self) -> dbc.Card:
        """Create prediction horizon selector."""
        return dbc.Card([
            dbc.CardHeader("Prediction Horizon"),
            dbc.CardBody([
                dbc.Label("Forecast Period:", html_for="horizon-radio"),
                dbc.RadioItems(
                    id="horizon-radio",
                    options=[
                        {"label": "1 Day", "value": 1},
                        {"label": "3 Days", "value": 3},
                        {"label": "1 Week", "value": 7},
                        {"label": "2 Weeks", "value": 14},
                        {"label": "1 Month", "value": 30}
                    ],
                    value=self.default_horizon,
                    inline=True,
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("How far ahead to predict"),
                html.Hr(),
                dbc.Label("Custom Horizon:", className="mt-2"),
                dbc.InputGroup([
                    dbc.Input(
                        id="custom-horizon-input",
                        type="number",
                        min=1,
                        max=90,
                        placeholder="Days"
                    ),
                    dbc.Button("Apply", id="apply-horizon-btn", color="primary", size="sm")
                ], size="sm")
            ])
        ], className="mb-3")
    
    def create_pattern_filters(self) -> dbc.Card:
        """Create pattern type filters."""
        return dbc.Card([
            dbc.CardHeader("Pattern Type Filters"),
            dbc.CardBody([
                dbc.Label("Select Pattern Types:", html_for="pattern-checklist"),
                dbc.Checklist(
                    id="pattern-checklist",
                    options=[
                        {"label": pattern.replace("_", " ").title(), "value": pattern}
                        for pattern in self.pattern_types
                    ],
                    value=self.default_pattern_types,
                    inline=False,
                    persistence=True,
                    persistence_type="session"
                ),
                html.Hr(),
                dbc.ButtonGroup([
                    dbc.Button("Select All", id="select-all-patterns", size="sm", color="info"),
                    dbc.Button("Clear All", id="clear-all-patterns", size="sm", color="warning")
                ], className="mt-2")
            ])
        ], className="mb-3")
    
    def create_confidence_slider(self) -> dbc.Card:
        """Create confidence threshold slider."""
        return dbc.Card([
            dbc.CardHeader("Confidence Threshold"),
            dbc.CardBody([
                dbc.Label("Minimum Confidence Level:", html_for="confidence-slider"),
                dcc.Slider(
                    id="confidence-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=self.default_confidence,
                    marks={
                        0: "0%",
                        0.25: "25%",
                        0.5: "50%",
                        0.75: "75%",
                        1: "100%"
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("Filter patterns by confidence score"),
                html.Hr(),
                dbc.Progress(
                    id="confidence-progress",
                    value=self.default_confidence * 100,
                    striped=True,
                    animated=True,
                    className="mt-2"
                ),
                html.Div(id="confidence-display", className="text-center mt-2")
            ])
        ], className="mb-3")
    
    def create_mode_switches(self) -> dbc.Card:
        """Create mode toggle switches."""
        return dbc.Card([
            dbc.CardHeader("Operation Modes"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Real-time Detection:"),
                        dbc.Switch(
                            id="realtime-toggle",
                            label="",
                            value=False,
                            persistence=True,
                            persistence_type="session"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Live Mode:"),
                        dbc.Switch(
                            id="live-mode-toggle",
                            label="",
                            value=False,
                            persistence=True,
                            persistence_type="session"
                        )
                    ], width=6)
                ]),
                html.Hr(),
                dbc.Alert(
                    id="mode-status",
                    children="Historical Mode Active",
                    color="info",
                    className="mt-2"
                )
            ])
        ], className="mb-3")
    
    def create_complexity_selector(self) -> dbc.Card:
        """Create pattern complexity selector."""
        return dbc.Card([
            dbc.CardHeader("Pattern Complexity"),
            dbc.CardBody([
                dbc.Label("Complexity Level:", html_for="complexity-select"),
                dbc.Select(
                    id="complexity-select",
                    options=[
                        {"label": level.title(), "value": level}
                        for level in self.complexity_levels
                    ],
                    value=self.default_complexity,
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("Adjust pattern detection sensitivity"),
                html.Hr(),
                dbc.Label("Complexity Details:", className="mt-2"),
                html.Div(id="complexity-info", className="small text-muted")
            ])
        ], className="mb-3")
    
    def create_forecast_selector(self) -> dbc.Card:
        """Create forecast method selector."""
        return dbc.Card([
            dbc.CardHeader("Forecast Method"),
            dbc.CardBody([
                dbc.Label("Select Algorithm:", html_for="forecast-method"),
                dbc.RadioItems(
                    id="forecast-method",
                    options=[
                        {"label": method.upper(), "value": method}
                        for method in self.forecast_methods
                    ],
                    value=self.default_forecast_method,
                    persistence=True,
                    persistence_type="session"
                ),
                dbc.FormText("Choose prediction algorithm"),
                html.Hr(),
                dbc.Button(
                    "Algorithm Info",
                    id="algo-info-btn",
                    color="info",
                    size="sm",
                    className="mt-2"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(id="algo-description"),
                        className="mt-2"
                    ),
                    id="algo-info-collapse",
                    is_open=False
                )
            ])
        ], className="mb-3")
    
    def create_advanced_controls(self) -> dbc.Card:
        """Create advanced control panel."""
        return dbc.Card([
            dbc.CardHeader([
                "Advanced Settings",
                dbc.Button(
                    "⚙️",
                    id="advanced-toggle",
                    color="link",
                    size="sm",
                    className="float-end"
                )
            ]),
            dbc.Collapse(
                dbc.CardBody([
                    dbc.Label("Update Frequency (seconds):"),
                    dbc.Input(
                        id="update-frequency",
                        type="number",
                        min=1,
                        max=60,
                        value=5,
                        persistence=True
                    ),
                    html.Hr(),
                    dbc.Label("Cache Duration (minutes):"),
                    dbc.Input(
                        id="cache-duration",
                        type="number",
                        min=1,
                        max=60,
                        value=15,
                        persistence=True
                    ),
                    html.Hr(),
                    dbc.Label("Max Patterns to Display:"),
                    dbc.Input(
                        id="max-patterns",
                        type="number",
                        min=1,
                        max=100,
                        value=20,
                        persistence=True
                    ),
                    html.Hr(),
                    dbc.Button(
                        "Reset to Defaults",
                        id="reset-btn",
                        color="danger",
                        size="sm",
                        className="mt-2"
                    )
                ]),
                id="advanced-collapse",
                is_open=False
            )
        ], className="mb-3")
    
    def create_control_panel(self) -> html.Div:
        """Create complete control panel layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    self.create_ticker_selector(),
                    self.create_lookback_control(),
                    self.create_horizon_selector()
                ], width=12, lg=4),
                dbc.Col([
                    self.create_pattern_filters(),
                    self.create_confidence_slider()
                ], width=12, lg=4),
                dbc.Col([
                    self.create_mode_switches(),
                    self.create_complexity_selector(),
                    self.create_forecast_selector(),
                    self.create_advanced_controls()
                ], width=12, lg=4)
            ]),
            # Hidden stores for state management
            dcc.Store(id="control-state", storage_type="session"),
            dcc.Store(id="validation-state", storage_type="memory"),
            dcc.Interval(id="update-interval", interval=5000, disabled=True)
        ])
    
    @staticmethod
    def register_callbacks(app):
        """Register all control callbacks with the Dash app."""
        
        # Ticker selection callback
        @app.callback(
            [Output("custom-ticker-input", "style"),
             Output("ticker-dropdown", "options")],
            [Input("add-ticker-btn", "n_clicks"),
             Input("custom-ticker-input", "value")],
            [State("ticker-dropdown", "options")]
        )
        def handle_custom_ticker(n_clicks, custom_ticker, current_options):
            """Handle custom ticker addition."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return {"display": "none"}, current_options
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "add-ticker-btn" and n_clicks:
                return {"display": "block"}, current_options
            
            if trigger_id == "custom-ticker-input" and custom_ticker:
                # Add new ticker to options
                new_option = {"label": custom_ticker.upper(), "value": custom_ticker.upper()}
                if new_option not in current_options:
                    current_options.append(new_option)
                return {"display": "none"}, current_options
            
            return {"display": "none"}, current_options
        
        # Lookback synchronization
        @app.callback(
            [Output("lookback-input", "value"),
             Output("lookback-slider", "value")],
            [Input("lookback-slider", "value"),
             Input("lookback-input", "value")]
        )
        def sync_lookback_controls(slider_value, input_value):
            """Synchronize lookback slider and input."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return slider_value, slider_value
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "lookback-slider":
                return slider_value, slider_value
            elif trigger_id == "lookback-input" and input_value:
                return input_value, input_value
            
            return slider_value, slider_value
        
        # Pattern selection callbacks
        @app.callback(
            Output("pattern-checklist", "value"),
            [Input("select-all-patterns", "n_clicks"),
             Input("clear-all-patterns", "n_clicks")],
            [State("pattern-checklist", "options")]
        )
        def update_pattern_selection(select_clicks, clear_clicks, options):
            """Update pattern selection based on button clicks."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return dash.no_update
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "select-all-patterns":
                return [opt["value"] for opt in options]
            elif trigger_id == "clear-all-patterns":
                return []
            
            return dash.no_update
        
        # Confidence display update
        @app.callback(
            [Output("confidence-progress", "value"),
             Output("confidence-display", "children")],
            [Input("confidence-slider", "value")]
        )
        def update_confidence_display(confidence):
            """Update confidence progress bar and display."""
            percentage = confidence * 100
            return percentage, f"{percentage:.0f}% Confidence Threshold"
        
        # Mode status update
        @app.callback(
            [Output("mode-status", "children"),
             Output("mode-status", "color"),
             Output("update-interval", "disabled")],
            [Input("realtime-toggle", "value"),
             Input("live-mode-toggle", "value")]
        )
        def update_mode_status(realtime, live):
            """Update mode status display."""
            if live:
                return "Live Mode Active", "success", False
            elif realtime:
                return "Real-time Detection Active", "warning", False
            else:
                return "Historical Mode Active", "info", True
        
        # Complexity info update
        @app.callback(
            Output("complexity-info", "children"),
            [Input("complexity-select", "value")]
        )
        def update_complexity_info(complexity):
            """Update complexity information display."""
            info_map = {
                "simple": "Basic patterns with clear structure (3-5 points)",
                "medium": "Standard patterns with moderate complexity (5-10 points)",
                "complex": "Advanced patterns with intricate structure (10+ points)"
            }
            return info_map.get(complexity, "")
        
        # Algorithm info toggle
        @app.callback(
            [Output("algo-info-collapse", "is_open"),
             Output("algo-description", "children")],
            [Input("algo-info-btn", "n_clicks"),
             Input("forecast-method", "value")],
            [State("algo-info-collapse", "is_open")]
        )
        def toggle_algo_info(n_clicks, method, is_open):
            """Toggle algorithm information display."""
            ctx = dash.callback_context
            
            descriptions = {
                "lstm": "Long Short-Term Memory network for sequence prediction",
                "gru": "Gated Recurrent Unit - faster alternative to LSTM",
                "transformer": "Attention-based model for complex patterns",
                "ensemble": "Combination of multiple models for robust predictions",
                "markov": "Probabilistic model based on state transitions"
            }
            
            if ctx.triggered[0]["prop_id"].split(".")[0] == "algo-info-btn":
                is_open = not is_open
            
            return is_open, descriptions.get(method, "")
        
        # Advanced settings toggle
        @app.callback(
            Output("advanced-collapse", "is_open"),
            [Input("advanced-toggle", "n_clicks")],
            [State("advanced-collapse", "is_open")]
        )
        def toggle_advanced_settings(n_clicks, is_open):
            """Toggle advanced settings panel."""
            if n_clicks:
                return not is_open
            return is_open
        
        # Custom horizon application
        @app.callback(
            Output("horizon-radio", "value"),
            [Input("apply-horizon-btn", "n_clicks")],
            [State("custom-horizon-input", "value")]
        )
        def apply_custom_horizon(n_clicks, custom_value):
            """Apply custom horizon value."""
            if n_clicks and custom_value:
                return custom_value
            return dash.no_update
        
        # Control state management
        @app.callback(
            Output("control-state", "data"),
            [Input("ticker-dropdown", "value"),
             Input("lookback-slider", "value"),
             Input("horizon-radio", "value"),
             Input("pattern-checklist", "value"),
             Input("confidence-slider", "value"),
             Input("realtime-toggle", "value"),
             Input("live-mode-toggle", "value"),
             Input("complexity-select", "value"),
             Input("forecast-method", "value"),
             Input("update-frequency", "value"),
             Input("cache-duration", "value"),
             Input("max-patterns", "value")]
        )
        def update_control_state(ticker, lookback, horizon, patterns, confidence,
                               realtime, live, complexity, forecast, update_freq,
                               cache_dur, max_patterns):
            """Update centralized control state."""
            return {
                "ticker": ticker,
                "lookback": lookback,
                "horizon": horizon,
                "patterns": patterns,
                "confidence": confidence,
                "realtime": realtime,
                "live": live,
                "complexity": complexity,
                "forecast": forecast,
                "update_frequency": update_freq,
                "cache_duration": cache_dur,
                "max_patterns": max_patterns,
                "timestamp": datetime.now().isoformat()
            }
        
        # Reset to defaults
        @app.callback(
            [Output("ticker-dropdown", "value"),
             Output("lookback-slider", "value"),
             Output("horizon-radio", "value"),
             Output("pattern-checklist", "value"),
             Output("confidence-slider", "value"),
             Output("realtime-toggle", "value"),
             Output("live-mode-toggle", "value"),
             Output("complexity-select", "value"),
             Output("forecast-method", "value"),
             Output("update-frequency", "value"),
             Output("cache-duration", "value"),
             Output("max-patterns", "value")],
            [Input("reset-btn", "n_clicks")]
        )
        def reset_to_defaults(n_clicks):
            """Reset all controls to default values."""
            if n_clicks:
                controls = DashboardControls()
                return (
                    controls.default_ticker,
                    controls.default_lookback,
                    controls.default_horizon,
                    controls.default_pattern_types,
                    controls.default_confidence,
                    False,  # realtime
                    False,  # live
                    controls.default_complexity,
                    controls.default_forecast_method,
                    5,      # update frequency
                    15,     # cache duration
                    20      # max patterns
                )
            return dash.no_update
        
        # Validation state
        @app.callback(
            Output("validation-state", "data"),
            [Input("control-state", "data")]
        )
        def validate_control_state(state):
            """Validate control state and return validation results."""
            if not state:
                return {"valid": False, "errors": ["No state available"]}
            
            errors = []
            
            # Validate lookback
            if state.get("lookback", 0) < 7:
                errors.append("Lookback must be at least 7 days")
            
            # Validate horizon
            if state.get("horizon", 0) < 1:
                errors.append("Horizon must be at least 1 day")
            
            # Validate patterns
            if not state.get("patterns"):
                errors.append("At least one pattern type must be selected")
            
            # Validate confidence
            confidence = state.get("confidence", 0)
            if confidence < 0 or confidence > 1:
                errors.append("Confidence must be between 0 and 1")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "timestamp": datetime.now().isoformat()
            }


# Helper functions for control state management
def get_control_state(control_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate control state."""
    if not control_state:
        controls = DashboardControls()
        return {
            "ticker": controls.default_ticker,
            "lookback": controls.default_lookback,
            "horizon": controls.default_horizon,
            "patterns": controls.default_pattern_types,
            "confidence": controls.default_confidence,
            "realtime": False,
            "live": False,
            "complexity": controls.default_complexity,
            "forecast": controls.default_forecast_method,
            "update_frequency": 5,
            "cache_duration": 15,
            "max_patterns": 20
        }
    return control_state


def validate_input_ranges(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input ranges for all controls."""
    errors = []
    
    # Lookback validation
    lookback = state.get("lookback", 0)
    if lookback < 7 or lookback > 365:
        errors.append(f"Lookback {lookback} out of range [7, 365]")
    
    # Horizon validation
    horizon = state.get("horizon", 0)
    if horizon < 1 or horizon > 90:
        errors.append(f"Horizon {horizon} out of range [1, 90]")
    
    # Confidence validation
    confidence = state.get("confidence", 0)
    if confidence < 0 or confidence > 1:
        errors.append(f"Confidence {confidence} out of range [0, 1]")
    
    # Update frequency validation
    update_freq = state.get("update_frequency", 5)
    if update_freq < 1 or update_freq > 60:
        errors.append(f"Update frequency {update_freq} out of range [1, 60]")
    
    return len(errors) == 0, errors


def format_control_summary(state: Dict[str, Any]) -> str:
    """Format control state into a readable summary."""
    summary = f"""
    Current Settings:
    - Ticker: {state.get('ticker', 'N/A')}
    - Lookback: {state.get('lookback', 'N/A')} days
    - Horizon: {state.get('horizon', 'N/A')} days
    - Patterns: {', '.join(state.get('patterns', []))}
    - Confidence: {state.get('confidence', 0) * 100:.0f}%
    - Mode: {'Live' if state.get('live') else 'Real-time' if state.get('realtime') else 'Historical'}
    - Complexity: {state.get('complexity', 'N/A')}
    - Forecast: {state.get('forecast', 'N/A').upper()}
    """
    return summary.strip()
