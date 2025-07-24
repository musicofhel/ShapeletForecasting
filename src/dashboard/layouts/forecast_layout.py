"""
Forecast Dashboard Layout Module

This module defines the layout structure for the forecasting dashboard,
including all UI components and responsive design elements.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from typing import List, Dict, Any

def create_header() -> dbc.Navbar:
    """Create dashboard header with navigation"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-chart-line me-2"),
                        html.H4("Wavelet Forecast Dashboard", className="mb-0 text-white")
                    ], className="d-flex align-items-center")
                ], width="auto"),
                dbc.Col([
                    html.Div(id="performance-status", className="text-end")
                ], className="ms-auto")
            ], className="g-0 w-100 align-items-center")
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-3"
    )

def create_progress_panel() -> dbc.Card:
    """Create progress panel with terminal output"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-tasks me-2"),
                    "Processing Status"
                ], width="auto"),
                dbc.Col([
                    dbc.Button(
                        html.I(className="fas fa-chevron-up"),
                        id="toggle-progress",
                        color="link",
                        size="sm",
                        className="text-white"
                    )
                ], className="ms-auto")
            ], className="align-items-center")
        ], className="bg-dark text-white"),
        dbc.Collapse([
            dbc.CardBody([
                # Progress bar
                html.Div([
                    html.Div([
                        html.Span(id="progress-task-name", children="Initializing...", 
                                className="small text-muted"),
                        html.Span(id="progress-percentage", children="0%", 
                                className="small text-muted float-end")
                    ], className="d-flex justify-content-between mb-1"),
                    dbc.Progress(
                        id="task-progress-bar",
                        value=0,
                        striped=True,
                        animated=True,
                        color="primary",
                        className="mb-3"
                    )
                ]),
                
                # Terminal output
                html.Div([
                    html.Pre(
                        id="terminal-output",
                        children="$ Starting Wavelet Forecast Dashboard...\n",
                        style={
                            "backgroundColor": "#1e1e1e",
                            "color": "#00ff00",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "fontFamily": "Consolas, Monaco, 'Courier New', monospace",
                            "fontSize": "12px",
                            "height": "150px",
                            "overflowY": "auto",
                            "margin": "0"
                        }
                    )
                ]),
                
                # Estimated time
                html.Div([
                    html.Small(id="estimated-time", children="", className="text-muted")
                ], className="mt-2 text-end")
            ])
        ], id="progress-collapse", is_open=True)
    ], className="mb-3", style={"borderColor": "#333"})

def create_control_panel() -> dbc.Card:
    """Create control panel with filters and settings"""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-sliders-h me-2"),
            "Control Panel"
        ]),
        dbc.CardBody([
            # Symbol selection
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol", html_for="symbol-dropdown"),
                    dcc.Dropdown(
                        id="symbol-dropdown",
                        options=[
                            {"label": "BTC/USD", "value": "BTCUSD"},
                            {"label": "ETH/USD", "value": "ETHUSD"},
                            {"label": "SPY", "value": "SPY"},
                            {"label": "AAPL", "value": "AAPL"}
                        ],
                        value="BTCUSD",
                        clearable=False
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Timeframe", html_for="timeframe-dropdown"),
                    dcc.Dropdown(
                        id="timeframe-dropdown",
                        options=[
                            {"label": "1 Hour", "value": "1 Hour"},
                            {"label": "1 Day", "value": "1 Day"},
                            {"label": "1 Week", "value": "1 Week"},
                            {"label": "1 Month", "value": "1 Month"}
                        ],
                        value="1 Hour",
                        clearable=False
                    )
                ], md=6)
            ], className="mb-3"),
            
            # Pattern display options
            dbc.Row([
                dbc.Col([
                    dbc.Label("Display Options"),
                    dbc.Checklist(
                        id="pattern-toggle",
                        options=[
                            {"label": "Show Pattern Overlays", "value": "patterns"},
                            {"label": "Show Predictions", "value": "predictions"},
                            {"label": "Show Confidence Bands", "value": "confidence"}
                        ],
                        value=["patterns"],
                        inline=True
                    )
                ])
            ], className="mb-3"),
            
            # Prediction controls
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Prediction Horizon"),
                    dcc.Slider(
                        id="prediction-horizon-slider",
                        min=1,
                        max=50,
                        step=1,
                        value=10,
                        marks={i: str(i) for i in [1, 10, 20, 30, 40, 50]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-magic me-2"), "Generate Prediction"],
                        id="predict-button",
                        color="primary",
                        className="w-100"
                    )
                ])
            ])
        ])
    ])

def create_main_chart_section() -> dbc.Card:
    """Create main time series chart section"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-chart-area me-2"),
                    "Time Series Analysis"
                ], width="auto"),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            html.I(className="fas fa-sync-alt"),
                            id="refresh-chart",
                            color="light",
                            size="sm",
                            title="Refresh"
                        ),
                        dbc.Button(
                            html.I(className="fas fa-download"),
                            id="download-chart",
                            color="light",
                            size="sm",
                            title="Download"
                        ),
                        dbc.Button(
                            html.I(className="fas fa-expand"),
                            id="fullscreen-chart",
                            color="light",
                            size="sm",
                            title="Fullscreen"
                        )
                    ])
                ], className="ms-auto")
            ], className="align-items-center")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-main-chart",
                type="default",
                children=[
                    dcc.Graph(
                        id="main-time-series",
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                        }
                    )
                ]
            )
        ])
    ], className="mb-3")

def create_pattern_sequence_section() -> dbc.Card:
    """Create pattern sequence visualization section"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-project-diagram me-2"),
                    "Pattern Sequence Analysis"
                ], width="auto"),
                dbc.Col([
                    dbc.Label("Sequence Length:", className="me-2 mb-0"),
                    dcc.Slider(
                        id="sequence-length-slider",
                        min=3,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 11)},
                        className="flex-grow-1"
                    )
                ], className="ms-auto d-flex align-items-center", width=6)
            ], className="align-items-center")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-pattern-sequence",
                type="default",
                children=[
                    dcc.Graph(
                        id="pattern-sequence-viz",
                        config={'displayModeBar': False}
                    )
                ]
            )
        ])
    ], className="mb-3")

def create_prediction_panel() -> dbc.Card:
    """Create prediction display panel"""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-crystal-ball me-2"),
            "Pattern Predictions"
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-predictions",
                type="default",
                children=[
                    html.Div(id="prediction-display")
                ]
            )
        ])
    ], className="mb-3")

def create_metrics_panel() -> dbc.Card:
    """Create accuracy metrics panel"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Performance Metrics"
                ], width="auto"),
                dbc.Col([
                    dcc.Dropdown(
                        id="metric-type-dropdown",
                        options=[
                            {"label": "Accuracy Metrics", "value": "accuracy"},
                            {"label": "Return Analysis", "value": "returns"},
                            {"label": "Confusion Matrix", "value": "confusion"}
                        ],
                        value="accuracy",
                        clearable=False,
                        style={"minWidth": "200px"}
                    )
                ], className="ms-auto")
            ], className="align-items-center")
        ]),
        dbc.CardBody([
            dcc.Loading(
                id="loading-metrics",
                type="default",
                children=[
                    dcc.Graph(
                        id="accuracy-metrics",
                        config={'displayModeBar': False}
                    )
                ]
            )
        ])
    ])

def create_pattern_explorer() -> dbc.Card:
    """Create pattern exploration tools"""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-search me-2"),
            "Pattern Explorer"
        ]),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Pattern Library", tab_id="library"),
                dbc.Tab(label="Pattern Details", tab_id="details"),
                dbc.Tab(label="Historical Performance", tab_id="history")
            ], id="pattern-tabs", active_tab="library"),
            html.Div(id="pattern-tab-content", className="mt-3")
        ])
    ])

def create_footer() -> html.Div:
    """Create dashboard footer"""
    return html.Div([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P([
                    html.I(className="fas fa-info-circle me-2"),
                    "Wavelet Forecast Dashboard v1.0 | ",
                    html.A("Documentation", href="#", target="_blank"),
                    " | ",
                    html.A("GitHub", href="#", target="_blank")
                ], className="text-muted mb-0")
            ]),
            dbc.Col([
                html.P([
                    "Last updated: ",
                    html.Span(id="last-update-time", children="Never")
                ], className="text-muted mb-0 text-end")
            ])
        ])
    ], className="mt-4")

def create_forecast_layout() -> html.Div:
    """Create the main dashboard layout"""
    return html.Div([
        # Meta tags for mobile responsiveness
        html.Meta(name="viewport", content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"),
        
        # Header
        create_header(),
        
        # Main content
        dbc.Container([
            # Progress panel (shown when processing)
            html.Div([
                create_progress_panel()
            ], id="progress-panel-container", style={"display": "none"}),
            
            # Control panel row
            dbc.Row([
                dbc.Col([
                    create_control_panel()
                ], lg=3, md=4, className="mb-3 mb-lg-0"),
                
                # Main content area
                dbc.Col([
                    # Main chart
                    create_main_chart_section(),
                    
                    # Pattern sequence and predictions row
                    dbc.Row([
                        dbc.Col([
                            create_pattern_sequence_section()
                        ], lg=8, className="mb-3 mb-lg-0"),
                        dbc.Col([
                            create_prediction_panel()
                        ], lg=4)
                    ], className="mb-3"),
                    
                    # Metrics row
                    dbc.Row([
                        dbc.Col([
                            create_metrics_panel()
                        ], lg=6, className="mb-3 mb-lg-0"),
                        dbc.Col([
                            create_pattern_explorer()
                        ], lg=6)
                    ])
                ], lg=9, md=8)
            ]),
            
            # Footer
            create_footer()
        ], fluid=True),
        
        # Hidden components
        dcc.Store(id="pattern-overlay-store"),
        dcc.Store(id="prediction-store"),
        dcc.Store(id="error-store"),
        dcc.Store(id="progress-store", data={
            'current_task': 'Initializing...',
            'progress': 0,
            'subtasks': [],
            'estimated_time': 0
        }),
        
        # Intervals for auto-refresh
        dcc.Interval(id="refresh-interval", interval=30000),  # 30 seconds
        dcc.Interval(id="metrics-interval", interval=60000),  # 1 minute
        dcc.Interval(id="performance-interval", interval=5000),  # 5 seconds
        
        # Error modal
        dbc.Modal([
            dbc.ModalHeader("Error"),
            dbc.ModalBody(id="error-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-error-modal", className="ms-auto")
            )
        ], id="error-modal", is_open=False),
        
        # Loading overlay
        dcc.Loading(
            id="page-loading",
            type="default",
            fullscreen=True,
            style={"backgroundColor": "rgba(255, 255, 255, 0.8)"}
        )
    ], id="app-container")

# Responsive breakpoints for different screen sizes
RESPONSIVE_BREAKPOINTS = {
    'mobile': 576,
    'tablet': 768,
    'desktop': 992,
    'large': 1200
}

def get_responsive_layout_config() -> Dict[str, Any]:
    """Get responsive layout configuration"""
    return {
        'mobile': {
            'chart_height': 400,
            'card_spacing': 'mb-2',
            'font_size': '14px'
        },
        'tablet': {
            'chart_height': 500,
            'card_spacing': 'mb-3',
            'font_size': '15px'
        },
        'desktop': {
            'chart_height': 600,
            'card_spacing': 'mb-3',
            'font_size': '16px'
        }
    }
