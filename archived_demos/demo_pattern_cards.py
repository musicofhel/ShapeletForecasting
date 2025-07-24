"""
Demo script for Pattern Information Cards Component

This script demonstrates the comprehensive pattern information cards that display:
- Full pattern visualization with thumbnails
- Discovery timestamp and duration
- Statistical properties (mean, std, trend, energy)
- Quality metrics and confidence scores
- List of all occurrences with timestamps
- Associated predictions and their accuracy
- Expand/collapse functionality
- Pattern comparison and export features
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from src.dashboard.components.pattern_cards import PatternCards, PatternInfo, generate_demo_pattern_info
import webbrowser
from threading import Timer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
                ],
                suppress_callback_exceptions=True)

# Custom CSS for dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0E1117;
                color: #FAFAFA;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }
            
            .pattern-info-card {
                background-color: #1E1E1E;
                border: 1px solid #333;
                transition: all 0.3s ease;
            }
            
            .pattern-info-card:hover {
                border-color: #00D9FF;
                box-shadow: 0 4px 12px rgba(0, 217, 255, 0.2);
            }
            
            .card-header {
                background-color: #262626;
                border-bottom: 1px solid #333;
            }
            
            .card-body {
                background-color: #1E1E1E;
            }
            
            .nav-tabs .nav-link {
                color: #888;
                background-color: transparent;
                border: 1px solid transparent;
            }
            
            .nav-tabs .nav-link.active {
                color: #00D9FF;
                background-color: #1E1E1E;
                border-color: #333 #333 #1E1E1E;
            }
            
            .nav-tabs .nav-link:hover {
                color: #00D9FF;
                border-color: #333;
            }
            
            .badge {
                font-weight: 500;
            }
            
            .progress {
                background-color: #333;
            }
            
            .btn-link {
                color: #00D9FF;
                text-decoration: none;
            }
            
            .btn-link:hover {
                color: #00A8CC;
            }
            
            .list-group-item {
                background-color: #262626;
                border: 1px solid #333;
                color: #FAFAFA;
            }
            
            .table {
                color: #FAFAFA;
            }
            
            .modal-content {
                background-color: #1E1E1E;
                border: 1px solid #333;
            }
            
            .modal-header {
                background-color: #262626;
                border-bottom: 1px solid #333;
            }
            
            .modal-body {
                background-color: #1E1E1E;
            }
            
            .modal-footer {
                background-color: #262626;
                border-top: 1px solid #333;
            }
            
            .form-label {
                color: #888;
                font-size: 0.875rem;
                margin-bottom: 0.25rem;
            }
            
            .form-select, .form-control {
                background-color: #262626;
                border: 1px solid #333;
                color: #FAFAFA;
            }
            
            .form-select:focus, .form-control:focus {
                background-color: #262626;
                border-color: #00D9FF;
                color: #FAFAFA;
                box-shadow: 0 0 0 0.2rem rgba(0, 217, 255, 0.25);
            }
            
            .alert-info {
                background-color: #1E3A5F;
                border-color: #2E5A8F;
                color: #B8D4F1;
            }
            
            .border-bottom {
                border-color: #333 !important;
            }
            
            hr {
                border-color: #333;
                opacity: 0.5;
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1E1E1E;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #333;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #444;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize components
pattern_cards = PatternCards()

# Generate demo patterns
demo_patterns = generate_demo_pattern_info(num_patterns=8)

# Create app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Pattern Information Cards Demo", className="text-center mb-4"),
            html.P(
                "Comprehensive pattern analysis with detailed statistics, occurrences, and predictions",
                className="text-center text-muted mb-5"
            )
        ])
    ]),
    
    # Main content
    dbc.Row([
        dbc.Col([
            # Pattern cards
            pattern_cards.create_pattern_cards_layout(demo_patterns, max_cards=10),
            
            # Hidden components for callbacks
            dcc.Download(id="download-pattern-data"),
            html.Div(id="main-chart-highlight", style={"display": "none"}),
            
            # Modal for occurrence details
            dbc.Modal([
                dbc.ModalHeader("Occurrence Details"),
                dbc.ModalBody(id="occurrence-detail-content"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ml-auto")
                )
            ], id="occurrence-detail-modal", is_open=False),
            
            # Info section
            dbc.Card([
                dbc.CardBody([
                    html.H5("Features Demonstrated:", className="mb-3"),
                    html.Ul([
                        html.Li("Pattern thumbnails with trend lines"),
                        html.Li("Quality metrics and confidence scores"),
                        html.Li("Statistical properties (mean, std, trend, energy, entropy)"),
                        html.Li("Expandable cards with detailed analysis tabs"),
                        html.Li("Pattern occurrence timeline and history"),
                        html.Li("Prediction accuracy and performance metrics"),
                        html.Li("Market conditions and trading implications"),
                        html.Li("Pattern comparison and export functionality"),
                        html.Li("Interactive filtering and sorting"),
                        html.Li("Locate patterns on main chart"),
                    ]),
                    html.Hr(),
                    html.H6("Try these actions:"),
                    html.Ul([
                        html.Li("Click expand/collapse buttons to see detailed analysis"),
                        html.Li("Use the filter controls to sort and filter patterns"),
                        html.Li("Click 'View' on occurrences to see details"),
                        html.Li("Use the comparison button to select patterns"),
                        html.Li("Export individual patterns or all patterns"),
                        html.Li("Expand/collapse all cards at once"),
                    ])
                ])
            ], className="mt-5", style={"backgroundColor": "#262626", "border": "1px solid #333"})
        ], width=12)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.P(
                "Pattern Information Cards - Part of Financial Wavelet Prediction Dashboard",
                className="text-center text-muted"
            )
        ])
    ], className="mt-5")
], fluid=True, style={"maxWidth": "1400px"})

# Register callbacks
PatternCards.register_callbacks(app)

# Additional callback to close modal
@app.callback(
    Output("occurrence-detail-modal", "is_open", allow_duplicate=True),
    Input("close-modal", "n_clicks"),
    State("occurrence-detail-modal", "is_open"),
    prevent_initial_call=True
)
def close_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open

# Callback to handle main chart highlighting (demo only)
@app.callback(
    Output("main-chart-highlight", "children"),
    Input("main-chart-highlight", "data"),
    prevent_initial_call=True
)
def handle_chart_highlight(highlight_data):
    if highlight_data:
        logger.info(f"Pattern located on chart: {highlight_data}")
    return ""

# Callback to handle pattern comparison (demo only)
@app.callback(
    Output("comparison-patterns-store", "children"),
    Input("comparison-patterns-store", "data"),
    prevent_initial_call=True
)
def handle_comparison_update(comparison_data):
    if comparison_data:
        logger.info(f"Patterns selected for comparison: {comparison_data}")
    return ""

def open_browser():
    """Open the web browser to the app URL"""
    webbrowser.open_new("http://localhost:8050")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PATTERN INFORMATION CARDS DEMO")
    print("="*50)
    print("\nFeatures:")
    print("- Comprehensive pattern analysis cards")
    print("- Statistical properties and quality metrics")
    print("- Occurrence history and predictions")
    print("- Expandable detailed views")
    print("- Pattern comparison and export")
    print("- Interactive filtering and sorting")
    print("\nStarting server at http://localhost:8050")
    print("="*50 + "\n")
    
    # Open browser after a short delay
    Timer(1.5, open_browser).start()
    
    # Run the app
    app.run_server(debug=True, port=8050)
