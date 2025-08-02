"""
Dashboard Architecture
Shows the structure of the Dash/Plotly dashboard application
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import React, Flask
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.generic.blank import Blank
from diagrams.aws.analytics import Analytics
from diagrams.onprem.inmemory import Redis

with Diagram("Financial Wavelet Prediction - Dashboard Architecture", 
             filename="architecture_diagrams/3_dashboard_architecture", 
             show=False,
             direction="TB"):
    
    # Users
    users = User("Users")
    
    # Main Dashboard App
    with Cluster("Dash Application"):
        app = Flask("forecast_app.py")
        
        # Layouts
        with Cluster("Layouts"):
            forecast_layout = Python("forecast_layout.py")
            
        # Components
        with Cluster("Components"):
            sidebar = Python("sidebar.py")
            controls = Python("controls.py")
            pattern_cards = Python("pattern_cards.py")
            
        # Visualizations
        with Cluster("Visualizations"):
            sequence_view = Python("sequence_view.py")
            forecast_view = Python("forecast_view.py")
            accuracy_metrics = Python("accuracy_metrics.py")
            scalogram = Python("scalogram.py")
            timeseries = Python("timeseries.py")
            pattern_gallery = Python("pattern_gallery.py")
            analytics = Python("analytics.py")
            pattern_comparison = Python("pattern_comparison.py")
            
        # Callbacks
        with Cluster("Callbacks"):
            prediction_callbacks = Python("prediction_callbacks.py")
            
    # Core Services
    with Cluster("Core Services"):
        # Pattern Analysis
        with Cluster("Pattern Analysis"):
            pattern_classifier = Python("pattern_classifier.py")
            pattern_matcher = Python("pattern_matcher.py")
            pattern_predictor = Python("pattern_predictor.py")
            pattern_detection = Python("pattern_detection.py")
            wavelet_analyzer = Python("wavelet_sequence_analyzer.py")
            
        # Data Management
        with Cluster("Data Management"):
            data_utils = Python("data_utils.py")
            data_utils_yf = Python("data_utils_yfinance.py")
            data_utils_poly = Python("data_utils_polygon.py")
            model_loader = Python("model_loader.py")
            
        # Tools
        with Cluster("Tools"):
            pattern_compare = Python("pattern_compare.py")
            pattern_explorer = Python("pattern_explorer.py")
            
        # Evaluation
        with Cluster("Evaluation"):
            forecast_backtester = Python("forecast_backtester.py")
            model_comparison = Python("model_comparison.py")
            
        # Advanced Features
        with Cluster("Advanced"):
            multi_step = Python("multi_step_forecast.py")
            
        # Real-time
        with Cluster("Real-time"):
            pattern_monitor = Python("pattern_monitor.py")
            
        # Search
        with Cluster("Search"):
            pattern_search = Python("pattern_search.py")
            
        # Export
        with Cluster("Export"):
            report_generator = Python("report_generator.py")
            
        # Optimization
        with Cluster("Optimization"):
            cache_manager = Python("cache_manager.py")
    
    # External Services
    with Cluster("External Services"):
        cache = Redis("Redis Cache")
        yfinance_api = Analytics("YFinance API")
        polygon_api = Analytics("Polygon API")
    
    # Flow
    users >> app
    app >> forecast_layout
    forecast_layout >> [sidebar, controls, pattern_cards]
    
    [sidebar, controls] >> prediction_callbacks
    prediction_callbacks >> [pattern_classifier, pattern_matcher, pattern_predictor]
    
    pattern_cards >> [sequence_view, forecast_view, accuracy_metrics]
    
    [pattern_classifier, pattern_matcher, pattern_predictor] >> wavelet_analyzer
    wavelet_analyzer >> pattern_detection
    
    [data_utils_yf, data_utils_poly] >> data_utils
    data_utils >> [yfinance_api, polygon_api]
    
    model_loader >> [pattern_predictor, wavelet_analyzer]
    
    cache_manager >> cache
    data_utils >> cache_manager
    
    [pattern_compare, pattern_explorer] >> pattern_matcher
    [forecast_backtester, model_comparison] >> pattern_predictor
    
    multi_step >> pattern_predictor
    pattern_monitor >> [pattern_matcher, data_utils]
    pattern_search >> pattern_matcher
    report_generator >> [forecast_view, accuracy_metrics]
    
    # Visualization connections
    [scalogram, timeseries, pattern_gallery, analytics, pattern_comparison] >> Edge(label="Display") >> app
