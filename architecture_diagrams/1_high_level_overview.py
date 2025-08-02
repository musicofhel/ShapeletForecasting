"""
High-Level System Architecture Overview
Shows the main components and their relationships
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import React, FastAPI
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.monitoring import Grafana
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.aws.analytics import Analytics
from diagrams.aws.ml import Sagemaker

with Diagram("Financial Wavelet Prediction - High Level Architecture", 
             filename="architecture_diagrams/1_high_level_overview", 
             show=False,
             direction="TB"):
    
    # Frontend
    with Cluster("Frontend Layer"):
        dashboard = React("Dash Dashboard\n(Plotly)")
        
    # API Layer
    with Cluster("API Layer"):
        api = FastAPI("FastAPI\nREST API")
        
    # Core Processing
    with Cluster("Core Processing"):
        with Cluster("Wavelet Analysis"):
            wavelet_analyzer = Python("Wavelet\nAnalyzer")
            pattern_detector = Python("Pattern\nDetector")
            shapelet_extractor = Python("Shapelet\nExtractor")
            
        with Cluster("Machine Learning"):
            ensemble = Sagemaker("Ensemble\nModel")
            xgboost = Python("XGBoost\nPredictor")
            transformer = Python("Transformer\nPredictor")
            
        with Cluster("Pattern Processing"):
            pattern_matcher = Python("Pattern\nMatcher")
            pattern_classifier = Python("Pattern\nClassifier")
            sequence_analyzer = Python("Sequence\nAnalyzer")
    
    # Data Layer
    with Cluster("Data Layer"):
        with Cluster("Data Sources"):
            yfinance = Analytics("YFinance\nAPI")
            polygon = Analytics("Polygon.io\nAPI")
            
        with Cluster("Storage"):
            market_db = PostgreSQL("Market\nData")
            pattern_storage = Storage("Pattern\nStorage")
            model_storage = Storage("Model\nStorage")
            
        with Cluster("Cache"):
            cache = Redis("Redis\nCache")
    
    # Advanced Features
    with Cluster("Advanced Features"):
        realtime = Rack("Realtime\nPipeline")
        backtester = Python("Backtesting\nEngine")
        optimizer = Python("Portfolio\nOptimizer")
        
    # Monitoring
    monitoring = Grafana("System\nMonitoring")
    
    # Connections
    dashboard >> Edge(label="HTTP/WebSocket") >> api
    api >> Edge(label="Process") >> [wavelet_analyzer, pattern_matcher, ensemble]
    
    [yfinance, polygon] >> Edge(label="Fetch") >> market_db
    market_db >> Edge(label="Load") >> wavelet_analyzer
    
    wavelet_analyzer >> pattern_detector >> shapelet_extractor
    shapelet_extractor >> pattern_storage
    
    pattern_storage >> [pattern_matcher, pattern_classifier]
    [pattern_matcher, pattern_classifier] >> sequence_analyzer
    
    sequence_analyzer >> [xgboost, transformer] >> ensemble
    ensemble >> model_storage
    
    [market_db, pattern_storage, model_storage] >> cache
    cache >> api
    
    realtime >> [market_db, pattern_matcher]
    backtester >> [market_db, ensemble]
    optimizer >> ensemble
    
    [api, realtime, backtester] >> monitoring
