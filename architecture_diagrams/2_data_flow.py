"""
Data Flow Architecture
Shows how data flows through the system from ingestion to prediction
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import DataPipeline, Analytics
from diagrams.aws.storage import S3
from diagrams.programming.flowchart import StartEnd, Decision, Action, Document
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.generic.blank import Blank

with Diagram("Financial Wavelet Prediction - Data Flow", 
             filename="architecture_diagrams/2_data_flow", 
             show=False,
             direction="LR"):
    
    # Data Sources
    with Cluster("External Data Sources"):
        start = StartEnd("Start")
        yfinance = Analytics("YFinance API")
        polygon = Analytics("Polygon.io API")
    
    # Data Ingestion
    with Cluster("Data Ingestion Layer"):
        data_manager = Action("Data Manager")
        rate_limiter = Decision("Rate Limiter")
        validator = Action("Data Validator")
    
    # Storage
    with Cluster("Storage Layer"):
        raw_storage = PostgreSQL("Raw Market Data")
        cache = Redis("Redis Cache")
    
    # Wavelet Processing
    with Cluster("Wavelet Analysis"):
        preprocessor = Action("Preprocessor")
        wavelet_transform = Action("Wavelet Transform")
        pattern_extractor = Action("Pattern Extractor")
        shapelet_miner = Action("Shapelet Miner")
    
    # Pattern Storage
    with Cluster("Pattern Management"):
        pattern_db = S3("Pattern Storage")
        pattern_index = Document("Pattern Index")
    
    # Feature Engineering
    with Cluster("Feature Engineering"):
        feature_extractor = Action("Feature Extractor")
        technical_indicators = Action("Technical Indicators")
        transition_matrix = Action("Transition Matrix")
    
    # Model Processing
    with Cluster("ML Pipeline"):
        pattern_matcher = Action("Pattern Matcher")
        sequence_builder = Action("Sequence Builder")
        ensemble_predictor = Action("Ensemble Predictor")
    
    # Output
    with Cluster("Output Layer"):
        forecast = Document("Forecast Results")
        api_response = DataPipeline("API Response")
        dashboard_update = Action("Dashboard Update")
        end = StartEnd("End")
    
    # Flow connections
    start >> [yfinance, polygon]
    [yfinance, polygon] >> data_manager
    data_manager >> rate_limiter
    rate_limiter >> Edge(label="Pass") >> validator
    rate_limiter >> Edge(label="Wait") >> data_manager
    
    validator >> [raw_storage, cache]
    raw_storage >> preprocessor
    cache >> Edge(label="Fast Access") >> preprocessor
    
    preprocessor >> wavelet_transform
    wavelet_transform >> pattern_extractor
    pattern_extractor >> shapelet_miner
    shapelet_miner >> [pattern_db, pattern_index]
    
    preprocessor >> feature_extractor
    feature_extractor >> [technical_indicators, transition_matrix]
    
    [pattern_db, technical_indicators, transition_matrix] >> pattern_matcher
    pattern_matcher >> sequence_builder
    sequence_builder >> ensemble_predictor
    
    ensemble_predictor >> forecast
    forecast >> [api_response, dashboard_update]
    [api_response, dashboard_update] >> end
