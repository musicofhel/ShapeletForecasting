"""
Machine Learning Pipeline Architecture
Shows the ML models and their training/prediction flow
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.ml import Sagemaker, MachineLearning
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.programming.flowchart import Decision, Action
from diagrams.onprem.analytics import Spark

with Diagram("Financial Wavelet Prediction - ML Pipeline", 
             filename="architecture_diagrams/4_ml_pipeline", 
             show=False,
             direction="TB"):
    
    # Input Data
    with Cluster("Input Data"):
        market_data = Storage("Market Data")
        pattern_data = Storage("Pattern Data")
        features = Storage("Feature Store")
    
    # Feature Engineering
    with Cluster("Feature Engineering"):
        feature_pipeline = Python("feature_pipeline.py")
        feature_selector = Python("feature_selector.py")
        pattern_features = Python("pattern_features.py")
        technical_indicators = Python("technical_indicators.py")
        transition_matrix = Python("transition_matrix.py")
    
    # Model Training
    with Cluster("Model Training"):
        model_trainer = Python("model_trainer.py")
        hyperparameter_opt = Python("hyperparameter_optimizer.py")
        
        with Cluster("Individual Models"):
            xgboost = MachineLearning("XGBoost\nPredictor")
            transformer = MachineLearning("Transformer\nPredictor")
            sequence_pred = MachineLearning("Sequence\nPredictor")
            pattern_pred = MachineLearning("Pattern\nPredictor")
        
        ensemble = Sagemaker("Ensemble\nModel")
    
    # Model Evaluation
    with Cluster("Model Evaluation"):
        model_evaluator = Python("model_evaluator.py")
        backtester = Python("backtester.py")
        performance_reporter = Python("performance_reporter.py")
        risk_analyzer = Python("risk_analyzer.py")
    
    # Model Optimization
    with Cluster("Model Optimization"):
        model_compressor = Python("model_compressor.py")
        
    # Prediction Pipeline
    with Cluster("Prediction Pipeline"):
        predictor_service = Python("predictor_service.py")
        
        with Cluster("Advanced Features"):
            adaptive_learner = Python("adaptive_learner.py")
            market_regime = Python("market_regime_detector.py")
            multi_timeframe = Python("multi_timeframe_analyzer.py")
            portfolio_opt = Python("portfolio_optimizer.py")
            risk_manager = Python("risk_manager.py")
    
    # Model Storage
    with Cluster("Model Management"):
        model_storage = Storage("Model Storage")
        model_registry = Action("Model Registry")
        model_versioning = Decision("Version Control")
    
    # Real-time Processing
    with Cluster("Real-time Processing"):
        realtime_pipeline = Spark("realtime_pipeline.py")
        trading_simulator = Python("trading_simulator.py")
    
    # Flow - Feature Engineering
    [market_data, pattern_data] >> feature_pipeline
    feature_pipeline >> [pattern_features, technical_indicators, transition_matrix]
    [pattern_features, technical_indicators, transition_matrix] >> feature_selector
    feature_selector >> features
    
    # Flow - Training
    features >> model_trainer
    model_trainer >> hyperparameter_opt
    hyperparameter_opt >> [xgboost, transformer, sequence_pred, pattern_pred]
    [xgboost, transformer, sequence_pred, pattern_pred] >> ensemble
    
    # Flow - Evaluation
    ensemble >> model_evaluator
    model_evaluator >> [backtester, performance_reporter, risk_analyzer]
    
    # Flow - Optimization
    model_evaluator >> Edge(label="Optimize") >> model_compressor
    model_compressor >> model_storage
    
    # Flow - Storage
    ensemble >> model_versioning
    model_versioning >> Edge(label="New Version") >> model_registry
    model_versioning >> Edge(label="Update") >> model_storage
    model_registry >> model_storage
    
    # Flow - Prediction
    model_storage >> predictor_service
    predictor_service >> [adaptive_learner, market_regime, multi_timeframe]
    [market_regime, multi_timeframe] >> portfolio_opt
    portfolio_opt >> risk_manager
    
    # Flow - Real-time
    predictor_service >> realtime_pipeline
    realtime_pipeline >> trading_simulator
    trading_simulator >> Edge(label="Feedback") >> adaptive_learner
