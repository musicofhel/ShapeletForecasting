"""
FastAPI application for Financial Wavelet Prediction API
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import os
import sys
import json
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthResponse, ErrorResponse,
    MetricsResponse, StreamPredictionRequest, StreamPredictionResponse,
    ModelUpdateRequest, ModelUpdateResponse
)

# Import pattern-related services
try:
    from ..features.pattern_matcher import PatternMatcher
    from ..features.pattern_predictor import PatternPredictor
    from ..dashboard.search.pattern_search import PatternSearch
    from ..dashboard.visualizations.analytics_simple import PatternAnalytics
except ImportError:
    PatternMatcher = None
    PatternPredictor = None
    PatternSearch = None
    PatternAnalytics = None
from .predictor_service import PredictorService, StreamingPredictor
from .monitoring import MetricsCollector, RequestLogger, AlertManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
predictor_service: Optional[PredictorService] = None
metrics_collector: Optional[MetricsCollector] = None
request_logger: Optional[RequestLogger] = None
alert_manager: Optional[AlertManager] = None
streaming_predictors = {}
pattern_matcher: Optional[PatternMatcher] = None
pattern_predictor: Optional[PatternPredictor] = None
pattern_search: Optional[PatternSearch] = None
pattern_analytics: Optional[PatternAnalytics] = None
websocket_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global predictor_service, metrics_collector, request_logger, alert_manager
    global pattern_matcher, pattern_predictor, pattern_search, pattern_analytics
    
    # Startup
    logger.info("Starting Financial Wavelet Prediction API...")
    
    # Initialize services
    model_dir = os.environ.get("MODEL_DIR", "models/")
    predictor_service = PredictorService(model_dir=model_dir)
    metrics_collector = MetricsCollector()
    request_logger = RequestLogger(log_file="api_requests.log")
    alert_manager = AlertManager(metrics_collector)
    
    # Initialize pattern services if available
    if PatternMatcher:
        pattern_matcher = PatternMatcher()
    if PatternPredictor:
        pattern_predictor = PatternPredictor()
    if PatternSearch:
        pattern_search = PatternSearch()
    if PatternAnalytics:
        pattern_analytics = PatternAnalytics()
    
    logger.info("API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    # Close all WebSocket connections
    for connection in websocket_connections:
        await connection.close()


# Create FastAPI app
app = FastAPI(
    title="Financial Wavelet Prediction API",
    description="REST API for financial time series prediction using wavelet analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all requests"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log request
    if request_logger:
        request_logger.log_request(
            request_id=request_id,
            method=request.method,
            endpoint=str(request.url.path),
            headers=dict(request.headers)
        )
    
    # Record request metrics
    if metrics_collector:
        body_size = int(request.headers.get("content-length", 0))
        metrics_collector.record_request(
            endpoint=str(request.url.path),
            method=request.method,
            request_size=body_size
        )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        response_time = time.time() - start_time
        if request_logger:
            request_logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                response_time=response_time
            )
        
        # Record response metrics
        if metrics_collector:
            metrics_collector.record_response(
                status_code=response.status_code,
                response_time=response_time,
                response_size=int(response.headers.get("content-length", 0))
            )
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(response_time)
        
        return response
        
    except Exception as e:
        # Log error
        response_time = time.time() - start_time
        if request_logger:
            request_logger.log_response(
                request_id=request_id,
                status_code=500,
                response_time=response_time,
                error=str(e)
            )
        
        # Record error metrics
        if metrics_collector:
            metrics_collector.record_response(
                status_code=500,
                response_time=response_time,
                response_size=0,
                error=str(e)
            )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e),
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )


# Dependency to get predictor service
def get_predictor() -> PredictorService:
    """Get predictor service instance"""
    if predictor_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service not initialized"
        )
    return predictor_service


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status"""
    if predictor_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    # Get loaded models
    models_loaded = {
        model_type: True for model_type in predictor_service.models.keys()
    }
    
    # Calculate uptime
    uptime = 0
    if metrics_collector:
        metrics = metrics_collector.get_metrics()
        uptime = metrics.get('uptime_seconds', 0)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=models_loaded,
        version="1.0.0",
        uptime_seconds=uptime
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    predictor: PredictorService = Depends(get_predictor)
):
    """Make a single prediction"""
    start_time = time.time()
    
    try:
        # Record model usage
        if metrics_collector:
            metrics_collector.model_usage[request.model_type.value] += 1
        
        # Make prediction
        prediction, confidence = predictor.predict(
            features=request.features,
            model_type=request.model_type.value
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_type=request.model_type.value,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: PredictorService = Depends(get_predictor)
):
    """Make batch predictions"""
    start_time = time.time()
    
    try:
        # Make predictions
        results = predictor.predict_batch(
            samples=request.samples,
            model_type=request.model_type.value,
            return_confidence=request.return_confidence
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        avg_time = processing_time / len(request.samples)
        
        return BatchPredictionResponse(
            predictions=results['predictions'],
            confidences=results.get('confidences'),
            model_type=request.model_type.value,
            timestamp=datetime.now(),
            total_samples=len(request.samples),
            processing_time_ms=processing_time,
            avg_time_per_sample_ms=avg_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


# Streaming prediction endpoint
@app.post("/predict/stream", response_model=StreamPredictionResponse, tags=["Predictions"])
async def predict_stream(
    request: StreamPredictionRequest,
    predictor: PredictorService = Depends(get_predictor)
):
    """Make streaming prediction with trend analysis"""
    try:
        # Get or create streaming predictor for this session
        session_id = str(uuid.uuid4())  # In production, use actual session management
        
        if session_id not in streaming_predictors:
            streaming_predictors[session_id] = StreamingPredictor(
                predictor_service=predictor,
                window_size=request.window_size
            )
        
        streamer = streaming_predictors[session_id]
        
        # Make streaming prediction
        result = streamer.predict_stream(
            features=request.features,
            model_type=request.model_type.value
        )
        
        return StreamPredictionResponse(
            prediction=result['prediction'],
            trend=result['trend'],
            change_from_previous=result['change_from_previous'],
            moving_average=result['moving_average'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Streaming prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming prediction failed"
        )


# Model info endpoint
@app.get("/models", response_model=dict, tags=["Models"])
async def get_models(predictor: PredictorService = Depends(get_predictor)):
    """Get information about all loaded models"""
    try:
        return predictor.get_all_models_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


# Specific model info endpoint
@app.get("/models/{model_type}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(
    model_type: str,
    predictor: PredictorService = Depends(get_predictor)
):
    """Get information about a specific model"""
    try:
        info = predictor.get_model_info(model_type)
        
        return ModelInfo(
            model_type=model_type,
            version=info.get('version', '1.0.0'),
            trained_date=datetime.fromisoformat(info.get('trained_date', datetime.now().isoformat())),
            features=info.get('features', []),
            performance_metrics=info.get('performance_metrics', {}),
            compression_stats=info.get('compression_stats')
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


# Update model endpoint
@app.post("/models/update", response_model=ModelUpdateResponse, tags=["Models"])
async def update_model(
    request: ModelUpdateRequest,
    predictor: PredictorService = Depends(get_predictor)
):
    """Update a model (requires authentication in production)"""
    try:
        # Get current version
        current_info = predictor.get_model_info(request.model_type.value)
        previous_version = current_info.get('version', 'unknown')
        
        # Update model
        success = predictor.update_model(
            model_type=request.model_type.value,
            model_path=request.model_path,
            version=request.version
        )
        
        return ModelUpdateResponse(
            success=success,
            model_type=request.model_type.value,
            previous_version=previous_version,
            new_version=request.version if success else previous_version,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Model update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}"
        )


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """Get API metrics"""
    if metrics_collector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector not initialized"
        )
    
    metrics = metrics_collector.get_metrics()
    
    return MetricsResponse(
        total_requests=metrics['total_requests'],
        successful_requests=metrics['successful_requests'],
        failed_requests=metrics['failed_requests'],
        avg_response_time_ms=metrics['avg_response_time_ms'],
        requests_per_minute=metrics['requests_per_minute'],
        model_usage=metrics['model_usage'],
        error_rate=metrics['error_rate']
    )


# Alerts endpoint
@app.get("/alerts", response_model=list, tags=["System"])
async def get_alerts(limit: int = 10):
    """Get recent alerts"""
    if alert_manager is None:
        return []
    
    # Check for new alerts
    alert_manager.check_alerts()
    
    # Return recent alerts
    return alert_manager.get_recent_alerts(limit=limit)


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "message": "Financial Wavelet Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Pattern endpoints
@app.get("/api/patterns/{ticker}", tags=["Patterns"])
async def get_patterns(
    ticker: str,
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get patterns for a specific ticker"""
    try:
        if not pattern_matcher:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pattern matcher service not available"
            )
        
        # Mock response for now - replace with actual pattern matching logic
        patterns = []
        for i in range(min(limit, 10)):
            patterns.append({
                "id": f"pattern_{ticker}_{i}",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "pattern_type": ["bullish", "bearish", "neutral"][i % 3],
                "confidence": 0.7 + (i * 0.02),
                "duration": 24 + (i * 12),
                "metrics": {
                    "strength": 0.8 + (i * 0.01),
                    "reliability": 0.75 + (i * 0.015),
                    "frequency": 5 + i
                }
            })
        
        return {
            "ticker": ticker,
            "patterns": patterns,
            "total": len(patterns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get patterns: {str(e)}"
        )


@app.get("/api/predictions/{ticker}", tags=["Predictions"])
async def get_predictions(
    ticker: str,
    horizon: int = 24,
    model_type: Optional[str] = "ensemble"
):
    """Get predictions for a specific ticker"""
    try:
        if not pattern_predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pattern predictor service not available"
            )
        
        # Mock response for now - replace with actual prediction logic
        predictions = []
        base_value = 100.0
        for i in range(horizon):
            predictions.append({
                "timestamp": datetime.now().isoformat(),
                "horizon": i + 1,
                "value": base_value + (i * 0.5),
                "confidence": 0.9 - (i * 0.01),
                "upper_bound": base_value + (i * 0.5) + 2.0,
                "lower_bound": base_value + (i * 0.5) - 2.0
            })
        
        return {
            "ticker": ticker,
            "predictions": predictions,
            "model_type": model_type,
            "horizon": horizon,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}"
        )


@app.post("/api/patterns/search", tags=["Patterns"])
async def search_patterns(
    query: Dict[str, Any]
):
    """Search for patterns based on criteria"""
    try:
        if not pattern_search:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pattern search service not available"
            )
        
        # Extract search parameters
        tickers = query.get("tickers", [])
        pattern_types = query.get("pattern_types", [])
        min_confidence = query.get("min_confidence", 0.0)
        max_results = query.get("max_results", 100)
        
        # Mock response for now - replace with actual search logic
        results = []
        for i in range(min(max_results, 20)):
            ticker = tickers[i % len(tickers)] if tickers else f"TICKER{i}"
            results.append({
                "id": f"pattern_search_{i}",
                "ticker": ticker,
                "pattern_type": pattern_types[i % len(pattern_types)] if pattern_types else "generic",
                "confidence": min_confidence + (i * 0.02),
                "timestamp": datetime.now().isoformat(),
                "relevance_score": 0.9 - (i * 0.02)
            })
        
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search patterns: {str(e)}"
        )


@app.get("/api/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(
    tickers: Optional[List[str]] = None,
    period: str = "1d"
):
    """Get analytics summary data"""
    try:
        if not pattern_analytics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pattern analytics service not available"
            )
        
        # Mock response for now - replace with actual analytics logic
        summary = {
            "period": period,
            "tickers": tickers or ["ALL"],
            "total_patterns": 1234,
            "pattern_distribution": {
                "bullish": 456,
                "bearish": 389,
                "neutral": 389
            },
            "average_confidence": 0.82,
            "top_patterns": [
                {
                    "type": "head_and_shoulders",
                    "count": 45,
                    "avg_confidence": 0.87
                },
                {
                    "type": "double_bottom",
                    "count": 38,
                    "avg_confidence": 0.84
                },
                {
                    "type": "triangle",
                    "count": 32,
                    "avg_confidence": 0.81
                }
            ],
            "performance_metrics": {
                "accuracy": 0.78,
                "precision": 0.82,
                "recall": 0.75,
                "f1_score": 0.78
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics summary: {str(e)}"
        )


# WebSocket endpoint for real-time updates
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pattern updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and send updates
        while True:
            try:
                # Wait for client messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    tickers = message.get("tickers", [])
                    await websocket.send_json({
                        "type": "subscription",
                        "status": "subscribed",
                        "tickers": tickers,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Start sending mock updates for subscribed tickers
                    for ticker in tickers:
                        await websocket.send_json({
                            "type": "pattern_update",
                            "ticker": ticker,
                            "pattern": {
                                "id": f"realtime_{ticker}_{int(time.time())}",
                                "type": "emerging_pattern",
                                "confidence": 0.75,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                
                elif message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_connections.remove(websocket)
        await websocket.close()


# Broadcast function for real-time updates
async def broadcast_pattern_update(pattern_data: Dict[str, Any]):
    """Broadcast pattern updates to all connected WebSocket clients"""
    disconnected = []
    
    for connection in websocket_connections:
        try:
            await connection.send_json({
                "type": "pattern_update",
                "data": pattern_data,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error broadcasting to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in websocket_connections:
            websocket_connections.remove(conn)


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
