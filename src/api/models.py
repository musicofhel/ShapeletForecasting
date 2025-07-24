"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: List[float] = Field(..., description="Input features for prediction")
    timestamp: Optional[datetime] = Field(None, description="Timestamp for the prediction")
    model_type: Optional[ModelType] = Field(ModelType.ENSEMBLE, description="Model to use for prediction")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("Features cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1, 0.2, -0.3, 0.4, 0.5],
                "timestamp": "2025-01-16T12:00:00",
                "model_type": "ensemble"
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: float = Field(..., description="Predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    model_type: str = Field(..., description="Model used for prediction")
    timestamp: datetime = Field(..., description="Timestamp of prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0.123,
                "confidence": 0.85,
                "model_type": "ensemble",
                "timestamp": "2025-01-16T12:00:00",
                "processing_time_ms": 15.3
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    samples: List[List[float]] = Field(..., description="List of feature vectors")
    timestamps: Optional[List[datetime]] = Field(None, description="Timestamps for each sample")
    model_type: Optional[ModelType] = Field(ModelType.ENSEMBLE, description="Model to use")
    return_confidence: bool = Field(False, description="Whether to return confidence scores")
    
    @validator('samples')
    def validate_samples(cls, v):
        if len(v) == 0:
            raise ValueError("Samples cannot be empty")
        
        # Check all samples have same length
        if len(v) > 1:
            first_len = len(v[0])
            for i, sample in enumerate(v[1:], 1):
                if len(sample) != first_len:
                    raise ValueError(f"Sample {i} has different length than first sample")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    [0.1, 0.2, -0.3, 0.4, 0.5],
                    [0.2, 0.3, -0.2, 0.5, 0.6]
                ],
                "model_type": "ensemble",
                "return_confidence": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[float] = Field(..., description="List of predictions")
    confidences: Optional[List[float]] = Field(None, description="Confidence scores if requested")
    model_type: str = Field(..., description="Model used")
    timestamp: datetime = Field(..., description="Timestamp of batch processing")
    total_samples: int = Field(..., description="Number of samples processed")
    processing_time_ms: float = Field(..., description="Total processing time")
    avg_time_per_sample_ms: float = Field(..., description="Average time per sample")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.123, 0.234],
                "confidences": [0.85, 0.92],
                "model_type": "ensemble",
                "timestamp": "2025-01-16T12:00:00",
                "total_samples": 2,
                "processing_time_ms": 30.6,
                "avg_time_per_sample_ms": 15.3
            }
        }


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str = Field(..., description="Type of model")
    version: str = Field(..., description="Model version")
    trained_date: datetime = Field(..., description="When model was trained")
    features: List[str] = Field(..., description="Expected feature names")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    compression_stats: Optional[Dict[str, Any]] = Field(None, description="Compression statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "ensemble",
                "version": "1.0.0",
                "trained_date": "2025-01-15T10:00:00",
                "features": ["feature_0", "feature_1", "feature_2"],
                "performance_metrics": {
                    "mse": 0.001,
                    "mae": 0.025,
                    "r2": 0.95
                },
                "compression_stats": {
                    "original_size_mb": 10.5,
                    "compressed_size_mb": 2.1,
                    "compression_ratio": 0.2
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-16T12:00:00",
                "models_loaded": {
                    "xgboost": True,
                    "lstm": True,
                    "transformer": True,
                    "ensemble": True
                },
                "version": "1.0.0",
                "uptime_seconds": 3600.0
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid input",
                "detail": "Features must be a list of numbers",
                "timestamp": "2025-01-16T12:00:00",
                "request_id": "req_123456"
            }
        }


class MetricsResponse(BaseModel):
    """API metrics response"""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    avg_response_time_ms: float = Field(..., description="Average response time")
    requests_per_minute: float = Field(..., description="Current request rate")
    model_usage: Dict[str, int] = Field(..., description="Usage count per model")
    error_rate: float = Field(..., description="Error rate (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 10000,
                "successful_requests": 9950,
                "failed_requests": 50,
                "avg_response_time_ms": 15.3,
                "requests_per_minute": 120.5,
                "model_usage": {
                    "xgboost": 2500,
                    "lstm": 2000,
                    "transformer": 1500,
                    "ensemble": 4000
                },
                "error_rate": 0.005
            }
        }


class StreamPredictionRequest(BaseModel):
    """Real-time streaming prediction request"""
    features: List[float] = Field(..., description="Current features")
    window_size: int = Field(10, description="Size of sliding window")
    model_type: Optional[ModelType] = Field(ModelType.ENSEMBLE, description="Model to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1, 0.2, -0.3, 0.4, 0.5],
                "window_size": 10,
                "model_type": "ensemble"
            }
        }


class StreamPredictionResponse(BaseModel):
    """Real-time streaming prediction response"""
    prediction: float = Field(..., description="Current prediction")
    trend: str = Field(..., description="Trend direction (up/down/stable)")
    change_from_previous: float = Field(..., description="Change from previous prediction")
    moving_average: float = Field(..., description="Moving average of predictions")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0.123,
                "trend": "up",
                "change_from_previous": 0.015,
                "moving_average": 0.118,
                "timestamp": "2025-01-16T12:00:00"
            }
        }


class ModelUpdateRequest(BaseModel):
    """Request to update model"""
    model_type: ModelType = Field(..., description="Model to update")
    model_path: str = Field(..., description="Path to new model file")
    version: str = Field(..., description="New model version")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "xgboost",
                "model_path": "/models/xgboost_v2.pkl",
                "version": "2.0.0"
            }
        }


class ModelUpdateResponse(BaseModel):
    """Model update response"""
    success: bool = Field(..., description="Whether update was successful")
    model_type: str = Field(..., description="Updated model type")
    previous_version: str = Field(..., description="Previous version")
    new_version: str = Field(..., description="New version")
    timestamp: datetime = Field(..., description="Update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "model_type": "xgboost",
                "previous_version": "1.0.0",
                "new_version": "2.0.0",
                "timestamp": "2025-01-16T12:00:00"
            }
        }
