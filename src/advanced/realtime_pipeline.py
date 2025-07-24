"""
Real-time data pipeline for streaming predictions.
Handles data ingestion, preprocessing, and real-time inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import asyncio
import websockets
import json
import threading
import queue
from datetime import datetime, timedelta
import ccxt
import logging
from src.dashboard.data_utils import data_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for data streaming."""
    source: str  # 'yahoo', 'binance', 'websocket', 'kafka'
    symbols: List[str]
    interval: str  # '1m', '5m', '15m', '1h', '1d'
    buffer_size: int
    batch_size: int
    max_latency_ms: int


@dataclass
class PredictionResult:
    """Real-time prediction result."""
    symbol: str
    timestamp: pd.Timestamp
    prediction: float
    confidence: float
    features: Dict[str, float]
    latency_ms: float
    model_version: str


class RealtimePipeline:
    """
    Real-time data pipeline for streaming financial predictions.
    """
    
    def __init__(self, 
                 stream_config: StreamConfig,
                 feature_extractor: Callable,
                 predictor: Callable,
                 output_handler: Optional[Callable] = None):
        """
        Initialize real-time pipeline.
        
        Args:
            stream_config: Streaming configuration
            feature_extractor: Function to extract features
            predictor: Prediction model/function
            output_handler: Function to handle predictions
        """
        self.config = stream_config
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.output_handler = output_handler
        
        # Data buffers
        self.data_buffers = {symbol: deque(maxlen=stream_config.buffer_size) 
                           for symbol in stream_config.symbols}
        self.feature_buffers = {symbol: deque(maxlen=stream_config.buffer_size) 
                              for symbol in stream_config.symbols}
        
        # Queues for async processing
        self.data_queue = asyncio.Queue()
        self.prediction_queue = asyncio.Queue()
        
        # Performance tracking
        self.latency_tracker = deque(maxlen=1000)
        self.error_count = 0
        self.processed_count = 0
        
        # State management
        self.is_running = False
        self.tasks = []
        
        # Initialize data sources
        self._init_data_sources()
        
    def _init_data_sources(self):
        """Initialize data source connections."""
        if self.config.source == 'binance':
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        elif self.config.source == 'yahoo':
            # Yahoo Finance doesn't need initialization
            pass
        elif self.config.source == 'websocket':
            self.ws_url = None  # To be set when starting
            
    async def start(self):
        """Start the real-time pipeline."""
        logger.info("Starting real-time pipeline...")
        self.is_running = True
        
        # Start data ingestion
        ingestion_task = asyncio.create_task(self._data_ingestion_loop())
        self.tasks.append(ingestion_task)
        
        # Start feature extraction
        feature_task = asyncio.create_task(self._feature_extraction_loop())
        self.tasks.append(feature_task)
        
        # Start prediction
        prediction_task = asyncio.create_task(self._prediction_loop())
        self.tasks.append(prediction_task)
        
        # Start output handling
        if self.output_handler:
            output_task = asyncio.create_task(self._output_handling_loop())
            self.tasks.append(output_task)
            
        # Start monitoring
        monitor_task = asyncio.create_task(self._monitoring_loop())
        self.tasks.append(monitor_task)
        
        logger.info("Real-time pipeline started")
        
    async def stop(self):
        """Stop the real-time pipeline."""
        logger.info("Stopping real-time pipeline...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Real-time pipeline stopped")
        
    async def _data_ingestion_loop(self):
        """Main loop for data ingestion."""
        while self.is_running:
            try:
                if self.config.source == 'yahoo':
                    await self._ingest_yahoo_data()
                elif self.config.source == 'binance':
                    await self._ingest_binance_data()
                elif self.config.source == 'websocket':
                    await self._ingest_websocket_data()
                    
                # Sleep based on interval
                sleep_time = self._get_sleep_time()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in data ingestion: {e}")
                self.error_count += 1
                await asyncio.sleep(1)
                
    async def _ingest_yahoo_data(self):
        """Ingest data from Yahoo Finance using DataManager."""
        for symbol in self.config.symbols:
            try:
                # Get latest data using data_manager
                data = data_manager.download_data(symbol, period='1d', use_cache=False)
                
                if data is not None and not data.empty:
                    latest = data.iloc[-1]
                    
                    # Create data point
                    data_point = {
                        'symbol': symbol,
                        'timestamp': pd.Timestamp.now(),
                        'open': latest['Open'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'close': latest['Close'],
                        'volume': latest['Volume']
                    }
                    
                    # Add to buffer and queue
                    self.data_buffers[symbol].append(data_point)
                    await self.data_queue.put(data_point)
                    
            except Exception as e:
                logger.error(f"Error ingesting Yahoo data for {symbol}: {e}")
                
    async def _ingest_binance_data(self):
        """Ingest data from Binance."""
        for symbol in self.config.symbols:
            try:
                # Get latest candles
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=self.config.interval,
                    limit=2
                )
                
                if ohlcv:
                    latest = ohlcv[-1]
                    
                    # Create data point
                    data_point = {
                        'symbol': symbol,
                        'timestamp': pd.Timestamp(latest[0], unit='ms'),
                        'open': latest[1],
                        'high': latest[2],
                        'low': latest[3],
                        'close': latest[4],
                        'volume': latest[5]
                    }
                    
                    # Add to buffer and queue
                    self.data_buffers[symbol].append(data_point)
                    await self.data_queue.put(data_point)
                    
            except Exception as e:
                logger.error(f"Error ingesting Binance data for {symbol}: {e}")
                
    async def _ingest_websocket_data(self):
        """Ingest data from WebSocket stream."""
        if not self.ws_url:
            logger.error("WebSocket URL not configured")
            return
            
        try:
            async with websockets.connect(self.ws_url) as websocket:
                while self.is_running:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Parse WebSocket data
                    data_point = self._parse_websocket_data(data)
                    if data_point:
                        symbol = data_point['symbol']
                        self.data_buffers[symbol].append(data_point)
                        await self.data_queue.put(data_point)
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
    def _parse_websocket_data(self, data: Dict) -> Optional[Dict]:
        """Parse WebSocket data into standard format."""
        # Implementation depends on WebSocket format
        # This is a placeholder
        return None
        
    async def _feature_extraction_loop(self):
        """Extract features from incoming data."""
        while self.is_running:
            try:
                # Get data from queue
                data_point = await asyncio.wait_for(
                    self.data_queue.get(),
                    timeout=1.0
                )
                
                # Extract features
                symbol = data_point['symbol']
                buffer_data = list(self.data_buffers[symbol])
                
                if len(buffer_data) >= 20:  # Minimum data for features
                    # Convert to DataFrame
                    df = pd.DataFrame(buffer_data)
                    
                    # Extract features
                    features = self.feature_extractor(df)
                    
                    # Create feature point
                    feature_point = {
                        'symbol': symbol,
                        'timestamp': data_point['timestamp'],
                        'features': features,
                        'raw_data': data_point
                    }
                    
                    # Add to feature buffer
                    self.feature_buffers[symbol].append(feature_point)
                    
                    # Queue for prediction
                    await self.prediction_queue.put(feature_point)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in feature extraction: {e}")
                
    async def _prediction_loop(self):
        """Generate predictions from features."""
        while self.is_running:
            try:
                # Get features from queue
                feature_point = await asyncio.wait_for(
                    self.prediction_queue.get(),
                    timeout=1.0
                )
                
                start_time = datetime.now()
                
                # Generate prediction
                features = feature_point['features']
                prediction, confidence = self.predictor(features)
                
                # Calculate latency
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Create prediction result
                result = PredictionResult(
                    symbol=feature_point['symbol'],
                    timestamp=feature_point['timestamp'],
                    prediction=float(prediction),
                    confidence=float(confidence),
                    features=features,
                    latency_ms=latency_ms,
                    model_version="1.0"  # Should be dynamic
                )
                
                # Track performance
                self.latency_tracker.append(latency_ms)
                self.processed_count += 1
                
                # Check latency threshold
                if latency_ms > self.config.max_latency_ms:
                    logger.warning(f"High latency: {latency_ms}ms for {result.symbol}")
                    
                # Handle output
                if self.output_handler:
                    await self.output_handler(result)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                self.error_count += 1
                
    async def _output_handling_loop(self):
        """Handle prediction outputs."""
        # This is handled by the output_handler callback
        pass
        
    async def _monitoring_loop(self):
        """Monitor pipeline performance."""
        while self.is_running:
            try:
                # Calculate metrics
                if self.latency_tracker:
                    avg_latency = np.mean(list(self.latency_tracker))
                    max_latency = np.max(list(self.latency_tracker))
                    p95_latency = np.percentile(list(self.latency_tracker), 95)
                else:
                    avg_latency = max_latency = p95_latency = 0
                    
                # Log metrics
                logger.info(f"Pipeline metrics - Processed: {self.processed_count}, "
                          f"Errors: {self.error_count}, "
                          f"Avg latency: {avg_latency:.2f}ms, "
                          f"P95 latency: {p95_latency:.2f}ms")
                
                # Check health
                error_rate = self.error_count / max(self.processed_count, 1)
                if error_rate > 0.1:
                    logger.warning(f"High error rate: {error_rate:.2%}")
                    
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                
    def _get_sleep_time(self) -> float:
        """Calculate sleep time based on interval."""
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        return interval_map.get(self.config.interval, 60)
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        stats = {
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'buffer_sizes': {symbol: len(buffer) 
                           for symbol, buffer in self.data_buffers.items()}
        }
        
        if self.latency_tracker:
            stats['latency'] = {
                'avg': np.mean(list(self.latency_tracker)),
                'max': np.max(list(self.latency_tracker)),
                'p50': np.percentile(list(self.latency_tracker), 50),
                'p95': np.percentile(list(self.latency_tracker), 95),
                'p99': np.percentile(list(self.latency_tracker), 99)
            }
            
        return stats
        
    def set_websocket_url(self, url: str):
        """Set WebSocket URL for streaming."""
        self.ws_url = url
        
    async def backfill_data(self, lookback_periods: int):
        """Backfill historical data for warm start."""
        logger.info(f"Backfilling {lookback_periods} periods...")
        
        for symbol in self.config.symbols:
            try:
                if self.config.source == 'yahoo':
                    # Use data_manager for backfilling
                    data = data_manager.download_data(
                        symbol, 
                        period=f"{lookback_periods}d"
                    )
                    
                    if data is not None and not data.empty:
                        # Add to buffer
                        for idx, row in data.iterrows():
                            data_point = {
                                'symbol': symbol,
                                'timestamp': idx,
                                'open': row['Open'],
                                'high': row['High'],
                                'low': row['Low'],
                                'close': row['Close'],
                                'volume': row['Volume']
                            }
                            self.data_buffers[symbol].append(data_point)
                        
                elif self.config.source == 'binance':
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=self.config.interval,
                        limit=lookback_periods
                    )
                    
                    for candle in ohlcv:
                        data_point = {
                            'symbol': symbol,
                            'timestamp': pd.Timestamp(candle[0], unit='ms'),
                            'open': candle[1],
                            'high': candle[2],
                            'low': candle[3],
                            'close': candle[4],
                            'volume': candle[5]
                        }
                        self.data_buffers[symbol].append(data_point)
                        
            except Exception as e:
                logger.error(f"Error backfilling {symbol}: {e}")
                
        logger.info("Backfill completed")
