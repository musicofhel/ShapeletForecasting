"""
Monitoring and logging for API
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import asyncio
from functools import wraps
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and track API metrics"""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.request_times = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.request_sizes = deque(maxlen=window_size)
        self.response_sizes = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.model_usage = defaultdict(int)
        self.endpoint_usage = defaultdict(int)
        self.status_codes = defaultdict(int)
        
        # Start time
        self.start_time = datetime.now()
        
        # Performance metrics
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Start background monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background system monitoring"""
        async def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.append(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.append(memory.percent)
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    await asyncio.sleep(60)
        
        # Run in background (if event loop exists)
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                asyncio.create_task(monitor_system())
        except:
            pass  # Skip if no event loop
    
    def record_request(self, endpoint: str, method: str, 
                      request_size: int, model_type: Optional[str] = None):
        """
        Record incoming request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            request_size: Size of request in bytes
            model_type: Model type if applicable
        """
        self.request_times.append(datetime.now())
        self.request_sizes.append(request_size)
        self.endpoint_usage[f"{method} {endpoint}"] += 1
        
        if model_type:
            self.model_usage[model_type] += 1
    
    def record_response(self, status_code: int, response_time: float, 
                       response_size: int, error: Optional[str] = None):
        """
        Record response
        
        Args:
            status_code: HTTP status code
            response_time: Response time in seconds
            response_size: Size of response in bytes
            error: Error message if any
        """
        self.response_times.append(response_time * 1000)  # Convert to ms
        self.response_sizes.append(response_size)
        self.status_codes[status_code] += 1
        
        if error:
            self.errors.append({
                'timestamp': datetime.now(),
                'error': error,
                'status_code': status_code
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        now = datetime.now()
        uptime = (now - self.start_time).total_seconds()
        
        # Calculate request rate
        recent_requests = [
            t for t in self.request_times 
            if (now - t).total_seconds() < 60
        ]
        requests_per_minute = len(recent_requests)
        
        # Response time stats
        response_times = list(self.response_times)
        avg_response_time = np.mean(response_times) if response_times else 0
        p50_response_time = np.percentile(response_times, 50) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        
        # Error rate
        total_requests = sum(self.status_codes.values())
        error_requests = sum(
            count for code, count in self.status_codes.items() 
            if code >= 400
        )
        error_rate = error_requests / total_requests if total_requests > 0 else 0
        
        # System metrics
        avg_cpu = np.mean(list(self.cpu_usage)) if self.cpu_usage else 0
        avg_memory = np.mean(list(self.memory_usage)) if self.memory_usage else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': total_requests,
            'successful_requests': total_requests - error_requests,
            'failed_requests': error_requests,
            'error_rate': error_rate,
            'requests_per_minute': requests_per_minute,
            'avg_response_time_ms': avg_response_time,
            'p50_response_time_ms': p50_response_time,
            'p95_response_time_ms': p95_response_time,
            'p99_response_time_ms': p99_response_time,
            'avg_request_size_bytes': np.mean(list(self.request_sizes)) if self.request_sizes else 0,
            'avg_response_size_bytes': np.mean(list(self.response_sizes)) if self.response_sizes else 0,
            'model_usage': dict(self.model_usage),
            'endpoint_usage': dict(self.endpoint_usage),
            'status_code_distribution': dict(self.status_codes),
            'recent_errors': [
                {
                    'timestamp': e['timestamp'].isoformat(),
                    'error': e['error'],
                    'status_code': e['status_code']
                }
                for e in list(self.errors)[-10:]
            ],
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'current_cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
                'current_memory_percent': self.memory_usage[-1] if self.memory_usage else 0
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        metrics = self.get_metrics()
        
        # Determine health status
        status = 'healthy'
        issues = []
        
        # Check error rate
        if metrics['error_rate'] > 0.1:  # 10% error rate
            status = 'degraded'
            issues.append(f"High error rate: {metrics['error_rate']:.2%}")
        
        # Check response time
        if metrics['p95_response_time_ms'] > 1000:  # 1 second
            status = 'degraded'
            issues.append(f"High response time: {metrics['p95_response_time_ms']:.0f}ms")
        
        # Check system resources
        if metrics['system_metrics']['current_cpu_percent'] > 80:
            status = 'degraded'
            issues.append(f"High CPU usage: {metrics['system_metrics']['current_cpu_percent']:.1f}%")
        
        if metrics['system_metrics']['current_memory_percent'] > 80:
            status = 'degraded'
            issues.append(f"High memory usage: {metrics['system_metrics']['current_memory_percent']:.1f}%")
        
        return {
            'status': status,
            'issues': issues,
            'metrics_summary': {
                'error_rate': metrics['error_rate'],
                'avg_response_time_ms': metrics['avg_response_time_ms'],
                'requests_per_minute': metrics['requests_per_minute'],
                'uptime_seconds': metrics['uptime_seconds']
            }
        }


class RequestLogger:
    """Log API requests and responses"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize request logger
        
        Args:
            log_file: Optional file to write logs to
        """
        self.log_file = log_file
        
        # Setup file handler if log file specified
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
    
    def log_request(self, request_id: str, method: str, endpoint: str,
                   headers: Dict[str, str], body: Optional[Any] = None):
        """Log incoming request"""
        log_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'headers': self._sanitize_headers(headers),
            'body_size': len(json.dumps(body)) if body else 0
        }
        
        logger.info(f"Request: {json.dumps(log_data)}")
    
    def log_response(self, request_id: str, status_code: int,
                    response_time: float, body: Optional[Any] = None,
                    error: Optional[str] = None):
        """Log response"""
        log_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'status_code': status_code,
            'response_time_ms': response_time * 1000,
            'body_size': len(json.dumps(body)) if body else 0,
            'error': error
        }
        
        if error:
            logger.error(f"Response: {json.dumps(log_data)}")
        else:
            logger.info(f"Response: {json.dumps(log_data)}")
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers"""
        sanitized = headers.copy()
        sensitive_keys = ['authorization', 'api-key', 'x-api-key', 'cookie']
        
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = '***'
        
        return sanitized


def monitor_endpoint(metrics_collector: MetricsCollector):
    """
    Decorator to monitor API endpoints
    
    Args:
        metrics_collector: MetricsCollector instance
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                status_code = 500
                raise
            finally:
                # Record metrics
                response_time = time.time() - start_time
                metrics_collector.record_response(
                    status_code=status_code,
                    response_time=response_time,
                    response_size=0,  # Would need to calculate actual size
                    error=error
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                status_code = 500
                raise
            finally:
                # Record metrics
                response_time = time.time() - start_time
                metrics_collector.record_response(
                    status_code=status_code,
                    response_time=response_time,
                    response_size=0,
                    error=error
                )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AlertManager:
    """Manage alerts based on metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        self.alerts = deque(maxlen=100)
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10%
            'response_time_p95': 1000,  # 1 second
            'cpu_usage': 80,  # 80%
            'memory_usage': 80,  # 80%
            'requests_per_minute': 1000  # High load
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        metrics = self.metrics_collector.get_metrics()
        new_alerts = []
        
        # Check error rate
        if metrics['error_rate'] > self.alert_thresholds['error_rate']:
            alert = {
                'type': 'error_rate',
                'severity': 'high',
                'message': f"Error rate {metrics['error_rate']:.2%} exceeds threshold",
                'timestamp': datetime.now().isoformat(),
                'value': metrics['error_rate'],
                'threshold': self.alert_thresholds['error_rate']
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        # Check response time
        if metrics['p95_response_time_ms'] > self.alert_thresholds['response_time_p95']:
            alert = {
                'type': 'response_time',
                'severity': 'medium',
                'message': f"P95 response time {metrics['p95_response_time_ms']:.0f}ms exceeds threshold",
                'timestamp': datetime.now().isoformat(),
                'value': metrics['p95_response_time_ms'],
                'threshold': self.alert_thresholds['response_time_p95']
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        # Check system resources
        current_cpu = metrics['system_metrics']['current_cpu_percent']
        if current_cpu > self.alert_thresholds['cpu_usage']:
            alert = {
                'type': 'cpu_usage',
                'severity': 'high',
                'message': f"CPU usage {current_cpu:.1f}% exceeds threshold",
                'timestamp': datetime.now().isoformat(),
                'value': current_cpu,
                'threshold': self.alert_thresholds['cpu_usage']
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        return new_alerts
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Simulate some requests
    for i in range(100):
        collector.record_request(
            endpoint="/predict",
            method="POST",
            request_size=1024,
            model_type="ensemble"
        )
        
        # Simulate response
        import random
        response_time = random.uniform(0.01, 0.1)  # 10-100ms
        status_code = 200 if random.random() > 0.05 else 500
        
        collector.record_response(
            status_code=status_code,
            response_time=response_time,
            response_size=512,
            error="Internal error" if status_code == 500 else None
        )
    
    # Get metrics
    metrics = collector.get_metrics()
    print("Metrics:", json.dumps(metrics, indent=2))
    
    # Check health
    health = collector.get_health_status()
    print("\nHealth Status:", json.dumps(health, indent=2))
    
    # Test alerts
    alert_manager = AlertManager(collector)
    alerts = alert_manager.check_alerts()
    print("\nAlerts:", json.dumps(alerts, indent=2))
