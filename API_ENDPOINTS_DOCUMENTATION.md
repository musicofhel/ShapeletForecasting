# Financial Wavelet Prediction API - Dashboard Endpoints Documentation

## Overview

This document describes the REST API endpoints created for the Financial Wavelet Prediction Dashboard. These endpoints provide access to pattern detection, predictions, search functionality, analytics, and real-time updates via WebSocket.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, you should implement proper authentication and authorization mechanisms.

## Endpoints

### 1. Pattern Endpoints

#### GET /api/patterns/{ticker}

Get patterns for a specific ticker symbol.

**Parameters:**
- `ticker` (path, required): The ticker symbol (e.g., "AAPL", "BTCUSD")
- `limit` (query, optional): Maximum number of patterns to return (default: 100)
- `start_date` (query, optional): Start date for pattern search (ISO format)
- `end_date` (query, optional): End date for pattern search (ISO format)

**Example Request:**
```bash
GET /api/patterns/AAPL?limit=10&start_date=2025-01-01
```

**Example Response:**
```json
{
  "ticker": "AAPL",
  "patterns": [
    {
      "id": "pattern_AAPL_0",
      "ticker": "AAPL",
      "timestamp": "2025-01-17T18:00:00",
      "pattern_type": "bullish",
      "confidence": 0.85,
      "duration": 24,
      "metrics": {
        "strength": 0.82,
        "reliability": 0.78,
        "frequency": 5
      }
    }
  ],
  "total": 10,
  "timestamp": "2025-01-17T18:00:00"
}
```

### 2. Prediction Endpoints

#### GET /api/predictions/{ticker}

Get predictions for a specific ticker.

**Parameters:**
- `ticker` (path, required): The ticker symbol
- `horizon` (query, optional): Prediction horizon in hours (default: 24)
- `model_type` (query, optional): Model to use for prediction (default: "ensemble")

**Example Request:**
```bash
GET /api/predictions/BTCUSD?horizon=48&model_type=lstm
```

**Example Response:**
```json
{
  "ticker": "BTCUSD",
  "predictions": [
    {
      "timestamp": "2025-01-17T19:00:00",
      "horizon": 1,
      "value": 45000.50,
      "confidence": 0.92,
      "upper_bound": 45500.00,
      "lower_bound": 44500.00
    }
  ],
  "model_type": "lstm",
  "horizon": 48,
  "timestamp": "2025-01-17T18:00:00"
}
```

### 3. Pattern Search

#### POST /api/patterns/search

Search for patterns based on multiple criteria.

**Request Body:**
```json
{
  "tickers": ["AAPL", "GOOGL"],
  "pattern_types": ["bullish", "bearish"],
  "min_confidence": 0.7,
  "max_results": 100
}
```

**Parameters:**
- `tickers` (array, optional): List of ticker symbols to search
- `pattern_types` (array, optional): Types of patterns to search for
- `min_confidence` (number, optional): Minimum confidence threshold (0-1)
- `max_results` (number, optional): Maximum number of results to return

**Example Response:**
```json
{
  "query": {
    "tickers": ["AAPL", "GOOGL"],
    "pattern_types": ["bullish", "bearish"],
    "min_confidence": 0.7,
    "max_results": 100
  },
  "results": [
    {
      "id": "pattern_search_0",
      "ticker": "AAPL",
      "pattern_type": "bullish",
      "confidence": 0.85,
      "timestamp": "2025-01-17T18:00:00",
      "relevance_score": 0.92
    }
  ],
  "total": 25,
  "timestamp": "2025-01-17T18:00:00"
}
```

### 4. Analytics

#### GET /api/analytics/summary

Get analytics summary data for patterns and predictions.

**Parameters:**
- `tickers` (query, optional): Comma-separated list of tickers
- `period` (query, optional): Time period for analytics (e.g., "1h", "1d", "1w")

**Example Request:**
```bash
GET /api/analytics/summary?tickers=AAPL,GOOGL&period=1d
```

**Example Response:**
```json
{
  "period": "1d",
  "tickers": ["AAPL", "GOOGL"],
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
    }
  ],
  "performance_metrics": {
    "accuracy": 0.78,
    "precision": 0.82,
    "recall": 0.75,
    "f1_score": 0.78
  },
  "timestamp": "2025-01-17T18:00:00"
}
```

### 5. WebSocket - Real-time Updates

#### WS /ws/realtime

WebSocket endpoint for real-time pattern updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');
```

**Message Types:**

1. **Connection Message** (Server → Client)
```json
{
  "type": "connection",
  "status": "connected",
  "timestamp": "2025-01-17T18:00:00"
}
```

2. **Subscribe to Tickers** (Client → Server)
```json
{
  "type": "subscribe",
  "tickers": ["AAPL", "BTCUSD", "EURUSD"]
}
```

3. **Subscription Confirmation** (Server → Client)
```json
{
  "type": "subscription",
  "status": "subscribed",
  "tickers": ["AAPL", "BTCUSD", "EURUSD"],
  "timestamp": "2025-01-17T18:00:00"
}
```

4. **Pattern Update** (Server → Client)
```json
{
  "type": "pattern_update",
  "ticker": "AAPL",
  "pattern": {
    "id": "realtime_AAPL_1234567890",
    "type": "emerging_pattern",
    "confidence": 0.75,
    "timestamp": "2025-01-17T18:00:00"
  }
}
```

5. **Heartbeat** (Server → Client)
```json
{
  "type": "heartbeat",
  "timestamp": "2025-01-17T18:00:00"
}
```

6. **Ping/Pong** (Client ↔ Server)
```json
// Client sends:
{"type": "ping"}

// Server responds:
{"type": "pong", "timestamp": "2025-01-17T18:00:00"}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2025-01-17T18:00:00",
  "request_id": "unique-request-id"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

Currently, there are no rate limits implemented. In production, you should implement appropriate rate limiting to prevent abuse.

## CORS

The API is configured to allow all origins (`*`). In production, you should restrict this to your specific frontend domain.

## Integration Example

### JavaScript/TypeScript

```javascript
// Fetch patterns
async function getPatterns(ticker) {
  const response = await fetch(`http://localhost:8000/api/patterns/${ticker}?limit=10`);
  const data = await response.json();
  return data;
}

// Search patterns
async function searchPatterns(query) {
  const response = await fetch('http://localhost:8000/api/patterns/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(query),
  });
  const data = await response.json();
  return data;
}

// WebSocket connection
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8000/ws/realtime');
  
  ws.onopen = () => {
    console.log('Connected to WebSocket');
    // Subscribe to tickers
    ws.send(JSON.stringify({
      type: 'subscribe',
      tickers: ['AAPL', 'BTCUSD']
    }));
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    if (data.type === 'pattern_update') {
      // Handle pattern update
      updateDashboard(data);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  return ws;
}
```

### Python

```python
import requests
import json
import websocket

# Fetch patterns
def get_patterns(ticker, limit=10):
    url = f"http://localhost:8000/api/patterns/{ticker}"
    params = {"limit": limit}
    response = requests.get(url, params=params)
    return response.json()

# Search patterns
def search_patterns(tickers, pattern_types, min_confidence=0.7):
    url = "http://localhost:8000/api/patterns/search"
    data = {
        "tickers": tickers,
        "pattern_types": pattern_types,
        "min_confidence": min_confidence
    }
    response = requests.post(url, json=data)
    return response.json()

# WebSocket connection
def on_message(ws, message):
    data = json.loads(message)
    print(f"Received: {data}")

def on_open(ws):
    # Subscribe to tickers
    ws.send(json.dumps({
        "type": "subscribe",
        "tickers": ["AAPL", "BTCUSD"]
    }))

ws = websocket.WebSocketApp("ws://localhost:8000/ws/realtime",
                            on_message=on_message,
                            on_open=on_open)
ws.run_forever()
```

## Testing

Use the provided `test_api_endpoints.py` script to test all endpoints:

```bash
# Start the API server
python -m src.api.app

# In another terminal, run the tests
python test_api_endpoints.py
```

## Next Steps

1. **Authentication**: Implement JWT or OAuth2 authentication
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **Caching**: Implement Redis caching for frequently accessed data
4. **Database Integration**: Connect to a real database for pattern storage
5. **Real Pattern Data**: Replace mock data with actual pattern detection logic
6. **Monitoring**: Add Prometheus metrics and Grafana dashboards
7. **API Versioning**: Implement API versioning (e.g., /api/v1/patterns)
8. **Documentation**: Generate OpenAPI/Swagger documentation

## Notes

- The current implementation returns mock data for demonstration purposes
- In production, integrate with actual pattern detection and prediction services
- Consider implementing pagination for large result sets
- Add request validation and sanitization
- Implement proper error handling and logging
- Consider using message queues for real-time updates at scale
