"""
Test script for the new REST API endpoints
"""

import requests
import json
import asyncio
import websockets
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_patterns_endpoint():
    """Test GET /api/patterns/{ticker}"""
    print("\n=== Testing GET /api/patterns/{ticker} ===")
    
    # Test with different parameters
    test_cases = [
        {"ticker": "AAPL", "params": {}},
        {"ticker": "BTCUSD", "params": {"limit": 5}},
        {"ticker": "EURUSD", "params": {"limit": 10, "start_date": "2025-01-01"}}
    ]
    
    for test in test_cases:
        url = f"{BASE_URL}/api/patterns/{test['ticker']}"
        try:
            response = requests.get(url, params=test['params'])
            print(f"\nRequest: GET {url}")
            print(f"Params: {test['params']}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: Found {len(data['patterns'])} patterns")
                print(f"First pattern: {json.dumps(data['patterns'][0], indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")


def test_predictions_endpoint():
    """Test GET /api/predictions/{ticker}"""
    print("\n=== Testing GET /api/predictions/{ticker} ===")
    
    # Test with different parameters
    test_cases = [
        {"ticker": "AAPL", "params": {}},
        {"ticker": "BTCUSD", "params": {"horizon": 12}},
        {"ticker": "EURUSD", "params": {"horizon": 48, "model_type": "lstm"}}
    ]
    
    for test in test_cases:
        url = f"{BASE_URL}/api/predictions/{test['ticker']}"
        try:
            response = requests.get(url, params=test['params'])
            print(f"\nRequest: GET {url}")
            print(f"Params: {test['params']}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {len(data['predictions'])} predictions")
                print(f"Model: {data['model_type']}")
                print(f"First prediction: {json.dumps(data['predictions'][0], indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")


def test_pattern_search_endpoint():
    """Test POST /api/patterns/search"""
    print("\n=== Testing POST /api/patterns/search ===")
    
    # Test with different search queries
    test_queries = [
        {
            "tickers": ["AAPL", "GOOGL"],
            "pattern_types": ["bullish", "bearish"],
            "min_confidence": 0.7,
            "max_results": 10
        },
        {
            "tickers": ["BTCUSD"],
            "pattern_types": ["head_and_shoulders"],
            "min_confidence": 0.8,
            "max_results": 5
        },
        {
            "pattern_types": ["double_bottom", "triangle"],
            "min_confidence": 0.6,
            "max_results": 20
        }
    ]
    
    url = f"{BASE_URL}/api/patterns/search"
    for query in test_queries:
        try:
            response = requests.post(url, json=query)
            print(f"\nRequest: POST {url}")
            print(f"Query: {json.dumps(query, indent=2)}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: Found {len(data['results'])} patterns")
                if data['results']:
                    print(f"First result: {json.dumps(data['results'][0], indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")


def test_analytics_endpoint():
    """Test GET /api/analytics/summary"""
    print("\n=== Testing GET /api/analytics/summary ===")
    
    # Test with different parameters
    test_cases = [
        {"params": {}},
        {"params": {"period": "1h"}},
        {"params": {"tickers": ["AAPL", "GOOGL"], "period": "1d"}},
        {"params": {"tickers": ["BTCUSD", "ETHUSD"], "period": "1w"}}
    ]
    
    url = f"{BASE_URL}/api/analytics/summary"
    for test in test_cases:
        try:
            # Convert list to comma-separated string for query params
            params = test['params'].copy()
            if 'tickers' in params:
                params['tickers'] = ','.join(params['tickers'])
            
            response = requests.get(url, params=params)
            print(f"\nRequest: GET {url}")
            print(f"Params: {test['params']}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response Summary:")
                print(f"  Total Patterns: {data['total_patterns']}")
                print(f"  Average Confidence: {data['average_confidence']}")
                print(f"  Pattern Distribution: {data['pattern_distribution']}")
                print(f"  Top Pattern: {data['top_patterns'][0]}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")


async def test_websocket_endpoint():
    """Test WebSocket endpoint /ws/realtime"""
    print("\n=== Testing WebSocket /ws/realtime ===")
    
    uri = "ws://localhost:8000/ws/realtime"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Receive connection message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"\nConnected: {data}")
            
            # Subscribe to tickers
            subscribe_msg = {
                "type": "subscribe",
                "tickers": ["AAPL", "BTCUSD", "EURUSD"]
            }
            await websocket.send(json.dumps(subscribe_msg))
            print(f"\nSent subscription: {subscribe_msg}")
            
            # Receive subscription confirmation and updates
            for i in range(5):  # Receive 5 messages
                message = await websocket.recv()
                data = json.loads(message)
                print(f"\nReceived: {data['type']}")
                if data['type'] == 'pattern_update':
                    print(f"  Ticker: {data.get('ticker')}")
                    print(f"  Pattern: {data.get('pattern')}")
            
            # Send ping
            ping_msg = {"type": "ping"}
            await websocket.send(json.dumps(ping_msg))
            print(f"\nSent ping")
            
            # Receive pong
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")
            
    except Exception as e:
        print(f"WebSocket Error: {e}")


def test_health_endpoint():
    """Test the health endpoint to ensure API is running"""
    print("\n=== Testing Health Endpoint ===")
    
    url = f"{BASE_URL}/health"
    try:
        response = requests.get(url)
        print(f"Request: GET {url}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Financial Wavelet Prediction API Endpoints")
    print("=" * 60)
    
    # First check if API is running
    if not test_health_endpoint():
        print("\n⚠️  API is not running! Please start the API server first.")
        print("Run: python -m src.api.app")
        return
    
    # Test all endpoints
    test_patterns_endpoint()
    test_predictions_endpoint()
    test_pattern_search_endpoint()
    test_analytics_endpoint()
    
    # Test WebSocket
    print("\nTesting WebSocket connection...")
    asyncio.run(test_websocket_endpoint())
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
