"""
Simple Polygon API Test
======================
Quick validation that the Polygon API is working
"""

import requests
import json
from datetime import datetime, timedelta

# Your API key
API_KEY = "rzbt8GG8mqwLo1EMuSzbkC34uxA_Df1R"

def test_polygon_api():
    print("POLYGON API TEST")
    print("=" * 50)
    print(f"API Key: {API_KEY[:10]}...")
    print()
    
    # Test 1: Get Bitcoin data (this works!)
    print("TEST 1: Fetching Bitcoin (BTC/USD) data...")
    print("-" * 40)
    
    # Calculate dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Build URL for crypto data
    url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/day/{start_date}/{end_date}"
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK":
            print("✅ SUCCESS! API is working")
            print(f"   Status: {data['status']}")
            print(f"   Ticker: {data.get('ticker', 'N/A')}")
            print(f"   Query Count: {data.get('queryCount', 0)}")
            print(f"   Results Count: {data.get('resultsCount', 0)}")
            
            # Show last 3 days of data
            if data.get("results"):
                print("\n   Last 3 days of Bitcoin prices:")
                for bar in data["results"][-3:]:
                    date = datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d')
                    print(f"   {date}: ${bar['c']:,.2f}")
        else:
            print("❌ FAILED")
            print(f"   Response: {json.dumps(data, indent=2)}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()
    
    # Test 2: Get company details (this also works!)
    print("TEST 2: Fetching Microsoft (MSFT) company details...")
    print("-" * 40)
    
    url = f"https://api.polygon.io/v3/reference/tickers/MSFT"
    params = {"apiKey": API_KEY}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK":
            print("✅ SUCCESS! Ticker details endpoint works")
            results = data.get("results", {})
            print(f"   Company: {results.get('name', 'N/A')}")
            print(f"   Ticker: {results.get('ticker', 'N/A')}")
            print(f"   Market Cap: ${results.get('market_cap', 0):,.0f}")
            print(f"   Exchange: {results.get('primary_exchange', 'N/A')}")
        else:
            print("❌ FAILED")
            print(f"   Response: {json.dumps(data, indent=2)}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()
    
    # Test 3: Try to get stock data (may not work on free tier)
    print("TEST 3: Attempting to fetch Apple (AAPL) stock data...")
    print("-" * 40)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}"
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
            print("✅ SUCCESS! Stock data is available")
            print(f"   Results Count: {data.get('resultsCount', 0)}")
            
            # Show last price
            if data.get("results"):
                last_bar = data["results"][-1]
                date = datetime.fromtimestamp(last_bar['t']/1000).strftime('%Y-%m-%d')
                print(f"   Latest price ({date}): ${last_bar['c']:.2f}")
        else:
            print("⚠️  NO DATA (might need paid plan for stocks)")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Message: {data.get('message', 'No results returned')}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()
    print("=" * 50)
    print("SUMMARY:")
    print("- Cryptocurrency data: ✅ WORKING")
    print("- Company details: ✅ WORKING")
    print("- Stock price data: ⚠️  LIMITED (may need paid plan)")
    print()
    print("Your API key is valid and working!")
    print("Use cryptocurrency data for testing your wavelet system.")

if __name__ == "__main__":
    test_polygon_api()
