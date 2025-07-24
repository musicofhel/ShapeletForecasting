"""
Test script for Financial Wavelet Prediction API
"""

import requests
import json
import time
import numpy as np
from datetime import datetime


class APITester:
    """Test the prediction API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test health endpoint"""
        print("\n=== Testing Health Endpoint ===")
        response = self.session.get(f"{self.base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    
    def test_single_prediction(self):
        """Test single prediction endpoint"""
        print("\n=== Testing Single Prediction ===")
        
        # Generate random features
        features = np.random.randn(10).tolist()
        
        payload = {
            "features": features,
            "model_type": "ensemble"
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        print("\n=== Testing Batch Prediction ===")
        
        # Generate batch of samples
        samples = [np.random.randn(10).tolist() for _ in range(5)]
        
        payload = {
            "samples": samples,
            "model_type": "ensemble",
            "return_confidence": True
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    
    def test_streaming_prediction(self):
        """Test streaming prediction endpoint"""
        print("\n=== Testing Streaming Prediction ===")
        
        # Simulate streaming data
        for i in range(3):
            features = np.random.randn(10).tolist()
            
            payload = {
                "features": features,
                "window_size": 10,
                "model_type": "ensemble"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/stream",
                json=payload
            )
            
            print(f"\nStream {i+1}:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            time.sleep(0.5)
        
        return True
    
    def test_model_info(self):
        """Test model info endpoints"""
        print("\n=== Testing Model Info ===")
        
        # Get all models
        response = self.session.get(f"{self.base_url}/models")
        print(f"All Models - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Get specific model info
        response = self.session.get(f"{self.base_url}/models/ensemble")
        print(f"\nEnsemble Model - Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return True
    
    def test_metrics(self):
        """Test metrics endpoint"""
        print("\n=== Testing Metrics ===")
        response = self.session.get(f"{self.base_url}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n=== Testing Error Handling ===")
        
        # Test with invalid features
        payload = {
            "features": [],  # Empty features
            "model_type": "ensemble"
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        
        print(f"Empty Features - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test with invalid model type
        payload = {
            "features": [1, 2, 3],
            "model_type": "invalid_model"
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        
        print(f"\nInvalid Model - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return True
    
    def run_performance_test(self, num_requests: int = 100):
        """Run performance test"""
        print(f"\n=== Performance Test ({num_requests} requests) ===")
        
        response_times = []
        errors = 0
        
        for i in range(num_requests):
            features = np.random.randn(10).tolist()
            payload = {
                "features": features,
                "model_type": "ensemble"
            }
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload
                )
                response_time = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    response_times.append(response_time)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"Request {i+1} failed: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{num_requests} requests")
        
        # Calculate statistics
        if response_times:
            avg_time = np.mean(response_times)
            p50_time = np.percentile(response_times, 50)
            p95_time = np.percentile(response_times, 95)
            p99_time = np.percentile(response_times, 99)
            
            print(f"\nPerformance Results:")
            print(f"Successful requests: {len(response_times)}/{num_requests}")
            print(f"Failed requests: {errors}")
            print(f"Average response time: {avg_time:.2f} ms")
            print(f"P50 response time: {p50_time:.2f} ms")
            print(f"P95 response time: {p95_time:.2f} ms")
            print(f"P99 response time: {p99_time:.2f} ms")
        
        return True
    
    def run_all_tests(self):
        """Run all tests"""
        print("Starting API Tests...")
        print(f"Base URL: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        tests = [
            ("Health Check", self.test_health),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Streaming Prediction", self.test_streaming_prediction),
            ("Model Info", self.test_model_info),
            ("Metrics", self.test_metrics),
            ("Error Handling", self.test_error_handling),
            ("Performance Test", lambda: self.run_performance_test(50))
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, "PASSED" if success else "FAILED"))
            except Exception as e:
                print(f"\nError in {test_name}: {e}")
                results.append((test_name, "ERROR"))
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        for test_name, result in results:
            print(f"{test_name}: {result}")
        
        # Get final metrics
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"\nFinal API Metrics:")
                print(f"Total requests: {metrics.get('total_requests', 0)}")
                print(f"Error rate: {metrics.get('error_rate', 0):.2%}")
                print(f"Avg response time: {metrics.get('avg_response_time_ms', 0):.2f} ms")
        except:
            pass


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Financial Wavelet Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", help="Run specific test", 
                       choices=["health", "predict", "batch", "stream", "models", "metrics", "errors", "performance"])
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for performance test")
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url)
    
    if args.test:
        # Run specific test
        test_map = {
            "health": tester.test_health,
            "predict": tester.test_single_prediction,
            "batch": tester.test_batch_prediction,
            "stream": tester.test_streaming_prediction,
            "models": tester.test_model_info,
            "metrics": tester.test_metrics,
            "errors": tester.test_error_handling,
            "performance": lambda: tester.run_performance_test(args.requests)
        }
        
        if args.test in test_map:
            test_map[args.test]()
    else:
        # Run all tests
        tester.run_all_tests()


if __name__ == "__main__":
    main()
