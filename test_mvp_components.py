"""
Quick test to verify MVP components work with real YFinance data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing MVP Components...")

# Test 1: Data Manager
print("\n1. Testing Data Manager...")
try:
    from src.dashboard.data_utils import data_manager
    data = data_manager.download_data("AAPL", period="1mo")
    if data is not None:
        print(f"✓ Data loaded: {len(data)} rows")
        print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("✗ Failed to load data")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Pattern Classifier
print("\n2. Testing Pattern Classifier...")
try:
    from src.dashboard.pattern_classifier import PatternClassifier
    classifier = PatternClassifier()
    print("✓ Pattern Classifier initialized")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Pattern Matcher
print("\n3. Testing Pattern Matcher...")
try:
    from src.dashboard.pattern_matcher import PatternMatcher
    matcher = PatternMatcher()
    print("✓ Pattern Matcher initialized")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Pattern Predictor
print("\n4. Testing Pattern Predictor...")
try:
    from src.dashboard.pattern_predictor import PatternPredictor
    predictor = PatternPredictor()
    print("✓ Pattern Predictor initialized")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Wavelet Analysis
print("\n5. Testing Wavelet Analysis...")
try:
    import pywt
    import numpy as np
    if data is not None:
        prices = data['Close'].values
        coeffs = pywt.wavedec(prices, 'db4', level=3)
        print(f"✓ Wavelet decomposition successful: {len(coeffs)} levels")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Streamlit availability
print("\n6. Testing Streamlit...")
try:
    import streamlit as st
    print("✓ Streamlit available")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 7: Plotly availability
print("\n7. Testing Plotly...")
try:
    import plotly.graph_objects as go
    fig = go.Figure()
    print("✓ Plotly available")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✅ Component testing complete!")
print("\nTo run the MVP demo, use:")
print("streamlit run mvp_demo.py")
