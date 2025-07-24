"""
Demo: Real Pattern Matching with YFinance Data
Run this to test the pattern matching with real market data
"""

import numpy as np
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.data_utils import DataManager
from sklearn.preprocessing import StandardScaler

# Initialize components
matcher = PatternMatcher()
dm = DataManager()

# Get real market data
ticker = 'BTC-USD'
print(f"Loading {ticker} data...")
data = dm.download_data(ticker, period='1d')

if data is not None and not data.empty:
    prices = data['Close'].values[-50:]  # Last 50 points
    
    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Find patterns
    print("\nSearching for patterns...")
    matches = matcher.match_pattern(normalized, top_k=5, min_similarity=0.6)
    
    print(f"\nFound {len(matches)} matches:")
    for i, match in enumerate(matches):
        print(f"\n{i+1}. {match.template_id}")
        print(f"   Similarity: {match.similarity_score:.3f}")
        print(f"   Expected return: {match.historical_outcomes.get('avg_return', 0):.2f}%")
else:
    print("Failed to load data")
