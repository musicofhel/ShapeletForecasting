"""
Test script to verify pattern data can be loaded
"""

import pickle
import numpy as np

# Load the pattern data
with open('data/pattern_sequences.pkl', 'rb') as f:
    data = pickle.load(f)

print("Pattern Data Verification")
print("=" * 60)

# Check structure
print("\nData keys:", list(data.keys()))

# Check training data
if 'training_data' in data:
    training = data['training_data']
    if training:
        print("\nTraining data keys:", list(training.keys()))
        print(f"Number of sequences: {len(training.get('sequences', []))}")
        print(f"Pattern vocabulary size: {len(training.get('pattern_vocabulary', {}))}")
        
        # Check transition matrix
        if training.get('transition_matrix') is not None:
            tm = training['transition_matrix']
            if isinstance(tm, np.ndarray):
                print(f"Transition matrix shape: {tm.shape}")
            else:
                print(f"Transition matrix type: {type(tm)}")

# Check ticker metadata
if 'ticker_metadata' in data:
    print("\nTicker metadata:")
    for ticker, meta in data['ticker_metadata'].items():
        print(f"  {ticker}:")
        print(f"    Latest price: ${meta['latest_price']:.2f}")
        print(f"    Change: {meta['price_change']:+.2f}%")
        print(f"    Volatility: {meta['volatility']:.4f}")

# Check wavelet analyzer state
if 'wavelet_analyzer_state' in data:
    state = data['wavelet_analyzer_state']
    print(f"\nWavelet analyzer state:")
    print(f"  Patterns: {len(state.get('patterns', []))}")
    print(f"  Sequences: {len(state.get('pattern_sequences', []))}")
    print(f"  Vocabulary size: {len(state.get('pattern_vocabulary', {}))}")

print("\n✅ Data loaded successfully!")
print("✅ Ready for model training (Window 2)")
