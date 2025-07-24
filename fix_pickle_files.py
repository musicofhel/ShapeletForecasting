"""
Fix corrupted pickle files by regenerating them with dummy data
"""
import pickle
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create models directory if it doesn't exist
os.makedirs("models/pattern_predictor", exist_ok=True)

# Pattern types
pattern_types = [
    'head_shoulders', 'double_top', 'double_bottom',
    'triangle_ascending', 'triangle_descending',
    'flag_bull', 'flag_bear', 'wedge_rising', 'wedge_falling',
    'pattern_10', 'pattern_11', 'pattern_12', 'pattern_13',
    'pattern_14', 'pattern_15', 'pattern_16', 'pattern_17',
    'pattern_18', 'pattern_19', 'pattern_20'
]

# Create and save label encoder
print("Creating label encoder...")
label_encoder = LabelEncoder()
label_encoder.fit(pattern_types[:20])  # Fit with 20 classes to match the model
with open("models/pattern_predictor/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f, protocol=4)
print("✓ Label encoder saved")

# Create and save feature scaler
print("Creating feature scaler...")
feature_scaler = StandardScaler()
# Create dummy data with 8 features (matching the model input size)
dummy_features = np.random.randn(1000, 8)
feature_scaler.fit(dummy_features)
with open("models/pattern_predictor/feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f, protocol=4)
print("✓ Feature scaler saved")

# Create Markov model
print("Creating Markov model...")
markov_model = {}
for pattern in pattern_types[:9]:  # Use only the first 9 patterns
    # Create transition probabilities to other patterns
    transitions = {}
    remaining_prob = 1.0
    for i, next_pattern in enumerate(pattern_types[:9]):
        if i < 8:
            prob = np.random.uniform(0.05, 0.2)
            remaining_prob -= prob
        else:
            prob = remaining_prob
        transitions[next_pattern] = max(0.01, prob)
    
    # Normalize probabilities
    total = sum(transitions.values())
    for key in transitions:
        transitions[key] /= total
    
    markov_model[pattern] = transitions

with open("models/pattern_predictor/markov_model.json", "w") as f:
    json.dump(markov_model, f, indent=2)
print("✓ Markov model saved")

# Update config to ensure consistency
config = {
    "seq_length": 10,
    "n_features": 8,
    "n_classes": 20,
    "ensemble_weights": {
        "lstm": 0.4,
        "gru": 0.2,
        "transformer": 0.3,
        "markov": 0.1
    },
    "model_params": {
        "input_size": 8,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": 20,
        "dropout": 0.2
    }
}

with open("models/pattern_predictor/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✓ Config updated")

print("\nAll pickle files have been regenerated successfully!")
print("\nVerifying files...")

# Verify the files can be loaded
try:
    with open("models/pattern_predictor/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print(f"✓ Label encoder loaded successfully. Classes: {len(le.classes_)}")
    
    with open("models/pattern_predictor/feature_scaler.pkl", "rb") as f:
        fs = pickle.load(f)
    print(f"✓ Feature scaler loaded successfully. Features: {fs.n_features_in_}")
    
    with open("models/pattern_predictor/markov_model.json", "r") as f:
        mm = json.load(f)
    print(f"✓ Markov model loaded successfully. Patterns: {len(mm)}")
    
except Exception as e:
    print(f"✗ Error verifying files: {e}")
