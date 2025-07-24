"""Inspect the actual format of model files"""
import os
import torch
import pickle

model_path = "models/pattern_predictor/lstm_model.pth"

print(f"Inspecting {model_path}")
print(f"File size: {os.path.getsize(model_path)} bytes")

# Read first few bytes to check format
with open(model_path, 'rb') as f:
    header = f.read(20)
    print(f"File header (hex): {header.hex()}")
    print(f"File header (ascii): {repr(header)}")

# Try different loading methods
print("\nTrying different loading methods:")

# Method 1: Direct torch.load
try:
    print("\n1. torch.load with weights_only=False:")
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"   Success! Type: {type(data)}")
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())[:5]}...")
except Exception as e:
    print(f"   Failed: {e}")

# Method 2: Try as pickle
try:
    print("\n2. pickle.load:")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"   Success! Type: {type(data)}")
except Exception as e:
    print(f"   Failed: {e}")

# Method 3: Check if it's a state dict or full model
try:
    print("\n3. Checking content structure:")
    # Create a dummy model to test loading
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(128, 9)
            
    model = TestModel()
    
    # Try loading the state dict
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    print("   Successfully loaded as state_dict!")
    
except Exception as e:
    print(f"   Failed to load as state_dict: {e}")
