"""Debug pickle file loading issues"""
import pickle
import os

# Check label encoder
encoder_path = "models/pattern_predictor/label_encoder.pkl"
print(f"Checking {encoder_path}")
print(f"File size: {os.path.getsize(encoder_path)} bytes")

# Read raw bytes
with open(encoder_path, 'rb') as f:
    header = f.read(20)
    print(f"Header (hex): {header.hex()}")
    print(f"Header (ascii): {repr(header)}")

# Try different pickle protocols
for protocol in range(6):
    try:
        with open(encoder_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded with default protocol")
        print(f"  Type: {type(data)}")
        if hasattr(data, 'classes_'):
            print(f"  Classes: {data.classes_}")
        break
    except Exception as e:
        print(f"✗ Failed with protocol {protocol}: {e}")

print("\n" + "="*50 + "\n")

# Check feature scaler
scaler_path = "models/pattern_predictor/feature_scaler.pkl"
print(f"Checking {scaler_path}")
print(f"File size: {os.path.getsize(scaler_path)} bytes")

# Read raw bytes
with open(scaler_path, 'rb') as f:
    header = f.read(20)
    print(f"Header (hex): {header.hex()}")
    print(f"Header (ascii): {repr(header)}")

# Try loading
try:
    with open(scaler_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Successfully loaded")
    print(f"  Type: {type(data)}")
    if hasattr(data, 'mean_'):
        print(f"  Feature means shape: {data.mean_.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*50 + "\n")

# Try to understand the file format
print("Analyzing file format...")

# Check if it's a torch file mistakenly saved as .pkl
import torch

for file in ['label_encoder.pkl', 'feature_scaler.pkl']:
    filepath = os.path.join("models/pattern_predictor", file)
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"✓ {file} is actually a PyTorch file!")
        print(f"  Type: {type(data)}")
        print(f"  Content: {data if isinstance(data, dict) else 'Not a dict'}")
    except:
        print(f"✗ {file} is not a PyTorch file")
