"""Check model file formats"""
import torch
import os
import pickle

print("PyTorch version:", torch.__version__)

model_dir = "models/pattern_predictor"
model_files = ["lstm_model.pth", "gru_model.pth", "transformer_model.pth"]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"\nChecking {model_file}:")
    print(f"  File size: {os.path.getsize(model_path)} bytes")
    
    try:
        # Try loading as PyTorch model
        state = torch.load(model_path, map_location='cpu')
        print("  ✓ Loaded as PyTorch model")
        print(f"  Keys: {list(state.keys())[:5]}...")  # Show first 5 keys
    except Exception as e:
        print(f"  ✗ PyTorch load error: {e}")
        
        # Try loading as pickle
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            print("  ✓ Loaded as pickle file")
            print(f"  Type: {type(data)}")
        except Exception as e2:
            print(f"  ✗ Pickle load error: {e2}")

# Check other files
print("\nChecking other files:")
other_files = ["label_encoder.pkl", "feature_scaler.pkl", "markov_model.json", "config.json"]

for file in other_files:
    file_path = os.path.join(model_dir, file)
    if os.path.exists(file_path):
        print(f"  {file}: {os.path.getsize(file_path)} bytes")
