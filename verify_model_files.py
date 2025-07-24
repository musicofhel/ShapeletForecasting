"""
Verify that all model files are properly saved and ready for dashboard integration
"""

import os
import json
import pickle
from pathlib import Path

def verify_model_files():
    """Check all required model files exist"""
    model_path = Path("models/pattern_predictor")
    
    required_files = {
        "lstm_model.pth": "LSTM model weights",
        "gru_model.pth": "GRU model weights", 
        "transformer_model.pth": "Transformer model weights",
        "markov_model.json": "Markov chain model",
        "label_encoder.pkl": "Pattern type encoder",
        "feature_scaler.pkl": "Feature scaler",
        "config.json": "Model configuration",
        "sp500_training_metadata.json": "Training metadata"
    }
    
    print("VERIFYING MODEL FILES")
    print("=" * 60)
    
    all_good = True
    
    for file, description in required_files.items():
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"✓ {file:<30} ({size:>8.1f} KB) - {description}")
        else:
            print(f"✗ {file:<30} MISSING - {description}")
            all_good = False
    
    # Check data files
    print("\nVERIFYING DATA FILES")
    print("=" * 60)
    
    data_files = {
        "data/sp500_production_data.pkl": "S&P 500 market data",
        "data/sp500_pattern_sequences.pkl": "Pattern sequences",
        "data/sp500_pattern_summary.json": "Pattern summary"
    }
    
    for file, description in data_files.items():
        file_path = Path(file)
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {file:<40} ({size:>8.1f} MB) - {description}")
        else:
            print(f"✗ {file:<40} MISSING - {description}")
            all_good = False
    
    # Load and display training metadata
    metadata_path = model_path / "sp500_training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\nTRAINING SUMMARY")
        print("=" * 60)
        print(f"Training timestamp: {metadata['training_timestamp']}")
        print(f"Training time: {metadata['training_time']:.1f} seconds")
        print(f"Data source: {metadata['data_source']}")
        print(f"Tickers count: {metadata['tickers_count']}")
        print(f"Train sequences: {metadata['n_train_sequences']}")
        print(f"Test sequences: {metadata['n_test_sequences']}")
        
        print("\nMODEL PERFORMANCE")
        print("=" * 60)
        for metric, value in metadata['evaluation_results']['accuracy'].items():
            print(f"{metric}: {value:.2%}")
        
        print("\nENSEMBLE WEIGHTS")
        print("=" * 60)
        for model, weight in metadata['ensemble_weights'].items():
            print(f"{model}: {weight:.2f}")
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL FILES VERIFIED - Ready for dashboard integration!")
        print("\nNext step: Run Window 3 to integrate models with dashboard")
    else:
        print("❌ Some files are missing - please check the training process")
    
    return all_good

if __name__ == "__main__":
    verify_model_files()
