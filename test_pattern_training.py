"""
Test script for pattern predictor training
Verifies that the training pipeline works correctly
"""

import os
import pickle
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Import components
from train_pattern_predictor import PatternTrainingPipeline
from src.dashboard.pattern_predictor import PatternPredictor


def test_data_loading():
    """Test loading pattern sequences data"""
    print("\n1. Testing data loading...")
    
    data_path = "data/pattern_sequences.pkl"
    if not os.path.exists(data_path):
        print(f"❌ Pattern sequences file not found at {data_path}")
        print("   Please run wavelet_pattern_pipeline.py first")
        return False
    
    # Load and inspect data
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Successfully loaded pattern sequences")
        
        # Check structure
        if 'training_data' in data and data['training_data']:
            sequences = data['training_data']['sequences']
            print(f"✓ Found {len(sequences)} sequences")
            
            # Sample sequence info
            if sequences:
                sample_seq = sequences[0]
                print(f"✓ Sample sequence has {len(sample_seq['pattern_features'])} patterns")
                print(f"✓ Pattern features include: {list(sample_seq['pattern_features'][0].keys())}")
            
            return True
        else:
            print("❌ No training data found in file")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def test_sequence_preparation():
    """Test sequence preparation for training"""
    print("\n2. Testing sequence preparation...")
    
    pipeline = PatternTrainingPipeline()
    
    if not pipeline.load_pattern_data():
        print("❌ Failed to load pattern data")
        return False
    
    sequences = pipeline.prepare_training_sequences()
    
    if not sequences:
        print("❌ No sequences prepared")
        return False
    
    print(f"✓ Prepared {len(sequences)} training sequences")
    
    # Check sequence format
    sample_seq = sequences[0]
    print(f"✓ Sample sequence length: {len(sample_seq)}")
    
    # Check pattern format
    sample_pattern = sample_seq[0]
    required_fields = ['type', 'scale', 'amplitude', 'duration', 'energy', 
                      'entropy', 'skewness', 'kurtosis']
    
    missing_fields = [f for f in required_fields if f not in sample_pattern]
    if missing_fields:
        print(f"❌ Missing fields in pattern: {missing_fields}")
        return False
    
    print(f"✓ All required fields present in patterns")
    print(f"✓ Pattern types: {set(p['type'] for seq in sequences[:10] for p in seq)}")
    
    return True


def test_model_training():
    """Test model training with small dataset"""
    print("\n3. Testing model training (reduced epochs)...")
    
    pipeline = PatternTrainingPipeline(
        seq_length=5  # Shorter sequences for faster testing
    )
    
    if not pipeline.load_pattern_data():
        print("❌ Failed to load pattern data")
        return False
    
    # Train with reduced parameters for testing
    print("   Training models with reduced parameters...")
    training_results = pipeline.train_models(
        epochs=5,  # Reduced epochs for testing
        batch_size=16,
        learning_rate=0.001
    )
    
    if not training_results:
        print("❌ Training failed")
        return False
    
    print("✓ Training completed successfully")
    print(f"✓ Training time: {training_results['training_time']:.2f} seconds")
    
    # Check results
    eval_results = training_results['evaluation_results']
    print(f"✓ 1-step accuracy: {eval_results['accuracy']['1-step']:.2%}")
    print(f"✓ Ensemble weights: {training_results['ensemble_weights']}")
    
    return True


def test_model_loading():
    """Test loading saved models"""
    print("\n4. Testing model loading...")
    
    model_path = "models/pattern_predictor"
    
    if not os.path.exists(model_path):
        print("❌ Model directory not found")
        return False
    
    # Check saved files
    expected_files = [
        'lstm_model.pth',
        'gru_model.pth', 
        'transformer_model.pth',
        'markov_model.json',
        'label_encoder.pkl',
        'feature_scaler.pkl',
        'config.json'
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    print("✓ All model files present")
    
    # Try loading models
    try:
        predictor = PatternPredictor()
        predictor.load(model_path)
        print("✓ Models loaded successfully")
        
        # Test prediction
        # Create dummy sequence for testing
        dummy_sequence = []
        for i in range(10):
            dummy_sequence.append({
                'type': f'cluster_{i % 3}',
                'scale': np.random.rand(),
                'amplitude': np.random.rand(),
                'duration': 1,
                'energy': np.random.rand(),
                'entropy': np.random.rand(),
                'skewness': np.random.randn(),
                'kurtosis': np.random.randn()
            })
        
        result = predictor.predict(dummy_sequence, horizon=1)
        print("✓ Model prediction successful")
        print(f"  Predicted: {result['predictions'][0]['pattern_type']}")
        print(f"  Confidence: {result['predictions'][0]['confidence']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


def test_training_report():
    """Test training report generation"""
    print("\n5. Testing training report...")
    
    report_path = Path("models/pattern_predictor/training_report.txt")
    
    if not report_path.exists():
        print("❌ Training report not found")
        return False
    
    # Read and check report
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    print("✓ Training report found")
    
    # Check key sections
    required_sections = [
        "Dataset Statistics",
        "Model Performance",
        "Accuracy by Horizon",
        "Confidence Calibration",
        "Ensemble Weights"
    ]
    
    missing_sections = [s for s in required_sections if s not in report_content]
    
    if missing_sections:
        print(f"❌ Missing report sections: {missing_sections}")
        return False
    
    print("✓ All report sections present")
    
    # Display report summary
    print("\nReport Preview:")
    print("-" * 40)
    lines = report_content.split('\n')[:20]
    for line in lines:
        print(line)
    print("-" * 40)
    
    return True


def main():
    """Run all tests"""
    print("PATTERN PREDICTOR TRAINING TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Sequence Preparation", test_sequence_preparation),
        ("Model Training", test_model_training),
        ("Model Loading", test_model_loading),
        ("Training Report", test_training_report)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The training pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Run the full training with more epochs for better performance")
        print("2. Proceed to Window 3 for dashboard integration")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
