"""
Test Suite for Pattern Predictor Module

Tests all components of the pattern prediction system:
- LSTM/GRU models
- Transformer model
- Markov chain model
- Ensemble predictions
- Confidence calibration
- Performance benchmarks
"""

import pytest
import numpy as np
import pandas as pd
import torch
import time
import tempfile
import os
import json
from typing import List, Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.pattern_predictor import (
    PatternDataset,
    LSTMPredictor,
    GRUPredictor,
    TransformerPredictor,
    MarkovChainPredictor,
    PatternPredictor
)


class TestPatternDataset:
    """Test pattern dataset preparation"""
    
    @pytest.fixture
    def sample_sequences(self):
        """Create sample pattern sequences"""
        sequences = []
        pattern_types = ['uptrend', 'downtrend', 'consolidation', 'breakout', 'reversal']
        
        for _ in range(50):
            seq = []
            for _ in range(20):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': np.random.randint(1, 5),
                    'amplitude': np.random.uniform(0.1, 2.0),
                    'duration': np.random.randint(5, 50),
                    'energy': np.random.uniform(0.1, 1.0),
                    'entropy': np.random.uniform(0.1, 1.0),
                    'skewness': np.random.uniform(-2, 2),
                    'kurtosis': np.random.uniform(-2, 5)
                }
                seq.append(pattern)
            sequences.append(seq)
        
        return sequences
    
    def test_dataset_creation(self, sample_sequences):
        """Test dataset creation and preparation"""
        dataset = PatternDataset(sample_sequences, seq_length=10)
        
        assert len(dataset) > 0
        assert dataset.X.shape[1] == 10  # seq_length
        assert dataset.X.shape[2] == 8   # num_features
        assert len(dataset.y) == len(dataset.X)
    
    def test_feature_extraction(self, sample_sequences):
        """Test feature extraction from patterns"""
        dataset = PatternDataset(sample_sequences, seq_length=10)
        
        # Test single sample
        x, y = dataset[0]
        assert x.shape == (10, 8)
        assert y.shape == (1,)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_label_encoding(self, sample_sequences):
        """Test pattern type encoding"""
        dataset = PatternDataset(sample_sequences, seq_length=10)
        
        # Check all pattern types are encoded
        unique_types = set()
        for seq in sample_sequences:
            for pattern in seq:
                unique_types.add(pattern['type'])
        
        assert len(dataset.label_encoder.classes_) == len(unique_types)


class TestNeuralModels:
    """Test neural network models"""
    
    @pytest.fixture
    def model_params(self):
        return {
            'input_size': 8,
            'hidden_size': 64,
            'num_classes': 5,
            'num_layers': 2,
            'dropout': 0.2
        }
    
    def test_lstm_model(self, model_params):
        """Test LSTM model architecture"""
        model = LSTMPredictor(**model_params)
        
        # Test forward pass
        batch_size = 16
        seq_length = 10
        x = torch.randn(batch_size, seq_length, model_params['input_size'])
        
        output = model(x)
        assert output.shape == (batch_size, model_params['num_classes'])
    
    def test_gru_model(self, model_params):
        """Test GRU model architecture"""
        model = GRUPredictor(**model_params)
        
        # Test forward pass
        batch_size = 16
        seq_length = 10
        x = torch.randn(batch_size, seq_length, model_params['input_size'])
        
        output = model(x)
        assert output.shape == (batch_size, model_params['num_classes'])
    
    def test_transformer_model(self):
        """Test Transformer model architecture"""
        model = TransformerPredictor(
            input_size=8,
            num_classes=5,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        # Test forward pass
        batch_size = 16
        seq_length = 10
        x = torch.randn(batch_size, seq_length, 8)
        
        output = model(x)
        assert output.shape == (batch_size, 5)


class TestMarkovChain:
    """Test Markov chain model"""
    
    def test_markov_training(self):
        """Test Markov chain training"""
        sequences = [
            ['A', 'B', 'C', 'A', 'B', 'C'],
            ['A', 'A', 'B', 'B', 'C', 'C'],
            ['C', 'B', 'A', 'C', 'B', 'A']
        ]
        
        model = MarkovChainPredictor(order=2)
        model.fit(sequences)
        
        # Check transition matrix
        assert len(model.transition_matrix) > 0
        assert len(model.pattern_types) == 3
        
        # Check probabilities sum to 1
        for state, transitions in model.transition_matrix.items():
            total_prob = sum(transitions.values())
            assert abs(total_prob - 1.0) < 1e-6
    
    def test_markov_prediction(self):
        """Test Markov chain prediction"""
        sequences = [
            ['A', 'B', 'C', 'A', 'B', 'C'],
            ['A', 'B', 'C', 'A', 'B', 'C']
        ]
        
        model = MarkovChainPredictor(order=2)
        model.fit(sequences)
        
        # Test prediction
        probs = model.predict_proba(['A', 'B'])
        assert 'C' in probs
        assert probs['C'] > 0.9  # Should predict C with high probability
        
        # Test with unseen state
        probs = model.predict_proba(['C', 'A'])
        assert sum(probs.values()) == pytest.approx(1.0)


class TestPatternPredictor:
    """Test main pattern predictor"""
    
    @pytest.fixture
    def predictor(self):
        return PatternPredictor(seq_length=5, device='cpu')
    
    @pytest.fixture
    def training_sequences(self):
        """Create training sequences with patterns"""
        sequences = []
        pattern_types = ['uptrend', 'downtrend', 'consolidation']
        
        # Create sequences with some patterns
        for _ in range(20):
            seq = []
            current_type = np.random.choice(pattern_types)
            
            for i in range(15):
                # Sometimes continue pattern, sometimes change
                if i % 3 == 0:
                    current_type = np.random.choice(pattern_types)
                
                pattern = {
                    'type': current_type,
                    'scale': np.random.randint(1, 4),
                    'amplitude': np.random.uniform(0.5, 1.5),
                    'duration': np.random.randint(10, 30),
                    'energy': np.random.uniform(0.3, 0.8),
                    'entropy': np.random.uniform(0.2, 0.7),
                    'skewness': np.random.uniform(-1, 1),
                    'kurtosis': np.random.uniform(0, 3)
                }
                seq.append(pattern)
            sequences.append(seq)
        
        return sequences
    
    def test_training(self, predictor, training_sequences):
        """Test model training"""
        predictor.train(
            training_sequences,
            epochs=5,  # Reduced for testing
            batch_size=8,
            learning_rate=0.001,
            validation_split=0.2
        )
        
        # Check models are trained
        assert predictor.lstm_model is not None
        assert predictor.gru_model is not None
        assert predictor.transformer_model is not None
        assert len(predictor.markov_model.transition_matrix) > 0
        
        # Check training history
        assert len(predictor.training_history['lstm']) == 5
        assert len(predictor.training_history['gru']) == 5
        assert len(predictor.training_history['transformer']) == 5
    
    def test_prediction_single_horizon(self, predictor, training_sequences):
        """Test single horizon prediction"""
        # Train first
        predictor.train(training_sequences, epochs=5)
        
        # Test prediction
        test_seq = training_sequences[0][:10]
        result = predictor.predict(test_seq, horizon=1)
        
        assert 'predictions' in result
        assert len(result['predictions']) == 1
        
        pred = result['predictions'][0]
        assert 'pattern_type' in pred
        assert 'confidence' in pred
        assert 'confidence_interval' in pred
        assert 'probabilities' in pred
        assert 'model_confidences' in pred
        
        # Check confidence bounds
        assert 0 <= pred['confidence'] <= 1
        assert pred['confidence_interval'][0] <= pred['confidence']
        assert pred['confidence'] <= pred['confidence_interval'][1]
    
    def test_prediction_multi_horizon(self, predictor, training_sequences):
        """Test multi-horizon prediction"""
        # Train first
        predictor.train(training_sequences, epochs=5)
        
        # Test prediction
        test_seq = training_sequences[0][:10]
        result = predictor.predict(test_seq, horizon=3)
        
        assert len(result['predictions']) == 3
        
        for i, pred in enumerate(result['predictions']):
            assert pred['horizon'] == i + 1
    
    def test_prediction_latency(self, predictor, training_sequences):
        """Test prediction latency requirement"""
        # Train first
        predictor.train(training_sequences, epochs=5)
        
        # Test prediction speed
        test_seq = training_sequences[0][:10]
        
        start_time = time.time()
        result = predictor.predict(test_seq, horizon=1)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 50  # Must be under 50ms
    
    def test_variable_sequence_lengths(self, predictor):
        """Test handling of different sequence lengths"""
        pattern_types = ['A', 'B', 'C']
        
        # Create sequences of different lengths
        sequences = []
        for length in [5, 10, 20, 50, 100]:
            seq = []
            for _ in range(length):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': 1,
                    'amplitude': 1.0,
                    'duration': 10,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        # Train with variable lengths
        predictor.train(sequences * 4, epochs=3)  # Repeat for more data
        
        # Test prediction works for different lengths
        for seq in sequences:
            if len(seq) >= predictor.seq_length:
                result = predictor.predict(seq, horizon=1)
                assert 'predictions' in result


class TestCalibration:
    """Test confidence calibration"""
    
    def test_calibration_evaluation(self):
        """Test calibration metrics"""
        predictor = PatternPredictor(seq_length=5, device='cpu')
        
        # Create sequences
        sequences = []
        pattern_types = ['A', 'B', 'C']
        
        for _ in range(30):
            seq = []
            for _ in range(20):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': 1,
                    'amplitude': 1.0,
                    'duration': 10,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        # Train
        predictor.train(sequences[:20], epochs=5)
        
        # Evaluate calibration
        calibration_result = predictor.evaluate_calibration(sequences[20:])
        
        assert 'ece' in calibration_result
        assert 'mean_confidence' in calibration_result
        assert 'mean_accuracy' in calibration_result
        assert 'calibration_curve' in calibration_result
        
        # Check ECE is reasonable (not perfect due to limited training)
        assert calibration_result['ece'] < 0.5


class TestEnsemble:
    """Test ensemble functionality"""
    
    def test_ensemble_weights_optimization(self):
        """Test ensemble weight optimization"""
        predictor = PatternPredictor(seq_length=5, device='cpu')
        
        # Create sequences
        sequences = []
        pattern_types = ['A', 'B', 'C']
        
        for _ in range(20):
            seq = []
            for _ in range(15):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': 1,
                    'amplitude': 1.0,
                    'duration': 10,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        # Train
        predictor.train(sequences, epochs=5)
        
        # Check weights sum to 1
        total_weight = sum(predictor.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        # Check all weights are positive
        for weight in predictor.ensemble_weights.values():
            assert weight > 0


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_save_load(self):
        """Test saving and loading models"""
        predictor = PatternPredictor(seq_length=5, device='cpu')
        
        # Create and train
        sequences = []
        pattern_types = ['A', 'B', 'C']
        
        for _ in range(20):
            seq = []
            for _ in range(15):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': 1,
                    'amplitude': 1.0,
                    'duration': 10,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        predictor.train(sequences, epochs=3)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            predictor.save(save_path)
            
            # Check files exist
            assert os.path.exists(os.path.join(save_path, 'lstm_model.pth'))
            assert os.path.exists(os.path.join(save_path, 'gru_model.pth'))
            assert os.path.exists(os.path.join(save_path, 'transformer_model.pth'))
            assert os.path.exists(os.path.join(save_path, 'markov_model.json'))
            assert os.path.exists(os.path.join(save_path, 'config.json'))
            
            # Load into new predictor
            new_predictor = PatternPredictor(device='cpu')
            new_predictor.load(save_path)
            
            # Test prediction works
            test_seq = sequences[0][:10]
            result = new_predictor.predict(test_seq, horizon=1)
            assert 'predictions' in result


class TestAccuracyBenchmark:
    """Test accuracy benchmarks"""
    
    def test_pattern_type_accuracy(self):
        """Test pattern type prediction accuracy"""
        predictor = PatternPredictor(seq_length=5, device='cpu')
        
        # Create sequences with clear patterns
        sequences = []
        pattern_cycle = ['uptrend', 'consolidation', 'downtrend', 'consolidation']
        
        for _ in range(50):
            seq = []
            for i in range(20):
                pattern = {
                    'type': pattern_cycle[i % len(pattern_cycle)],
                    'scale': 2,
                    'amplitude': 1.0,
                    'duration': 20,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        # Split train/test
        train_sequences = sequences[:40]
        test_sequences = sequences[40:]
        
        # Train
        predictor.train(train_sequences, epochs=10)
        
        # Evaluate accuracy
        correct = 0
        total = 0
        
        for seq in test_sequences:
            for i in range(predictor.seq_length, len(seq)):
                input_seq = seq[i-predictor.seq_length:i]
                true_pattern = seq[i]
                
                result = predictor.predict(input_seq, horizon=1)
                pred_pattern = result['predictions'][0]['pattern_type']
                
                if pred_pattern == true_pattern['type']:
                    correct += 1
                total += 1
        
        accuracy = correct / total
        print(f"Pattern type accuracy: {accuracy*100:.2f}%")
        
        # Should achieve good accuracy on cyclic patterns
        assert accuracy > 0.6  # Relaxed for test environment


class TestCrossValidation:
    """Test cross-validation framework"""
    
    def test_cross_validation(self):
        """Test k-fold cross validation"""
        from sklearn.model_selection import KFold
        
        # Create sequences
        sequences = []
        pattern_types = ['A', 'B', 'C']
        
        for _ in range(50):
            seq = []
            for _ in range(15):
                pattern = {
                    'type': np.random.choice(pattern_types),
                    'scale': 1,
                    'amplitude': 1.0,
                    'duration': 10,
                    'energy': 0.5,
                    'entropy': 0.5,
                    'skewness': 0,
                    'kurtosis': 1
                }
                seq.append(pattern)
            sequences.append(seq)
        
        # K-fold validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
            print(f"\nFold {fold + 1}/5")
            
            train_seqs = [sequences[i] for i in train_idx]
            val_seqs = [sequences[i] for i in val_idx]
            
            # Train new predictor for each fold
            predictor = PatternPredictor(seq_length=5, device='cpu')
            predictor.train(train_seqs, epochs=5)
            
            # Evaluate
            correct = 0
            total = 0
            
            for seq in val_seqs:
                for i in range(predictor.seq_length, len(seq)):
                    input_seq = seq[i-predictor.seq_length:i]
                    true_pattern = seq[i]
                    
                    result = predictor.predict(input_seq, horizon=1)
                    pred_pattern = result['predictions'][0]['pattern_type']
                    
                    if pred_pattern == true_pattern['type']:
                        correct += 1
                    total += 1
            
            if total > 0:
                accuracy = correct / total
                accuracies.append(accuracy)
                print(f"Fold accuracy: {accuracy*100:.2f}%")
        
        # Check cross-validation completed
        assert len(accuracies) == 5
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")


if __name__ == "__main__":
    # Run specific test
    test = TestPatternPredictor()
    predictor = test.predictor()
    sequences = test.training_sequences()
    
    print("Training pattern predictor...")
    test.test_training(predictor, sequences)
    
    print("\nTesting predictions...")
    test.test_prediction_single_horizon(predictor, sequences)
    
    print("\nTesting latency...")
    test.test_prediction_latency(predictor, sequences)
    
    print("\nAll tests completed!")
