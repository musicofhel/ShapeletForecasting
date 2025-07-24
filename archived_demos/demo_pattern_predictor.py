"""
Demo Script for Pattern Predictor

Demonstrates the pattern prediction system with:
- Multiple model types (LSTM, GRU, Transformer, Markov)
- Ensemble predictions
- Confidence scoring
- Multi-horizon predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.pattern_predictor import PatternPredictor


def generate_synthetic_patterns(n_sequences=100, seq_length=50):
    """Generate synthetic pattern sequences for demo"""
    print("\n1. GENERATING SYNTHETIC PATTERN DATA")
    print("=" * 80)
    
    # Define pattern types and their characteristics
    pattern_types = {
        'uptrend': {
            'amplitude': (0.8, 1.5),
            'duration': (20, 40),
            'energy': (0.6, 0.9),
            'entropy': (0.3, 0.5),
            'skewness': (0.5, 1.5),
            'kurtosis': (2, 4)
        },
        'downtrend': {
            'amplitude': (0.8, 1.5),
            'duration': (20, 40),
            'energy': (0.6, 0.9),
            'entropy': (0.3, 0.5),
            'skewness': (-1.5, -0.5),
            'kurtosis': (2, 4)
        },
        'consolidation': {
            'amplitude': (0.2, 0.5),
            'duration': (10, 25),
            'energy': (0.2, 0.4),
            'entropy': (0.6, 0.8),
            'skewness': (-0.5, 0.5),
            'kurtosis': (1, 2)
        },
        'breakout': {
            'amplitude': (1.5, 2.5),
            'duration': (5, 15),
            'energy': (0.8, 1.0),
            'entropy': (0.2, 0.4),
            'skewness': (1, 2),
            'kurtosis': (3, 5)
        },
        'reversal': {
            'amplitude': (1.0, 2.0),
            'duration': (15, 30),
            'energy': (0.7, 0.9),
            'entropy': (0.4, 0.6),
            'skewness': (-0.5, 0.5),
            'kurtosis': (2, 3)
        }
    }
    
    # Generate sequences with realistic transitions
    sequences = []
    transition_probs = {
        'uptrend': {'uptrend': 0.4, 'consolidation': 0.3, 'reversal': 0.2, 'breakout': 0.1},
        'downtrend': {'downtrend': 0.4, 'consolidation': 0.3, 'reversal': 0.2, 'breakout': 0.1},
        'consolidation': {'consolidation': 0.3, 'breakout': 0.3, 'uptrend': 0.2, 'downtrend': 0.2},
        'breakout': {'uptrend': 0.4, 'downtrend': 0.3, 'consolidation': 0.2, 'reversal': 0.1},
        'reversal': {'uptrend': 0.3, 'downtrend': 0.3, 'consolidation': 0.3, 'breakout': 0.1}
    }
    
    for i in range(n_sequences):
        seq = []
        current_type = np.random.choice(list(pattern_types.keys()))
        
        for j in range(seq_length):
            # Get pattern characteristics
            chars = pattern_types[current_type]
            
            pattern = {
                'type': current_type,
                'scale': np.random.randint(1, 5),
                'amplitude': np.random.uniform(*chars['amplitude']),
                'duration': np.random.uniform(*chars['duration']),
                'energy': np.random.uniform(*chars['energy']),
                'entropy': np.random.uniform(*chars['entropy']),
                'skewness': np.random.uniform(*chars['skewness']),
                'kurtosis': np.random.uniform(*chars['kurtosis'])
            }
            seq.append(pattern)
            
            # Transition to next pattern
            if j < seq_length - 1:
                probs = transition_probs[current_type]
                current_type = np.random.choice(
                    list(probs.keys()),
                    p=list(probs.values())
                )
        
        sequences.append(seq)
    
    print(f"Generated {n_sequences} sequences with {seq_length} patterns each")
    print(f"Pattern types: {list(pattern_types.keys())}")
    
    return sequences


def train_pattern_predictor(sequences):
    """Train the pattern predictor"""
    print("\n2. TRAINING PATTERN PREDICTOR")
    print("=" * 80)
    
    # Initialize predictor
    predictor = PatternPredictor(seq_length=10, device='cpu')
    
    # Split data
    n_train = int(0.8 * len(sequences))
    train_sequences = sequences[:n_train]
    test_sequences = sequences[n_train:]
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # Train models
    print("\nTraining models...")
    predictor.train(
        train_sequences,
        epochs=20,  # Reduced for demo
        batch_size=16,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    return predictor, test_sequences


def evaluate_predictions(predictor, test_sequences):
    """Evaluate prediction accuracy"""
    print("\n3. EVALUATING PREDICTIONS")
    print("=" * 80)
    
    # Single-step predictions
    correct_predictions = 0
    total_predictions = 0
    confidence_scores = []
    
    for seq in test_sequences[:10]:  # Use subset for demo
        for i in range(predictor.seq_length, len(seq) - 1):
            input_seq = seq[i-predictor.seq_length:i]
            true_pattern = seq[i]
            
            # Get prediction
            result = predictor.predict(input_seq, horizon=1)
            pred = result['predictions'][0]
            
            if pred['pattern_type'] == true_pattern['type']:
                correct_predictions += 1
            total_predictions += 1
            confidence_scores.append(pred['confidence'])
    
    accuracy = correct_predictions / total_predictions
    avg_confidence = np.mean(confidence_scores)
    
    print(f"Single-step accuracy: {accuracy*100:.2f}%")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # Calibration evaluation
    print("\nEvaluating calibration...")
    calibration_result = predictor.evaluate_calibration(test_sequences[:10])
    print(f"Expected Calibration Error (ECE): {calibration_result['ece']:.3f}")
    print(f"Mean accuracy: {calibration_result['mean_accuracy']*100:.2f}%")
    
    return calibration_result


def demonstrate_predictions(predictor, test_sequences):
    """Demonstrate various prediction scenarios"""
    print("\n4. PREDICTION DEMONSTRATIONS")
    print("=" * 80)
    
    # Select a test sequence
    test_seq = test_sequences[0]
    
    # Single-step prediction
    print("\n--- Single-Step Prediction ---")
    input_seq = test_seq[:10]
    result = predictor.predict(input_seq, horizon=1)
    pred = result['predictions'][0]
    
    print(f"Input sequence types: {[p['type'] for p in input_seq[-5:]]}")
    print(f"True next pattern: {test_seq[10]['type']}")
    print(f"Predicted pattern: {pred['pattern_type']}")
    print(f"Confidence: {pred['confidence']:.3f}")
    print(f"Confidence interval: [{pred['confidence_interval'][0]:.3f}, {pred['confidence_interval'][1]:.3f}]")
    
    # Show top probabilities
    print("\nTop pattern probabilities:")
    sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for pattern, prob in sorted_probs[:3]:
        print(f"  {pattern}: {prob:.3f}")
    
    # Model confidences
    print("\nModel confidences:")
    for model, conf in pred['model_confidences'].items():
        print(f"  {model}: {conf:.3f}")
    
    # Multi-horizon prediction
    print("\n--- Multi-Horizon Prediction ---")
    result = predictor.predict(input_seq, horizon=5)
    
    print("Predictions for next 5 patterns:")
    for pred in result['predictions']:
        print(f"  Horizon {pred['horizon']}: {pred['pattern_type']} (conf: {pred['confidence']:.3f})")
    
    # Measure prediction latency
    print("\n--- Prediction Latency Test ---")
    latencies = []
    for _ in range(100):
        start_time = time.time()
        _ = predictor.predict(input_seq, horizon=1)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
    
    print(f"Average latency: {np.mean(latencies):.2f}ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.2f}ms")
    print(f"Max latency: {np.max(latencies):.2f}ms")


def visualize_results(predictor, test_sequences, calibration_result):
    """Create visualizations"""
    print("\n5. CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pattern Predictor Analysis', fontsize=16)
    
    # 1. Training history - LSTM
    ax = axes[0, 0]
    history = predictor.training_history['lstm']
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    ax.plot(epochs, train_acc, label='Train', marker='o')
    ax.plot(epochs, val_acc, label='Validation', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LSTM Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Model comparison
    ax = axes[0, 1]
    models = list(predictor.ensemble_weights.keys())
    weights = list(predictor.ensemble_weights.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars = ax.bar(models, weights, color=colors)
    ax.set_ylabel('Ensemble Weight')
    ax.set_title('Ensemble Model Weights')
    ax.set_ylim(0, 0.5)
    
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.2f}', ha='center', va='bottom')
    
    # 3. Calibration curve
    ax = axes[0, 2]
    frac_pos = calibration_result['calibration_curve']['fraction_of_positives']
    mean_pred = calibration_result['calibration_curve']['mean_predicted_value']
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', label='Model calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve (ECE={calibration_result["ece"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Pattern transition matrix
    ax = axes[1, 0]
    
    # Calculate transition frequencies from test data
    pattern_types = ['uptrend', 'downtrend', 'consolidation', 'breakout', 'reversal']
    transition_matrix = np.zeros((len(pattern_types), len(pattern_types)))
    
    for seq in test_sequences[:20]:
        for i in range(len(seq) - 1):
            from_idx = pattern_types.index(seq[i]['type'])
            to_idx = pattern_types.index(seq[i+1]['type'])
            transition_matrix[from_idx, to_idx] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                 where=row_sums != 0, out=transition_matrix)
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pattern_types)))
    ax.set_yticks(range(len(pattern_types)))
    ax.set_xticklabels(pattern_types, rotation=45, ha='right')
    ax.set_yticklabels(pattern_types)
    ax.set_xlabel('To Pattern')
    ax.set_ylabel('From Pattern')
    ax.set_title('Pattern Transition Matrix')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability')
    
    # 5. Confidence distribution
    ax = axes[1, 1]
    
    # Collect confidence scores
    confidences = []
    for seq in test_sequences[:10]:
        for i in range(predictor.seq_length, min(len(seq), predictor.seq_length + 20)):
            input_seq = seq[i-predictor.seq_length:i]
            result = predictor.predict(input_seq, horizon=1)
            confidences.append(result['predictions'][0]['confidence'])
    
    ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidences):.3f}')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Multi-horizon accuracy
    ax = axes[1, 2]
    
    horizons = [1, 2, 3, 4, 5]
    accuracies = []
    
    for h in horizons:
        correct = 0
        total = 0
        
        for seq in test_sequences[:5]:  # Small sample for demo
            if len(seq) >= predictor.seq_length + h:
                input_seq = seq[:predictor.seq_length]
                result = predictor.predict(input_seq.copy(), horizon=h)
                
                for i in range(h):
                    if i < len(result['predictions']):
                        pred_type = result['predictions'][i]['pattern_type']
                        true_type = seq[predictor.seq_length + i]['type']
                        if pred_type == true_type:
                            correct += 1
                        total += 1
        
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
    
    ax.plot(horizons, accuracies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Accuracy')
    ax.set_title('Multi-Horizon Prediction Accuracy')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('pattern_predictor_demo.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'pattern_predictor_demo.png'")
    
    return fig


def save_model(predictor):
    """Save the trained model"""
    print("\n6. SAVING MODEL")
    print("=" * 80)
    
    save_path = 'models/saved_pattern_predictors/demo_model'
    predictor.save(save_path)
    print(f"Model saved to: {save_path}")
    
    # Verify save by loading
    new_predictor = PatternPredictor(device='cpu')
    new_predictor.load(save_path)
    print("Model successfully loaded and verified")


def main():
    """Run the complete demo"""
    print("\n" + "="*80)
    print("PATTERN PREDICTOR DEMO")
    print("="*80)
    
    # Generate data
    sequences = generate_synthetic_patterns(n_sequences=100, seq_length=50)
    
    # Train predictor
    predictor, test_sequences = train_pattern_predictor(sequences)
    
    # Evaluate predictions
    calibration_result = evaluate_predictions(predictor, test_sequences)
    
    # Demonstrate predictions
    demonstrate_predictions(predictor, test_sequences)
    
    # Visualize results
    visualize_results(predictor, test_sequences, calibration_result)
    
    # Save model
    save_model(predictor)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    
    # Display the visualization
    plt.show()


if __name__ == "__main__":
    main()
