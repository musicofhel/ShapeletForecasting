"""
Train Pattern Predictor Models
Loads pattern sequences from wavelet_pattern_pipeline.py output and trains prediction models
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time

# Import the pattern predictor
from src.dashboard.pattern_predictor import PatternPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternTrainingPipeline:
    """Pipeline for training pattern prediction models"""
    
    def __init__(self, 
                 data_path: str = "data/pattern_sequences.pkl",
                 model_save_path: str = "models/pattern_predictor",
                 seq_length: int = 10):
        """
        Initialize training pipeline
        
        Args:
            data_path: Path to pattern sequences data
            model_save_path: Path to save trained models
            seq_length: Sequence length for prediction
        """
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.seq_length = seq_length
        
        # Load data
        self.data = None
        self.training_sequences = []
        self.pattern_statistics = {}
        
    def load_pattern_data(self) -> bool:
        """Load pattern sequences from pickle file"""
        logger.info(f"Loading pattern data from {self.data_path}")
        
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
            
            logger.info("Pattern data loaded successfully")
            
            # Extract key information
            if 'training_data' in self.data and self.data['training_data']:
                sequences = self.data['training_data']['sequences']
                logger.info(f"Found {len(sequences)} pattern sequences")
                
                # Extract pattern statistics
                self.pattern_statistics = {
                    'total_sequences': len(sequences),
                    'ticker_metadata': self.data['training_data'].get('ticker_metadata', {}),
                    'config': self.data['training_data'].get('config', {})
                }
                
                return True
            else:
                logger.error("No training data found in pattern sequences file")
                return False
                
        except Exception as e:
            logger.error(f"Error loading pattern data: {e}")
            return False
    
    def prepare_training_sequences(self) -> List[List[Dict]]:
        """Convert pattern sequences to format expected by PatternPredictor"""
        logger.info("Preparing training sequences...")
        
        if not self.data or 'training_data' not in self.data:
            logger.error("No data loaded")
            return []
        
        sequences = self.data['training_data']['sequences']
        training_sequences = []
        
        for seq_data in sequences:
            # Convert sequence to pattern dictionaries
            pattern_sequence = []
            
            for i, features in enumerate(seq_data['pattern_features']):
                # Create pattern dictionary with all required fields
                pattern = {
                    'type': f"cluster_{features['cluster_id']}",  # Use cluster ID as pattern type
                    'scale': features['scale'],
                    'amplitude': features['coefficients_stats']['max'] - features['coefficients_stats']['min'],
                    'duration': 1,  # Default duration
                    'energy': features['energy'],
                    'entropy': self._calculate_entropy(features['coefficients_stats']),
                    'skewness': self._calculate_skewness(features['coefficients_stats']),
                    'kurtosis': self._calculate_kurtosis(features['coefficients_stats']),
                    'ticker': features['ticker'],
                    'cluster_id': features['cluster_id']
                }
                pattern_sequence.append(pattern)
            
            if len(pattern_sequence) >= self.seq_length + 1:
                training_sequences.append(pattern_sequence)
        
        logger.info(f"Prepared {len(training_sequences)} training sequences")
        
        # Calculate pattern type distribution
        pattern_types = {}
        for seq in training_sequences:
            for pattern in seq:
                ptype = pattern['type']
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        logger.info(f"Found {len(pattern_types)} unique pattern types")
        logger.info(f"Pattern type distribution: {dict(sorted(pattern_types.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        self.training_sequences = training_sequences
        return training_sequences
    
    def _calculate_entropy(self, stats: Dict) -> float:
        """Calculate entropy from coefficient statistics"""
        # Simple entropy approximation based on variance
        variance = stats['std'] ** 2 if stats['std'] > 0 else 1e-6
        return 0.5 * np.log(2 * np.pi * np.e * variance)
    
    def _calculate_skewness(self, stats: Dict) -> float:
        """Calculate skewness approximation from statistics"""
        # Approximate skewness from mean and std
        if stats['std'] > 0:
            return (stats['mean'] - (stats['max'] + stats['min']) / 2) / stats['std']
        return 0.0
    
    def _calculate_kurtosis(self, stats: Dict) -> float:
        """Calculate kurtosis approximation from statistics"""
        # Approximate kurtosis (excess kurtosis)
        if stats['std'] > 0:
            range_val = stats['max'] - stats['min']
            return (range_val / (4 * stats['std'])) ** 2 - 3
        return 0.0
    
    def split_sequences(self, sequences: List[List[Dict]], 
                       test_split: float = 0.2) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Split sequences into train and test sets"""
        n_sequences = len(sequences)
        
        # Handle small datasets
        if n_sequences < 5:
            logger.warning(f"Only {n_sequences} sequences available. Using all for training with validation split.")
            # Use all sequences for training, rely on validation split
            return sequences, sequences  # Return same sequences for both train and test
        
        n_test = max(1, int(n_sequences * test_split))  # At least 1 test sequence
        
        # Shuffle sequences
        indices = np.random.permutation(n_sequences)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_sequences = [sequences[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        
        logger.info(f"Split data: {len(train_sequences)} train, {len(test_sequences)} test sequences")
        
        return train_sequences, test_sequences
    
    def train_models(self, 
                    epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train pattern prediction models"""
        logger.info("Starting model training...")
        
        # Prepare sequences
        sequences = self.prepare_training_sequences()
        if not sequences:
            logger.error("No sequences available for training")
            return {}
        
        # Split data
        train_sequences, test_sequences = self.split_sequences(sequences)
        
        # Initialize predictor
        predictor = PatternPredictor(seq_length=self.seq_length)
        
        # Train models
        start_time = time.time()
        
        try:
            predictor.train(
                train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=0.2
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            logger.info("Evaluating model performance...")
            evaluation_results = self.evaluate_models(predictor, test_sequences)
            
            # Save models
            logger.info(f"Saving models to {self.model_save_path}")
            predictor.save(str(self.model_save_path))
            
            # Save training metadata
            metadata = {
                'training_time': training_time,
                'training_timestamp': datetime.now().isoformat(),
                'n_train_sequences': len(train_sequences),
                'n_test_sequences': len(test_sequences),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'evaluation_results': evaluation_results,
                'pattern_statistics': self.pattern_statistics,
                'ensemble_weights': predictor.ensemble_weights,
                'training_history': predictor.training_history
            }
            
            with open(self.model_save_path / 'training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {}
    
    def evaluate_models(self, predictor: PatternPredictor, 
                       test_sequences: List[List[Dict]]) -> Dict[str, Any]:
        """Evaluate trained models on test sequences"""
        results = {
            'accuracy': {'1-step': 0, '3-step': 0, '5-step': 0},
            'confidence_metrics': {},
            'pattern_type_accuracy': {},
            'sample_predictions': []
        }
        
        # Calculate accuracy for different horizons
        for horizon in [1, 3, 5]:
            correct = 0
            total = 0
            
            for seq in test_sequences:
                if len(seq) < self.seq_length + horizon:
                    continue
                
                for i in range(len(seq) - self.seq_length - horizon + 1):
                    input_seq = seq[i:i + self.seq_length]
                    true_patterns = seq[i + self.seq_length:i + self.seq_length + horizon]
                    
                    # Get predictions
                    pred_result = predictor.predict(input_seq, horizon=horizon)
                    predictions = pred_result['predictions']
                    
                    # Check accuracy for each step
                    for j in range(min(len(predictions), len(true_patterns))):
                        if predictions[j]['pattern_type'] == true_patterns[j]['type']:
                            correct += 1
                        total += 1
                    
                    # Store sample predictions
                    if len(results['sample_predictions']) < 5 and horizon == 1:
                        results['sample_predictions'].append({
                            'input_types': [p['type'] for p in input_seq[-3:]],
                            'predicted': predictions[0]['pattern_type'],
                            'actual': true_patterns[0]['type'],
                            'confidence': predictions[0]['confidence'],
                            'correct': predictions[0]['pattern_type'] == true_patterns[0]['type']
                        })
            
            if total > 0:
                results['accuracy'][f'{horizon}-step'] = correct / total
        
        # Evaluate confidence calibration
        calibration_results = predictor.evaluate_calibration(test_sequences)
        results['confidence_metrics'] = calibration_results
        
        # Calculate per-pattern-type accuracy
        pattern_correct = {}
        pattern_total = {}
        
        for seq in test_sequences:
            if len(seq) < self.seq_length + 1:
                continue
            
            for i in range(len(seq) - self.seq_length):
                input_seq = seq[i:i + self.seq_length]
                true_pattern = seq[i + self.seq_length]
                
                # Get prediction
                pred_result = predictor.predict(input_seq, horizon=1)
                pred = pred_result['predictions'][0]
                
                # Update counts
                true_type = true_pattern['type']
                if true_type not in pattern_total:
                    pattern_total[true_type] = 0
                    pattern_correct[true_type] = 0
                
                pattern_total[true_type] += 1
                if pred['pattern_type'] == true_type:
                    pattern_correct[true_type] += 1
        
        # Calculate per-type accuracy
        for ptype in pattern_total:
            if pattern_total[ptype] > 0:
                results['pattern_type_accuracy'][ptype] = {
                    'accuracy': pattern_correct[ptype] / pattern_total[ptype],
                    'support': pattern_total[ptype]
                }
        
        return results
    
    def generate_report(self, training_results: Dict[str, Any]):
        """Generate training report"""
        report = []
        report.append("=" * 60)
        report.append("PATTERN PREDICTOR TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"\nTraining completed at: {training_results['training_timestamp']}")
        report.append(f"Training time: {training_results['training_time']:.2f} seconds")
        
        report.append("\n## Dataset Statistics")
        report.append(f"- Training sequences: {training_results['n_train_sequences']}")
        report.append(f"- Test sequences: {training_results['n_test_sequences']}")
        report.append(f"- Tickers analyzed: {len(training_results['pattern_statistics'].get('ticker_metadata', {}))}")
        
        report.append("\n## Model Performance")
        eval_results = training_results['evaluation_results']
        
        report.append("\n### Accuracy by Horizon")
        for horizon, acc in eval_results['accuracy'].items():
            report.append(f"- {horizon}: {acc:.2%}")
        
        report.append("\n### Confidence Calibration")
        conf_metrics = eval_results['confidence_metrics']
        report.append(f"- Expected Calibration Error: {conf_metrics['ece']:.4f}")
        report.append(f"- Mean Confidence: {conf_metrics['mean_confidence']:.2%}")
        report.append(f"- Mean Accuracy: {conf_metrics['mean_accuracy']:.2%}")
        
        report.append("\n### Ensemble Weights")
        for model, weight in training_results['ensemble_weights'].items():
            report.append(f"- {model}: {weight:.2f}")
        
        report.append("\n### Sample Predictions")
        for i, sample in enumerate(eval_results['sample_predictions'][:3]):
            report.append(f"\nExample {i+1}:")
            report.append(f"  Input: {' -> '.join(sample['input_types'][-3:])}")
            report.append(f"  Predicted: {sample['predicted']} (confidence: {sample['confidence']:.2%})")
            report.append(f"  Actual: {sample['actual']}")
            report.append(f"  Correct: {'✓' if sample['correct'] else '✗'}")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_path = self.model_save_path / 'training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        
        return report_path


def main():
    """Run the training pipeline"""
    print("PATTERN PREDICTOR TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PatternTrainingPipeline(
        data_path="data/pattern_sequences.pkl",
        model_save_path="models/pattern_predictor",
        seq_length=10
    )
    
    # Load data
    if not pipeline.load_pattern_data():
        print("Failed to load pattern data. Please run wavelet_pattern_pipeline.py first.")
        return
    
    # Train models
    print("\nStarting model training...")
    training_results = pipeline.train_models(
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    if training_results:
        # Generate report
        report_path = pipeline.generate_report(training_results)
        print(f"\nTraining report saved to: {report_path}")
        print(f"Models saved to: {pipeline.model_save_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Models saved in: {pipeline.model_save_path}")
        print("\nNext steps:")
        print("1. Check the training report for detailed results")
        print("2. Use the trained models in the dashboard")
        print("3. Run Window 3 to integrate with the dashboard")
    else:
        print("\nTraining failed. Check the logs for details.")


if __name__ == "__main__":
    main()
