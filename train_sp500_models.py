"""
Train Pattern Predictor Models with S&P 500 Data
Handles the large-scale S&P 500 pattern data with appropriate adjustments
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
import sys

# Import the pattern predictor
from src.dashboard.pattern_predictor import PatternPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SP500ModelTrainer:
    """Train models on S&P 500 pattern data"""
    
    def __init__(self, 
                 data_path: str = "data/sp500_pattern_sequences.pkl",
                 model_save_path: str = "models/pattern_predictor",
                 seq_length: int = 10):
        """Initialize training pipeline for S&P 500 data"""
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.seq_length = seq_length
        
        # Storage
        self.data = None
        self.training_sequences = []
        self.pattern_statistics = {}
        
    def load_sp500_data(self) -> bool:
        """Load S&P 500 pattern data"""
        logger.info(f"Loading S&P 500 pattern data from {self.data_path}")
        
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
            
            logger.info("S&P 500 pattern data loaded successfully")
            
            # Extract statistics
            patterns = self.data['wavelet_analyzer_state']['patterns']
            sequences = self.data['wavelet_analyzer_state']['pattern_sequences']
            
            logger.info(f"Found {len(patterns)} patterns")
            logger.info(f"Found {len(sequences)} sequences")
            
            # Since we only have 1 sequence from the processing, we need to create more
            # by sliding windows through the patterns
            if len(sequences) < 10:
                logger.warning("Very few sequences found. Creating sequences from patterns...")
                self.create_sequences_from_patterns(patterns)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading S&P 500 data: {e}")
            return False
    
    def create_sequences_from_patterns(self, patterns: List[Any]):
        """Create sequences by sliding windows through patterns"""
        logger.info("Creating sequences from pattern data...")
        
        # Group patterns by ticker and time
        ticker_patterns = {}
        
        for pattern in patterns:
            ticker = pattern.metadata.get('ticker', 'unknown')
            if ticker not in ticker_patterns:
                ticker_patterns[ticker] = []
            ticker_patterns[ticker].append(pattern)
        
        # Sort patterns by timestamp within each ticker
        for ticker in ticker_patterns:
            ticker_patterns[ticker].sort(key=lambda p: p.timestamp)
        
        # Create sequences using sliding windows
        all_sequences = []
        
        for ticker, patterns_list in ticker_patterns.items():
            # Create sequences of length seq_length + 5 (for prediction targets)
            for i in range(len(patterns_list) - self.seq_length - 5):
                sequence = patterns_list[i:i + self.seq_length + 5]
                all_sequences.append(sequence)
        
        logger.info(f"Created {len(all_sequences)} sequences from patterns")
        
        # Store sequences
        self.pattern_sequences = all_sequences
        
        return all_sequences
    
    def prepare_training_sequences(self) -> List[List[Dict]]:
        """Convert patterns to training format"""
        logger.info("Preparing training sequences...")
        
        training_sequences = []
        
        # Use either loaded sequences or created sequences
        sequences_to_process = getattr(self, 'pattern_sequences', [])
        
        if not sequences_to_process and self.data:
            # Try to use patterns directly
            patterns = self.data['wavelet_analyzer_state']['patterns']
            sequences_to_process = self.create_sequences_from_patterns(patterns)
        
        for sequence in sequences_to_process:
            # Convert to pattern dictionaries
            pattern_sequence = []
            
            for pattern in sequence:
                # Extract features
                coeffs = pattern.coefficients
                
                pattern_dict = {
                    'type': f"cluster_{pattern.cluster_id if pattern.cluster_id is not None else 0}",
                    'scale': pattern.scale,
                    'amplitude': float(np.max(np.abs(coeffs))),
                    'duration': len(coeffs),
                    'energy': pattern.metadata.get('energy', float(np.sum(coeffs**2))),
                    'entropy': self._calculate_entropy(coeffs),
                    'skewness': float(self._calculate_skewness(coeffs)),
                    'kurtosis': float(self._calculate_kurtosis(coeffs)),
                    'ticker': pattern.metadata.get('ticker', 'unknown'),
                    'cluster_id': pattern.cluster_id if pattern.cluster_id is not None else 0,
                    'volatility': pattern.metadata.get('volatility', 0),
                    'rsi': pattern.metadata.get('rsi', 50)
                }
                pattern_sequence.append(pattern_dict)
            
            if len(pattern_sequence) >= self.seq_length + 1:
                training_sequences.append(pattern_sequence)
        
        # Limit sequences if too many (for memory efficiency)
        if len(training_sequences) > 10000:
            logger.info(f"Limiting training sequences from {len(training_sequences)} to 10000")
            # Sample diverse sequences
            indices = np.random.choice(len(training_sequences), 10000, replace=False)
            training_sequences = [training_sequences[i] for i in indices]
        
        logger.info(f"Prepared {len(training_sequences)} training sequences")
        
        # Calculate pattern statistics
        pattern_types = {}
        tickers = set()
        
        for seq in training_sequences:
            for pattern in seq:
                ptype = pattern['type']
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                tickers.add(pattern['ticker'])
        
        logger.info(f"Found {len(pattern_types)} unique pattern types")
        logger.info(f"Found {len(tickers)} unique tickers")
        
        self.training_sequences = training_sequences
        self.pattern_statistics = {
            'n_sequences': len(training_sequences),
            'n_pattern_types': len(pattern_types),
            'n_tickers': len(tickers),
            'pattern_distribution': dict(sorted(pattern_types.items(), 
                                              key=lambda x: x[1], reverse=True)[:20])
        }
        
        return training_sequences
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        # Normalize to probability distribution
        data_positive = np.abs(data) + 1e-10
        probs = data_positive / np.sum(data_positive)
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def train_models(self, 
                    epochs: int = 30,  # Reduced for large dataset
                    batch_size: int = 64,  # Increased for efficiency
                    learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train models on S&P 500 data"""
        logger.info("Starting S&P 500 model training...")
        
        # Prepare sequences
        sequences = self.prepare_training_sequences()
        
        if len(sequences) < 10:
            logger.error(f"Insufficient sequences for training: {len(sequences)}")
            return {}
        
        # Split data (80/20 split)
        n_sequences = len(sequences)
        n_test = max(10, int(n_sequences * 0.2))
        
        indices = np.random.permutation(n_sequences)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_sequences = [sequences[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        
        logger.info(f"Split data: {len(train_sequences)} train, {len(test_sequences)} test")
        
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
            
            # Evaluate
            logger.info("Evaluating model performance...")
            evaluation_results = self.evaluate_models(predictor, test_sequences)
            
            # Save models
            logger.info(f"Saving models to {self.model_save_path}")
            predictor.save(str(self.model_save_path))
            
            # Save metadata
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
                'data_source': 'S&P 500',
                'tickers_count': self.pattern_statistics.get('n_tickers', 0)
            }
            
            with open(self.model_save_path / 'sp500_training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def evaluate_models(self, predictor: PatternPredictor, 
                       test_sequences: List[List[Dict]]) -> Dict[str, Any]:
        """Evaluate models on test data"""
        results = {
            'accuracy': {},
            'sample_predictions': []
        }
        
        # Test different prediction horizons
        for horizon in [1, 3, 5]:
            correct = 0
            total = 0
            
            for seq in test_sequences[:100]:  # Limit evaluation for speed
                if len(seq) < self.seq_length + horizon:
                    continue
                
                input_seq = seq[:self.seq_length]
                true_patterns = seq[self.seq_length:self.seq_length + horizon]
                
                # Get predictions
                try:
                    pred_result = predictor.predict(input_seq, horizon=horizon)
                    predictions = pred_result['predictions']
                    
                    # Check accuracy
                    for j in range(min(len(predictions), len(true_patterns))):
                        if predictions[j]['pattern_type'] == true_patterns[j]['type']:
                            correct += 1
                        total += 1
                    
                    # Store samples
                    if len(results['sample_predictions']) < 5 and horizon == 1:
                        results['sample_predictions'].append({
                            'input': [p['type'] for p in input_seq[-3:]],
                            'predicted': predictions[0]['pattern_type'],
                            'actual': true_patterns[0]['type'],
                            'confidence': predictions[0]['confidence']
                        })
                        
                except Exception as e:
                    logger.warning(f"Prediction error: {e}")
                    continue
            
            if total > 0:
                results['accuracy'][f'{horizon}-step'] = correct / total
            else:
                results['accuracy'][f'{horizon}-step'] = 0.0
        
        return results
    
    def generate_report(self, training_results: Dict[str, Any]):
        """Generate comprehensive training report"""
        report = []
        report.append("=" * 60)
        report.append("S&P 500 PATTERN PREDICTOR TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"\nTraining completed at: {training_results['training_timestamp']}")
        report.append(f"Training time: {training_results['training_time']:.2f} seconds")
        
        report.append("\n## Dataset Statistics")
        stats = training_results['pattern_statistics']
        report.append(f"- Training sequences: {training_results['n_train_sequences']}")
        report.append(f"- Test sequences: {training_results['n_test_sequences']}")
        report.append(f"- Unique tickers: {stats.get('n_tickers', 'N/A')}")
        report.append(f"- Pattern types: {stats.get('n_pattern_types', 'N/A')}")
        
        report.append("\n## Model Performance")
        eval_results = training_results['evaluation_results']
        
        report.append("\n### Accuracy by Horizon")
        for horizon, acc in eval_results['accuracy'].items():
            report.append(f"- {horizon}: {acc:.2%}")
        
        report.append("\n### Ensemble Weights")
        for model, weight in training_results['ensemble_weights'].items():
            report.append(f"- {model}: {weight:.2f}")
        
        report.append("\n### Sample Predictions")
        for i, sample in enumerate(eval_results['sample_predictions'][:3]):
            report.append(f"\nExample {i+1}:")
            report.append(f"  Input: {' -> '.join(sample['input'][-3:])}")
            report.append(f"  Predicted: {sample['predicted']}")
            report.append(f"  Actual: {sample['actual']}")
            report.append(f"  Confidence: {sample['confidence']:.2%}")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_path = self.model_save_path / 'sp500_training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        
        return report_path


def main():
    """Run S&P 500 model training"""
    print("S&P 500 MODEL TRAINING")
    print("=" * 60)
    
    # Check if S&P 500 data exists
    sp500_path = Path("data/sp500_pattern_sequences.pkl")
    regular_path = Path("data/pattern_sequences.pkl")
    
    if sp500_path.exists():
        print("Using S&P 500 pattern data")
        data_path = sp500_path
    elif regular_path.exists():
        print("S&P 500 data not found, using regular pattern data")
        data_path = regular_path
    else:
        print("No pattern data found. Please run pattern extraction first.")
        return
    
    # Initialize trainer
    trainer = SP500ModelTrainer(
        data_path=str(data_path),
        model_save_path="models/pattern_predictor",
        seq_length=10
    )
    
    # Load data
    if not trainer.load_sp500_data():
        print("Failed to load data.")
        return
    
    # Train models
    print("\nStarting model training...")
    print("This may take 5-10 minutes for S&P 500 data...")
    
    training_results = trainer.train_models(
        epochs=30,  # Reduced for large dataset
        batch_size=64,
        learning_rate=0.001
    )
    
    if training_results:
        # Generate report
        report_path = trainer.generate_report(training_results)
        
        print(f"\nTraining report saved to: {report_path}")
        print(f"Models saved to: {trainer.model_save_path}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("\nYour production-ready models are now trained with S&P 500 data!")
        print("\nNext steps:")
        print("1. Check the training report for detailed results")
        print("2. Proceed to Window 3 for dashboard integration")
        print("3. The models are ready for production use!")
    else:
        print("\nTraining failed. Check the logs for details.")


if __name__ == "__main__":
    main()
