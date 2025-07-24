"""
Process S&P 500 data through wavelet pattern extraction pipeline
Adapted to handle the sp500_production_data.pkl format
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

from wavelet_pattern_pipeline import WaveletPatternPipeline
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SP500PatternProcessor:
    """Process S&P 500 data through wavelet analysis"""
    
    def __init__(self, 
                 input_path: str = "data/sp500_production_data.pkl",
                 output_path: str = "data/sp500_pattern_sequences.pkl"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Initialize wavelet pipeline
        self.wavelet_pipeline = WaveletPatternPipeline(
            n_clusters=20,  # More clusters for diverse data
            min_pattern_length=10,
            max_pattern_length=50
        )
        
        # Storage
        self.all_patterns = []
        self.all_sequences = []
        self.ticker_results = {}
        
    def load_sp500_data(self) -> Dict[str, Any]:
        """Load S&P 500 data from pickle file"""
        logger.info(f"Loading S&P 500 data from {self.input_path}")
        
        with open(self.input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded data for {len(data['ticker_data'])} tickers")
        logger.info(f"Total sequences: {data['metadata']['total_sequences']}")
        
        return data
    
    def process_ticker_period(self, ticker: str, period_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single ticker-period combination"""
        df = period_data['data']
        period = period_data['period']
        
        # Prepare data for wavelet analysis
        price_data = df['Close'].values
        timestamps = np.arange(len(price_data))
        
        # Run wavelet analysis
        patterns = self.wavelet_pipeline.wavelet_analyzer.extract_wavelet_patterns(
            price_data, timestamps
        )
        
        if not patterns:
            return None
        
        # Add metadata to patterns
        for pattern in patterns:
            pattern.metadata.update({
                'ticker': ticker,
                'period': period,
                'volatility': df['volatility'].mean() if 'volatility' in df else 0,
                'rsi': df['rsi'].mean() if 'rsi' in df else 50
            })
        
        return {
            'patterns': patterns,
            'count': len(patterns),
            'period': period
        }
    
    def process_all_tickers(self, sp500_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all tickers through wavelet analysis"""
        logger.info("Processing all tickers through wavelet analysis...")
        
        total_patterns = 0
        processed_tickers = 0
        
        for ticker, ticker_data in sp500_data['ticker_data'].items():
            ticker_patterns = []
            
            for period_data in ticker_data:
                result = self.process_ticker_period(ticker, period_data)
                
                if result:
                    ticker_patterns.extend(result['patterns'])
                    total_patterns += result['count']
            
            if ticker_patterns:
                self.all_patterns.extend(ticker_patterns)
                self.ticker_results[ticker] = {
                    'pattern_count': len(ticker_patterns),
                    'periods_analyzed': len(ticker_data)
                }
                processed_tickers += 1
            
            # Progress update
            if processed_tickers % 10 == 0:
                logger.info(f"Processed {processed_tickers} tickers, {total_patterns} patterns found")
        
        logger.info(f"Total patterns extracted: {total_patterns}")
        return {
            'total_patterns': total_patterns,
            'processed_tickers': processed_tickers
        }
    
    def cluster_and_sequence(self):
        """Cluster patterns and identify sequences"""
        logger.info("Clustering patterns...")
        
        if not self.all_patterns:
            logger.error("No patterns to cluster")
            return
        
        # Cluster patterns
        cluster_mapping = self.wavelet_pipeline.wavelet_analyzer.cluster_patterns(
            self.all_patterns
        )
        
        # Identify sequences
        sequences = self.wavelet_pipeline.wavelet_analyzer.identify_sequences(
            min_sequence_length=3
        )
        
        self.all_sequences = sequences
        
        logger.info(f"Found {len(sequences)} pattern sequences")
        
        # Calculate transition matrix
        if sequences:
            transition_matrix = self.wavelet_pipeline.wavelet_analyzer.calculate_transition_matrix()
        else:
            transition_matrix = None
        
        return {
            'cluster_mapping': cluster_mapping,
            'sequence_count': len(sequences),
            'transition_matrix': transition_matrix
        }
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare comprehensive training data"""
        logger.info("Preparing training data...")
        
        # Group sequences by ticker
        ticker_sequences = {}
        for seq in self.all_sequences:
            # Get ticker from first pattern in sequence
            if seq.pattern_ids:
                pattern = next((p for p in self.all_patterns if p.pattern_id == seq.pattern_ids[0]), None)
                if pattern and 'ticker' in pattern.metadata:
                    ticker = pattern.metadata['ticker']
                    if ticker not in ticker_sequences:
                        ticker_sequences[ticker] = []
                    ticker_sequences[ticker].append(seq)
        
        # Prepare sequence data with enhanced features
        sequence_data = []
        
        for seq in self.all_sequences:
            pattern_features = []
            
            for pattern_id in seq.pattern_ids:
                pattern = next((p for p in self.all_patterns if p.pattern_id == pattern_id), None)
                if pattern:
                    features = {
                        'cluster_id': pattern.cluster_id,
                        'scale': pattern.scale,
                        'energy': pattern.metadata.get('energy', 0),
                        'ticker': pattern.metadata.get('ticker', 'unknown'),
                        'period': pattern.metadata.get('period', 'unknown'),
                        'volatility': pattern.metadata.get('volatility', 0),
                        'rsi': pattern.metadata.get('rsi', 50),
                        'coefficients_stats': {
                            'mean': float(np.mean(pattern.coefficients)),
                            'std': float(np.std(pattern.coefficients)),
                            'max': float(np.max(pattern.coefficients)),
                            'min': float(np.min(pattern.coefficients)),
                            'skew': float(self.calculate_skew(pattern.coefficients)),
                            'kurtosis': float(self.calculate_kurtosis(pattern.coefficients))
                        }
                    }
                    pattern_features.append(features)
            
            if pattern_features:
                sequence_data.append({
                    'sequence_id': seq.sequence_id,
                    'pattern_features': pattern_features,
                    'pattern_ids': seq.pattern_ids,
                    'timestamps': seq.timestamps,
                    'metadata': seq.metadata
                })
        
        training_data = {
            'sequences': sequence_data,
            'pattern_vocabulary': self.wavelet_pipeline.wavelet_analyzer.pattern_vocabulary,
            'transition_matrix': self.wavelet_pipeline.wavelet_analyzer.transition_matrix,
            'cluster_mapping': getattr(self.wavelet_pipeline.wavelet_analyzer, 'cluster_to_idx', {}),
            'ticker_sequences': {k: len(v) for k, v in ticker_sequences.items()},
            'extraction_timestamp': datetime.now().isoformat(),
            'config': {
                'wavelet': self.wavelet_pipeline.wavelet_analyzer.wavelet,
                'scales': self.wavelet_pipeline.wavelet_analyzer.scales.tolist(),
                'n_clusters': self.wavelet_pipeline.wavelet_analyzer.n_clusters,
                'min_pattern_length': self.wavelet_pipeline.wavelet_analyzer.min_pattern_length,
                'max_pattern_length': self.wavelet_pipeline.wavelet_analyzer.max_pattern_length,
                'source': 'S&P 500',
                'ticker_count': len(ticker_sequences)
            }
        }
        
        return training_data
    
    def calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def save_results(self):
        """Save processed results"""
        logger.info(f"Saving results to {self.output_path}")
        
        # Prepare comprehensive save data
        save_data = {
            'training_data': self.prepare_training_data(),
            'wavelet_analyzer_state': {
                'patterns': self.all_patterns,
                'pattern_vocabulary': self.wavelet_pipeline.wavelet_analyzer.pattern_vocabulary,
                'pattern_sequences': self.all_sequences,
                'transition_matrix': self.wavelet_pipeline.wavelet_analyzer.transition_matrix,
                'cluster_centers': self.wavelet_pipeline.wavelet_analyzer.cluster_centers
            },
            'ticker_results': self.ticker_results,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Save as pickle
        with open(self.output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save summary
        summary_path = self.output_path.parent / 'sp500_pattern_summary.json'
        summary = {
            'total_patterns': len(self.all_patterns),
            'total_sequences': len(self.all_sequences),
            'tickers_processed': len(self.ticker_results),
            'extraction_timestamp': save_data['extraction_timestamp'],
            'config': save_data['training_data']['config']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved successfully")
        logger.info(f"  - Patterns: {len(self.all_patterns)}")
        logger.info(f"  - Sequences: {len(self.all_sequences)}")
        logger.info(f"  - Tickers: {len(self.ticker_results)}")
    
    def run(self):
        """Run the complete processing pipeline"""
        logger.info("Starting S&P 500 pattern processing...")
        
        # Load data
        sp500_data = self.load_sp500_data()
        
        # Process all tickers
        process_stats = self.process_all_tickers(sp500_data)
        
        # Cluster and sequence
        cluster_stats = self.cluster_and_sequence()
        
        # Save results
        self.save_results()
        
        return {
            'process_stats': process_stats,
            'cluster_stats': cluster_stats,
            'output_path': self.output_path
        }


def main():
    """Run S&P 500 pattern processing"""
    print("S&P 500 PATTERN PROCESSING")
    print("=" * 60)
    print("\nThis will process the S&P 500 data through wavelet analysis.")
    print("Expected time: 5-10 minutes\n")
    
    try:
        processor = SP500PatternProcessor()
        results = processor.run()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"\nPatterns extracted: {results['process_stats']['total_patterns']}")
        print(f"Sequences found: {results['cluster_stats']['sequence_count']}")
        print(f"Tickers processed: {results['process_stats']['processed_tickers']}")
        print(f"\nResults saved to: {results['output_path']}")
        print("\nNext step: Re-train models with this comprehensive dataset")
        print("Run: python train_pattern_predictor.py --input data/sp500_pattern_sequences.pkl")
        
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        raise


if __name__ == "__main__":
    main()
