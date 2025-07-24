"""
Wavelet Pattern Extraction Pipeline for Financial Data
Fetches data from YFinance, performs wavelet analysis, and extracts pattern sequences
Adapted from demo_yfinance_wavelet_integration.py
"""

import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import time

# Import components
from src.dashboard.data_utils_yfinance import YFinanceDataManager
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer, PatternSequence
from src.dashboard.pattern_classifier import PatternClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveletPatternPipeline:
    """
    Complete pipeline for extracting wavelet patterns from financial data
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 output_dir: str = "data",
                 wavelet: str = 'morl',
                 scales: Optional[np.ndarray] = None,
                 n_clusters: int = 15,
                 min_pattern_length: int = 10,
                 max_pattern_length: int = 50):
        """
        Initialize the pattern extraction pipeline
        
        Args:
            cache_dir: Directory for YFinance cache
            output_dir: Directory for output files
            wavelet: Wavelet type for CWT
            scales: Scales for wavelet transform
            n_clusters: Number of pattern clusters
            min_pattern_length: Minimum pattern length
            max_pattern_length: Maximum pattern length
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_manager = YFinanceDataManager(cache_dir=cache_dir)
        self.wavelet_analyzer = WaveletSequenceAnalyzer(
            wavelet=wavelet,
            scales=scales if scales is not None else np.arange(1, 33),
            n_clusters=n_clusters,
            min_pattern_length=min_pattern_length,
            max_pattern_length=max_pattern_length,
            overlap_ratio=0.5,
            pca_components=10
        )
        self.pattern_classifier = PatternClassifier()
        
        # Storage for results
        self.extracted_patterns = []
        self.pattern_sequences = []
        self.ticker_metadata = {}
        
    def fetch_and_prepare_data(self, ticker: str, period_days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch data from yfinance and prepare for wavelet analysis
        Adapted from demo_yfinance_wavelet_integration.py
        """
        logger.info(f"Fetching {ticker} data for the last {period_days} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        try:
            # Download data using yfinance directly for better control
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Extract data from MultiIndex columns
                close_data = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close']
                volume_data = data[('Volume', ticker)] if ('Volume', ticker) in data.columns else data['Volume']
                high_data = data[('High', ticker)] if ('High', ticker) in data.columns else data['High']
                low_data = data[('Low', ticker)] if ('Low', ticker) in data.columns else data['Low']
                open_data = data[('Open', ticker)] if ('Open', ticker) in data.columns else data['Open']
                
                # Flatten to 1D arrays
                prepared_data = pd.DataFrame({
                    'timestamp': data.index,
                    'price': close_data.values.flatten(),
                    'volume': volume_data.values.flatten(),
                    'high': high_data.values.flatten(),
                    'low': low_data.values.flatten(),
                    'open': open_data.values.flatten()
                })
            else:
                # Single ticker, normal columns
                prepared_data = pd.DataFrame({
                    'timestamp': data.index,
                    'price': data['Close'].values.flatten(),
                    'volume': data['Volume'].values.flatten(),
                    'high': data['High'].values.flatten(),
                    'low': data['Low'].values.flatten(),
                    'open': data['Open'].values.flatten()
                })
            
            # Add returns calculation
            prepared_data['returns'] = prepared_data['price'].pct_change().fillna(0)
            
            # Reset index for compatibility
            prepared_data = prepared_data.reset_index(drop=True)
            
            logger.info(f"Data fetched: {len(prepared_data)} data points")
            logger.info(f"Date range: {prepared_data['timestamp'].min()} to {prepared_data['timestamp'].max()}")
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def extract_patterns_from_ticker(self, 
                                   ticker: str,
                                   period_days: int = 180) -> Dict[str, Any]:
        """
        Extract patterns from a single ticker
        """
        logger.info(f"Extracting patterns from {ticker}")
        
        # Fetch and prepare data
        data = self.fetch_and_prepare_data(ticker, period_days)
        if data is None or len(data) < 50:
            logger.warning(f"Insufficient data for {ticker}")
            return None
        
        # Extract patterns from price data
        price_data = data['price'].values
        timestamps = np.arange(len(price_data))
        
        # Run wavelet analysis
        patterns = self.wavelet_analyzer.extract_wavelet_patterns(price_data, timestamps)
        
        if not patterns:
            logger.warning(f"No patterns found for {ticker}")
            return None
        
        # Cluster patterns
        cluster_mapping = self.wavelet_analyzer.cluster_patterns(patterns)
        
        # Identify sequences
        sequences = self.wavelet_analyzer.identify_sequences(min_sequence_length=3)
        
        # Calculate transition matrix
        if sequences:
            transition_matrix = self.wavelet_analyzer.calculate_transition_matrix()
        else:
            transition_matrix = None
        
        # Classify patterns (adapted from demo)
        classified_patterns = []
        for i, pattern in enumerate(patterns[:100]):  # Limit for performance
            if pattern.cluster_id is not None and pattern.cluster_id != -1:
                classification = self.pattern_classifier.classify_pattern(
                    pattern.coefficients
                )
                classified_patterns.append({
                    'pattern_id': pattern.pattern_id,
                    'cluster_id': pattern.cluster_id,
                    'classification': classification,
                    'timestamp': pattern.timestamp,
                    'scale': pattern.scale,
                    'energy': pattern.metadata.get('energy', 0)
                })
        
        # Get statistics
        stats = self.wavelet_analyzer.get_pattern_statistics()
        
        # Prepare results
        results = {
            'ticker': ticker,
            'period_days': period_days,
            'data_points': len(price_data),
            'patterns_found': len(patterns),
            'sequences_found': len(sequences),
            'cluster_mapping': cluster_mapping,
            'transition_matrix': transition_matrix,
            'classified_patterns': classified_patterns,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store ticker metadata
        self.ticker_metadata[ticker] = {
            'latest_price': data['price'].iloc[-1],
            'price_change': (data['price'].iloc[-1] - data['price'].iloc[0]) / data['price'].iloc[0] * 100,
            'volatility': np.std(data['returns']),
            'data_range': {
                'start': str(data['timestamp'].min()),
                'end': str(data['timestamp'].max())
            }
        }
        
        return results
    
    def extract_patterns_from_multiple_tickers(self, 
                                             tickers: List[str],
                                             period_days: int = 180) -> Dict[str, Any]:
        """
        Extract patterns from multiple tickers (adapted from real_time_monitoring_demo)
        """
        logger.info(f"Extracting patterns from {len(tickers)} tickers")
        
        all_results = {}
        all_patterns = []
        all_sequences = []
        
        for ticker in tickers:
            logger.info(f"\n--- Analyzing {ticker} ---")
            
            try:
                # Extract patterns
                results = self.extract_patterns_from_ticker(ticker, period_days)
                
                if results:
                    all_results[ticker] = results
                    
                    # Collect patterns and sequences
                    patterns = self.wavelet_analyzer.patterns
                    sequences = self.wavelet_analyzer.pattern_sequences
                    
                    # Add ticker information to patterns
                    for pattern in patterns:
                        pattern.metadata['ticker'] = ticker
                    
                    all_patterns.extend(patterns)
                    all_sequences.extend(sequences)
                    
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                all_results[ticker] = {'error': str(e)}
        
        # Store combined results
        self.extracted_patterns = all_patterns
        self.pattern_sequences = all_sequences
        
        return all_results
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """
        Prepare pattern sequences for model training
        """
        logger.info("Preparing training data from extracted patterns")
        
        if not self.pattern_sequences:
            logger.warning("No pattern sequences available")
            return None
        
        # Prepare sequence data
        sequence_data = []
        
        for seq in self.pattern_sequences:
            # Get pattern features for each pattern in sequence
            pattern_features = []
            
            for pattern_id in seq.pattern_ids:
                # Find pattern
                pattern = next((p for p in self.extracted_patterns if p.pattern_id == pattern_id), None)
                if pattern:
                    features = {
                        'cluster_id': pattern.cluster_id,
                        'scale': pattern.scale,
                        'energy': pattern.metadata.get('energy', 0),
                        'ticker': pattern.metadata.get('ticker', 'unknown'),
                        'coefficients_stats': {
                            'mean': float(np.mean(pattern.coefficients)),
                            'std': float(np.std(pattern.coefficients)),
                            'max': float(np.max(pattern.coefficients)),
                            'min': float(np.min(pattern.coefficients))
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
        
        # Prepare training data structure
        training_data = {
            'sequences': sequence_data,
            'pattern_vocabulary': self.wavelet_analyzer.pattern_vocabulary,
            'transition_matrix': self.wavelet_analyzer.transition_matrix,
            'cluster_mapping': getattr(self.wavelet_analyzer, 'cluster_to_idx', {}),
            'ticker_metadata': self.ticker_metadata,
            'extraction_timestamp': datetime.now().isoformat(),
            'config': {
                'wavelet': self.wavelet_analyzer.wavelet,
                'scales': self.wavelet_analyzer.scales.tolist(),
                'n_clusters': self.wavelet_analyzer.n_clusters,
                'min_pattern_length': self.wavelet_analyzer.min_pattern_length,
                'max_pattern_length': self.wavelet_analyzer.max_pattern_length
            }
        }
        
        return training_data
    
    def save_results(self, filename: str = "pattern_sequences.pkl"):
        """
        Save extracted patterns and sequences
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving results to {output_path}")
        
        # Prepare data for saving
        save_data = {
            'training_data': self.prepare_training_data(),
            'wavelet_analyzer_state': {
                'patterns': self.wavelet_analyzer.patterns,
                'pattern_vocabulary': self.wavelet_analyzer.pattern_vocabulary,
                'pattern_sequences': self.wavelet_analyzer.pattern_sequences,
                'transition_matrix': self.wavelet_analyzer.transition_matrix,
                'cluster_centers': self.wavelet_analyzer.cluster_centers
            },
            'ticker_metadata': self.ticker_metadata,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_path = self.output_dir / "pattern_metadata.json"
        metadata = {
            'total_patterns': len(self.extracted_patterns),
            'total_sequences': len(self.pattern_sequences),
            'tickers_analyzed': list(self.ticker_metadata.keys()),
            'extraction_timestamp': save_data['extraction_timestamp'],
            'config': save_data['training_data']['config'] if save_data['training_data'] else {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved successfully")
        logger.info(f"  - Patterns: {len(self.extracted_patterns)}")
        logger.info(f"  - Sequences: {len(self.pattern_sequences)}")
        logger.info(f"  - Tickers: {len(self.ticker_metadata)}")
        
        return output_path
    
    def display_summary(self, results: Dict[str, Any]):
        """
        Display pattern monitoring summary (adapted from demo)
        """
        print("\n" + "=" * 60)
        print("PATTERN EXTRACTION SUMMARY")
        print("=" * 60)
        
        for ticker, result in results.items():
            if 'error' in result:
                print(f"\n{ticker}: Error - {result['error']}")
            else:
                print(f"\n{ticker}:")
                print(f"  Data Points: {result['data_points']}")
                print(f"  Patterns Found: {result['patterns_found']}")
                print(f"  Sequences Found: {result['sequences_found']}")
                
                if ticker in self.ticker_metadata:
                    meta = self.ticker_metadata[ticker]
                    print(f"  Latest Price: ${meta['latest_price']:.2f}")
                    print(f"  Period Change: {meta['price_change']:+.2f}%")
                    print(f"  Volatility: {meta['volatility']:.4f}")


def main():
    """
    Run the pattern extraction pipeline
    """
    print("WAVELET PATTERN EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = WaveletPatternPipeline(
        n_clusters=15,
        min_pattern_length=10,
        max_pattern_length=50
    )
    
    # Define tickers to analyze (from demo)
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "SPY", "QQQ", "TSLA"]
    
    # Extract patterns
    results = pipeline.extract_patterns_from_multiple_tickers(tickers, period_days=180)
    
    # Display summary
    pipeline.display_summary(results)
    
    # Save results
    output_path = pipeline.save_results()
    
    print(f"\nPipeline complete! Results saved to {output_path}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
