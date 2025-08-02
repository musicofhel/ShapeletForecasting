"""
Shapelet Discovery Module

This module discovers discriminative shapelets from time series data.
Shapelets are labeled using SAX (Symbolic Aggregate approXimation) and
stored in a library tagged by ticker and timeframe.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

# Import SAX transformer from existing module
from ..advanced.time_series_integration import SAXTransformer, SAXConfig

logger = logging.getLogger(__name__)


@dataclass
class Shapelet:
    """Data class representing a discovered shapelet"""
    shapelet_id: str
    sax_label: str
    ticker: str
    timeframe: str
    length: int
    raw_data: List[float]
    start_date: str
    end_date: str
    statistics: Dict[str, float]
    discovered_at: str
    source_context: Dict[str, Any]  # Additional context about where it was found
    start_idx: int = 0  # Add start index
    end_idx: int = 0    # Add end index
    start_time: Optional[str] = None  # Add start time
    end_time: Optional[str] = None    # Add end time


class ShapeletDiscoverer:
    """
    Discovers shapelets from time series data using sliding window approach
    and information gain metrics.
    """
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 50,
                 step_size: int = 1,
                 sax_segments: int = 20,
                 sax_alphabet_size: int = 5,
                 min_frequency: int = 3):
        """
        Initialize the shapelet discoverer.
        
        Args:
            min_length: Minimum shapelet length
            max_length: Maximum shapelet length
            step_size: Step size for sliding window
            sax_segments: Number of segments for SAX
            sax_alphabet_size: Size of SAX alphabet
            min_frequency: Minimum frequency for a shapelet to be included
        """
        self.min_length = min_length
        self.max_length = max_length
        self.step_size = step_size
        self.min_frequency = min_frequency
        
        # Initialize SAX transformer
        self.sax_config = SAXConfig(
            n_segments=sax_segments,
            alphabet_size=sax_alphabet_size
        )
        self.sax_transformer = SAXTransformer(self.sax_config)
        
        # Storage for discovered shapelets
        self.shapelet_library = {}
        self.sax_to_shapelets = {}  # Map SAX labels to shapelet IDs
        
    def discover_shapelets(self, 
                          data: pd.DataFrame,
                          ticker: str,
                          timeframe: str,
                          price_col: str = 'close') -> List[Shapelet]:
        """
        Discover shapelets from time series data.
        
        Args:
            data: DataFrame with time series data
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1D', '1H', '5M')
            price_col: Column name for price data
            
        Returns:
            List of discovered shapelets
        """
        logger.info(f"Discovering shapelets for {ticker} ({timeframe})")
        
        # Extract price series
        prices = data[price_col].values
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['timestamp'])
        
        # Normalize the entire series
        scaler = StandardScaler()
        normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Extract all candidate shapelets
        candidates = self._extract_candidates(normalized_prices, timestamps)
        
        # Group by SAX representation
        sax_groups = self._group_by_sax(candidates)
        
        # Select representative shapelets
        discovered_shapelets = []
        
        for sax_label, shapelet_group in sax_groups.items():
            if len(shapelet_group) >= self.min_frequency:
                # Select the most representative shapelet from the group
                representative = self._select_representative(shapelet_group)
                
                # Calculate statistics
                statistics = self._calculate_statistics(
                    shapelet_group, 
                    normalized_prices,
                    prices
                )
                
                # Create shapelet object
                shapelet = Shapelet(
                    shapelet_id=f"{ticker}_{timeframe}_{sax_label}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    sax_label=sax_label,
                    ticker=ticker,
                    timeframe=timeframe,
                    length=len(representative['data']),
                    raw_data=representative['data'].tolist(),
                    start_date=str(representative['start_time']),
                    end_date=str(representative['end_time']),
                    statistics=statistics,
                    discovered_at=datetime.now().isoformat(),
                    source_context={
                        'frequency': len(shapelet_group),
                        'positions': [s['start_idx'] for s in shapelet_group[:5]],  # First 5 occurrences
                        'scaler_params': {
                            'mean': float(scaler.mean_[0]),
                            'scale': float(scaler.scale_[0])
                        }
                    },
                    start_idx=representative['start_idx'],
                    end_idx=representative['end_idx'],
                    start_time=str(representative['start_time']),
                    end_time=str(representative['end_time'])
                )
                
                discovered_shapelets.append(shapelet)
                
        logger.info(f"Discovered {len(discovered_shapelets)} unique shapelets")
        return discovered_shapelets
    
    def _extract_candidates(self, 
                           normalized_prices: np.ndarray,
                           timestamps: pd.DatetimeIndex) -> List[Dict]:
        """Extract all candidate shapelets using sliding window"""
        candidates = []
        
        for length in range(self.min_length, min(self.max_length + 1, len(normalized_prices))):
            for start_idx in range(0, len(normalized_prices) - length + 1, self.step_size):
                end_idx = start_idx + length
                
                # Extract subsequence
                subsequence = normalized_prices[start_idx:end_idx]
                
                # Get SAX representation
                sax_label = self.sax_transformer.transform(subsequence)
                
                candidates.append({
                    'data': subsequence,
                    'sax': sax_label,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx - 1],
                    'length': length
                })
                
        return candidates
    
    def _group_by_sax(self, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """Group candidates by their SAX representation"""
        sax_groups = {}
        
        for candidate in candidates:
            sax_label = candidate['sax']
            if sax_label not in sax_groups:
                sax_groups[sax_label] = []
            sax_groups[sax_label].append(candidate)
            
        return sax_groups
    
    def _select_representative(self, shapelet_group: List[Dict]) -> Dict:
        """Select the most representative shapelet from a group"""
        # First, ensure all shapelets in the group have the same length
        # Group by length
        length_groups = {}
        for shapelet in shapelet_group:
            length = len(shapelet['data'])
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(shapelet)
        
        # Find the most common length
        most_common_length = max(length_groups.keys(), key=lambda k: len(length_groups[k]))
        same_length_group = length_groups[most_common_length]
        
        # Calculate the centroid of all shapelets with the same length
        all_data = np.array([s['data'] for s in same_length_group])
        centroid = np.mean(all_data, axis=0)
        
        # Find the shapelet closest to the centroid
        min_dist = float('inf')
        representative = None
        
        for shapelet in same_length_group:
            dist = np.linalg.norm(shapelet['data'] - centroid)
            if dist < min_dist:
                min_dist = dist
                representative = shapelet
                
        return representative
    
    def _calculate_statistics(self, 
                            shapelet_group: List[Dict],
                            normalized_prices: np.ndarray,
                            original_prices: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for a shapelet group"""
        # Calculate average return after pattern
        returns_after = []
        
        for shapelet in shapelet_group:
            end_idx = shapelet['end_idx']
            
            # Look ahead window (e.g., 5 periods)
            look_ahead = 5
            if end_idx + look_ahead < len(original_prices):
                start_price = original_prices[end_idx]
                end_price = original_prices[end_idx + look_ahead]
                return_pct = (end_price - start_price) / start_price
                returns_after.append(return_pct)
        
        # Calculate statistics
        statistics = {
            'frequency': len(shapelet_group),
            'avg_return_after': float(np.mean(returns_after)) if returns_after else 0.0,
            'std_return_after': float(np.std(returns_after)) if returns_after else 0.0,
            'win_rate': float(np.sum(np.array(returns_after) > 0) / len(returns_after)) if returns_after else 0.5,
            'avg_length': float(np.mean([s['length'] for s in shapelet_group])),
            'confidence': float(len(shapelet_group) / len(normalized_prices))  # Rough confidence metric
        }
        
        return statistics
    
    def add_to_library(self, shapelets: List[Shapelet]):
        """Add discovered shapelets to the library"""
        for shapelet in shapelets:
            self.shapelet_library[shapelet.shapelet_id] = shapelet
            
            # Update SAX to shapelet mapping
            if shapelet.sax_label not in self.sax_to_shapelets:
                self.sax_to_shapelets[shapelet.sax_label] = []
            self.sax_to_shapelets[shapelet.sax_label].append(shapelet.shapelet_id)
            
        logger.info(f"Added {len(shapelets)} shapelets to library")
    
    def find_shapelet_matches(self, 
                            data: np.ndarray,
                            threshold: float = 0.9) -> List[Tuple[Shapelet, float, int]]:
        """
        Find matching shapelets in new data.
        
        Args:
            data: Time series data to search
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (shapelet, similarity_score, position) tuples
        """
        matches = []
        
        # Normalize input data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Slide through the data
        for shapelet_id, shapelet in self.shapelet_library.items():
            shapelet_len = shapelet.length
            
            if len(normalized_data) < shapelet_len:
                continue
                
            for i in range(len(normalized_data) - shapelet_len + 1):
                subsequence = normalized_data[i:i + shapelet_len]
                
                # Calculate SAX representation
                sax_label = self.sax_transformer.transform(subsequence)
                
                # Quick SAX-based filtering
                if sax_label == shapelet.sax_label:
                    # Calculate exact similarity
                    similarity = self._calculate_similarity(
                        subsequence, 
                        np.array(shapelet.raw_data)
                    )
                    
                    if similarity >= threshold:
                        matches.append((shapelet, similarity, i))
                        
        return matches
    
    def _calculate_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate similarity between two sequences"""
        # Ensure same length
        if len(seq1) != len(seq2):
            return 0.0
            
        # Use correlation as similarity metric
        correlation = np.corrcoef(seq1, seq2)[0, 1]
        
        # Convert to 0-1 range
        similarity = (correlation + 1) / 2
        
        return similarity
    
    def save_library(self, filepath: str):
        """Save shapelet library to file"""
        library_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'n_shapelets': len(self.shapelet_library),
                'sax_config': {
                    'n_segments': self.sax_config.n_segments,
                    'alphabet_size': self.sax_config.alphabet_size
                }
            },
            'shapelets': [asdict(s) for s in self.shapelet_library.values()]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(library_data, f, indent=2)
            
        logger.info(f"Saved shapelet library to {filepath}")
    
    def load_library(self, filepath: str):
        """Load shapelet library from file"""
        with open(filepath, 'r') as f:
            library_data = json.load(f)
            
        self.shapelet_library.clear()
        self.sax_to_shapelets.clear()
        
        for shapelet_dict in library_data['shapelets']:
            shapelet = Shapelet(**shapelet_dict)
            self.shapelet_library[shapelet.shapelet_id] = shapelet
            
            if shapelet.sax_label not in self.sax_to_shapelets:
                self.sax_to_shapelets[shapelet.sax_label] = []
            self.sax_to_shapelets[shapelet.sax_label].append(shapelet.shapelet_id)
            
        logger.info(f"Loaded {len(self.shapelet_library)} shapelets from {filepath}")
    
    def get_shapelets_by_ticker(self, ticker: str) -> List[Shapelet]:
        """Get all shapelets for a specific ticker"""
        return [s for s in self.shapelet_library.values() if s.ticker == ticker]
    
    def get_shapelets_by_sax(self, sax_label: str) -> List[Shapelet]:
        """Get all shapelets with a specific SAX label"""
        shapelet_ids = self.sax_to_shapelets.get(sax_label, [])
        return [self.shapelet_library[sid] for sid in shapelet_ids if sid in self.shapelet_library]
    
    def get_library_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the shapelet library"""
        if not self.shapelet_library:
            return {'total_shapelets': 0}
            
        all_shapelets = list(self.shapelet_library.values())
        
        # Group by ticker
        ticker_counts = {}
        for s in all_shapelets:
            ticker_counts[s.ticker] = ticker_counts.get(s.ticker, 0) + 1
            
        # Group by timeframe
        timeframe_counts = {}
        for s in all_shapelets:
            timeframe_counts[s.timeframe] = timeframe_counts.get(s.timeframe, 0) + 1
            
        # SAX label distribution
        sax_counts = {sax: len(ids) for sax, ids in self.sax_to_shapelets.items()}
        
        return {
            'total_shapelets': len(all_shapelets),
            'unique_sax_labels': len(self.sax_to_shapelets),
            'ticker_distribution': ticker_counts,
            'timeframe_distribution': timeframe_counts,
            'sax_label_distribution': dict(sorted(sax_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_shapelet_length': np.mean([s.length for s in all_shapelets]),
            'avg_frequency': np.mean([s.statistics['frequency'] for s in all_shapelets])
        }
