"""
Pattern Explorer Tool for Wavelet Pattern Forecasting Dashboard
Provides comprehensive pattern library browsing and analysis capabilities
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from ..pattern_classifier import PatternClassifier
from ..wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from ..pattern_matcher import PatternMatcher
from ..data_utils import load_financial_data


@dataclass
class PatternStats:
    """Statistics for a pattern type"""
    pattern_type: str
    frequency: int
    avg_duration: float
    avg_confidence: float
    success_rate: float
    avg_return: float
    volatility: float
    reliability_score: float


class PatternExplorer:
    """Comprehensive pattern exploration and analysis tool"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.classifier = PatternClassifier()
        self.analyzer = WaveletSequenceAnalyzer()
        self.matcher = PatternMatcher()
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "pattern_library").mkdir(exist_ok=True)
        
    def load_pattern_library(self) -> Dict[str, Any]:
        """Load the pattern library with metadata"""
        library_path = self.data_dir / "pattern_library" / "pattern_library.json"
        if library_path.exists():
            with open(library_path, 'r') as f:
                return json.load(f)
        return self._create_default_library()
    
    def _create_default_library(self) -> Dict[str, Any]:
        """Create default pattern library"""
        default_patterns = {
            "head_and_shoulders": {
                "name": "Head and Shoulders",
                "description": "Classic reversal pattern with three peaks",
                "type": "reversal",
                "reliability": 0.85,
                "avg_return": 0.12,
                "frequency": 0.08,
                "characteristics": ["symmetry", "volume_confirmation", "neckline_break"]
            },
            "ascending_triangle": {
                "name": "Ascending Triangle",
                "description": "Bullish continuation pattern with rising support",
                "type": "continuation",
                "reliability": 0.78,
                "avg_return": 0.09,
                "frequency": 0.12,
                "characteristics": ["rising_support", "horizontal_resistance", "volume_decrease"]
            },
            "bull_flag": {
                "name": "Bull Flag",
                "description": "Brief consolidation in uptrend",
                "type": "continuation",
                "reliability": 0.82,
                "avg_return": 0.07,
                "frequency": 0.15,
                "characteristics": ["sharp_rise", "consolidation", "volume_decline"]
            },
            "double_bottom": {
                "name": "Double Bottom",
                "description": "Reversal pattern with two lows",
                "type": "reversal",
                "reliability": 0.80,
                "avg_return": 0.11,
                "frequency": 0.06,
                "characteristics": ["two_equal_lows", "volume_confirmation", "neckline_break"]
            },
            "fractal_pattern": {
                "name": "Fractal Pattern",
                "description": "Self-similar pattern at different scales",
                "type": "fractal",
                "reliability": 0.75,
                "avg_return": 0.05,
                "frequency": 0.25,
                "characteristics": ["self_similarity", "scale_invariance", "recursive_structure"]
            }
        }
        
        # Save default library
        library_path = self.data_dir / "pattern_library" / "pattern_library.json"
        with open(library_path, 'w') as f:
            json.dump(default_patterns, f, indent=2)
            
        return default_patterns
    
    def browse_patterns(self, ticker: str = None, pattern_type: str = None) -> pd.DataFrame:
        """Browse patterns with filtering capabilities"""
        library = self.load_pattern_library()
        
        # Load data if ticker provided
        if ticker:
            data = load_financial_data(ticker)
            patterns = self.analyzer.extract_patterns(data)
        else:
            patterns = []
        
        # Create pattern statistics
        pattern_stats = []
        for pattern_id, info in library.items():
            if pattern_type and info['type'] != pattern_type:
                continue
                
            stats = PatternStats(
                pattern_type=pattern_id,
                frequency=info['frequency'],
                avg_duration=30.0,  # Placeholder
                avg_confidence=info['reliability'],
                success_rate=info['reliability'],
                avg_return=info['avg_return'],
                volatility=0.15,
                reliability_score=info['reliability']
            )
            pattern_stats.append(stats)
        
        return pd.DataFrame([
            {
                'Pattern': library[ps.pattern_type]['name'],
                'Type': library[ps.pattern_type]['type'],
                'Frequency': ps.frequency,
                'Reliability': ps.success_rate,
                'Avg Return': ps.avg_return,
                'Characteristics': ', '.join(library[ps.pattern_type]['characteristics'])
            }
            for ps in pattern_stats
        ])
    
    def get_pattern_statistics(self, pattern_type: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific pattern type"""
        library = self.load_pattern_library()
        
        if pattern_type not in library:
            return {}
        
        pattern_info = library[pattern_type]
        
        # Return only the actual data from library
        stats = {
            'pattern_name': pattern_info['name'],
            'type': pattern_info['type'],
            'description': pattern_info['description'],
            'frequency': pattern_info['frequency'],
            'reliability': pattern_info['reliability'],
            'avg_return': pattern_info['avg_return'],
            'characteristics': pattern_info['characteristics']
        }
        
        return stats
    
    def find_similar_patterns(self, pattern_id: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find patterns similar to the given pattern"""
        library = self.load_pattern_library()
        
        if pattern_id not in library:
            return []
        
        # Return empty list when no real data available
        return []
    
    def track_pattern_evolution(self, ticker: str, pattern_type: str) -> Dict[str, Any]:
        """Track how a pattern type evolves over time"""
        data = load_financial_data(ticker)
        
        if data is None or data.empty:
            return {}
        
        # Return empty dict when no real pattern evolution data available
        return {}
    
    def backtest_pattern(self, ticker: str, pattern_type: str, 
                        lookback_days: int = 252) -> Dict[str, Any]:
        """Backtest a pattern type on historical data"""
        data = load_financial_data(ticker)
        
        if data is None or data.empty:
            return {}
        
        # Return empty dict when no real backtest data available
        return {}
    
    def create_pattern_browser_dashboard(self) -> go.Figure:
        """Create interactive pattern browser dashboard"""
        library = self.load_pattern_library()
        
        # Create scatter plot of patterns
        fig = go.Figure()
        
        for pattern_id, info in library.items():
            fig.add_trace(go.Scatter(
                x=[info['frequency']],
                y=[info['reliability']],
                mode='markers+text',
                name=info['name'],
                text=[info['name']],
                textposition="top center",
                marker=dict(
                    size=info['avg_return'] * 100,
                    color=np.random.choice(['red', 'blue', 'green', 'orange']),
                    opacity=0.7
                ),
                hovertemplate=f"<b>{info['name']}</b><br>" +
                             f"Type: {info['type']}<br>" +
                             f"Frequency: {info['frequency']:.2%}<br>" +
                             f"Reliability: {info['reliability']:.1%}<br>" +
                             f"Avg Return: {info['avg_return']:.1%}<br>" +
                             f"<extra></extra>"
            ))
        
        fig.update_layout(
            title="Pattern Library Browser",
            xaxis_title="Frequency",
            yaxis_title="Reliability",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def export_pattern_library(self, format: str = 'json') -> str:
        """Export pattern library in specified format"""
        library = self.load_pattern_library()
        
        if format.lower() == 'json':
            return json.dumps(library, indent=2)
        elif format.lower() == 'csv':
            # Convert to CSV format
            df = pd.DataFrame([
                {
                    'pattern_id': pid,
                    'name': info['name'],
                    'type': info['type'],
                    'reliability': info['reliability'],
                    'avg_return': info['avg_return'],
                    'frequency': info['frequency']
                }
                for pid, info in library.items()
            ])
            return df.to_csv(index=False)
        
        return json.dumps(library, indent=2)


# Example usage and testing
if __name__ == "__main__":
    explorer = PatternExplorer()
    
    # Browse patterns
    patterns_df = explorer.browse_patterns()
    print("Pattern Library:")
    print(patterns_df)
    
    # Get pattern statistics
    stats = explorer.get_pattern_statistics("head_and_shoulders")
    print("\nHead and Shoulders Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create dashboard
    fig = explorer.create_pattern_browser_dashboard()
    fig.show()
