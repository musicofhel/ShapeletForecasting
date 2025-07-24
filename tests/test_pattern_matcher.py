"""
Unit tests for Pattern Matcher
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import time
import tempfile
import shutil
from typing import List, Dict, Any
import sys
sys.path.append('.')

from src.dashboard.pattern_matcher import (
    PatternMatcher, DTWMatcher, TemplateMatcher, PatternMatch
)


class TestDTWMatcher(unittest.TestCase):
    """Test DTW matching algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dtw_matcher = DTWMatcher()
        
    def test_identical_sequences(self):
        """Test DTW with identical sequences"""
        seq = np.array([1, 2, 3, 4, 5])
        distance, path = self.dtw_matcher.compute_dtw(seq, seq)
        
        self.assertEqual(distance, 0.0)
        self.assertEqual(len(path), len(seq))
        
        # Check diagonal alignment
        for i, (x, y) in enumerate(path):
            self.assertEqual(x, i)
            self.assertEqual(y, i)
    
    def test_shifted_sequences(self):
        """Test DTW with shifted sequences"""
        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([0, 1, 2, 3, 4, 5])  # Shifted right
        
        distance, path = self.dtw_matcher.compute_dtw(seq1, seq2)
        
        self.assertGreater(distance, 0)
        self.assertIsNotNone(path)
    
    def test_different_lengths(self):
        """Test DTW with sequences of different lengths"""
        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([1, 2, 3])
        
        distance, path = self.dtw_matcher.compute_dtw(seq1, seq2)
        
        self.assertGreater(distance, 0)
        self.assertEqual(path[0], (0, 0))  # Start alignment
        self.assertEqual(path[-1], (4, 2))  # End alignment
    
    def test_window_constraints(self):
        """Test DTW with different window constraints"""
        seq1 = np.sin(np.linspace(0, 2*np.pi, 50))
        seq2 = np.sin(np.linspace(0, 2*np.pi, 50) + 0.1)
        
        # Test Sakoe-Chiba band
        dtw_sakoe = DTWMatcher(window_type='sakoe_chiba', window_size=5)
        dist_sakoe, _ = dtw_sakoe.compute_dtw(seq1, seq2)
        
        # Test Itakura parallelogram
        dtw_itakura = DTWMatcher(window_type='itakura', window_size=5)
        dist_itakura, _ = dtw_itakura.compute_dtw(seq1, seq2)
        
        # Test no constraints
        dtw_none = DTWMatcher(window_type='none')
        dist_none, _ = dtw_none.compute_dtw(seq1, seq2)
        
        # Constrained should have higher distance due to limited paths
        self.assertGreaterEqual(dist_sakoe, dist_none)
        self.assertGreaterEqual(dist_itakura, dist_none)
    
    def test_performance(self):
        """Test DTW performance with different sequence lengths"""
        lengths = [50, 100, 200]
        times = []
        
        for length in lengths:
            seq1 = np.random.randn(length)
            seq2 = np.random.randn(length)
            
            start_time = time.time()
            self.dtw_matcher.compute_dtw(seq1, seq2)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Should complete in reasonable time
            self.assertLess(elapsed, 1.0)  # Less than 1 second
        
        # Time should increase roughly quadratically
        # But with window constraints, should be more linear
        print(f"DTW times for lengths {lengths}: {times}")


class TestTemplateMatcher(unittest.TestCase):
    """Test template matching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.template_matcher = TemplateMatcher()
        
    def test_add_template(self):
        """Test adding templates"""
        pattern = np.sin(np.linspace(0, 2*np.pi, 50))
        outcomes = {'returns': [0.01, 0.02, 0.015]}
        
        self.template_matcher.add_template('test_template', pattern, outcomes)
        
        self.assertIn('test_template', self.template_matcher.templates)
        self.assertIn('test_template', self.template_matcher.template_features)
        
        # Check features were extracted
        features = self.template_matcher.template_features['test_template']
        self.assertEqual(len(features), 10)  # 6 statistical + 4 shape features
    
    def test_feature_extraction(self):
        """Test pattern feature extraction"""
        # Create pattern with known properties
        pattern = np.array([0, 1, 0, -1, 0, 1, 0])  # Oscillating pattern
        features = self.template_matcher._extract_features(pattern)
        
        # Check statistical features
        self.assertAlmostEqual(features[0], np.mean(pattern))  # Mean
        self.assertAlmostEqual(features[1], np.std(pattern))   # Std
        self.assertEqual(features[2], -1)  # Min
        self.assertEqual(features[3], 1)   # Max
        
        # Check shape features
        self.assertGreater(features[6], 0)  # Proportion of increases
        self.assertGreater(features[7], 0)  # Total variation
    
    def test_peak_valley_detection(self):
        """Test peak and valley detection"""
        # Create pattern with clear peaks and valleys
        pattern = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0])
        
        peaks = self.template_matcher._find_peaks(pattern)
        valleys = self.template_matcher._find_valleys(pattern)
        
        self.assertEqual(len(peaks), 2)  # Two peaks at indices 1 and 5
        self.assertEqual(len(valleys), 2)  # Two valleys at indices 3 and 7
        self.assertIn(1, peaks)
        self.assertIn(5, peaks)
        self.assertIn(3, valleys)
        self.assertIn(7, valleys)
    
    def test_find_similar_templates(self):
        """Test finding similar templates"""
        # Add multiple templates
        for i in range(10):
            pattern = np.sin(np.linspace(0, 2*np.pi, 50) + i*0.1)
            self.template_matcher.add_template(
                f'template_{i}', pattern, {'returns': [0.01]}
            )
        
        # Query with similar pattern
        query_pattern = np.sin(np.linspace(0, 2*np.pi, 50))
        query_features = self.template_matcher._extract_features(query_pattern)
        
        similar = self.template_matcher.find_similar_templates(query_features, top_k=5)
        
        self.assertEqual(len(similar), 5)
        # First match should have highest similarity
        self.assertGreater(similar[0][1], similar[-1][1])


class TestPatternMatcher(unittest.TestCase):
    """Test main pattern matching system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.matcher = PatternMatcher(template_dir=Path(self.temp_dir))
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test pattern matcher initialization"""
        self.assertIsInstance(self.matcher.dtw_matcher, DTWMatcher)
        self.assertIsInstance(self.matcher.template_matcher, TemplateMatcher)
        self.assertTrue(self.matcher.template_dir.exists())
    
    def test_add_and_save_templates(self):
        """Test adding and saving templates"""
        pattern = np.sin(np.linspace(0, 2*np.pi, 50))
        outcomes = {
            'returns': [0.01, 0.02, 0.015, 0.025, 0.005],
            'volatility': 0.008,
            'sharpe': 1.5
        }
        metadata = {'source': 'test', 'date': '2024-01-01'}
        
        self.matcher.add_template('test_pattern', pattern, outcomes, metadata)
        
        # Check template was added
        self.assertIn('test_pattern', self.matcher.template_matcher.templates)
        
        # Check template was saved
        template_file = self.matcher.template_dir / 'templates.pkl'
        self.assertTrue(template_file.exists())
        
        # Test loading templates
        new_matcher = PatternMatcher(template_dir=Path(self.temp_dir))
        self.assertIn('test_pattern', new_matcher.template_matcher.templates)
    
    def test_pattern_matching_accuracy(self):
        """Test pattern matching accuracy"""
        # Add a test template
        pattern = np.sin(np.linspace(0, 2*np.pi, 50))
        self.matcher.add_template(
            'test_sine', pattern,
            {'returns': np.random.normal(0.01, 0.005, 10).tolist()}
        )
        
        # Test matching with similar pattern
        query = np.sin(np.linspace(0, 2*np.pi, 45)) + np.random.normal(0, 0.05, 45)
        matches = self.matcher.match_pattern(query, top_k=5, min_similarity=0.6)
        
        if len(matches) > 0:
            # Check match properties
            best_match = matches[0]
            self.assertGreaterEqual(best_match.similarity_score, 0.6)
            self.assertIsInstance(best_match.dtw_distance, float)
            self.assertIsInstance(best_match.correlation, float)
            self.assertIsNotNone(best_match.alignment_path)
            self.assertIsNotNone(best_match.historical_outcomes)
    
    def test_matching_performance(self):
        """Test matching performance with multiple templates"""
        # Add multiple templates
        for i in range(100):
            pattern = np.sin(np.linspace(0, 2*np.pi, 50) + i*0.1)
            self.matcher.add_template(
                f'template_{i}', pattern,
                {'returns': np.random.normal(0.01, 0.005, 10).tolist()}
            )
        
        query = np.random.randn(50)
        
        # Test sequential matching
        start_time = time.time()
        matches_seq = self.matcher.match_pattern(
            query, top_k=5, use_parallel=False
        )
        seq_time = time.time() - start_time
        
        # Test parallel matching
        start_time = time.time()
        matches_par = self.matcher.match_pattern(
            query, top_k=5, use_parallel=True
        )
        par_time = time.time() - start_time
        
        print(f"Sequential time: {seq_time:.3f}s, Parallel time: {par_time:.3f}s")
        
        # Should complete quickly
        self.assertLess(par_time, 1.0)
    
    def test_different_length_patterns(self):
        """Test matching patterns of different lengths"""
        # Add templates of various lengths
        for length in [30, 40, 50, 60, 70]:
            pattern = np.sin(np.linspace(0, 2*np.pi, length))
            self.matcher.add_template(
                f'sine_{length}', pattern,
                {'returns': np.random.normal(0.01, 0.005, 10).tolist()}
            )
        
        # Query with medium length
        query = np.sin(np.linspace(0, 2*np.pi, 45))
        matches = self.matcher.match_pattern(query, top_k=5)
        
        self.assertGreater(len(matches), 0)
        
        # Should handle different lengths gracefully
        for match in matches:
            self.assertIsNotNone(match.alignment_path)
            self.assertGreater(match.similarity_score, 0)
    
    def test_forecast_ranges(self):
        """Test forecast range calculation"""
        # Create templates with known outcomes
        for i in range(5):
            pattern = np.random.randn(50)
            returns = np.random.normal(0.02 + i*0.005, 0.01, 20)
            self.matcher.add_template(
                f'pattern_{i}', pattern,
                {'returns': returns.tolist()}
            )
        
        # Get some matches
        query = np.random.randn(50)
        matches = self.matcher.match_pattern(query, top_k=3, min_similarity=0.0)
        
        # Calculate forecast ranges
        forecast = self.matcher.get_forecast_ranges(matches, confidence_levels=[0.68, 0.95])
        
        self.assertIsNotNone(forecast['mean_forecast'])
        self.assertIn('68%', forecast['confidence_intervals'])
        self.assertIn('95%', forecast['confidence_intervals'])
        
        # Check confidence interval properties
        ci_68 = forecast['confidence_intervals']['68%']
        ci_95 = forecast['confidence_intervals']['95%']
        
        self.assertLess(ci_68['lower'], forecast['mean_forecast'])
        self.assertGreater(ci_68['upper'], forecast['mean_forecast'])
        self.assertLess(ci_95['lower'], ci_68['lower'])
        self.assertGreater(ci_95['upper'], ci_68['upper'])
        
        # Check pattern outcomes
        self.assertEqual(len(forecast['pattern_outcomes']), len(matches))
        for outcome in forecast['pattern_outcomes']:
            self.assertIn('template_id', outcome)
            self.assertIn('similarity', outcome)
            self.assertIn('mean_return', outcome)
            self.assertIn('std_return', outcome)
    
    def test_cross_ticker_matching(self):
        """Test pattern matching across different tickers"""
        # Create templates from different "tickers"
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        for ticker in tickers:
            for i in range(5):
                # Each ticker has slightly different pattern characteristics
                base_freq = 1 + tickers.index(ticker) * 0.1
                pattern = np.sin(np.linspace(0, 2*np.pi*base_freq, 50))
                pattern += np.random.normal(0, 0.1, 50)
                
                self.matcher.add_template(
                    f'{ticker}_pattern_{i}',
                    pattern,
                    {'returns': np.random.normal(0.01, 0.005, 10).tolist()},
                    {'ticker': ticker, 'pattern_id': i}
                )
        
        # Query pattern
        query = np.sin(np.linspace(0, 2*np.pi*1.2, 50)) + np.random.normal(0, 0.1, 50)
        matches = self.matcher.match_pattern(query, top_k=10)
        
        # Should find matches from multiple tickers
        matched_tickers = set()
        for match in matches:
            if 'ticker' in match.metadata:
                matched_tickers.add(match.metadata['ticker'])
        
        self.assertGreater(len(matched_tickers), 1)
        print(f"Matched patterns from tickers: {matched_tickers}")
    
    def test_memory_efficiency(self):
        """Test memory usage with many templates"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add many templates
        for i in range(1000):
            pattern = np.random.randn(50)
            self.matcher.add_template(
                f'template_{i}', pattern,
                {'returns': np.random.normal(0.01, 0.005, 10).tolist()}
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase for 1000 templates: {memory_increase:.1f} MB")
        
        # Should be memory efficient
        self.assertLess(memory_increase, 100)
        
        # Test that matching still works efficiently
        query = np.random.randn(50)
        start_time = time.time()
        matches = self.matcher.match_pattern(query, top_k=5)
        elapsed = time.time() - start_time
        
        self.assertLess(elapsed, 0.5)  # Should still be fast
    
    def test_pattern_normalization(self):
        """Test pattern normalization"""
        # Create patterns with different scales
        pattern1 = np.sin(np.linspace(0, 2*np.pi, 50)) * 100  # Large scale
        pattern2 = np.sin(np.linspace(0, 2*np.pi, 50)) * 0.01  # Small scale
        
        norm1 = self.matcher._normalize_pattern(pattern1)
        norm2 = self.matcher._normalize_pattern(pattern2)
        
        # After normalization, should have similar properties
        self.assertAlmostEqual(np.mean(norm1), 0, places=10)
        self.assertAlmostEqual(np.mean(norm2), 0, places=10)
        self.assertAlmostEqual(np.std(norm1), 1, places=5)
        self.assertAlmostEqual(np.std(norm2), 1, places=5)
        
        # Should have high correlation after normalization
        correlation = np.corrcoef(norm1, norm2)[0, 1]
        self.assertGreater(correlation, 0.99)
    
    def test_pattern_variations(self):
        """Test handling of pattern variations"""
        base_pattern = np.sin(np.linspace(0, 2*np.pi, 50))
        
        # Add base pattern
        self.matcher.add_template(
            'base_pattern', base_pattern,
            {'returns': [0.01, 0.02, 0.015]}
        )
        
        # Add variations manually
        for i in range(5):
            # Add noise
            variation = base_pattern + np.random.normal(0, 0.1, len(base_pattern))
            # Add scaling
            variation *= np.random.uniform(0.9, 1.1)
            
            self.matcher.add_template(
                f'variation_{i}', variation,
                {'returns': np.random.normal(0.015, 0.005, 10).tolist()}
            )
        
        # Query with base pattern
        matches = self.matcher.match_pattern(base_pattern, top_k=6)
        
        # Should match base pattern and variations
        self.assertGreater(len(matches), 0)
        
        # Base pattern should be best match
        if matches:
            self.assertEqual(matches[0].template_id, 'base_pattern')


class TestPatternMatchingIntegration(unittest.TestCase):
    """Integration tests for pattern matching system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.matcher = PatternMatcher(template_dir=Path(self.temp_dir))
        
        # Create comprehensive template library
        self._create_template_library()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_template_library(self):
        """Create a comprehensive template library"""
        # Market patterns
        patterns = {
            'bull_flag': self._create_bull_flag(),
            'bear_flag': self._create_bear_flag(),
            'double_bottom': self._create_double_bottom(),
            'ascending_triangle': self._create_ascending_triangle(),
            'channel': self._create_channel_pattern()
        }
        
        # Create variations
        for pattern_name, base_pattern in patterns.items():
            for i in range(20):
                # Vary length, noise, and scale
                length_factor = np.random.uniform(0.8, 1.2)
                noise_level = np.random.uniform(0.05, 0.15)
                scale = np.random.uniform(0.5, 2.0)
                
                # Create variation
                new_length = int(len(base_pattern) * length_factor)
                x_old = np.linspace(0, 1, len(base_pattern))
                x_new = np.linspace(0, 1, new_length)
                varied = np.interp(x_new, x_old, base_pattern) * scale
                varied += np.random.normal(0, noise_level, new_length)
                
                # Generate realistic outcomes based on pattern type
                if 'bull' in pattern_name or 'ascending' in pattern_name:
                    mean_return = np.random.uniform(0.01, 0.03)
                elif 'bear' in pattern_name:
                    mean_return = np.random.uniform(-0.03, -0.01)
                else:
                    mean_return = np.random.uniform(-0.01, 0.01)
                
                returns = np.random.normal(mean_return, 0.01, 30)
                
                self.matcher.add_template(
                    f'{pattern_name}_{i}',
                    varied,
                    {
                        'returns': returns.tolist(),
                        'volatility': np.std(returns),
                        'sharpe': mean_return / np.std(returns) if np.std(returns) > 0 else 0,
                        'max_drawdown': np.min(np.minimum.accumulate(returns))
                    },
                    {
                        'pattern_type': pattern_name,
                        'market_condition': np.random.choice(['trending', 'ranging', 'volatile']),
                        'timeframe': np.random.choice(['1H', '4H', '1D']),
                        'success_rate': np.random.uniform(0.6, 0.8)
                    }
                )
    
    def _create_bull_flag(self):
        """Create bull flag pattern"""
        # Sharp rise followed by consolidation
        rise = np.linspace(0, 1, 20)
        consolidation = np.linspace(1, 0.8, 30) + np.sin(np.linspace(0, 2*np.pi, 30)) * 0.05
        return np.concatenate([rise, consolidation])
    
    def _create_bear_flag(self):
        """Create bear flag pattern"""
        # Sharp drop followed by consolidation
        drop = np.linspace(1, 0, 20)
        consolidation = np.linspace(0, 0.2, 30) + np.sin(np.linspace(0, 2*np.pi, 30)) * 0.05
        return np.concatenate([drop, consolidation])
    
    def _create_double_bottom(self):
        """Create double bottom pattern"""
        x = np.linspace(0, 4*np.pi, 60)
        pattern = -np.abs(np.sin(x/2)) + 1
        pattern[30:35] = pattern[30:35] * 0.9  # Slight difference in bottoms
        return pattern
    
    def _create_ascending_triangle(self):
        """Create ascending triangle pattern"""
        resistance = np.ones(50) * 0.8
        support = np.linspace(0.2, 0.75, 50)
        pattern = np.minimum(resistance, support + np.random.normal(0, 0.05, 50))
        return pattern
    
    def _create_channel_pattern(self):
        """Create channel pattern"""
        x = np.linspace(0, 4*np.pi, 80)
        trend = np.linspace(0, 0.5, 80)
        oscillation = np.sin(x) * 0.2
        return trend + oscillation
    
    def test_real_world_pattern_matching(self):
        """Test pattern matching with realistic scenarios"""
        # Create a query that looks like a bull flag
        query = self._create_bull_flag()
        query += np.random.normal(0, 0.1, len(query))  # Add noise
        
        matches = self.matcher.match_pattern(query, top_k=10, min_similarity=0.7)
        
        # Should primarily match bull flag patterns
        bull_flag_matches = sum(1 for m in matches if 'bull_flag' in m.template_id)
        self.assertGreater(bull_flag_matches, len(matches) // 2)
        
        # Get forecast
        forecast = self.matcher.get_forecast_ranges(matches)
        
        # Bull flag should predict positive returns
        self.assertGreater(forecast['mean_forecast'], 0)
        
        print(f"Bull flag query matched {bull_flag_matches}/{len(matches)} bull flag patterns")
        print(f"Forecast: {forecast['mean_forecast']:.3%} "
              f"[{forecast['confidence_intervals']['95%']['lower']:.3%}, "
              f"{forecast['confidence_intervals']['95%']['upper']:.3%}]")
    
    def test_pattern_discrimination(self):
        """Test ability to discriminate between different patterns"""
        test_patterns = {
            'bull_flag': self._create_bull_flag(),
            'bear_flag': self._create_bear_flag(),
            'double_bottom': self._create_double_bottom()
        }
        
        results = {}
        
        for pattern_name, pattern in test_patterns.items():
            # Add noise
            query = pattern + np.random.normal(0, 0.08, len(pattern))
            matches = self.matcher.match_pattern(query, top_k=5, min_similarity=0.6)
            
            # Count correct matches
            correct_matches = sum(1 for m in matches if pattern_name in m.template_id)
            accuracy = correct_matches / len(matches) if matches else 0
            
            results[pattern_name] = accuracy
            
            print(f"{pattern_name}: {accuracy:.1%} accuracy ({correct_matches}/{len(matches)})")
        
        # Should achieve good discrimination
        for pattern_name, accuracy in results.items():
            self.assertGreater(accuracy, 0.6)  # At least 60% accuracy


if __name__ == '__main__':
    unittest.main()
