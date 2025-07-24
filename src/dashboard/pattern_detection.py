"""
Pattern Detection Module

This module implements various technical pattern detection algorithms
for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)

class PatternDetector:
    """Detect various technical patterns in price data"""
    
    def __init__(self, min_pattern_length: int = 10, max_pattern_length: int = 50):
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        
    def detect_all_patterns(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect all supported patterns in the price data"""
        patterns = []
        
        # Detect each pattern type
        pattern_detectors = [
            ('head_shoulders', self.detect_head_shoulders),
            ('double_top', self.detect_double_top),
            ('double_bottom', self.detect_double_bottom),
            ('triangle_ascending', self.detect_triangle_ascending),
            ('triangle_descending', self.detect_triangle_descending),
            ('flag_bull', self.detect_flag_pattern),
            ('flag_bear', self.detect_flag_pattern),
            ('wedge_rising', self.detect_wedge_pattern),
            ('wedge_falling', self.detect_wedge_pattern)
        ]
        
        for pattern_type, detector in pattern_detectors:
            try:
                if 'bear' in pattern_type or 'falling' in pattern_type or 'descending' in pattern_type:
                    detected = detector(prices, timestamps, bearish=True)
                else:
                    detected = detector(prices, timestamps, bearish=False)
                    
                for pattern in detected:
                    pattern['type'] = pattern_type
                    patterns.append(pattern)
                    
            except Exception as e:
                logger.error(f"Error detecting {pattern_type}: {e}")
                
        return patterns
    
    def find_local_extrema(self, prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima in price data"""
        # Find peaks (local maxima)
        peaks, _ = find_peaks(prices, distance=order)
        
        # Find valleys (local minima)
        valleys, _ = find_peaks(-prices, distance=order)
        
        return peaks, valleys
    
    def detect_head_shoulders(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None, 
                            bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect head and shoulders patterns"""
        patterns = []
        peaks, valleys = self.find_local_extrema(prices)
        
        # Need at least 3 peaks and 2 valleys for head and shoulders
        if len(peaks) < 3 or len(valleys) < 2:
            return patterns
        
        # Look for head and shoulders pattern
        for i in range(len(peaks) - 2):
            for j in range(len(valleys) - 1):
                # Check if valleys are between the peaks
                if valleys[j] > peaks[i] and valleys[j] < peaks[i+1] and \
                   valleys[j+1] > peaks[i+1] and valleys[j+1] < peaks[i+2]:
                    
                    # Get the three peaks (shoulders and head)
                    left_shoulder = peaks[i]
                    head = peaks[i+1]
                    right_shoulder = peaks[i+2]
                    
                    # Check if middle peak is highest (head)
                    if prices[head] > prices[left_shoulder] and prices[head] > prices[right_shoulder]:
                        # Check if shoulders are roughly at same level (within 5%)
                        shoulder_diff = abs(prices[left_shoulder] - prices[right_shoulder]) / prices[left_shoulder]
                        
                        if shoulder_diff < 0.05:
                            pattern = {
                                'name': 'Head and Shoulders',
                                'start_idx': left_shoulder,
                                'end_idx': right_shoulder,
                                'confidence': 0.8 - shoulder_diff * 2,  # Higher confidence for more symmetric shoulders
                                'key_points': {
                                    'left_shoulder': left_shoulder,
                                    'head': head,
                                    'right_shoulder': right_shoulder,
                                    'neckline': (valleys[j], valleys[j+1])
                                }
                            }
                            
                            if timestamps is not None:
                                pattern['start_time'] = timestamps[left_shoulder]
                                pattern['end_time'] = timestamps[right_shoulder]
                                
                            patterns.append(pattern)
                            
        return patterns
    
    def detect_double_top(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                         bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect double top patterns"""
        patterns = []
        peaks, valleys = self.find_local_extrema(prices)
        
        # Need at least 2 peaks and 1 valley
        if len(peaks) < 2 or len(valleys) < 1:
            return patterns
        
        # Look for double top pattern
        for i in range(len(peaks) - 1):
            for j in range(len(valleys)):
                # Check if valley is between the peaks
                if valleys[j] > peaks[i] and valleys[j] < peaks[i+1]:
                    # Check if peaks are at similar levels (within 3%)
                    peak_diff = abs(prices[peaks[i]] - prices[peaks[i+1]]) / prices[peaks[i]]
                    
                    if peak_diff < 0.03:
                        pattern = {
                            'name': 'Double Top',
                            'start_idx': peaks[i],
                            'end_idx': peaks[i+1],
                            'confidence': 0.85 - peak_diff * 5,
                            'key_points': {
                                'first_top': peaks[i],
                                'second_top': peaks[i+1],
                                'valley': valleys[j]
                            }
                        }
                        
                        if timestamps is not None:
                            pattern['start_time'] = timestamps[peaks[i]]
                            pattern['end_time'] = timestamps[peaks[i+1]]
                            
                        patterns.append(pattern)
                        
        return patterns
    
    def detect_double_bottom(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                           bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect double bottom patterns"""
        patterns = []
        peaks, valleys = self.find_local_extrema(prices)
        
        # Need at least 2 valleys and 1 peak
        if len(valleys) < 2 or len(peaks) < 1:
            return patterns
        
        # Look for double bottom pattern
        for i in range(len(valleys) - 1):
            for j in range(len(peaks)):
                # Check if peak is between the valleys
                if peaks[j] > valleys[i] and peaks[j] < valleys[i+1]:
                    # Check if valleys are at similar levels (within 3%)
                    valley_diff = abs(prices[valleys[i]] - prices[valleys[i+1]]) / prices[valleys[i]]
                    
                    if valley_diff < 0.03:
                        pattern = {
                            'name': 'Double Bottom',
                            'start_idx': valleys[i],
                            'end_idx': valleys[i+1],
                            'confidence': 0.85 - valley_diff * 5,
                            'key_points': {
                                'first_bottom': valleys[i],
                                'second_bottom': valleys[i+1],
                                'peak': peaks[j]
                            }
                        }
                        
                        if timestamps is not None:
                            pattern['start_time'] = timestamps[valleys[i]]
                            pattern['end_time'] = timestamps[valleys[i+1]]
                            
                        patterns.append(pattern)
                        
        return patterns
    
    def detect_triangle_ascending(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                                 bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect ascending triangle patterns"""
        patterns = []
        
        # Scan through price data with sliding windows
        for window_size in range(self.min_pattern_length, min(self.max_pattern_length, len(prices))):
            for start in range(len(prices) - window_size):
                end = start + window_size
                window_prices = prices[start:end]
                
                # Find peaks and valleys in window
                peaks, valleys = self.find_local_extrema(window_prices, order=3)
                
                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Check if peaks form horizontal resistance
                    peak_prices = window_prices[peaks]
                    peak_std = np.std(peak_prices) / np.mean(peak_prices)
                    
                    # Check if valleys form ascending support
                    valley_prices = window_prices[valleys]
                    valley_indices = valleys + start
                    
                    if len(valley_prices) >= 2 and peak_std < 0.02:  # Peaks are horizontal
                        # Fit line to valleys
                        slope, intercept, r_value, _, _ = linregress(valleys, valley_prices)
                        
                        if slope > 0 and abs(r_value) > 0.8:  # Ascending valleys with good fit
                            pattern = {
                                'name': 'Ascending Triangle',
                                'start_idx': start,
                                'end_idx': end,
                                'confidence': 0.7 + abs(r_value) * 0.2,
                                'key_points': {
                                    'resistance_level': np.mean(peak_prices),
                                    'support_slope': slope,
                                    'peaks': peaks + start,
                                    'valleys': valleys + start
                                }
                            }
                            
                            if timestamps is not None:
                                pattern['start_time'] = timestamps[start]
                                pattern['end_time'] = timestamps[end-1]
                                
                            patterns.append(pattern)
                            
        return patterns
    
    def detect_triangle_descending(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                                  bearish: bool = True) -> List[Dict[str, Any]]:
        """Detect descending triangle patterns"""
        patterns = []
        
        # Scan through price data with sliding windows
        for window_size in range(self.min_pattern_length, min(self.max_pattern_length, len(prices))):
            for start in range(len(prices) - window_size):
                end = start + window_size
                window_prices = prices[start:end]
                
                # Find peaks and valleys in window
                peaks, valleys = self.find_local_extrema(window_prices, order=3)
                
                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Check if valleys form horizontal support
                    valley_prices = window_prices[valleys]
                    valley_std = np.std(valley_prices) / np.mean(valley_prices)
                    
                    # Check if peaks form descending resistance
                    peak_prices = window_prices[peaks]
                    
                    if len(peak_prices) >= 2 and valley_std < 0.02:  # Valleys are horizontal
                        # Fit line to peaks
                        slope, intercept, r_value, _, _ = linregress(peaks, peak_prices)
                        
                        if slope < 0 and abs(r_value) > 0.8:  # Descending peaks with good fit
                            pattern = {
                                'name': 'Descending Triangle',
                                'start_idx': start,
                                'end_idx': end,
                                'confidence': 0.7 + abs(r_value) * 0.2,
                                'key_points': {
                                    'support_level': np.mean(valley_prices),
                                    'resistance_slope': slope,
                                    'peaks': peaks + start,
                                    'valleys': valleys + start
                                }
                            }
                            
                            if timestamps is not None:
                                pattern['start_time'] = timestamps[start]
                                pattern['end_time'] = timestamps[end-1]
                                
                            patterns.append(pattern)
                            
        return patterns
    
    def detect_flag_pattern(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                           bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect flag patterns (bull or bear)"""
        patterns = []
        
        # Flag pattern: strong move followed by consolidation
        for i in range(20, len(prices) - 20):
            # Look for strong initial move (pole)
            pole_start = i - 20
            pole_end = i
            pole_move = (prices[pole_end] - prices[pole_start]) / prices[pole_start]
            
            # Bull flag: strong up move, Bear flag: strong down move
            if (not bearish and pole_move > 0.1) or (bearish and pole_move < -0.1):
                # Look for consolidation (flag)
                flag_prices = prices[pole_end:min(pole_end + 20, len(prices))]
                
                if len(flag_prices) >= 10:
                    # Check if price consolidates in a channel
                    flag_std = np.std(flag_prices) / np.mean(flag_prices)
                    
                    if flag_std < 0.03:  # Low volatility consolidation
                        # Fit trend line to flag
                        x = np.arange(len(flag_prices))
                        slope, intercept, r_value, _, _ = linregress(x, flag_prices)
                        
                        # Flag should have slight counter-trend
                        if (not bearish and slope < 0) or (bearish and slope > 0):
                            pattern = {
                                'name': 'Bull Flag' if not bearish else 'Bear Flag',
                                'start_idx': pole_start,
                                'end_idx': pole_end + len(flag_prices) - 1,
                                'confidence': 0.7 + min(abs(pole_move), 0.2),
                                'key_points': {
                                    'pole_start': pole_start,
                                    'pole_end': pole_end,
                                    'flag_start': pole_end,
                                    'flag_end': pole_end + len(flag_prices) - 1,
                                    'pole_strength': pole_move
                                }
                            }
                            
                            if timestamps is not None:
                                pattern['start_time'] = timestamps[pole_start]
                                pattern['end_time'] = timestamps[pattern['end_idx']]
                                
                            patterns.append(pattern)
                            
        return patterns
    
    def detect_wedge_pattern(self, prices: np.ndarray, timestamps: Optional[np.ndarray] = None,
                           bearish: bool = False) -> List[Dict[str, Any]]:
        """Detect wedge patterns (rising or falling)"""
        patterns = []
        
        # Scan through price data with sliding windows
        for window_size in range(self.min_pattern_length, min(self.max_pattern_length, len(prices))):
            for start in range(len(prices) - window_size):
                end = start + window_size
                window_prices = prices[start:end]
                
                # Find peaks and valleys
                peaks, valleys = self.find_local_extrema(window_prices, order=3)
                
                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Fit lines to peaks and valleys
                    peak_prices = window_prices[peaks]
                    valley_prices = window_prices[valleys]
                    
                    peak_slope, _, peak_r, _, _ = linregress(peaks, peak_prices)
                    valley_slope, _, valley_r, _, _ = linregress(valleys, valley_prices)
                    
                    # Check for converging lines with good fit
                    if abs(peak_r) > 0.8 and abs(valley_r) > 0.8:
                        # Rising wedge: both lines ascending, but converging
                        # Falling wedge: both lines descending, but converging
                        if (not bearish and peak_slope > 0 and valley_slope > 0 and valley_slope > peak_slope) or \
                           (bearish and peak_slope < 0 and valley_slope < 0 and valley_slope < peak_slope):
                            
                            pattern = {
                                'name': 'Rising Wedge' if not bearish else 'Falling Wedge',
                                'start_idx': start,
                                'end_idx': end,
                                'confidence': 0.7 + (abs(peak_r) + abs(valley_r)) * 0.1,
                                'key_points': {
                                    'upper_slope': peak_slope,
                                    'lower_slope': valley_slope,
                                    'peaks': peaks + start,
                                    'valleys': valleys + start
                                }
                            }
                            
                            if timestamps is not None:
                                pattern['start_time'] = timestamps[start]
                                pattern['end_time'] = timestamps[end-1]
                                
                            patterns.append(pattern)
                            
        return patterns
    
    def calculate_pattern_strength(self, pattern: Dict[str, Any], prices: np.ndarray) -> float:
        """Calculate the strength/quality of a detected pattern"""
        # Base strength on confidence
        strength = pattern['confidence']
        
        # Adjust based on pattern completion
        if 'key_points' in pattern:
            # More key points detected = stronger pattern
            num_points = sum(1 for v in pattern['key_points'].values() if v is not None)
            strength *= (0.5 + 0.5 * (num_points / len(pattern['key_points'])))
            
        # Adjust based on pattern size relative to data
        pattern_size = pattern['end_idx'] - pattern['start_idx']
        size_ratio = pattern_size / len(prices)
        
        # Prefer patterns that are not too small or too large
        if 0.05 < size_ratio < 0.3:
            strength *= 1.1
        elif size_ratio < 0.02 or size_ratio > 0.5:
            strength *= 0.8
            
        return min(strength, 1.0)
