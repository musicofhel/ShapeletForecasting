"""
Pattern Backtester Module

Backtests pattern-based predictions and strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PatternBacktester:
    """
    Backtests pattern predictions and evaluates performance.
    """
    
    def __init__(self):
        """Initialize the backtester."""
        self.results = {}
        self.metrics = {}
        logger.info("Initialized PatternBacktester")
    
    def backtest_predictions(self, 
                           data: pd.DataFrame,
                           patterns: List[Dict],
                           predictions: List[Dict]) -> Dict:
        """
        Backtest pattern predictions against actual data.
        
        Args:
            data: Price data DataFrame
            patterns: List of detected patterns
            predictions: List of pattern predictions
            
        Returns:
            Dictionary with backtest results
        """
        results = {
            'predictions': [],
            'accuracy_metrics': {},
            'performance_metrics': {}
        }
        
        # Match predictions with actual outcomes
        for i, pred in enumerate(predictions):
            if i < len(patterns) - 1:
                actual_pattern = patterns[i + 1]
                
                result = {
                    'prediction': pred,
                    'actual': actual_pattern,
                    'correct': pred.get('predicted_pattern') == actual_pattern.get('type'),
                    'confidence': pred.get('confidence', 0.0),
                    'timestamp': pred.get('timestamp')
                }
                
                results['predictions'].append(result)
        
        # Calculate accuracy metrics
        if results['predictions']:
            results['accuracy_metrics'] = self._calculate_accuracy_metrics(
                results['predictions']
            )
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(
            data, patterns, predictions
        )
        
        self.results = results
        return results
    
    def _calculate_accuracy_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate prediction accuracy metrics."""
        metrics = {}
        
        # Overall accuracy
        correct = sum(1 for p in predictions if p['correct'])
        total = len(predictions)
        metrics['overall_accuracy'] = correct / total if total > 0 else 0.0
        
        # Accuracy by pattern type
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        
        for ptype in pattern_types:
            type_preds = [p for p in predictions 
                         if p['prediction'].get('predicted_pattern') == ptype]
            
            if type_preds:
                type_correct = sum(1 for p in type_preds if p['correct'])
                metrics[f'{ptype}_accuracy'] = type_correct / len(type_preds)
                metrics[f'{ptype}_count'] = len(type_preds)
            else:
                metrics[f'{ptype}_accuracy'] = 0.0
                metrics[f'{ptype}_count'] = 0
        
        # Confidence calibration
        confidence_bins = np.linspace(0, 1, 11)
        calibration_data = []
        
        for i in range(len(confidence_bins) - 1):
            bin_preds = [p for p in predictions 
                        if confidence_bins[i] <= p['confidence'] < confidence_bins[i+1]]
            
            if bin_preds:
                bin_accuracy = sum(1 for p in bin_preds if p['correct']) / len(bin_preds)
                bin_confidence = np.mean([p['confidence'] for p in bin_preds])
                
                calibration_data.append({
                    'confidence_range': (confidence_bins[i], confidence_bins[i+1]),
                    'mean_confidence': bin_confidence,
                    'accuracy': bin_accuracy,
                    'count': len(bin_preds)
                })
        
        metrics['calibration_data'] = calibration_data
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = sum(bin_data['count'] for bin_data in calibration_data)
        
        for bin_data in calibration_data:
            weight = bin_data['count'] / total_samples if total_samples > 0 else 0
            ece += weight * abs(bin_data['mean_confidence'] - bin_data['accuracy'])
        
        metrics['expected_calibration_error'] = ece
        
        return metrics
    
    def _calculate_performance_metrics(self,
                                     data: pd.DataFrame,
                                     patterns: List[Dict],
                                     predictions: List[Dict]) -> Dict:
        """Calculate performance metrics for pattern-based trading."""
        metrics = {}
        
        # Simple strategy: trade based on predicted patterns
        trades = []
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = None
        
        for pred in predictions:
            pred_type = pred.get('predicted_pattern')
            timestamp = pred.get('timestamp')
            
            # Find price at prediction time
            if timestamp and 'timestamp' in data.columns:
                price_data = data[data['timestamp'] == timestamp]
                if not price_data.empty:
                    current_price = price_data.iloc[0]['close']
                else:
                    continue
            else:
                continue
            
            # Trading logic
            if position == 0:  # No position
                if pred_type in ['trend_up', 'reversal_bottom']:
                    # Enter long
                    position = 1
                    entry_price = current_price
                    trades.append({
                        'type': 'entry_long',
                        'price': current_price,
                        'timestamp': timestamp,
                        'pattern': pred_type
                    })
                elif pred_type in ['trend_down', 'reversal_top']:
                    # Enter short
                    position = -1
                    entry_price = current_price
                    trades.append({
                        'type': 'entry_short',
                        'price': current_price,
                        'timestamp': timestamp,
                        'pattern': pred_type
                    })
            
            elif position != 0:  # Have position
                # Exit conditions
                exit_signal = False
                
                if position == 1:  # Long position
                    if pred_type in ['trend_down', 'reversal_top']:
                        exit_signal = True
                elif position == -1:  # Short position
                    if pred_type in ['trend_up', 'reversal_bottom']:
                        exit_signal = True
                
                if exit_signal:
                    # Calculate return
                    if position == 1:
                        trade_return = (current_price - entry_price) / entry_price
                    else:  # Short
                        trade_return = (entry_price - current_price) / entry_price
                    
                    trades.append({
                        'type': 'exit',
                        'price': current_price,
                        'timestamp': timestamp,
                        'pattern': pred_type,
                        'return': trade_return
                    })
                    
                    position = 0
                    entry_price = None
        
        # Calculate metrics from trades
        if trades:
            returns = [t['return'] for t in trades if 'return' in t]
            
            if returns:
                metrics['total_trades'] = len(returns)
                metrics['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
                metrics['average_return'] = np.mean(returns)
                metrics['total_return'] = np.sum(returns)
                metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns) * np.sqrt(252) 
                                         if np.std(returns) > 0 else 0.0)
                metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
                
                # Return distribution
                metrics['return_stats'] = {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'skew': self._calculate_skewness(returns),
                    'kurtosis': self._calculate_kurtosis(returns)
                }
            else:
                metrics = self._empty_performance_metrics()
        else:
            metrics = self._empty_performance_metrics()
        
        metrics['trades'] = trades
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of returns."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of returns."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return float(np.mean(((data - mean) / std) ** 4) - 3)
    
    def _empty_performance_metrics(self) -> Dict:
        """Return empty performance metrics structure."""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'average_return': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'return_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'skew': 0.0,
                'kurtosis': 0.0
            }
        }
    
    def generate_report(self) -> str:
        """Generate a text report of backtest results."""
        if not self.results:
            return "No backtest results available."
        
        report = []
        report.append("=" * 60)
        report.append("PATTERN PREDICTION BACKTEST REPORT")
        report.append("=" * 60)
        
        # Accuracy metrics
        acc_metrics = self.results.get('accuracy_metrics', {})
        report.append("\nACCURACY METRICS:")
        report.append(f"Overall Accuracy: {acc_metrics.get('overall_accuracy', 0):.2%}")
        report.append(f"Expected Calibration Error: {acc_metrics.get('expected_calibration_error', 0):.3f}")
        
        report.append("\nAccuracy by Pattern Type:")
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        
        for ptype in pattern_types:
            acc = acc_metrics.get(f'{ptype}_accuracy', 0)
            count = acc_metrics.get(f'{ptype}_count', 0)
            report.append(f"  {ptype}: {acc:.2%} ({count} predictions)")
        
        # Performance metrics
        perf_metrics = self.results.get('performance_metrics', {})
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"Total Trades: {perf_metrics.get('total_trades', 0)}")
        report.append(f"Win Rate: {perf_metrics.get('win_rate', 0):.2%}")
        report.append(f"Average Return: {perf_metrics.get('average_return', 0):.2%}")
        report.append(f"Total Return: {perf_metrics.get('total_return', 0):.2%}")
        report.append(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {perf_metrics.get('max_drawdown', 0):.2%}")
        
        # Return statistics
        return_stats = perf_metrics.get('return_stats', {})
        if return_stats:
            report.append("\nReturn Distribution:")
            report.append(f"  Mean: {return_stats.get('mean', 0):.2%}")
            report.append(f"  Std Dev: {return_stats.get('std', 0):.2%}")
            report.append(f"  Min: {return_stats.get('min', 0):.2%}")
            report.append(f"  Max: {return_stats.get('max', 0):.2%}")
            report.append(f"  Skewness: {return_stats.get('skew', 0):.2f}")
            report.append(f"  Kurtosis: {return_stats.get('kurtosis', 0):.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
