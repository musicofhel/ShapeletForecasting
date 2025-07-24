"""
Comprehensive Demo: Financial Wavelet Pattern Forecasting
========================================================

This demo showcases the complete forecasting capabilities:
1. Load sample data for multiple tickers
2. Extract and analyze pattern sequences
3. Generate next-pattern predictions
4. Show prediction accuracy over time
5. Demonstrate real-time capabilities
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import json
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.wavelet_analysis.pattern_detector import WaveletPatternDetector
from src.models.pattern_predictor import PatternPredictor
from src.features.pattern_features import PatternFeatureExtractor
from src.evaluation.backtester import PatternBacktester

class WaveletForecastingDemo:
    """Main demo class for wavelet pattern forecasting"""
    
    def __init__(self):
        self.pattern_detector = WaveletPatternDetector()
        self.feature_extractor = PatternFeatureExtractor()
        self.pattern_predictor = PatternPredictor()
        self.backtester = PatternBacktester()
        
        # Demo configuration
        self.tickers = ['BTC-USD', 'ETH-USD', 'SPY', 'AAPL']
        self.pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                             'reversal_bottom', 'consolidation']
        self.colors = {
            'trend_up': '#00ff00',
            'trend_down': '#ff0000',
            'reversal_top': '#ff9900',
            'reversal_bottom': '#0099ff',
            'consolidation': '#9900ff'
        }
        
    def generate_sample_data(self, ticker: str, n_points: int = 2000) -> pd.DataFrame:
        """Generate realistic sample financial data with embedded patterns"""
        print(f"\n📊 Generating sample data for {ticker}...")
        
        # Base price movement
        np.random.seed(hash(ticker) % 1000)
        time_index = pd.date_range(end=datetime.now(), periods=n_points, freq='1h')
        
        # Generate base trend
        trend = np.cumsum(np.random.randn(n_points) * 0.02)
        
        # Add cyclical patterns
        cycles = (
            0.1 * np.sin(2 * np.pi * np.arange(n_points) / 100) +
            0.05 * np.sin(2 * np.pi * np.arange(n_points) / 50) +
            0.03 * np.sin(2 * np.pi * np.arange(n_points) / 25)
        )
        
        # Add noise
        noise = np.random.randn(n_points) * 0.01
        
        # Combine components
        base_price = 100 if ticker in ['SPY', 'AAPL'] else 10000 if ticker == 'BTC-USD' else 1000
        price = base_price * np.exp(trend + cycles + noise)
        
        # Calculate volume (correlated with price changes)
        volume = np.abs(np.diff(price, prepend=price[0])) * np.random.uniform(1e6, 1e7, n_points)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': time_index,
            'open': price * (1 + np.random.randn(n_points) * 0.001),
            'high': price * (1 + np.abs(np.random.randn(n_points)) * 0.002),
            'low': price * (1 - np.abs(np.random.randn(n_points)) * 0.002),
            'close': price,
            'volume': volume,
            'ticker': ticker
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def extract_pattern_sequences(self, df: pd.DataFrame) -> Dict:
        """Extract wavelet patterns and create sequences"""
        print("\n🔍 Extracting wavelet pattern sequences...")
        
        # Perform wavelet transform
        scales = np.arange(1, 65)
        coefficients, frequencies = pywt.cwt(df['close'].values, scales, 'morl')
        
        # Detect patterns
        patterns = []
        window_size = 50
        
        for i in range(window_size, len(df) - window_size, 10):
            window_data = df.iloc[i-window_size:i+window_size]
            window_coeffs = coefficients[:, i-window_size:i+window_size]
            
            # Analyze pattern characteristics
            pattern_type = self._classify_pattern(window_data, window_coeffs)
            
            if pattern_type:
                pattern = {
                    'start_idx': i - window_size,
                    'end_idx': i + window_size,
                    'type': pattern_type,
                    'timestamp': window_data.iloc[window_size]['timestamp'],
                    'strength': self._calculate_pattern_strength(window_coeffs),
                    'features': self._extract_pattern_features(window_data, window_coeffs)
                }
                patterns.append(pattern)
        
        # Create pattern sequences
        sequences = self._create_pattern_sequences(patterns)
        
        return {
            'patterns': patterns,
            'sequences': sequences,
            'coefficients': coefficients,
            'scales': scales
        }
    
    def _classify_pattern(self, window_data: pd.DataFrame, coeffs: np.ndarray) -> Optional[str]:
        """Classify pattern type based on price and wavelet data"""
        prices = window_data['close'].values
        mid_idx = len(prices) // 2
        
        # Calculate price changes
        left_trend = np.polyfit(range(mid_idx), prices[:mid_idx], 1)[0]
        right_trend = np.polyfit(range(mid_idx), prices[mid_idx:], 1)[0]
        overall_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # Analyze wavelet energy
        energy = np.sum(coeffs**2, axis=0)
        energy_change = energy[mid_idx:].mean() - energy[:mid_idx].mean()
        
        # Pattern classification logic
        if overall_trend > 0.1 and energy_change > 0:
            return 'trend_up'
        elif overall_trend < -0.1 and energy_change < 0:
            return 'trend_down'
        elif left_trend > 0.1 and right_trend < -0.1:
            return 'reversal_top'
        elif left_trend < -0.1 and right_trend > 0.1:
            return 'reversal_bottom'
        elif abs(overall_trend) < 0.05:
            return 'consolidation'
        
        return None
    
    def _calculate_pattern_strength(self, coeffs: np.ndarray) -> float:
        """Calculate pattern strength from wavelet coefficients"""
        # Use energy concentration as strength metric
        total_energy = np.sum(coeffs**2)
        dominant_scale_energy = np.max(np.sum(coeffs**2, axis=1))
        return dominant_scale_energy / total_energy if total_energy > 0 else 0
    
    def _extract_pattern_features(self, window_data: pd.DataFrame, coeffs: np.ndarray) -> Dict:
        """Extract features from pattern for prediction"""
        prices = window_data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            'price_change': (prices[-1] - prices[0]) / prices[0],
            'volatility': np.std(returns),
            'volume_ratio': window_data['volume'].iloc[-10:].mean() / window_data['volume'].iloc[:10].mean(),
            'wavelet_energy': np.sum(coeffs**2),
            'dominant_scale': np.argmax(np.sum(coeffs**2, axis=1)),
            'energy_concentration': self._calculate_pattern_strength(coeffs)
        }
        
        return features
    
    def _create_pattern_sequences(self, patterns: List[Dict]) -> List[List[Dict]]:
        """Create sequences of patterns for sequence modeling"""
        sequences = []
        sequence_length = 5
        
        for i in range(sequence_length, len(patterns)):
            sequence = patterns[i-sequence_length:i]
            sequences.append(sequence)
        
        return sequences
    
    def generate_predictions(self, data: Dict[str, Dict]) -> Dict:
        """Generate next-pattern predictions for all tickers"""
        print("\n🔮 Generating pattern predictions...")
        
        predictions = {}
        
        for ticker, ticker_data in data.items():
            sequences = ticker_data['sequences']
            
            if len(sequences) < 10:
                continue
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for i in range(len(sequences) - 1):
                # Extract features from sequence
                seq_features = self._extract_sequence_features(sequences[i])
                X_train.append(seq_features)
                
                # Next pattern type as target
                next_pattern = sequences[i+1][-1]['type']
                y_train.append(next_pattern)
            
            if len(X_train) < 5:
                continue
            
            # Train simple model for demo
            X_train = np.array(X_train)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make prediction for the last sequence
            last_seq_features = self._extract_sequence_features(sequences[-1])
            prediction_proba = model.predict_proba([last_seq_features])[0]
            predicted_pattern = model.predict([last_seq_features])[0]
            
            predictions[ticker] = {
                'predicted_pattern': predicted_pattern,
                'confidence': max(prediction_proba),
                'probabilities': dict(zip(model.classes_, prediction_proba)),
                'last_patterns': [p['type'] for p in sequences[-1]],
                'model': model
            }
        
        return predictions
    
    def _extract_sequence_features(self, sequence: List[Dict]) -> np.ndarray:
        """Extract features from pattern sequence"""
        features = []
        
        # Pattern type distribution
        type_counts = {pt: 0 for pt in self.pattern_types}
        for pattern in sequence:
            type_counts[pattern['type']] += 1
        features.extend(list(type_counts.values()))
        
        # Average pattern strength
        avg_strength = np.mean([p['strength'] for p in sequence])
        features.append(avg_strength)
        
        # Trend features
        price_changes = [p['features']['price_change'] for p in sequence]
        features.extend([
            np.mean(price_changes),
            np.std(price_changes),
            price_changes[-1]
        ])
        
        # Volatility trend
        volatilities = [p['features']['volatility'] for p in sequence]
        features.extend([
            np.mean(volatilities),
            volatilities[-1]
        ])
        
        return np.array(features)
    
    def calculate_accuracy_metrics(self, data: Dict[str, Dict], predictions: Dict) -> Dict:
        """Calculate prediction accuracy over historical data"""
        print("\n📈 Calculating prediction accuracy...")
        
        accuracy_metrics = {}
        
        for ticker in predictions:
            if ticker not in data:
                continue
            
            sequences = data[ticker]['sequences']
            model = predictions[ticker]['model']
            
            # Prepare test data
            X_test = []
            y_true = []
            
            # Use last 20% of sequences for testing
            test_start = int(len(sequences) * 0.8)
            
            for i in range(test_start, len(sequences) - 1):
                seq_features = self._extract_sequence_features(sequences[i])
                X_test.append(seq_features)
                y_true.append(sequences[i+1][-1]['type'])
            
            if len(X_test) < 5:
                continue
            
            # Make predictions
            X_test = np.array(X_test)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=self.pattern_types)
            
            # Calculate per-class metrics
            class_accuracies = {}
            for i, pattern_type in enumerate(self.pattern_types):
                true_positives = conf_matrix[i, i]
                total_actual = conf_matrix[i, :].sum()
                class_accuracies[pattern_type] = true_positives / total_actual if total_actual > 0 else 0
            
            accuracy_metrics[ticker] = {
                'overall_accuracy': accuracy,
                'class_accuracies': class_accuracies,
                'confusion_matrix': conf_matrix.tolist(),
                'confidence_scores': [max(proba) for proba in y_proba],
                'predictions': y_pred.tolist(),
                'actual': y_true
            }
        
        return accuracy_metrics
    
    def create_forecast_visualization(self, ticker: str, data: Dict, 
                                    prediction: Dict, accuracy: Dict) -> go.Figure:
        """Create comprehensive forecast visualization"""
        df = self.sample_data[ticker]
        patterns = data['patterns']
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=[
                f'{ticker} Price with Pattern Detection',
                'Wavelet Scalogram',
                'Pattern Sequence',
                'Prediction Accuracy'
            ],
            vertical_spacing=0.05
        )
        
        # 1. Price chart with patterns
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add pattern overlays
        for pattern in patterns[-20:]:  # Show last 20 patterns
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[start_idx:end_idx]['timestamp'],
                    y=df.iloc[start_idx:end_idx]['close'],
                    mode='lines',
                    line=dict(color=self.colors[pattern['type']], width=3),
                    name=pattern['type'],
                    showlegend=False,
                    opacity=0.6
                ),
                row=1, col=1
            )
        
        # 2. Wavelet scalogram
        coeffs = data['coefficients']
        scales = data['scales']
        
        # Sample coefficients for visualization
        sample_coeffs = coeffs[:, ::10]  # Every 10th point
        sample_times = df['timestamp'].iloc[::10]
        
        fig.add_trace(
            go.Heatmap(
                x=sample_times,
                y=scales,
                z=np.abs(sample_coeffs),
                colorscale='Viridis',
                showscale=False,
                name='Scalogram'
            ),
            row=2, col=1
        )
        
        # 3. Pattern sequence visualization
        sequence_data = []
        for i, seq in enumerate(data['sequences'][-10:]):
            for j, pattern in enumerate(seq):
                sequence_data.append({
                    'sequence': i,
                    'position': j,
                    'type': pattern['type'],
                    'strength': pattern['strength']
                })
        
        if sequence_data:
            seq_df = pd.DataFrame(sequence_data)
            
            for pattern_type in self.pattern_types:
                pattern_data = seq_df[seq_df['type'] == pattern_type]
                if not pattern_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=pattern_data['sequence'],
                            y=pattern_data['position'],
                            mode='markers',
                            marker=dict(
                                size=pattern_data['strength'] * 20,
                                color=self.colors[pattern_type],
                                symbol='circle'
                            ),
                            name=pattern_type,
                            showlegend=True
                        ),
                        row=3, col=1
                    )
        
        # 4. Accuracy visualization
        if ticker in accuracy:
            acc_data = accuracy[ticker]
            
            # Accuracy over time
            window_size = 10
            rolling_accuracy = []
            
            for i in range(window_size, len(acc_data['predictions'])):
                window_acc = accuracy_score(
                    acc_data['actual'][i-window_size:i],
                    acc_data['predictions'][i-window_size:i]
                )
                rolling_accuracy.append(window_acc)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rolling_accuracy))),
                    y=rolling_accuracy,
                    mode='lines+markers',
                    name='Rolling Accuracy',
                    line=dict(color='blue', width=2)
                ),
                row=4, col=1
            )
            
            # Add overall accuracy line
            fig.add_hline(
                y=acc_data['overall_accuracy'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Overall: {acc_data['overall_accuracy']:.2%}",
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Wavelet Pattern Forecasting - {ticker}',
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Scale", row=2, col=1)
        fig.update_xaxes(title_text="Sequence", row=3, col=1)
        fig.update_yaxes(title_text="Position", row=3, col=1)
        fig.update_xaxes(title_text="Window", row=4, col=1)
        fig.update_yaxes(title_text="Accuracy", row=4, col=1)
        
        return fig
    
    def demonstrate_realtime_capabilities(self):
        """Demonstrate real-time pattern detection and prediction"""
        print("\n⚡ Demonstrating real-time capabilities...")
        
        # Simulate real-time data stream
        ticker = 'BTC-USD'
        base_df = self.sample_data[ticker].iloc[:-100].copy()  # Keep last 100 points for streaming
        stream_data = self.sample_data[ticker].iloc[-100:].copy()
        
        # Initialize real-time tracking
        realtime_patterns = []
        realtime_predictions = []
        processing_times = []
        
        print("\n📡 Starting real-time simulation...")
        print("Processing 100 new data points...")
        
        # Create progress visualization
        fig = go.Figure()
        
        for i in range(0, len(stream_data), 5):  # Process 5 points at a time
            start_time = time.time()
            
            # Add new data
            new_data = stream_data.iloc[:i+5]
            current_df = pd.concat([base_df, new_data], ignore_index=True)
            
            # Detect patterns in recent window
            recent_window = current_df.iloc[-200:]
            
            # Quick pattern detection
            if len(recent_window) >= 100:
                # Simplified pattern detection for speed
                prices = recent_window['close'].values
                scales = np.arange(1, 33)
                coeffs, _ = pywt.cwt(prices, scales, 'morl')
                
                # Detect pattern in last window
                pattern_type = self._classify_pattern(recent_window.iloc[-100:], coeffs[:, -100:])
                
                if pattern_type:
                    pattern = {
                        'timestamp': recent_window.iloc[-1]['timestamp'],
                        'type': pattern_type,
                        'price': recent_window.iloc[-1]['close'],
                        'confidence': np.random.uniform(0.7, 0.95)  # Simulated confidence
                    }
                    realtime_patterns.append(pattern)
                    
                    # Make prediction
                    if len(realtime_patterns) >= 5:
                        # Simple prediction based on pattern history
                        recent_types = [p['type'] for p in realtime_patterns[-5:]]
                        
                        # Transition probabilities (simplified)
                        next_pattern = self._predict_next_pattern(recent_types)
                        
                        prediction = {
                            'timestamp': pattern['timestamp'],
                            'predicted': next_pattern,
                            'confidence': np.random.uniform(0.6, 0.9)
                        }
                        realtime_predictions.append(prediction)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            processing_times.append(processing_time)
            
            # Update progress
            if i % 20 == 0:
                print(f"  Processed {i+5}/{len(stream_data)} points | "
                      f"Avg latency: {np.mean(processing_times):.1f}ms | "
                      f"Patterns detected: {len(realtime_patterns)}")
        
        # Create real-time performance visualization
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=[
                'Real-time Pattern Detection',
                'Processing Latency',
                'Prediction Confidence'
            ]
        )
        
        # 1. Price with real-time patterns
        fig.add_trace(
            go.Scatter(
                x=current_df['timestamp'],
                y=current_df['close'],
                mode='lines',
                name='Price',
                line=dict(color='white', width=1)
            ),
            row=1, col=1
        )
        
        # Add detected patterns
        for pattern in realtime_patterns:
            fig.add_annotation(
                x=pattern['timestamp'],
                y=pattern['price'],
                text=pattern['type'][:3].upper(),
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.colors[pattern['type']],
                bgcolor=self.colors[pattern['type']],
                opacity=0.8,
                row=1, col=1
            )
        
        # 2. Processing latency
        fig.add_trace(
            go.Scatter(
                x=list(range(len(processing_times))),
                y=processing_times,
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='cyan', width=2)
            ),
            row=2, col=1
        )
        
        # Add target latency line
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 100ms",
            row=2, col=1
        )
        
        # 3. Prediction confidence over time
        if realtime_predictions:
            fig.add_trace(
                go.Scatter(
                    x=[p['timestamp'] for p in realtime_predictions],
                    y=[p['confidence'] for p in realtime_predictions],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='green', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Real-time Pattern Detection Performance',
            height=900,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Performance summary
        print("\n📊 Real-time Performance Summary:")
        print(f"  • Average latency: {np.mean(processing_times):.1f}ms")
        print(f"  • Max latency: {np.max(processing_times):.1f}ms")
        print(f"  • Patterns detected: {len(realtime_patterns)}")
        print(f"  • Predictions made: {len(realtime_predictions)}")
        print(f"  • Detection rate: {len(realtime_patterns)/len(stream_data)*100:.1f}%")
        
        return fig, {
            'patterns': realtime_patterns,
            'predictions': realtime_predictions,
            'latencies': processing_times
        }
    
    def _predict_next_pattern(self, recent_patterns: List[str]) -> str:
        """Simple pattern prediction based on transitions"""
        # Simplified transition matrix
        transitions = {
            'trend_up': {'trend_up': 0.6, 'reversal_top': 0.3, 'consolidation': 0.1},
            'trend_down': {'trend_down': 0.6, 'reversal_bottom': 0.3, 'consolidation': 0.1},
            'reversal_top': {'trend_down': 0.7, 'consolidation': 0.3},
            'reversal_bottom': {'trend_up': 0.7, 'consolidation': 0.3},
            'consolidation': {'trend_up': 0.3, 'trend_down': 0.3, 'consolidation': 0.4}
        }
        
        last_pattern = recent_patterns[-1]
        probs = transitions.get(last_pattern, {'consolidation': 1.0})
        
        # Sample from distribution
        patterns = list(probs.keys())
        probabilities = list(probs.values())
        
        return np.random.choice(patterns, p=probabilities)
    
    def save_demo_results(self, results: Dict):
        """Save demo results and visualizations"""
        print("\n💾 Saving demo results...")
        
        # Create results directory
        os.makedirs('demo_results', exist_ok=True)
        
        # Save metrics
        with open('demo_results/forecast_metrics.json', 'w') as f:
            json.dump({
                'accuracy_metrics': results['accuracy_metrics'],
                'predictions': {
                    ticker: {
                        'predicted_pattern': pred['predicted_pattern'],
                        'confidence': float(pred['confidence']),
                        'probabilities': pred['probabilities']
                    }
                    for ticker, pred in results['predictions'].items()
                },
                'realtime_performance': {
                    'avg_latency_ms': float(np.mean(results['realtime_stats']['latencies'])),
                    'max_latency_ms': float(np.max(results['realtime_stats']['latencies'])),
                    'patterns_detected': len(results['realtime_stats']['patterns']),
                    'predictions_made': len(results['realtime_stats']['predictions'])
                }
            }, f, indent=2)
        
        # Save visualizations
        for ticker, fig in results['visualizations'].items():
            fig.write_html(f'demo_results/forecast_{ticker}.html')
        
        results['realtime_fig'].write_html('demo_results/realtime_performance.html')
        
        print("✅ Results saved to demo_results/")
    
    def run_demo(self):
        """Run the complete forecasting demo"""
        print("=" * 80)
        print("🚀 WAVELET PATTERN FORECASTING DEMO")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. Generate sample data
        print("\n📊 Step 1: Loading sample data for multiple tickers")
        self.sample_data = {}
        for ticker in self.tickers:
            self.sample_data[ticker] = self.generate_sample_data(ticker)
        
        # 2. Extract pattern sequences
        print("\n🔍 Step 2: Extracting and analyzing pattern sequences")
        pattern_data = {}
        for ticker in self.tickers:
            pattern_data[ticker] = self.extract_pattern_sequences(self.sample_data[ticker])
            print(f"  • {ticker}: Found {len(pattern_data[ticker]['patterns'])} patterns")
        
        # 3. Generate predictions
        print("\n🔮 Step 3: Generating next-pattern predictions")
        predictions = self.generate_predictions(pattern_data)
        
        for ticker, pred in predictions.items():
            print(f"\n  {ticker} Prediction:")
            print(f"    • Next pattern: {pred['predicted_pattern']}")
            print(f"    • Confidence: {pred['confidence']:.2%}")
            print(f"    • Pattern sequence: {' → '.join(pred['last_patterns'][-3:])} → ?")
        
        # 4. Calculate accuracy
        print("\n📈 Step 4: Calculating prediction accuracy over time")
        accuracy_metrics = self.calculate_accuracy_metrics(pattern_data, predictions)
        
        for ticker, metrics in accuracy_metrics.items():
            print(f"\n  {ticker} Accuracy:")
            print(f"    • Overall: {metrics['overall_accuracy']:.2%}")
            print(f"    • By pattern type:")
            for pattern_type, acc in metrics['class_accuracies'].items():
                print(f"      - {pattern_type}: {acc:.2%}")
        
        # 5. Create visualizations
        print("\n📊 Step 5: Creating forecast visualizations")
        visualizations = {}
        for ticker in self.tickers[:2]:  # Create detailed viz for first 2 tickers
            if ticker in predictions and ticker in accuracy_metrics:
                viz = self.create_forecast_visualization(
                    ticker, pattern_data[ticker], 
                    predictions[ticker], accuracy_metrics
                )
                visualizations[ticker] = viz
                viz.write_html(f'forecast_demo_{ticker}.html')
                print(f"  • Created visualization for {ticker}")
        
        # 6. Demonstrate real-time capabilities
        print("\n⚡ Step 6: Demonstrating real-time capabilities")
        realtime_fig, realtime_stats = self.demonstrate_realtime_capabilities()
        realtime_fig.write_html('realtime_demo.html')
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'pattern_data': pattern_data,
            'predictions': predictions,
            'accuracy_metrics': accuracy_metrics,
            'visualizations': visualizations,
            'realtime_fig': realtime_fig,
            'realtime_stats': realtime_stats
        }
        
        # Save results
        self.save_demo_results(results)
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE!")
        print("=" * 80)
        print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
        print("\n📊 Summary:")
        print(f"  • Analyzed {len(self.tickers)} tickers")
        print(f"  • Detected {sum(len(pd[ticker]['patterns']) for pd in [pattern_data])} total patterns")
        print(f"  • Generated predictions with {np.mean([pred['confidence'] for pred in predictions.values()]):.1%} avg confidence")
        print(f"  • Achieved {np.mean([metrics['overall_accuracy'] for metrics in accuracy_metrics.values()]):.1%} avg accuracy")
        print(f"  • Real-time latency: {np.mean(realtime_stats['latencies']):.1f}ms average")
        
        print("\n📁 Output files:")
        print("  • forecast_demo_BTC-USD.html - Bitcoin forecast visualization")
        print("  • forecast_demo_ETH-USD.html - Ethereum forecast visualization")
        print("  • realtime_demo.html - Real-time performance demo")
        print("  • demo_results/ - All metrics and results")
        
        return results


if __name__ == "__main__":
    # Run the demo
    demo = WaveletForecastingDemo()
    results = demo.run_demo()
