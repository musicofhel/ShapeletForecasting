"""
Pattern Analytics Module

Provides comprehensive analytics for pattern-based trading including:
- Pattern frequency analysis
- Quality distribution metrics
- Prediction accuracy tracking
- Trading signal generation
- Risk metrics calculation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

warnings.filterwarnings('ignore')


class PatternAnalytics:
    """Comprehensive pattern analytics and visualization"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.pattern_colors = {
            'double_top': '#e74c3c',
            'double_bottom': '#2ecc71',
            'head_shoulders': '#3498db',
            'inverse_head_shoulders': '#9b59b6',
            'triangle': '#f39c12',
            'wedge': '#1abc9c',
            'flag': '#34495e',
            'pennant': '#e67e22'
        }
    
    def analyze_pattern_frequency(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern frequency by ticker and type"""
        
        # Frequency by ticker
        ticker_freq = patterns.groupby('ticker').size().to_dict()
        
        # Frequency by pattern type
        type_freq = patterns.groupby('pattern_type').size().to_dict()
        
        # Frequency by ticker and type
        ticker_type_freq = patterns.groupby(['ticker', 'pattern_type']).size().unstack(fill_value=0)
        
        # Time-based frequency (daily, weekly, monthly)
        patterns['date'] = pd.to_datetime(patterns['timestamp'])
        daily_freq = patterns.groupby(patterns['date'].dt.date).size()
        weekly_freq = patterns.groupby(patterns['date'].dt.to_period('W')).size()
        monthly_freq = patterns.groupby(patterns['date'].dt.to_period('M')).size()
        
        return {
            'ticker_frequency': ticker_freq,
            'type_frequency': type_freq,
            'ticker_type_matrix': ticker_type_freq,
            'daily_frequency': daily_freq,
            'weekly_frequency': weekly_freq,
            'monthly_frequency': monthly_freq
        }
    
    def analyze_pattern_quality(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern quality distribution"""
        
        # Quality score statistics
        quality_stats = {
            'mean': patterns['quality_score'].mean(),
            'std': patterns['quality_score'].std(),
            'median': patterns['quality_score'].median(),
            'q1': patterns['quality_score'].quantile(0.25),
            'q3': patterns['quality_score'].quantile(0.75)
        }
        
        # Quality by pattern type
        quality_by_type = patterns.groupby('pattern_type')['quality_score'].agg(['mean', 'std', 'count'])
        
        # Quality by ticker
        quality_by_ticker = patterns.groupby('ticker')['quality_score'].agg(['mean', 'std', 'count'])
        
        # Quality distribution
        quality_bins = pd.cut(patterns['quality_score'], bins=10)
        quality_dist = quality_bins.value_counts().sort_index()
        
        return {
            'statistics': quality_stats,
            'by_type': quality_by_type,
            'by_ticker': quality_by_ticker,
            'distribution': quality_dist
        }
    
    def analyze_prediction_accuracy(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction accuracy metrics"""
        
        # Filter patterns with predictions
        predicted = patterns[patterns['prediction'].notna()].copy()
        
        if len(predicted) == 0:
            return {'error': 'No predictions available'}
        
        # Calculate accuracy metrics
        predicted['actual_direction'] = predicted['actual_return'].apply(lambda x: 1 if x > 0 else 0)
        predicted['predicted_direction'] = predicted['prediction'].apply(lambda x: 1 if x > 0 else 0)
        
        # Overall accuracy
        accuracy = (predicted['actual_direction'] == predicted['predicted_direction']).mean()
        
        # Accuracy by pattern type
        accuracy_by_type = predicted.groupby('pattern_type').apply(
            lambda x: (x['actual_direction'] == x['predicted_direction']).mean()
        )
        
        # Accuracy by quality score bins
        predicted['quality_bin'] = pd.cut(predicted['quality_score'], bins=5)
        accuracy_by_quality = predicted.groupby('quality_bin').apply(
            lambda x: (x['actual_direction'] == x['predicted_direction']).mean()
        )
        
        # Confusion matrix
        cm = confusion_matrix(predicted['actual_direction'], predicted['predicted_direction'])
        
        # Precision, recall, F1
        report = classification_report(
            predicted['actual_direction'], 
            predicted['predicted_direction'],
            output_dict=True
        )
        
        return {
            'overall_accuracy': accuracy,
            'by_pattern_type': accuracy_by_type,
            'by_quality': accuracy_by_quality,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def generate_trading_signals(self, patterns: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern-based trading signals"""
        
        signals = []
        
        for _, pattern in patterns.iterrows():
            # Base signal strength on quality score and pattern type
            base_strength = pattern['quality_score']
            
            # Adjust for pattern type
            pattern_multipliers = {
                'double_top': -1.2,  # Bearish
                'double_bottom': 1.2,  # Bullish
                'head_shoulders': -1.1,  # Bearish
                'inverse_head_shoulders': 1.1,  # Bullish
                'triangle': 0.8,  # Neutral, depends on breakout
                'wedge': 0.9,
                'flag': 1.0,
                'pennant': 0.9
            }
            
            multiplier = pattern_multipliers.get(pattern['pattern_type'], 1.0)
            signal_strength = base_strength * multiplier
            
            # Determine signal type
            if signal_strength > 0.7:
                signal_type = 'STRONG_BUY'
            elif signal_strength > 0.3:
                signal_type = 'BUY'
            elif signal_strength < -0.7:
                signal_type = 'STRONG_SELL'
            elif signal_strength < -0.3:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            # Calculate stop loss and take profit
            volatility = pattern.get('volatility', 0.02)
            if 'BUY' in signal_type:
                stop_loss = pattern['current_price'] * (1 - 2 * volatility)
                take_profit = pattern['current_price'] * (1 + 3 * volatility)
            else:
                stop_loss = pattern['current_price'] * (1 + 2 * volatility)
                take_profit = pattern['current_price'] * (1 - 3 * volatility)
            
            signals.append({
                'timestamp': pattern['timestamp'],
                'ticker': pattern['ticker'],
                'pattern_type': pattern['pattern_type'],
                'signal_type': signal_type,
                'signal_strength': abs(signal_strength),
                'entry_price': pattern['current_price'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': abs((take_profit - pattern['current_price']) / 
                                       (pattern['current_price'] - stop_loss))
            })
        
        return pd.DataFrame(signals)
    
    def calculate_risk_metrics(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for each pattern type"""
        
        risk_metrics = {}
        
        for pattern_type in patterns['pattern_type'].unique():
            type_patterns = patterns[patterns['pattern_type'] == pattern_type]
            
            if 'actual_return' in type_patterns.columns:
                returns = type_patterns['actual_return'].dropna()
                
                if len(returns) > 0:
                    # Calculate various risk metrics
                    metrics = {
                        'avg_return': returns.mean(),
                        'volatility': returns.std(),
                        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                        'max_drawdown': (returns.cumsum().cummax() - returns.cumsum()).max(),
                        'win_rate': (returns > 0).mean(),
                        'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
                        'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
                        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) 
                                       if returns[returns < 0].sum() != 0 else np.inf,
                        'var_95': np.percentile(returns, 5),
                        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
                    }
                else:
                    metrics = self._empty_risk_metrics()
            else:
                metrics = self._empty_risk_metrics()
            
            risk_metrics[pattern_type] = metrics
        
        return risk_metrics
    
    def _empty_risk_metrics(self) -> Dict[str, float]:
        """Return empty risk metrics structure"""
        return {
            'avg_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'var_95': 0,
            'cvar_95': 0
        }
    
    def create_frequency_timeline(self, patterns: pd.DataFrame) -> go.Figure:
        """Create pattern occurrence timeline visualization"""
        
        patterns['date'] = pd.to_datetime(patterns['timestamp'])
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Pattern Occurrences Over Time', 'Cumulative Pattern Count'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Plot pattern occurrences
        for pattern_type in patterns['pattern_type'].unique():
            type_data = patterns[patterns['pattern_type'] == pattern_type]
            
            # Daily counts
            daily_counts = type_data.groupby(type_data['date'].dt.date).size()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    name=pattern_type,
                    mode='lines+markers',
                    line=dict(color=self.pattern_colors.get(pattern_type, '#666')),
                    stackgroup='one'
                ),
                row=1, col=1
            )
        
        # Cumulative count
        cumulative = patterns.groupby(patterns['date'].dt.date).size().cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                name='Cumulative',
                mode='lines',
                line=dict(color=self.color_scheme['primary'], width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Pattern Occurrence Timeline',
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Cumulative Count', row=2, col=1)
        
        return fig
    
    def create_quality_trends(self, patterns: pd.DataFrame) -> go.Figure:
        """Create quality score trends visualization"""
        
        patterns['date'] = pd.to_datetime(patterns['timestamp'])
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Quality Score Trends by Pattern Type',
                'Quality Score Distribution',
                'Quality Score Heatmap',
                'Quality Score vs Prediction Accuracy'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # 1. Quality trends by pattern type
        for pattern_type in patterns['pattern_type'].unique():
            type_data = patterns[patterns['pattern_type'] == pattern_type]
            daily_quality = type_data.groupby(type_data['date'].dt.date)['quality_score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_quality.index,
                    y=daily_quality.values,
                    name=pattern_type,
                    mode='lines+markers',
                    line=dict(color=self.pattern_colors.get(pattern_type, '#666'))
                ),
                row=1, col=1
            )
        
        # 2. Quality score distribution
        fig.add_trace(
            go.Histogram(
                x=patterns['quality_score'],
                nbinsx=30,
                name='Quality Distribution',
                marker_color=self.color_scheme['primary'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Quality score heatmap (ticker vs pattern type)
        pivot_quality = patterns.pivot_table(
            values='quality_score',
            index='ticker',
            columns='pattern_type',
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_quality.values,
                x=pivot_quality.columns,
                y=pivot_quality.index,
                colorscale='RdYlGn',
                showscale=True,
                text=np.round(pivot_quality.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=1
        )
        
        # 4. Quality vs Accuracy (if predictions available)
        if 'prediction' in patterns.columns and 'actual_return' in patterns.columns:
            predicted = patterns[patterns['prediction'].notna()].copy()
            predicted['accuracy'] = (predicted['prediction'] * predicted['actual_return'] > 0).astype(int)
            
            # Bin quality scores
            predicted['quality_bin'] = pd.cut(predicted['quality_score'], bins=10)
            accuracy_by_quality = predicted.groupby('quality_bin')['accuracy'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=[interval.mid for interval in accuracy_by_quality.index],
                    y=accuracy_by_quality.values,
                    mode='lines+markers',
                    name='Accuracy vs Quality',
                    line=dict(color=self.color_scheme['success'], width=3),
                    marker=dict(size=10),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Pattern Quality Analysis',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Quality Score', row=1, col=2)
        fig.update_xaxes(title_text='Pattern Type', row=2, col=1)
        fig.update_xaxes(title_text='Quality Score', row=2, col=2)
        
        fig.update_yaxes(title_text='Avg Quality Score', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='Ticker', row=2, col=1)
        fig.update_yaxes(title_text='Prediction Accuracy', row=2, col=2)
        
        return fig
    
    def create_prediction_performance(self, patterns: pd.DataFrame) -> go.Figure:
        """Create prediction performance visualization"""
        
        # Filter patterns with predictions
        predicted = patterns[patterns['prediction'].notna()].copy()
        
        if len(predicted) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No prediction data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Calculate metrics
        predicted['actual_direction'] = predicted['actual_return'].apply(lambda x: 1 if x > 0 else -1)
        predicted['predicted_direction'] = predicted['prediction'].apply(lambda x: 1 if x > 0 else -1)
        predicted['correct'] = predicted['actual_direction'] == predicted['predicted_direction']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Prediction Accuracy by Pattern Type',
                'Confusion Matrix',
                'Prediction vs Actual Returns',
                'Cumulative Performance'
            ),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Accuracy by pattern type
        accuracy_by_type = predicted.groupby('pattern_type')['correct'].agg(['mean', 'count'])
        
        fig.add_trace(
            go.Bar(
                x=accuracy_by_type.index,
                y=accuracy_by_type['mean'],
                name='Accuracy',
                marker_color=[self.pattern_colors.get(pt, '#666') for pt in accuracy_by_type.index],
                text=[f"{acc:.1%}<br>n={n}" for acc, n in zip(accuracy_by_type['mean'], accuracy_by_type['count'])],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Confusion matrix
        cm = confusion_matrix(predicted['actual_direction'], predicted['predicted_direction'])
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Sell', 'Buy'],
                y=['Sell', 'Buy'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=2
        )
        
        # 3. Prediction vs Actual scatter
        fig.add_trace(
            go.Scatter(
                x=predicted['prediction'],
                y=predicted['actual_return'],
                mode='markers',
                marker=dict(
                    color=predicted['correct'].map({True: self.color_scheme['success'], 
                                                   False: self.color_scheme['danger']}),
                    size=8,
                    opacity=0.6
                ),
                name='Predictions',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add diagonal line
        min_val = min(predicted['prediction'].min(), predicted['actual_return'].min())
        max_val = max(predicted['prediction'].max(), predicted['actual_return'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect Prediction',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Cumulative performance
        predicted_sorted = predicted.sort_values('timestamp')
        predicted_sorted['cumulative_return'] = (1 + predicted_sorted['actual_return']).cumprod() - 1
        predicted_sorted['strategy_return'] = predicted_sorted.apply(
            lambda x: x['actual_return'] if x['correct'] else -x['actual_return'], axis=1
        )
        predicted_sorted['cumulative_strategy'] = (1 + predicted_sorted['strategy_return']).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(
                x=predicted_sorted['timestamp'],
                y=predicted_sorted['cumulative_return'],
                name='Buy & Hold',
                line=dict(color=self.color_scheme['info'], width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=predicted_sorted['timestamp'],
                y=predicted_sorted['cumulative_strategy'],
                name='Pattern Strategy',
                line=dict(color=self.color_scheme['success'], width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Prediction Performance Analysis',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Pattern Type', row=1, col=1)
        fig.update_xaxes(title_text='Predicted', row=1, col=2)
        fig.update_xaxes(title_text='Predicted Return', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=2)
        
        fig.update_yaxes(title_text='Accuracy', row=1, col=1)
        fig.update_yaxes(title_text='Actual', row=1, col=2)
        fig.update_yaxes(title_text='Actual Return', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative Return', row=2, col=2)
        
        return fig
    
    def create_pattern_correlation_matrix(self, patterns: pd.DataFrame) -> go.Figure:
        """Create pattern correlation matrix visualization"""
        
        # Create pattern occurrence matrix
        patterns['date'] = pd.to_datetime(patterns['timestamp'])
        patterns['date_hour'] = patterns['date'].dt.floor('H')
        
        # Count patterns by type and time
        pattern_counts = patterns.pivot_table(
            index='date_hour',
            columns='pattern_type',
            values='pattern_id',
            aggfunc='count',
            fill_value=0
        )
        
        # Calculate correlation matrix
        corr_matrix = pattern_counts.corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Correlation')
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Pattern Type Correlation Matrix',
            height=600,
            width=800,
            template='plotly_white',
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def create_risk_metrics_dashboard(self, patterns: pd.DataFrame) -> go.Figure:
        """Create comprehensive risk metrics dashboard"""
        
        risk_metrics = self.calculate_risk_metrics(patterns)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk-Return Profile',
                'Win Rate by Pattern Type',
                'Profit Factor Comparison',
                'Value at Risk (95%)'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Prepare data
        pattern_types = list(risk_metrics.keys())
        returns = [risk_metrics[pt]['avg_return'] for pt in pattern_types]
        volatilities = [risk_metrics[pt]['volatility'] for pt in pattern_types]
        win_rates = [risk_metrics[pt]['win_rate'] for pt in pattern_types]
        profit_factors = [risk_metrics[pt]['profit_factor'] for pt in pattern_types]
        var_95s = [risk_metrics[pt]['var_95'] for pt in pattern_types]
        
        # 1. Risk-Return scatter
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=pattern_types,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=[self.pattern_colors.get(pt, '#666') for pt in pattern_types]
                ),
                name='Patterns',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Win rate bar chart
        fig.add_trace(
            go.Bar(
                x=pattern_types,
                y=win_rates,
                marker_color=[self.pattern_colors.get(pt, '#666') for pt in pattern_types],
                text=[f"{wr:.1%}" for wr in win_rates],
                textposition='outside',
                name='Win Rate',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Profit factor
        fig.add_trace(
            go.Bar(
                x=pattern_types,
                y=profit_factors,
                marker_color=[self.pattern_colors.get(pt, '#666') for pt in pattern_types],
                text=[f"{pf:.2f}" for pf in profit_factors],
                textposition='outside',
                name='Profit Factor',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Value at Risk
        fig.add_trace(
            go.Bar(
                x=pattern_types,
                y=var_95s,
                marker_color=[self.pattern_colors.get(pt, '#666') for pt in pattern_types],
                text=[f"{var:.1%}" for var in var_95s],
                textposition='outside',
                name='VaR 95%',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Risk Metrics Dashboard',
            height=900,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Volatility', row=1, col=1)
        fig.update_xaxes(title_text='Pattern Type', row=1, col=2)
        fig.update_xaxes(title_text='Pattern Type', row=2, col=1)
        fig.update_xaxes(title_text='Pattern Type', row=2, col=2)
        
        fig.update_yaxes(title_text='Average Return', row=1, col=1)
        fig.update_yaxes(title_text='Win Rate', row=1, col=2)
        fig.update_yaxes(title_text='Profit Factor', row=2, col=1)
        fig.update_yaxes(title_text='VaR (95%)', row=2, col=2)
        
        return fig
    
    def create_signal_dashboard(self, patterns: pd.DataFrame) -> go.Figure:
        """Create trading signals dashboard"""
        
        signals = self.generate_trading_signals(patterns)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Signal Distribution',
                'Signals by Pattern Type',
                'Risk-Reward Ratios',
                'Signal Timeline'
            ),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # 1. Signal distribution pie chart
        signal_counts = signals['signal_type'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                marker=dict(colors=[
                    self.color_scheme['success'] if 'BUY' in s else self.color_scheme['danger'] 
                    if 'SELL' in s else self.color_scheme['warning']
                    for s in signal_counts.index
                ]),
                hole=0.3
            ),
            row=1, col=1
        )
        
        # 2. Signals by pattern type
        signals_by_type = signals.groupby(['pattern_type', 'signal_type']).size().unstack(fill_value=0)
        
        for signal_type in signals_by_type.columns:
            color = self.color_scheme['success'] if 'BUY' in signal_type else \
                   self.color_scheme['danger'] if 'SELL' in signal_type else \
                   self.color_scheme['warning']
            
            fig.add_trace(
                go.Bar(
                    x=signals_by_type.index,
                    y=signals_by_type[signal_type],
                    name=signal_type,
                    marker_color=color
                ),
                row=1, col=2
            )
        
        # 3. Risk-reward ratio histogram
        fig.add_trace(
            go.Histogram(
                x=signals['risk_reward_ratio'],
                nbinsx=20,
                name='Risk-Reward Distribution',
                marker_color=self.color_scheme['info'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Signal timeline
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        
        for signal_type in signals['signal_type'].unique():
            type_signals = signals[signals['signal_type'] == signal_type]
            
            color = self.color_scheme['success'] if 'BUY' in signal_type else \
                   self.color_scheme['danger'] if 'SELL' in signal_type else \
                   self.color_scheme['warning']
            
            fig.add_trace(
                go.Scatter(
                    x=type_signals['timestamp'],
                    y=type_signals['signal_strength'],
                    mode='markers',
                    name=signal_type,
                    marker=dict(
                        color=color,
                        size=10,
                        symbol='diamond' if 'STRONG' in signal_type else 'circle'
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Trading Signals Dashboard',
            height=900,
            showlegend=True,
            template='plotly_white',
            barmode='stack'
        )
        
        fig.update_xaxes(title_text='Pattern Type', row=1, col=2)
        fig.update_xaxes(title_text='Risk-Reward Ratio', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=2)
        
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=2, col=1)
        fig.update_yaxes(title_text='Signal Strength', row=2, col=2)
        
        return fig
    
    def create_comprehensive_dashboard(self, patterns: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create all analytics visualizations"""
        
        dashboard = {}
        
        # Generate all visualizations
        dashboard['frequency_timeline'] = self.create_frequency_timeline(patterns)
        dashboard['quality_trends'] = self.create_quality_trends(patterns)
        dashboard['prediction_performance'] = self.create_prediction_performance(patterns)
        dashboard['correlation_matrix'] = self.create_pattern_correlation_matrix(patterns)
        dashboard['risk_metrics'] = self.create_risk_metrics_dashboard(patterns)
        dashboard['signal_dashboard'] = self.create_signal_dashboard(patterns)
        
        return dashboard
    
    def generate_analytics_report(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        report = {
            'summary': self._generate_summary(patterns),
            'frequency_analysis': self.analyze_pattern_frequency(patterns),
            'quality_analysis': self.analyze_pattern_quality(patterns),
            'prediction_analysis': self.analyze_prediction_accuracy(patterns),
            'risk_analysis': self.calculate_risk_metrics(patterns),
            'signals': self.generate_trading_signals(patterns).to_dict('records')
        }
        
        return report
    
    def _generate_summary(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Generate executive summary of pattern analytics"""
        
        # Basic statistics
        total_patterns = len(patterns)
        unique_tickers = patterns['ticker'].nunique()
        pattern_types = patterns['pattern_type'].nunique()
        
        # Time range
        patterns['timestamp'] = pd.to_datetime(patterns['timestamp'])
        date_range = {
            'start': patterns['timestamp'].min().strftime('%Y-%m-%d'),
            'end': patterns['timestamp'].max().strftime('%Y-%m-%d'),
            'days': (patterns['timestamp'].max() - patterns['timestamp'].min()).days
        }
        
        # Quality metrics
        quality_metrics = {
            'avg_quality': patterns['quality_score'].mean(),
            'high_quality_patterns': len(patterns[patterns['quality_score'] > 0.7]),
            'low_quality_patterns': len(patterns[patterns['quality_score'] < 0.3])
        }
        
        # Performance metrics (if available)
        performance_metrics = {}
        if 'prediction' in patterns.columns and 'actual_return' in patterns.columns:
            predicted = patterns[patterns['prediction'].notna()]
            if len(predicted) > 0:
                predicted['correct'] = (predicted['prediction'] * predicted['actual_return'] > 0)
                performance_metrics = {
                    'predictions_made': len(predicted),
                    'accuracy': predicted['correct'].mean(),
                    'avg_return': predicted['actual_return'].mean(),
                    'total_return': (1 + predicted['actual_return']).prod() - 1
                }
        
        return {
            'total_patterns': total_patterns,
            'unique_tickers': unique_tickers,
            'pattern_types': pattern_types,
            'date_range': date_range,
            'quality_metrics': quality_metrics,
            'performance_metrics': performance_metrics
        }




if __name__ == "__main__":
    # Initialize analytics
    analytics = PatternAnalytics()
    
    print("Pattern Analytics module loaded successfully")
    print("Use analytics.generate_analytics_report(patterns_df) to generate reports")
    print("Use analytics.create_comprehensive_dashboard(patterns_df) to create visualizations")
