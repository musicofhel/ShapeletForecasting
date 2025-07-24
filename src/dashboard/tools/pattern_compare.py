"""
Enhanced Pattern Comparison Tool for Wavelet Pattern Forecasting Dashboard

This module provides a comprehensive pattern comparison interface that includes:
1. Side-by-side pattern comparison
2. Overlay multiple patterns
3. Statistical comparison metrics
4. Pattern morphing visualization
5. Difference highlighting
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import DTW functionality
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. Using fallback DTW implementation.")

# Import from existing pattern comparison
from ..visualizations.pattern_comparison import PatternComparison as BasePatternComparison


class EnhancedPatternComparison(BasePatternComparison):
    """
    Enhanced pattern comparison with additional features for the tools directory
    """
    
    def __init__(self):
        """Initialize the enhanced pattern comparison interface"""
        super().__init__()
        self.statistical_metrics = {}
        self.morphing_data = {}
        self.difference_highlights = {}
        
    def calculate_statistical_comparison_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive statistical comparison metrics between patterns
        
        Returns:
            Dictionary of statistical metrics for each pattern pair
        """
        metrics = {}
        
        for i, pattern1_id in enumerate(self.selected_patterns):
            for j, pattern2_id in enumerate(self.selected_patterns):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                pair_key = f"{pattern1_id}_vs_{pattern2_id}"
                data1 = self.pattern_data[pattern1_id]['normalized']
                data2 = self.pattern_data[pattern2_id]['normalized']
                
                # Ensure same length for comparison
                min_len = min(len(data1), len(data2))
                data1_aligned = data1[:min_len]
                data2_aligned = data2[:min_len]
                
                # Calculate various statistical tests and metrics
                metrics[pair_key] = {
                    # Basic statistics
                    'mean_difference': float(np.mean(data1_aligned) - np.mean(data2_aligned)),
                    'std_difference': float(np.std(data1_aligned) - np.std(data2_aligned)),
                    'median_difference': float(np.median(data1_aligned) - np.median(data2_aligned)),
                    
                    # Distribution tests
                    'ks_statistic': float(stats.ks_2samp(data1_aligned, data2_aligned).statistic),
                    'ks_pvalue': float(stats.ks_2samp(data1_aligned, data2_aligned).pvalue),
                    
                    # Correlation metrics
                    'pearson_correlation': float(stats.pearsonr(data1_aligned, data2_aligned)[0]),
                    'spearman_correlation': float(stats.spearmanr(data1_aligned, data2_aligned)[0]),
                    'kendall_tau': float(stats.kendalltau(data1_aligned, data2_aligned)[0]),
                    
                    # Distance metrics
                    'rmse': float(np.sqrt(np.mean((data1_aligned - data2_aligned)**2))),
                    'mae': float(np.mean(np.abs(data1_aligned - data2_aligned))),
                    'max_absolute_difference': float(np.max(np.abs(data1_aligned - data2_aligned))),
                    
                    # Shape metrics
                    'cross_correlation_max': float(np.max(np.correlate(data1_aligned, data2_aligned, mode='same'))),
                    'phase_shift': self._calculate_phase_shift(data1_aligned, data2_aligned),
                    
                    # Trend analysis
                    'trend_similarity': self._calculate_trend_similarity(data1_aligned, data2_aligned),
                    'volatility_ratio': float(np.std(data1_aligned) / np.std(data2_aligned)) if np.std(data2_aligned) > 0 else np.inf
                }
                
        self.statistical_metrics = metrics
        return metrics
    
    def _calculate_phase_shift(self, data1: np.ndarray, data2: np.ndarray) -> int:
        """Calculate phase shift between two patterns"""
        correlation = np.correlate(data1, data2, mode='same')
        shift = np.argmax(correlation) - len(data1) // 2
        return int(shift)
    
    def _calculate_trend_similarity(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate trend similarity between two patterns"""
        # Calculate first differences (trends)
        trend1 = np.diff(data1)
        trend2 = np.diff(data2)
        
        # Calculate percentage of matching trend directions
        matching_trends = np.sum(np.sign(trend1) == np.sign(trend2))
        trend_similarity = matching_trends / len(trend1)
        
        return float(trend_similarity)
    
    def create_pattern_morphing_visualization(self, pattern1_id: str, pattern2_id: str, 
                                            n_steps: int = 10) -> go.Figure:
        """
        Create a visualization showing morphing between two patterns
        
        Args:
            pattern1_id: ID of the first pattern
            pattern2_id: ID of the second pattern
            n_steps: Number of morphing steps
            
        Returns:
            Plotly figure showing pattern morphing animation
        """
        if pattern1_id not in self.pattern_data or pattern2_id not in self.pattern_data:
            raise ValueError("Pattern IDs must be in the comparison data")
        
        data1 = self.pattern_data[pattern1_id]['normalized']
        data2 = self.pattern_data[pattern2_id]['normalized']
        
        # Ensure same length
        target_len = max(len(data1), len(data2))
        x_common = np.linspace(0, 1, target_len)
        
        # Interpolate to common length
        f1 = interp1d(np.linspace(0, 1, len(data1)), data1, kind='cubic')
        f2 = interp1d(np.linspace(0, 1, len(data2)), data2, kind='cubic')
        
        data1_interp = f1(x_common)
        data2_interp = f2(x_common)
        
        # Create morphing frames
        frames = []
        morphing_data = []
        
        for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
            morphed = (1 - alpha) * data1_interp + alpha * data2_interp
            morphing_data.append(morphed)
            
            frame = go.Frame(
                data=[go.Scatter(
                    x=x_common,
                    y=morphed,
                    mode='lines',
                    line=dict(width=3, color='cyan'),
                    name='Morphed Pattern'
                )],
                name=str(i)
            )
            frames.append(frame)
        
        # Store morphing data
        self.morphing_data[f"{pattern1_id}_to_{pattern2_id}"] = morphing_data
        
        # Create figure with animation
        fig = go.Figure(
            data=[go.Scatter(
                x=x_common,
                y=data1_interp,
                mode='lines',
                line=dict(width=3, color='cyan'),
                name='Morphed Pattern'
            )],
            frames=frames
        )
        
        # Add reference patterns
        fig.add_trace(go.Scatter(
            x=x_common,
            y=data1_interp,
            mode='lines',
            line=dict(width=2, color='red', dash='dash'),
            name=pattern1_id,
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=x_common,
            y=data2_interp,
            mode='lines',
            line=dict(width=2, color='blue', dash='dash'),
            name=pattern2_id,
            opacity=0.5
        ))
        
        # Add animation controls
        fig.update_layout(
            title=f"Pattern Morphing: {pattern1_id} → {pattern2_id}",
            xaxis_title="Normalized Time",
            yaxis_title="Normalized Value",
            height=600,
            template="plotly_dark",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Step: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 50},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 50, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 50}
                        }],
                        'label': f'{i}/{n_steps-1}',
                        'method': 'animate'
                    }
                    for i, f in enumerate(frames)
                ]
            }]
        )
        
        return fig
    
    def create_difference_highlighting_visualization(self) -> go.Figure:
        """
        Create visualization highlighting differences between patterns
        
        Returns:
            Plotly figure with difference highlighting
        """
        n_patterns = len(self.selected_patterns)
        
        # Create subplots
        fig = make_subplots(
            rows=n_patterns-1, cols=2,
            subplot_titles=[f"{self.selected_patterns[0]} vs {pid}" for pid in self.selected_patterns[1:]] +
                          [f"Difference Map" for _ in self.selected_patterns[1:]],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            column_widths=[0.6, 0.4]
        )
        
        # Reference pattern (first in list)
        ref_pattern_id = self.selected_patterns[0]
        ref_data = self.pattern_data[ref_pattern_id]['normalized']
        
        # Compare with each other pattern
        for i, pattern_id in enumerate(self.selected_patterns[1:], 1):
            data = self.pattern_data[pattern_id]['normalized']
            
            # Ensure same length
            min_len = min(len(ref_data), len(data))
            ref_aligned = ref_data[:min_len]
            data_aligned = data[:min_len]
            x = np.arange(min_len)
            
            # Calculate differences
            differences = data_aligned - ref_aligned
            abs_differences = np.abs(differences)
            
            # Store difference data
            self.difference_highlights[f"{ref_pattern_id}_vs_{pattern_id}"] = {
                'differences': differences,
                'abs_differences': abs_differences,
                'max_diff_idx': np.argmax(abs_differences),
                'max_diff_value': np.max(abs_differences)
            }
            
            # Plot patterns with difference shading
            fig.add_trace(
                go.Scatter(
                    x=x, y=ref_aligned,
                    mode='lines',
                    name=ref_pattern_id,
                    line=dict(width=2, color='blue'),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=data_aligned,
                    mode='lines',
                    name=pattern_id,
                    line=dict(width=2, color='red'),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Add difference shading
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([ref_aligned, data_aligned[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,255,0,0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=i, col=1
            )
            
            # Create difference heatmap
            diff_matrix = np.abs(differences.reshape(-1, 1))
            
            fig.add_trace(
                go.Heatmap(
                    z=diff_matrix.T,
                    x=x,
                    y=['Difference'],
                    colorscale='Hot',
                    showscale=(i == 1),
                    colorbar=dict(title="Abs Diff", x=1.02)
                ),
                row=i, col=2
            )
            
            # Mark maximum difference
            max_idx = self.difference_highlights[f"{ref_pattern_id}_vs_{pattern_id}"]['max_diff_idx']
            fig.add_trace(
                go.Scatter(
                    x=[max_idx],
                    y=[ref_aligned[max_idx]],
                    mode='markers',
                    marker=dict(size=10, color='yellow', symbol='star'),
                    name='Max Difference',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Pattern Difference Highlighting",
            height=300 * (n_patterns - 1),
            template="plotly_dark",
            showlegend=True
        )
        
        # Update axes
        for i in range(1, n_patterns):
            fig.update_xaxes(title_text="Time Steps", row=i, col=1)
            fig.update_yaxes(title_text="Value", row=i, col=1)
            fig.update_xaxes(title_text="Time Steps", row=i, col=2)
            fig.update_yaxes(showticklabels=False, row=i, col=2)
        
        return fig
    
    def create_statistical_comparison_dashboard(self) -> go.Figure:
        """
        Create a comprehensive dashboard showing statistical comparisons
        
        Returns:
            Plotly figure with statistical comparison dashboard
        """
        # Calculate metrics if not already done
        if not self.statistical_metrics:
            self.calculate_statistical_comparison_metrics()
        
        # Prepare data for visualization
        metric_names = ['pearson_correlation', 'rmse', 'ks_statistic', 'trend_similarity']
        n_metrics = len(metric_names)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Correlation Matrix",
                "Distance Metrics",
                "Statistical Tests",
                "Trend Analysis"
            ],
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Correlation heatmap
        pattern_pairs = list(self.statistical_metrics.keys())
        correlations = []
        
        for pair in pattern_pairs:
            correlations.append([
                self.statistical_metrics[pair]['pearson_correlation'],
                self.statistical_metrics[pair]['spearman_correlation'],
                self.statistical_metrics[pair]['kendall_tau']
            ])
        
        fig.add_trace(
            go.Heatmap(
                z=correlations,
                x=['Pearson', 'Spearman', 'Kendall'],
                y=[p.replace('_vs_', ' vs ') for p in pattern_pairs],
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlations, 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=1, col=1
        )
        
        # 2. Distance metrics bar chart
        distances = []
        for pair in pattern_pairs:
            distances.append({
                'pair': pair.replace('_vs_', ' vs '),
                'RMSE': self.statistical_metrics[pair]['rmse'],
                'MAE': self.statistical_metrics[pair]['mae'],
                'Max Diff': self.statistical_metrics[pair]['max_absolute_difference']
            })
        
        df_distances = pd.DataFrame(distances)
        
        for metric in ['RMSE', 'MAE', 'Max Diff']:
            fig.add_trace(
                go.Bar(
                    x=df_distances['pair'],
                    y=df_distances[metric],
                    name=metric
                ),
                row=1, col=2
            )
        
        # 3. Statistical test results
        ks_stats = []
        p_values = []
        pairs_short = []
        
        for pair in pattern_pairs:
            ks_stats.append(self.statistical_metrics[pair]['ks_statistic'])
            p_values.append(self.statistical_metrics[pair]['ks_pvalue'])
            pairs_short.append(pair.split('_vs_')[1])  # Just show second pattern
        
        fig.add_trace(
            go.Scatter(
                x=ks_stats,
                y=p_values,
                mode='markers+text',
                text=pairs_short,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=p_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="p-value", x=0.48, y=0.25)
                ),
                name="KS Test Results"
            ),
            row=2, col=1
        )
        
        # Add significance threshold line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="p=0.05", row=2, col=1)
        
        # 4. Trend similarity
        trend_data = []
        for pair in pattern_pairs:
            trend_data.append({
                'pair': pair.replace('_vs_', ' vs '),
                'trend_similarity': self.statistical_metrics[pair]['trend_similarity'],
                'phase_shift': abs(self.statistical_metrics[pair]['phase_shift'])
            })
        
        df_trends = pd.DataFrame(trend_data)
        
        fig.add_trace(
            go.Bar(
                x=df_trends['pair'],
                y=df_trends['trend_similarity'],
                name='Trend Similarity',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Statistical Pattern Comparison Dashboard",
            height=800,
            template="plotly_dark",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Pattern Pair", row=1, col=2)
        fig.update_yaxes(title_text="Distance", row=1, col=2)
        fig.update_xaxes(title_text="KS Statistic", row=2, col=1)
        fig.update_yaxes(title_text="p-value", row=2, col=1)
        fig.update_xaxes(title_text="Pattern Pair", row=2, col=2)
        fig.update_yaxes(title_text="Similarity", row=2, col=2)
        
        return fig
    
    def export_comparison_results(self, format: str = 'json') -> str:
        """
        Export all comparison results in specified format
        
        Args:
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            Exported data as string
        """
        # Prepare comprehensive results
        results = {
            'metadata': {
                'n_patterns': len(self.selected_patterns),
                'pattern_ids': self.selected_patterns,
                'comparison_date': datetime.now().isoformat()
            },
            'similarity_metrics': self.similarity_metrics,
            'statistical_metrics': self.statistical_metrics,
            'difference_highlights': self.difference_highlights
        }
        
        if format.lower() == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Convert to flat structure for CSV
            rows = []
            for pair_key, metrics in self.statistical_metrics.items():
                row = {'pattern_pair': pair_key}
                row.update(metrics)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        
        elif format.lower() == 'html':
            # Create HTML report
            html = f"""
            <html>
            <head>
                <title>Pattern Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Pattern Comparison Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2>Patterns Compared</h2>
                <ul>
                    {''.join([f'<li>{pid}</li>' for pid in self.selected_patterns])}
                </ul>
                <h2>Statistical Comparison Results</h2>
                <table>
                    <tr>
                        <th>Pattern Pair</th>
                        <th>Pearson Correlation</th>
                        <th>RMSE</th>
                        <th>Trend Similarity</th>
                        <th>KS p-value</th>
                    </tr>
            """
            
            for pair, metrics in self.statistical_metrics.items():
                html += f"""
                    <tr>
                        <td>{pair.replace('_vs_', ' vs ')}</td>
                        <td>{metrics['pearson_correlation']:.3f}</td>
                        <td>{metrics['rmse']:.3f}</td>
                        <td>{metrics['trend_similarity']:.3f}</td>
                        <td>{metrics['ks_pvalue']:.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
        
        else:
            # Default to JSON
            import json
            return json.dumps(results, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    # Create enhanced comparison instance
    comparison = EnhancedPatternComparison()
    
    # Generate demo patterns
    x = np.linspace(0, 4*np.pi, 100)
    patterns = {
        'pattern_A': np.sin(x) + 0.5 * np.sin(3*x),
        'pattern_B': np.sin(x + np.pi/4) + 0.3 * np.sin(3*x),
        'pattern_C': 0.8 * np.sin(x) + 0.4 * np.sin(2*x),
        'pattern_D': np.sin(x) * np.exp(-x/10)
    }
    
    # Add patterns
    for pattern_id, data in patterns.items():
        comparison.add_pattern(pattern_id, data)
    
    # Calculate all metrics
    comparison.calculate_similarity_metrics()
    comparison.calculate_statistical_comparison_metrics()
    
    # Create visualizations
    fig1 = comparison.create_side_by_side_visualization()
    fig1.show()
    
    fig2 = comparison.create_pattern_morphing_visualization('pattern_A', 'pattern_B')
    fig2.show()
    
    fig3 = comparison.create_difference_highlighting_visualization()
    fig3.show()
    
    fig4 = comparison.create_statistical_comparison_dashboard()
    fig4.show()
    
    # Export results
    json_export = comparison.export_comparison_results('json')
    print("Exported results (first 500 chars):")
    print(json_export[:500])
