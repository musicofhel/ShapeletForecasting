"""
Pattern Comparison Interface

This module provides a comprehensive pattern comparison interface that allows:
1. Selection of multiple patterns for comparison
2. Side-by-side visualizations of patterns
3. Calculation of similarity metrics (DTW, correlation, etc.)
4. Display of correlation heatmaps
5. Identification of pattern evolution over time
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import DTW functionality from pyts
from pyts.metrics import dtw

class PatternComparison:
    """
    A comprehensive pattern comparison interface for analyzing multiple patterns
    """
    
    def __init__(self):
        """Initialize the pattern comparison interface"""
        self.selected_patterns = []
        self.pattern_data = {}
        self.similarity_metrics = {}
        self.correlation_matrix = None
        self.evolution_data = None
        
    def add_pattern(self, pattern_id: str, data: np.ndarray, 
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Add a pattern to the comparison
        
        Args:
            pattern_id: Unique identifier for the pattern
            data: Pattern data as numpy array
            metadata: Optional metadata about the pattern
        """
        self.pattern_data[pattern_id] = {
            'data': data,
            'metadata': metadata or {},
            'normalized': self._normalize_pattern(data)
        }
        if pattern_id not in self.selected_patterns:
            self.selected_patterns.append(pattern_id)
            
    def _normalize_pattern(self, data: np.ndarray) -> np.ndarray:
        """Normalize pattern data for comparison"""
        scaler = StandardScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def calculate_similarity_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate various similarity metrics between all selected patterns
        
        Returns:
            Dictionary of similarity metrics
        """
        metrics = {}
        
        for i, pattern1_id in enumerate(self.selected_patterns):
            metrics[pattern1_id] = {}
            
            for j, pattern2_id in enumerate(self.selected_patterns):
                if i == j:
                    metrics[pattern1_id][pattern2_id] = {
                        'dtw_distance': 0.0,
                        'correlation': 1.0,
                        'euclidean': 0.0,
                        'cosine_similarity': 1.0
                    }
                else:
                    data1 = self.pattern_data[pattern1_id]['normalized']
                    data2 = self.pattern_data[pattern2_id]['normalized']
                    
                    # Calculate DTW distance using pyts
                    dtw_dist = dtw(data1, data2, dist='square', method='classic')
                    
                    # Calculate correlation
                    correlation = np.corrcoef(data1, data2)[0, 1]
                    
                    # Calculate Euclidean distance
                    euclidean = np.linalg.norm(data1 - data2)
                    
                    # Calculate cosine similarity
                    cosine_sim = np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))
                    
                    metrics[pattern1_id][pattern2_id] = {
                        'dtw_distance': dtw_dist,
                        'correlation': correlation,
                        'euclidean': euclidean,
                        'cosine_similarity': cosine_sim
                    }
        
        self.similarity_metrics = metrics
        return metrics
    
    
    def create_side_by_side_visualization(self) -> go.Figure:
        """
        Create side-by-side visualizations of selected patterns
        
        Returns:
            Plotly figure with pattern comparisons
        """
        n_patterns = len(self.selected_patterns)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=n_patterns,
            subplot_titles=[f"Pattern: {pid}" for pid in self.selected_patterns] + 
                          [f"Normalized: {pid}" for pid in self.selected_patterns],
            vertical_spacing=0.15,
            horizontal_spacing=0.05
        )
        
        # Add original patterns
        for i, pattern_id in enumerate(self.selected_patterns):
            data = self.pattern_data[pattern_id]['data']
            x = np.arange(len(data))
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=data,
                    mode='lines',
                    name=f"{pattern_id} (Original)",
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1, col=i+1
            )
        
        # Add normalized patterns
        for i, pattern_id in enumerate(self.selected_patterns):
            data = self.pattern_data[pattern_id]['normalized']
            x = np.arange(len(data))
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=data,
                    mode='lines',
                    name=f"{pattern_id} (Normalized)",
                    line=dict(width=2, dash='dash'),
                    showlegend=True
                ),
                row=2, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title="Pattern Comparison: Side-by-Side Visualization",
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Steps")
        fig.update_yaxes(title_text="Value", row=1)
        fig.update_yaxes(title_text="Normalized Value", row=2)
        
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create a correlation heatmap for selected patterns
        
        Returns:
            Plotly figure with correlation heatmap
        """
        # Calculate correlation matrix
        n_patterns = len(self.selected_patterns)
        corr_matrix = np.zeros((n_patterns, n_patterns))
        
        for i, pattern1_id in enumerate(self.selected_patterns):
            for j, pattern2_id in enumerate(self.selected_patterns):
                if pattern1_id in self.similarity_metrics and pattern2_id in self.similarity_metrics[pattern1_id]:
                    corr_matrix[i, j] = self.similarity_metrics[pattern1_id][pattern2_id]['correlation']
                else:
                    data1 = self.pattern_data[pattern1_id]['normalized']
                    data2 = self.pattern_data[pattern2_id]['normalized']
                    corr_matrix[i, j] = np.corrcoef(data1, data2)[0, 1]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=self.selected_patterns,
            y=self.selected_patterns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Pattern Correlation Heatmap",
            xaxis_title="Pattern ID",
            yaxis_title="Pattern ID",
            height=500,
            template="plotly_dark"
        )
        
        return fig
    
    def create_similarity_matrix_visualization(self) -> go.Figure:
        """
        Create a comprehensive similarity matrix visualization
        
        Returns:
            Plotly figure with multiple similarity metrics
        """
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["DTW Distance", "Correlation", "Euclidean Distance", "Cosine Similarity"],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        n_patterns = len(self.selected_patterns)
        
        # Prepare matrices for each metric
        metrics_data = {
            'dtw_distance': np.zeros((n_patterns, n_patterns)),
            'correlation': np.zeros((n_patterns, n_patterns)),
            'euclidean': np.zeros((n_patterns, n_patterns)),
            'cosine_similarity': np.zeros((n_patterns, n_patterns))
        }
        
        # Fill matrices
        for i, pattern1_id in enumerate(self.selected_patterns):
            for j, pattern2_id in enumerate(self.selected_patterns):
                if pattern1_id in self.similarity_metrics and pattern2_id in self.similarity_metrics[pattern1_id]:
                    for metric in metrics_data:
                        metrics_data[metric][i, j] = self.similarity_metrics[pattern1_id][pattern2_id][metric]
        
        # Add heatmaps
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colorscales = ['Viridis', 'RdBu', 'Plasma', 'RdBu']
        
        for (metric, data), (row, col), colorscale in zip(metrics_data.items(), positions, colorscales):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    x=self.selected_patterns,
                    y=self.selected_patterns,
                    colorscale=colorscale,
                    text=np.round(data, 3),
                    texttemplate='%{text}',
                    textfont={"size": 8},
                    showscale=True,
                    colorbar=dict(len=0.4, y=0.75 if row == 1 else 0.25, x=1.02 if col == 2 else 0.48)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Comprehensive Similarity Matrix",
            height=800,
            template="plotly_dark"
        )
        
        return fig
    
    def analyze_pattern_evolution(self, time_labels: Optional[List[str]] = None) -> go.Figure:
        """
        Analyze and visualize pattern evolution over time
        
        Args:
            time_labels: Optional time labels for patterns
            
        Returns:
            Plotly figure showing pattern evolution
        """
        # Prepare data for PCA/t-SNE analysis
        pattern_matrix = np.array([self.pattern_data[pid]['normalized'] for pid in self.selected_patterns])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(pattern_matrix)
        
        # Apply t-SNE (adjust perplexity based on number of samples)
        n_samples = len(self.selected_patterns)
        if n_samples > 3:  # t-SNE needs at least 4 samples for default perplexity
            perplexity = min(30, n_samples - 1)  # Ensure perplexity is valid
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_coords = tsne.fit_transform(pattern_matrix)
        else:
            # For small sample sizes, use PCA coordinates as fallback
            tsne_coords = pca_coords
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["PCA Projection", "t-SNE Projection", 
                          "Pattern Trajectory (PCA)", "Feature Evolution"],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add PCA scatter
        fig.add_trace(
            go.Scatter(
                x=pca_coords[:, 0],
                y=pca_coords[:, 1],
                mode='markers+text',
                text=self.selected_patterns,
                textposition="top center",
                marker=dict(size=10, color=np.arange(len(self.selected_patterns)), colorscale='Viridis'),
                name="PCA"
            ),
            row=1, col=1
        )
        
        # Add t-SNE scatter
        fig.add_trace(
            go.Scatter(
                x=tsne_coords[:, 0],
                y=tsne_coords[:, 1],
                mode='markers+text',
                text=self.selected_patterns,
                textposition="top center",
                marker=dict(size=10, color=np.arange(len(self.selected_patterns)), colorscale='Plasma'),
                name="t-SNE"
            ),
            row=1, col=2
        )
        
        # Add pattern trajectory
        fig.add_trace(
            go.Scatter(
                x=pca_coords[:, 0],
                y=pca_coords[:, 1],
                mode='lines+markers',
                line=dict(width=2, color='cyan'),
                marker=dict(size=8, color=np.arange(len(self.selected_patterns)), colorscale='Turbo'),
                name="Evolution Path"
            ),
            row=2, col=1
        )
        
        # Add feature evolution
        # Calculate statistical features for each pattern
        features = []
        feature_names = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Max', 'Min']
        
        for pattern_id in self.selected_patterns:
            data = self.pattern_data[pattern_id]['normalized']
            features.append([
                np.mean(data),
                np.std(data),
                stats.skew(data),
                stats.kurtosis(data),
                np.max(data),
                np.min(data)
            ])
        
        features = np.array(features)
        
        # Plot feature evolution
        for i, feature_name in enumerate(feature_names):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(self.selected_patterns)),
                    y=features[:, i],
                    mode='lines+markers',
                    name=feature_name,
                    showlegend=True
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Pattern Evolution Analysis",
            height=800,
            template="plotly_dark",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title_text="PC1", row=2, col=1)
        fig.update_yaxes(title_text="PC2", row=2, col=1)
        fig.update_xaxes(title_text="Pattern Index", row=2, col=2)
        fig.update_yaxes(title_text="Feature Value", row=2, col=2)
        
        return fig
    
    def create_pattern_overlay(self) -> go.Figure:
        """
        Create an overlay visualization of all selected patterns
        
        Returns:
            Plotly figure with overlaid patterns
        """
        fig = go.Figure()
        
        # Define color palette
        colors = px.colors.qualitative.Set3
        
        # Add each pattern
        for i, pattern_id in enumerate(self.selected_patterns):
            data = self.pattern_data[pattern_id]['normalized']
            x = np.arange(len(data))
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=data,
                    mode='lines',
                    name=pattern_id,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    opacity=0.8
                )
            )
        
        # Add mean pattern
        mean_pattern = np.mean([self.pattern_data[pid]['normalized'] for pid in self.selected_patterns], axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mean_pattern)),
                y=mean_pattern,
                mode='lines',
                name='Mean Pattern',
                line=dict(width=3, color='white', dash='dash'),
                opacity=1.0
            )
        )
        
        # Add confidence bands
        std_pattern = np.std([self.pattern_data[pid]['normalized'] for pid in self.selected_patterns], axis=0)
        x = np.arange(len(mean_pattern))
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean_pattern + std_pattern, (mean_pattern - std_pattern)[::-1]]),
                fill='toself',
                fillcolor='rgba(255,255,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='±1 STD'
            )
        )
        
        fig.update_layout(
            title="Pattern Overlay Comparison",
            xaxis_title="Time Steps",
            yaxis_title="Normalized Value",
            height=500,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        return fig
    
    def create_similarity_network(self, threshold: float = 0.7) -> go.Figure:
        """
        Create a network visualization of pattern similarities
        
        Args:
            threshold: Similarity threshold for creating edges
            
        Returns:
            Plotly figure with network visualization
        """
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for pattern_id in self.selected_patterns:
            G.add_node(pattern_id)
        
        # Add edges based on similarity
        for i, pattern1_id in enumerate(self.selected_patterns):
            for j, pattern2_id in enumerate(self.selected_patterns[i+1:], i+1):
                if pattern1_id in self.similarity_metrics and pattern2_id in self.similarity_metrics[pattern1_id]:
                    similarity = self.similarity_metrics[pattern1_id][pattern2_id]['correlation']
                    if similarity > threshold:
                        G.add_edge(pattern1_id, pattern2_id, weight=similarity)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*5, color=f'rgba(125,125,125,{weight})'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=np.arange(len(G.nodes())),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Pattern Index")
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=f"Pattern Similarity Network (threshold={threshold})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            height=600
        )
        
        return fig
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report
        
        Returns:
            Dictionary containing comparison statistics and insights
        """
        report = {
            'n_patterns': len(self.selected_patterns),
            'pattern_ids': self.selected_patterns,
            'similarity_summary': {},
            'pattern_statistics': {},
            'insights': []
        }
        
        # Calculate similarity summary
        all_similarities = []
        for pattern1_id in self.similarity_metrics:
            for pattern2_id in self.similarity_metrics[pattern1_id]:
                if pattern1_id != pattern2_id:
                    all_similarities.append(self.similarity_metrics[pattern1_id][pattern2_id]['correlation'])
        
        if all_similarities:
            report['similarity_summary'] = {
                'mean_correlation': np.mean(all_similarities),
                'std_correlation': np.std(all_similarities),
                'min_correlation': np.min(all_similarities),
                'max_correlation': np.max(all_similarities)
            }
        
        # Calculate pattern statistics
        for pattern_id in self.selected_patterns:
            data = self.pattern_data[pattern_id]['normalized']
            report['pattern_statistics'][pattern_id] = {
                'length': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'trend': 'increasing' if data[-1] > data[0] else 'decreasing'
            }
        
        # Generate insights
        if report['similarity_summary']:
            mean_corr = report['similarity_summary']['mean_correlation']
            if mean_corr > 0.8:
                report['insights'].append("Patterns show high overall similarity (>0.8 correlation)")
            elif mean_corr < 0.3:
                report['insights'].append("Patterns show low overall similarity (<0.3 correlation)")
            
            # Find most similar pair
            max_sim = -1
            similar_pair = None
            for p1 in self.similarity_metrics:
                for p2 in self.similarity_metrics[p1]:
                    if p1 != p2 and self.similarity_metrics[p1][p2]['correlation'] > max_sim:
                        max_sim = self.similarity_metrics[p1][p2]['correlation']
                        similar_pair = (p1, p2)
            
            if similar_pair:
                report['insights'].append(f"Most similar patterns: {similar_pair[0]} and {similar_pair[1]} (correlation: {max_sim:.3f})")
        
        return report


def create_demo_patterns() -> Dict[str, np.ndarray]:
    """Create demo patterns for testing"""
    x = np.linspace(0, 4*np.pi, 100)
    
    patterns = {
        'sine_wave': np.sin(x),
        'cosine_wave': np.cos(x),
        'damped_sine': np.sin(x) * np.exp(-x/10),
        'growing_sine': np.sin(x) * np.exp(x/20),
        'square_wave': np.sign(np.sin(x)),
        'sawtooth': 2 * (x % (2*np.pi)) / (2*np.pi) - 1,
        'noisy_sine': np.sin(x) + np.random.normal(0, 0.1, len(x))
    }
    
    return patterns


if __name__ == "__main__":
    # Create demo instance
    comparison = PatternComparison()
    
    # Generate demo patterns
    patterns = create_demo_patterns()
    
    # Add patterns to comparison
    for pattern_id, data in patterns.items():
        comparison.add_pattern(pattern_id, data)
    
    # Calculate similarity metrics
    metrics = comparison.calculate_similarity_metrics()
    
    # Create visualizations
    fig1 = comparison.create_side_by_side_visualization()
    fig1.show()
    
    fig2 = comparison.create_correlation_heatmap()
    fig2.show()
    
    fig3 = comparison.create_similarity_matrix_visualization()
    fig3.show()
    
    fig4 = comparison.analyze_pattern_evolution()
    fig4.show()
    
    fig5 = comparison.create_pattern_overlay()
    fig5.show()
    
    fig6 = comparison.create_similarity_network(threshold=0.5)
    fig6.show()
    
    # Generate report
    report = comparison.generate_comparison_report()
    print("\nPattern Comparison Report:")
    print(f"Number of patterns: {report['n_patterns']}")
    print(f"Pattern IDs: {report['pattern_ids']}")
    print(f"\nSimilarity Summary:")
    for key, value in report['similarity_summary'].items():
        print(f"  {key}: {value:.3f}")
    print(f"\nInsights:")
    for insight in report['insights']:
        print(f"  - {insight}")
