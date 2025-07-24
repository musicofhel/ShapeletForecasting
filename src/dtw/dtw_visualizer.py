"""
DTW Visualizer

Provides visualization tools for DTW alignments, similarity matrices,
and pattern matching results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

from .dtw_calculator import DTWResult


class DTWVisualizer:
    """
    Visualization tools for DTW analysis results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize DTW Visualizer
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use
        figsize : Tuple[int, int]
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_alignment(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      path: np.ndarray,
                      title: str = "DTW Alignment",
                      labels: Tuple[str, str] = ("Series X", "Series Y"),
                      save_path: Optional[str] = None):
        """
        Plot DTW alignment between two time series
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        path : np.ndarray
            DTW alignment path
        title : str
            Plot title
        labels : Tuple[str, str]
            Labels for the two series
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Plot first time series
        ax1.plot(x, 'b-', linewidth=2, label=labels[0])
        ax1.set_ylabel(labels[0])
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot alignment connections
        for i, j in path:
            ax2.plot([i, j], [0, 1], 'k-', alpha=0.3, linewidth=0.5)
            
        # Highlight some key alignments
        step = max(1, len(path) // 20)  # Show ~20 connections
        for idx in range(0, len(path), step):
            i, j = path[idx]
            ax2.plot([i, j], [0, 1], 'r-', alpha=0.8, linewidth=2)
            
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(-1, max(len(x), len(y)))
        ax2.set_ylabel("Alignment")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels([labels[0], labels[1]])
        ax2.grid(True, alpha=0.3)
        
        # Plot second time series
        ax3.plot(y, 'g-', linewidth=2, label=labels[1])
        ax3.set_ylabel(labels[1])
        ax3.set_xlabel("Time")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_cost_matrix(self,
                        cost_matrix: np.ndarray,
                        path: Optional[np.ndarray] = None,
                        title: str = "DTW Cost Matrix",
                        save_path: Optional[str] = None):
        """
        Plot DTW cost matrix with optional path overlay
        
        Parameters:
        -----------
        cost_matrix : np.ndarray
            DTW cost matrix
        path : np.ndarray, optional
            Optimal warping path
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot cost matrix
        im = plt.imshow(cost_matrix.T, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Cost')
        
        # Overlay path if provided
        if path is not None:
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=3, label='Optimal Path')
            plt.legend()
            
        plt.xlabel("Series X")
        plt.ylabel("Series Y")
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_similarity_matrix(self,
                             similarity_matrix: np.ndarray,
                             labels: Optional[List[str]] = None,
                             title: str = "Pattern Similarity Matrix",
                             annotate: bool = False,
                             save_path: Optional[str] = None):
        """
        Plot similarity matrix heatmap
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Pairwise similarity matrix
        labels : List[str], optional
            Labels for patterns
        title : str
            Plot title
        annotate : bool
            Whether to annotate cells with values
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(self.figsize[0], self.figsize[0]))  # Square figure
        
        # Create heatmap
        sns.heatmap(similarity_matrix,
                   annot=annotate,
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   square=True,
                   cbar_kws={'label': 'Similarity'},
                   xticklabels=labels,
                   yticklabels=labels)
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_pattern_comparison(self,
                              patterns: List[np.ndarray],
                              labels: Optional[List[str]] = None,
                              title: str = "Pattern Comparison",
                              normalize: bool = True,
                              save_path: Optional[str] = None):
        """
        Plot multiple patterns for comparison
        
        Parameters:
        -----------
        patterns : List[np.ndarray]
            List of patterns to compare
        labels : List[str], optional
            Labels for patterns
        title : str
            Plot title
        normalize : bool
            Whether to normalize patterns for comparison
        save_path : str, optional
            Path to save the plot
        """
        n_patterns = len(patterns)
        colors = plt.cm.tab10(np.linspace(0, 1, n_patterns))
        
        plt.figure(figsize=self.figsize)
        
        for i, pattern in enumerate(patterns):
            if pattern.ndim > 1:
                pattern = pattern.mean(axis=1)  # Average multivariate
                
            if normalize:
                # Z-score normalization
                pattern = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
                
            label = labels[i] if labels else f"Pattern {i+1}"
            plt.plot(pattern, color=colors[i], linewidth=2, label=label, alpha=0.8)
            
        plt.xlabel("Time")
        plt.ylabel("Value (normalized)" if normalize else "Value")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_dtw_matrix_comparison(self,
                                 dtw_results: Dict[str, np.ndarray],
                                 labels: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        Compare multiple DTW distance matrices
        
        Parameters:
        -----------
        dtw_results : Dict[str, np.ndarray]
            Dictionary of DTW method names to distance matrices
        labels : List[str], optional
            Pattern labels
        save_path : str, optional
            Path to save the plot
        """
        n_methods = len(dtw_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
            
        for idx, (method_name, distance_matrix) in enumerate(dtw_results.items()):
            ax = axes[idx]
            
            im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f"{method_name} DTW")
            
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
                
            plt.colorbar(im, ax=ax, label='Distance')
            
        plt.suptitle("DTW Method Comparison", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_interactive_alignment(self,
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   path: np.ndarray,
                                   title: str = "Interactive DTW Alignment",
                                   labels: Tuple[str, str] = ("Series X", "Series Y")) -> go.Figure:
        """
        Create interactive DTW alignment visualization using Plotly
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        path : np.ndarray
            DTW alignment path
        title : str
            Plot title
        labels : Tuple[str, str]
            Labels for the two series
            
        Returns:
        --------
        Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.3, 0.4, 0.3],
            subplot_titles=(labels[0], "Alignment", labels[1]),
            vertical_spacing=0.05
        )
        
        # Plot first time series
        fig.add_trace(
            go.Scatter(x=list(range(len(x))), y=x, mode='lines', name=labels[0], line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Plot alignment connections
        for i, j in path[::max(1, len(path)//100)]:  # Subsample for performance
            fig.add_trace(
                go.Scatter(
                    x=[i, j],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
        # Highlight key alignments
        step = max(1, len(path) // 20)
        for idx in range(0, len(path), step):
            i, j = path[idx]
            fig.add_trace(
                go.Scatter(
                    x=[i, j],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hovertemplate=f'X[{i}] → Y[{j}]<extra></extra>'
                ),
                row=2, col=1
            )
            
        # Plot second time series
        fig.add_trace(
            go.Scatter(x=list(range(len(y))), y=y, mode='lines', name=labels[1], line=dict(color='green', width=2)),
            row=3, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="", row=2, col=1, tickvals=[0, 1], ticktext=[labels[0], labels[1]])
        fig.update_yaxes(title_text="Value", row=3, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def create_interactive_similarity_matrix(self,
                                          similarity_matrix: np.ndarray,
                                          labels: Optional[List[str]] = None,
                                          title: str = "Interactive Similarity Matrix") -> go.Figure:
        """
        Create interactive similarity matrix heatmap using Plotly
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Pairwise similarity matrix
        labels : List[str], optional
            Pattern labels
        title : str
            Plot title
            
        Returns:
        --------
        Plotly figure object
        """
        if labels is None:
            labels = [f"Pattern {i}" for i in range(len(similarity_matrix))]
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Similarity"),
            hovertemplate='%{x} vs %{y}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=-45),
            width=800,
            height=800
        )
        
        return fig
        
    def plot_cluster_patterns(self,
                            patterns: List[np.ndarray],
                            cluster_labels: np.ndarray,
                            cluster_centers: Dict[int, int],
                            n_examples: int = 3,
                            normalize: bool = True,
                            save_path: Optional[str] = None):
        """
        Plot example patterns from each cluster
        
        Parameters:
        -----------
        patterns : List[np.ndarray]
            All patterns
        cluster_labels : np.ndarray
            Cluster assignments
        cluster_centers : Dict[int, int]
            Indices of cluster centers
        n_examples : int
            Number of example patterns per cluster
        normalize : bool
            Whether to normalize patterns
        save_path : str, optional
            Path to save the plot
        """
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        fig, axes = plt.subplots(n_clusters, 1, figsize=(self.figsize[0], 3 * n_clusters))
        if n_clusters == 1:
            axes = [axes]
            
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for cluster_idx, cluster_label in enumerate(unique_clusters):
            ax = axes[cluster_idx]
            
            # Get cluster members
            cluster_mask = cluster_labels == cluster_label
            cluster_indices = np.where(cluster_mask)[0]
            
            # Plot cluster center
            center_idx = cluster_centers[cluster_label]
            center_pattern = patterns[center_idx]
            if center_pattern.ndim > 1:
                center_pattern = center_pattern.mean(axis=1)
            if normalize:
                center_pattern = (center_pattern - np.mean(center_pattern)) / (np.std(center_pattern) + 1e-8)
                
            ax.plot(center_pattern, 'k-', linewidth=3, label='Center', alpha=0.8)
            
            # Plot example patterns
            example_indices = np.random.choice(cluster_indices, 
                                             size=min(n_examples, len(cluster_indices)), 
                                             replace=False)
            
            for i, idx in enumerate(example_indices):
                if idx == center_idx:
                    continue
                    
                pattern = patterns[idx]
                if pattern.ndim > 1:
                    pattern = pattern.mean(axis=1)
                if normalize:
                    pattern = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
                    
                ax.plot(pattern, color=colors[i % 10], linewidth=1, alpha=0.6, 
                       label=f'Example {i+1}')
                
            ax.set_title(f'Cluster {cluster_label} (n={len(cluster_indices)})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value (normalized)' if normalize else 'Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('Cluster Pattern Examples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_dtw_performance_comparison(self,
                                      performance_data: Dict[str, Dict],
                                      save_path: Optional[str] = None):
        """
        Plot performance comparison of different DTW methods
        
        Parameters:
        -----------
        performance_data : Dict[str, Dict]
            Performance metrics for each DTW method
            Expected format: {method_name: {'time': [...], 'accuracy': [...]}}
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        methods = list(performance_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Time comparison
        times = [np.mean(data['time']) for data in performance_data.values()]
        time_stds = [np.std(data['time']) for data in performance_data.values()]
        
        bars1 = ax1.bar(methods, times, yerr=time_stds, capsize=5, color=colors)
        ax1.set_ylabel('Computation Time (seconds)')
        ax1.set_title('DTW Method Time Comparison')
        ax1.set_xticklabels(methods, rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.3f}s', ha='center', va='bottom')
                    
        # Accuracy comparison (if available)
        if all('accuracy' in data for data in performance_data.values()):
            accuracies = [np.mean(data['accuracy']) for data in performance_data.values()]
            acc_stds = [np.std(data['accuracy']) for data in performance_data.values()]
            
            bars2 = ax2.bar(methods, accuracies, yerr=acc_stds, capsize=5, color=colors)
            ax2.set_ylabel('Accuracy')
            ax2.set_title('DTW Method Accuracy Comparison')
            ax2.set_xticklabels(methods, rotation=45)
            ax2.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, acc in zip(bars2, accuracies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.3f}', ha='center', va='bottom')
                        
        plt.suptitle('DTW Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def save_all_visualizations(self,
                              output_dir: str,
                              x: np.ndarray,
                              y: np.ndarray,
                              dtw_result: DTWResult,
                              similarity_matrix: Optional[np.ndarray] = None,
                              cluster_results: Optional[Dict] = None):
        """
        Generate and save all relevant visualizations
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        dtw_result : DTWResult
            DTW computation result
        similarity_matrix : np.ndarray, optional
            Similarity matrix to visualize
        cluster_results : Dict, optional
            Clustering results to visualize
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # DTW alignment
        self.plot_alignment(x, y, dtw_result.path, 
                          save_path=output_path / "dtw_alignment.png")
        
        # Cost matrix if available
        if dtw_result.cost_matrix is not None:
            self.plot_cost_matrix(dtw_result.cost_matrix, dtw_result.path,
                                save_path=output_path / "cost_matrix.png")
                                
        # Similarity matrix if provided
        if similarity_matrix is not None:
            self.plot_similarity_matrix(similarity_matrix,
                                      save_path=output_path / "similarity_matrix.png")
                                      
        # Cluster visualizations if provided
        if cluster_results is not None:
            # Dendrogram for hierarchical clustering
            if 'linkage_matrix' in cluster_results:
                from .pattern_clusterer import PatternClusterer
                clusterer = PatternClusterer()
                clusterer.plot_dendrogram(cluster_results['linkage_matrix'],
                                        save_path=output_path / "dendrogram.png")
                                        
        print(f"Visualizations saved to {output_path}")
