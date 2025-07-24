"""
Pattern Visualizer Module

Provides visualization tools for wavelets, motifs, shapelets, and discovered patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PatternVisualizer:
    """
    Visualizes patterns discovered in financial time series analysis.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the PatternVisualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
        # Color palettes
        self.colors = sns.color_palette("husl", 10)
        self.cmap = plt.cm.viridis
        
        logger.info(f"Initialized PatternVisualizer with style '{style}'")
    
    def plot_wavelet_transform(self, coefficients: np.ndarray,
                              scales: np.ndarray,
                              time: Optional[np.ndarray] = None,
                              title: str = "Continuous Wavelet Transform",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot wavelet transform coefficients as a scalogram.
        
        Args:
            coefficients: CWT coefficients
            scales: Scale values
            time: Time array (optional)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Time axis
        if time is None:
            time = np.arange(coefficients.shape[1])
        
        # Plot scalogram
        im = ax1.imshow(np.abs(coefficients), aspect='auto', cmap='jet',
                       extent=[time[0], time[-1], scales[-1], scales[0]])
        ax1.set_ylabel('Scale')
        ax1.set_title(title)
        ax1.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Magnitude')
        
        # Plot scale-averaged power
        scale_avg_power = np.mean(np.abs(coefficients)**2, axis=0)
        ax2.plot(time, scale_avg_power, 'b-', linewidth=1.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Avg Power')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved wavelet transform plot to {save_path}")
        
        return fig
    
    def plot_motifs(self, data: Union[pd.Series, np.ndarray],
                   motifs: List[Dict],
                   title: str = "Discovered Motifs",
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot discovered motifs in the time series.
        
        Args:
            data: Original time series
            motifs: List of motif dictionaries
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            time = data.index
        else:
            values = np.asarray(data)
            time = np.arange(len(values))
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize,
                                gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot time series with motif highlights
        ax1 = axes[0]
        ax1.plot(time, values, 'k-', alpha=0.5, linewidth=0.5, label='Time Series')
        
        # Highlight motif occurrences
        for i, motif in enumerate(motifs[:5]):  # Show top 5 motifs
            color = self.colors[i % len(self.colors)]
            
            for occurrence in motif['occurrences']:
                window_size = motif['window_size']
                if occurrence + window_size <= len(values):
                    ax1.axvspan(time[occurrence], time[occurrence + window_size - 1],
                               alpha=0.3, color=color, label=f'Motif {i+1}' if occurrence == motif['occurrences'][0] else '')
        
        ax1.set_title(title)
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot motif patterns
        ax2 = axes[1]
        for i, motif in enumerate(motifs[:5]):
            color = self.colors[i % len(self.colors)]
            pattern = motif['pattern']
            pattern_time = np.arange(len(pattern))
            
            # Normalize for comparison
            pattern_norm = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
            
            ax2.plot(pattern_time, pattern_norm + i * 3, color=color,
                    linewidth=2, label=f'Motif {i+1} ({motif["num_occurrences"]} times)')
        
        ax2.set_xlabel('Time (relative)')
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('Motif Patterns')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved motifs plot to {save_path}")
        
        return fig
    
    def plot_shapelets(self, shapelets: List[Dict],
                      title: str = "Extracted Shapelets",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot extracted shapelets.
        
        Args:
            shapelets: List of shapelet dictionaries
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_shapelets = min(len(shapelets), 12)
        n_cols = 3
        n_rows = (n_shapelets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, shapelet in enumerate(shapelets[:n_shapelets]):
            ax = axes[i]
            pattern = shapelet['pattern']
            
            # Plot shapelet
            ax.plot(pattern, 'b-', linewidth=2)
            ax.fill_between(range(len(pattern)), pattern, alpha=0.3)
            
            # Add statistics
            quality = shapelet['quality']
            threshold = shapelet['threshold']
            
            ax.set_title(f'Shapelet {i+1}\nQuality: {quality:.3f}, Threshold: {threshold:.3f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_shapelets, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved shapelets plot to {save_path}")
        
        return fig
    
    def plot_pattern_heatmap(self, patterns: List[np.ndarray],
                           labels: Optional[List[str]] = None,
                           title: str = "Pattern Similarity Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of pattern similarities.
        
        Args:
            patterns: List of pattern arrays
            labels: Pattern labels
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Compute pairwise distances
        n_patterns = len(patterns)
        distances = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(n_patterns):
                if len(patterns[i]) == len(patterns[j]):
                    # Normalize patterns
                    p1 = (patterns[i] - np.mean(patterns[i])) / (np.std(patterns[i]) + 1e-8)
                    p2 = (patterns[j] - np.mean(patterns[j])) / (np.std(patterns[j]) + 1e-8)
                    distances[i, j] = np.linalg.norm(p1 - p2)
                else:
                    distances[i, j] = np.inf
        
        # Convert to similarity
        similarities = 1 / (1 + distances)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(similarities, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=labels if labels else range(n_patterns),
                   yticklabels=labels if labels else range(n_patterns),
                   ax=ax)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pattern heatmap to {save_path}")
        
        return fig
    
    def create_interactive_scalogram(self, coefficients: np.ndarray,
                                   scales: np.ndarray,
                                   time: Optional[np.ndarray] = None,
                                   original_data: Optional[np.ndarray] = None,
                                   title: str = "Interactive Wavelet Scalogram") -> go.Figure:
        """
        Create interactive scalogram using Plotly.
        
        Args:
            coefficients: CWT coefficients
            scales: Scale values
            time: Time array
            original_data: Original time series
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if time is None:
            time = np.arange(coefficients.shape[1])
        
        # Create subplots
        if original_data is not None:
            fig = make_subplots(rows=2, cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.1,
                               row_heights=[0.7, 0.3],
                               subplot_titles=('Scalogram', 'Original Signal'))
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Add scalogram
        fig.add_trace(
            go.Heatmap(
                z=np.abs(coefficients),
                x=time,
                y=scales,
                colorscale='Jet',
                name='Magnitude'
            ),
            row=1, col=1
        )
        
        # Add original signal if provided
        if original_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=original_data,
                    mode='lines',
                    name='Signal',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Scale',
            height=600 if original_data is not None else 400,
            showlegend=False
        )
        
        # Update y-axis for scalogram (log scale)
        fig.update_yaxes(type='log', row=1, col=1)
        
        return fig
    
    def create_interactive_motif_plot(self, data: Union[pd.Series, np.ndarray],
                                    motifs: List[Dict],
                                    title: str = "Interactive Motif Discovery") -> go.Figure:
        """
        Create interactive motif visualization using Plotly.
        
        Args:
            data: Original time series
            motifs: List of motif dictionaries
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            time = data.index
        else:
            values = np.asarray(data)
            time = np.arange(len(values))
        
        fig = go.Figure()
        
        # Add original time series
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                mode='lines',
                name='Time Series',
                line=dict(color='black', width=1),
                opacity=0.5
            )
        )
        
        # Add motif occurrences
        colors = px.colors.qualitative.Set3
        
        for i, motif in enumerate(motifs[:10]):  # Show top 10 motifs
            color = colors[i % len(colors)]
            
            # Add shapes for motif occurrences
            for occurrence in motif['occurrences']:
                window_size = motif['window_size']
                if occurrence + window_size <= len(values):
                    fig.add_vrect(
                        x0=time[occurrence],
                        x1=time[occurrence + window_size - 1],
                        fillcolor=color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                    )
            
            # Add motif pattern as separate trace
            pattern_time = time[motif['primary_index']:motif['primary_index'] + window_size]
            pattern_values = values[motif['primary_index']:motif['primary_index'] + window_size]
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_time,
                    y=pattern_values,
                    mode='lines',
                    name=f'Motif {i+1} ({motif["num_occurrences"]}x)',
                    line=dict(color=color, width=3),
                    visible='legendonly'
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_wavelet_features(self, features: Dict[str, np.ndarray],
                            title: str = "Wavelet Features",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot extracted wavelet features.
        
        Args:
            features: Dictionary of wavelet features
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, (name, values) in enumerate(features.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if name == 'ridges':
                # Special handling for ridge points
                if len(values) > 0:
                    ridges = np.array(values)
                    ax.scatter(ridges[:, 1], ridges[:, 0], alpha=0.6, s=20)
                    ax.set_xlabel('Time Index')
                    ax.set_ylabel('Scale Index')
                else:
                    ax.text(0.5, 0.5, 'No ridges detected', 
                           ha='center', va='center', transform=ax.transAxes)
            elif len(values.shape) == 1:
                # 1D feature
                ax.plot(values, 'b-', linewidth=1.5)
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
            else:
                # 2D feature
                im = ax.imshow(values, aspect='auto', cmap='viridis')
                plt.colorbar(im, ax=ax)
            
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved wavelet features plot to {save_path}")
        
        return fig
    
    def plot_pattern_evolution(self, chains: List[Dict],
                             title: str = "Pattern Evolution (Chains)",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot evolving patterns (chains).
        
        Args:
            chains: List of chain dictionaries
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot top 4 chains
        for i, chain in enumerate(chains[:4]):
            ax = axes[i]
            
            patterns = chain['patterns']
            n_patterns = len(patterns)
            
            # Create color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, n_patterns))
            
            # Plot each pattern in the chain
            for j, pattern in enumerate(patterns):
                time = np.arange(len(pattern))
                ax.plot(time, pattern + j * 0.5, color=colors[j],
                       linewidth=2, alpha=0.8, label=f'Step {j+1}')
            
            ax.set_title(f'Chain {i+1} (Length: {chain["length"]})')
            ax.set_xlabel('Time (relative)')
            ax.set_ylabel('Value (offset)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add evolution metric
            ax.text(0.02, 0.98, f'Total Evolution: {chain["total_evolution"]:.3f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pattern evolution plot to {save_path}")
        
        return fig
    
    def create_pattern_summary_dashboard(self, data: Union[pd.Series, np.ndarray],
                                       wavelet_results: Dict,
                                       motifs: List[Dict],
                                       shapelets: List[Dict],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard summarizing all pattern discoveries.
        
        Args:
            data: Original time series
            wavelet_results: Dictionary with wavelet analysis results
            motifs: List of discovered motifs
            shapelets: List of extracted shapelets
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original time series
        ax1 = fig.add_subplot(gs[0, :])
        if isinstance(data, pd.Series):
            ax1.plot(data.index, data.values, 'k-', linewidth=1)
        else:
            ax1.plot(data, 'k-', linewidth=1)
        ax1.set_title('Original Time Series', fontsize=14)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Wavelet scalogram
        ax2 = fig.add_subplot(gs[1, :2])
        if 'power' in wavelet_results:
            im = ax2.imshow(wavelet_results['power'], aspect='auto', cmap='jet',
                           extent=[0, wavelet_results['power'].shape[1],
                                  wavelet_results['scales'][-1],
                                  wavelet_results['scales'][0]])
            ax2.set_title('Wavelet Power Spectrum', fontsize=14)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Scale')
            ax2.invert_yaxis()
            plt.colorbar(im, ax=ax2)
        
        # 3. Top motifs
        ax3 = fig.add_subplot(gs[1, 2])
        if motifs:
            motif_counts = [m['num_occurrences'] for m in motifs[:5]]
            motif_labels = [f'Motif {i+1}' for i in range(len(motif_counts))]
            ax3.barh(motif_labels, motif_counts, color=self.colors[:len(motif_counts)])
            ax3.set_xlabel('Occurrences')
            ax3.set_title('Top Motifs', fontsize=14)
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Shapelet quality distribution
        ax4 = fig.add_subplot(gs[2, 0])
        if shapelets:
            qualities = [s['quality'] for s in shapelets]
            ax4.hist(qualities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Quality Score')
            ax4.set_ylabel('Count')
            ax4.set_title('Shapelet Quality Distribution', fontsize=14)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Pattern length distribution
        ax5 = fig.add_subplot(gs[2, 1])
        all_lengths = []
        if motifs:
            all_lengths.extend([m['window_size'] for m in motifs])
        if shapelets:
            all_lengths.extend([len(s['pattern']) for s in shapelets])
        
        if all_lengths:
            ax5.hist(all_lengths, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Pattern Length')
            ax5.set_ylabel('Count')
            ax5.set_title('Pattern Length Distribution', fontsize=14)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = f"""Pattern Discovery Summary
        
Total Motifs: {len(motifs)}
Total Shapelets: {len(shapelets)}
Avg Motif Occurrences: {np.mean([m['num_occurrences'] for m in motifs]):.1f}
Avg Shapelet Quality: {np.mean([s['quality'] for s in shapelets]):.3f}

Data Points: {len(data)}
Analysis Complete"""
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Pattern Discovery Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pattern summary dashboard to {save_path}")
        
        return fig


def main():
    """Demonstration of PatternVisualizer functionality."""
    # Create sample data
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    
    # Create sample patterns
    patterns = [
        np.sin(np.linspace(0, 2*np.pi, 50)),
        np.cos(np.linspace(0, 2*np.pi, 50)),
        np.linspace(0, 1, 50),
        np.linspace(1, 0, 50)
    ]
    
    # Initialize visualizer
    viz = PatternVisualizer()
    
    # Create pattern heatmap
    fig = viz.plot_pattern_heatmap(
        patterns,
        labels=['Sin', 'Cos', 'Up', 'Down'],
        title='Pattern Similarity'
    )
    plt.show()
    
    print("Visualization demonstration complete")


if __name__ == "__main__":
    main()
