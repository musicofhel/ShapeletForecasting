"""
Demo script for Pattern Comparison Interface

This script demonstrates the comprehensive pattern comparison functionality including:
- Loading and comparing multiple patterns
- Calculating similarity metrics
- Creating various visualizations
- Generating comparison reports
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.dashboard.visualizations.pattern_comparison import PatternComparison
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_financial_patterns():
    """Generate realistic financial patterns for demonstration"""
    
    # Time series length
    n_points = 100
    
    # Base time array
    t = np.linspace(0, 4*np.pi, n_points)
    
    patterns = {}
    
    # 1. Bull Market Pattern (steady uptrend with volatility)
    trend = 0.5 * t
    volatility = 0.3 * np.sin(5*t) + 0.1 * np.random.randn(n_points)
    patterns['bull_market'] = trend + volatility + 10
    
    # 2. Bear Market Pattern (downtrend with spikes)
    trend = -0.3 * t
    spikes = 0.4 * np.sin(3*t) * np.exp(-t/20)
    patterns['bear_market'] = trend + spikes + 15
    
    # 3. Consolidation Pattern (sideways movement)
    base = 12
    oscillation = 0.5 * np.sin(4*t) + 0.2 * np.cos(7*t)
    patterns['consolidation'] = base + oscillation
    
    # 4. Breakout Pattern (consolidation then sharp move up)
    consolidation_phase = 10 + 0.3 * np.sin(10*t[:60])
    breakout_phase = 10 + 0.3 * np.sin(10*t[60]) + 2 * (t[60:] - t[60])
    patterns['breakout'] = np.concatenate([consolidation_phase, breakout_phase])
    
    # 5. Head and Shoulders Pattern
    left_shoulder = 2 * np.exp(-(t-2)**2)
    head = 3 * np.exp(-(t-4)**2)
    right_shoulder = 2 * np.exp(-(t-6)**2)
    patterns['head_shoulders'] = 10 + left_shoulder + head + right_shoulder
    
    # 6. Double Bottom Pattern
    first_bottom = -2 * np.exp(-(t-3)**2)
    second_bottom = -2 * np.exp(-(t-7)**2)
    patterns['double_bottom'] = 12 + first_bottom + second_bottom
    
    # 7. Cup and Handle Pattern
    cup = -1.5 * (t - 2*np.pi)**2 / (4*np.pi**2) + 1.5
    handle = np.where(t > 3*np.pi, -0.3 * np.sin(10*(t-3*np.pi)), 0)
    patterns['cup_handle'] = 10 + cup + handle
    
    # 8. Volatility Expansion Pattern
    expanding_vol = 0.1 * t * np.sin(5*t)
    patterns['volatility_expansion'] = 11 + expanding_vol
    
    return patterns


def demonstrate_pattern_comparison():
    """Main demonstration function"""
    
    print("=== Pattern Comparison Interface Demo ===\n")
    
    # Create comparison instance
    comparison = PatternComparison()
    
    # Generate financial patterns
    patterns = generate_financial_patterns()
    
    # Add patterns to comparison with metadata
    pattern_metadata = {
        'bull_market': {'type': 'trend', 'direction': 'up', 'strength': 'strong'},
        'bear_market': {'type': 'trend', 'direction': 'down', 'strength': 'moderate'},
        'consolidation': {'type': 'range', 'direction': 'sideways', 'strength': 'weak'},
        'breakout': {'type': 'breakout', 'direction': 'up', 'strength': 'strong'},
        'head_shoulders': {'type': 'reversal', 'direction': 'down', 'strength': 'strong'},
        'double_bottom': {'type': 'reversal', 'direction': 'up', 'strength': 'moderate'},
        'cup_handle': {'type': 'continuation', 'direction': 'up', 'strength': 'strong'},
        'volatility_expansion': {'type': 'volatility', 'direction': 'neutral', 'strength': 'increasing'}
    }
    
    # Select patterns for comparison (user can select any subset)
    selected_patterns = ['bull_market', 'bear_market', 'breakout', 'head_shoulders', 'cup_handle']
    
    print(f"Selected patterns for comparison: {selected_patterns}\n")
    
    # Add selected patterns
    for pattern_id in selected_patterns:
        if pattern_id in patterns:
            comparison.add_pattern(
                pattern_id, 
                patterns[pattern_id],
                metadata=pattern_metadata.get(pattern_id, {})
            )
    
    # Calculate similarity metrics
    print("Calculating similarity metrics...")
    metrics = comparison.calculate_similarity_metrics()
    
    # Print similarity matrix
    print("\nSimilarity Matrix (Correlation):")
    print("-" * 80)
    print(f"{'Pattern':<20}", end="")
    for p in selected_patterns[:5]:  # Limit display width
        print(f"{p[:10]:<12}", end="")
    print()
    print("-" * 80)
    
    for p1 in selected_patterns:
        print(f"{p1:<20}", end="")
        for p2 in selected_patterns[:5]:
            if p1 in metrics and p2 in metrics[p1]:
                corr = metrics[p1][p2]['correlation']
                print(f"{corr:>11.3f}", end=" ")
            else:
                print(f"{'N/A':>11}", end=" ")
        print()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Side-by-side comparison
    fig1 = comparison.create_side_by_side_visualization()
    fig1.write_html("pattern_comparison_side_by_side.html")
    print("✓ Created: pattern_comparison_side_by_side.html")
    
    # 2. Correlation heatmap
    fig2 = comparison.create_correlation_heatmap()
    fig2.write_html("pattern_comparison_correlation.html")
    print("✓ Created: pattern_comparison_correlation.html")
    
    # 3. Comprehensive similarity matrix
    fig3 = comparison.create_similarity_matrix_visualization()
    fig3.write_html("pattern_comparison_similarity_matrix.html")
    print("✓ Created: pattern_comparison_similarity_matrix.html")
    
    # 4. Pattern evolution analysis
    fig4 = comparison.analyze_pattern_evolution()
    fig4.write_html("pattern_comparison_evolution.html")
    print("✓ Created: pattern_comparison_evolution.html")
    
    # 5. Pattern overlay
    fig5 = comparison.create_pattern_overlay()
    fig5.write_html("pattern_comparison_overlay.html")
    print("✓ Created: pattern_comparison_overlay.html")
    
    # 6. Similarity network
    fig6 = comparison.create_similarity_network(threshold=0.5)
    fig6.write_html("pattern_comparison_network.html")
    print("✓ Created: pattern_comparison_network.html")
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    report = comparison.generate_comparison_report()
    
    print("\n" + "="*80)
    print("PATTERN COMPARISON REPORT")
    print("="*80)
    print(f"\nNumber of patterns analyzed: {report['n_patterns']}")
    print(f"Pattern IDs: {', '.join(report['pattern_ids'])}")
    
    if report['similarity_summary']:
        print(f"\nSimilarity Summary:")
        print(f"  Mean correlation: {report['similarity_summary']['mean_correlation']:.3f}")
        print(f"  Std correlation: {report['similarity_summary']['std_correlation']:.3f}")
        print(f"  Min correlation: {report['similarity_summary']['min_correlation']:.3f}")
        print(f"  Max correlation: {report['similarity_summary']['max_correlation']:.3f}")
    
    print(f"\nPattern Statistics:")
    for pattern_id, stats in report['pattern_statistics'].items():
        print(f"\n  {pattern_id}:")
        print(f"    Length: {stats['length']}")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std: {stats['std']:.3f}")
        print(f"    Skewness: {stats['skewness']:.3f}")
        print(f"    Kurtosis: {stats['kurtosis']:.3f}")
        print(f"    Trend: {stats['trend']}")
    
    print(f"\nKey Insights:")
    for insight in report['insights']:
        print(f"  • {insight}")
    
    print("\n" + "="*80)
    
    # Create a comprehensive dashboard
    create_comparison_dashboard(comparison, selected_patterns)
    
    print("\n✓ Pattern comparison demo completed successfully!")
    print("✓ Check the generated HTML files for interactive visualizations")


def create_comparison_dashboard(comparison, selected_patterns):
    """Create a comprehensive comparison dashboard"""
    
    # Create a multi-panel dashboard
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Pattern Overlay", "Correlation Heatmap", "DTW Distance Matrix",
            "Pattern Evolution (PCA)", "Feature Statistics", "Similarity Network",
            "Pattern 1 vs 2", "Pattern 1 vs 3", "Time-based Evolution"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # 1. Pattern Overlay (row 1, col 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, pattern_id in enumerate(selected_patterns):
        data = comparison.pattern_data[pattern_id]['normalized']
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode='lines',
                name=pattern_id,
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Correlation Heatmap (row 1, col 2)
    n_patterns = len(selected_patterns)
    corr_matrix = np.zeros((n_patterns, n_patterns))
    for i, p1 in enumerate(selected_patterns):
        for j, p2 in enumerate(selected_patterns):
            if p1 in comparison.similarity_metrics and p2 in comparison.similarity_metrics[p1]:
                corr_matrix[i, j] = comparison.similarity_metrics[p1][p2]['correlation']
    
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix,
            x=selected_patterns,
            y=selected_patterns,
            colorscale='RdBu',
            zmid=0,
            showscale=False
        ),
        row=1, col=2
    )
    
    # 3. DTW Distance Matrix (row 1, col 3)
    dtw_matrix = np.zeros((n_patterns, n_patterns))
    for i, p1 in enumerate(selected_patterns):
        for j, p2 in enumerate(selected_patterns):
            if p1 in comparison.similarity_metrics and p2 in comparison.similarity_metrics[p1]:
                dtw_matrix[i, j] = comparison.similarity_metrics[p1][p2]['dtw_distance']
    
    fig.add_trace(
        go.Heatmap(
            z=dtw_matrix,
            x=selected_patterns,
            y=selected_patterns,
            colorscale='Viridis',
            showscale=False
        ),
        row=1, col=3
    )
    
    # 4. Pattern Evolution PCA (row 2, col 1)
    from sklearn.decomposition import PCA
    pattern_matrix = np.array([comparison.pattern_data[pid]['normalized'] for pid in selected_patterns])
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(pattern_matrix)
    
    fig.add_trace(
        go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            mode='markers+text',
            text=selected_patterns,
            textposition="top center",
            marker=dict(size=12, color=np.arange(len(selected_patterns)), colorscale='Viridis'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 5. Feature Statistics (row 2, col 2)
    features = []
    for pattern_id in selected_patterns:
        data = comparison.pattern_data[pattern_id]['normalized']
        features.append(np.std(data))
    
    fig.add_trace(
        go.Bar(
            x=selected_patterns,
            y=features,
            marker_color='lightblue',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 6. Similarity Network placeholder (row 2, col 3)
    # Simplified network visualization
    theta = np.linspace(0, 2*np.pi, len(selected_patterns), endpoint=False)
    x_pos = np.cos(theta)
    y_pos = np.sin(theta)
    
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            text=selected_patterns,
            textposition="top center",
            marker=dict(size=15, color='cyan'),
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Add edges for high correlations
    for i in range(len(selected_patterns)):
        for j in range(i+1, len(selected_patterns)):
            if i < len(selected_patterns) and j < len(selected_patterns):
                p1, p2 = selected_patterns[i], selected_patterns[j]
                if p1 in comparison.similarity_metrics and p2 in comparison.similarity_metrics[p1]:
                    corr = comparison.similarity_metrics[p1][p2]['correlation']
                    if corr > 0.5:
                        fig.add_trace(
                            go.Scatter(
                                x=[x_pos[i], x_pos[j]],
                                y=[y_pos[i], y_pos[j]],
                                mode='lines',
                                line=dict(color='gray', width=corr*3),
                                showlegend=False
                            ),
                            row=2, col=3
                        )
    
    # 7-9. Pairwise comparisons (row 3)
    if len(selected_patterns) >= 2:
        # Pattern 1 vs 2
        data1 = comparison.pattern_data[selected_patterns[0]]['normalized']
        data2 = comparison.pattern_data[selected_patterns[1]]['normalized']
        fig.add_trace(
            go.Scatter(x=data1, y=data2, mode='markers', 
                      marker=dict(color=np.arange(len(data1)), colorscale='Viridis'),
                      showlegend=False),
            row=3, col=1
        )
    
    if len(selected_patterns) >= 3:
        # Pattern 1 vs 3
        data3 = comparison.pattern_data[selected_patterns[2]]['normalized']
        fig.add_trace(
            go.Scatter(x=data1, y=data3, mode='markers',
                      marker=dict(color=np.arange(len(data1)), colorscale='Plasma'),
                      showlegend=False),
            row=3, col=2
        )
    
    # Time evolution
    for i, pattern_id in enumerate(selected_patterns[:3]):  # Limit to 3 for clarity
        data = comparison.pattern_data[pattern_id]['data']
        cumsum = np.cumsum(data - np.mean(data))
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(cumsum)),
                y=cumsum,
                mode='lines',
                name=f"{pattern_id} (cumulative)",
                line=dict(width=2),
                showlegend=False
            ),
            row=3, col=3
        )
    
    # Update layout
    fig.update_layout(
        title="Pattern Comparison Dashboard",
        height=1200,
        showlegend=True,
        template="plotly_dark"
    )
    
    # Save dashboard
    fig.write_html("pattern_comparison_dashboard.html")
    print("✓ Created: pattern_comparison_dashboard.html")


if __name__ == "__main__":
    demonstrate_pattern_comparison()
