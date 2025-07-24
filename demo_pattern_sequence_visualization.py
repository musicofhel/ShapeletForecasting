"""
Demo: Pattern Sequence Visualization

This demo showcases the pattern sequence visualization capabilities,
including timeline views, transition matrices, and flow diagrams.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer


def generate_sample_price_data(n_points=1000):
    """Generate sample price data with patterns"""
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='1H')
    
    # Generate price with trends and patterns
    trend = np.linspace(100, 120, n_points)
    noise = np.random.randn(n_points) * 2
    
    # Add some patterns
    price = trend + noise
    
    # Add head and shoulders pattern
    hs_start = 200
    hs_width = 50
    price[hs_start:hs_start+hs_width] += np.concatenate([
        np.linspace(0, 10, hs_width//3),
        np.linspace(10, 5, hs_width//3),
        np.linspace(5, 0, hs_width//3 + hs_width%3)
    ])
    
    # Add double top
    dt_start = 400
    dt_width = 40
    price[dt_start:dt_start+dt_width] += np.concatenate([
        np.linspace(0, 8, dt_width//4),
        np.linspace(8, 4, dt_width//4),
        np.linspace(4, 8, dt_width//4),
        np.linspace(8, 0, dt_width//4)
    ])
    
    # Add triangle pattern
    tr_start = 600
    tr_width = 60
    triangle = []
    for i in range(tr_width):
        triangle.append(5 * (1 - i/tr_width) * np.sin(i * np.pi / 5))
    price[tr_start:tr_start+tr_width] += triangle
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': price
    })


def create_pattern_sequence_demo():
    """Create comprehensive pattern sequence visualization demo"""
    print("Pattern Sequence Visualization Demo")
    print("=" * 50)
    
    # Initialize components
    visualizer = PatternSequenceVisualizer()
    classifier = PatternClassifier()
    analyzer = WaveletSequenceAnalyzer()
    
    # Generate sample data
    print("\n1. Generating sample price data...")
    df = generate_sample_price_data(1000)
    
    # Create realistic pattern data with classification
    print("\n2. Creating sample patterns...")
    base_time = df['timestamp'].iloc[0]
    patterns = []
    
    # Pattern types to cycle through
    pattern_types = [
        'head_shoulders', 'double_top', 'double_bottom',
        'triangle_ascending', 'triangle_descending',
        'flag_bull', 'flag_bear', 'wedge_rising', 'wedge_falling'
    ]
    
    # Create sample patterns at specific locations
    pattern_locations = [
        (50, 100), (150, 200), (250, 300), (350, 400), (450, 500),
        (550, 600), (650, 700), (750, 800), (850, 900), (200, 250),
        (300, 350), (400, 450), (500, 550), (600, 650), (700, 750)
    ]
    
    # Create patterns from predefined locations
    for i, (start_idx, end_idx) in enumerate(pattern_locations[:15]):
        pattern_type = pattern_types[i % len(pattern_types)]
        
        # Extract pattern data
        pattern_times = df['timestamp'].iloc[start_idx:end_idx]
        
        # Calculate confidence based on pattern quality
        confidence = np.random.uniform(0.65, 0.95)
        
        patterns.append({
            'id': f'P{i+1:03d}',
            'type': pattern_type,
            'start_time': pattern_times.iloc[0],
            'end_time': pattern_times.iloc[-1],
            'confidence': confidence,
            'ticker': 'DEMO',
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    # Calculate transitions
    print("\n3. Calculating pattern transitions...")
    transitions = {}
    
    for i in range(len(patterns) - 1):
        trans_key = f"{patterns[i]['id']}_to_{patterns[i+1]['id']}"
        # Use actual transition probabilities if available
        prob = np.random.uniform(0.3, 0.9)
        transitions[trans_key] = {'probability': prob}
    
    # Create transition matrix data for heatmap
    transition_matrix_data = {}
    for i, p1 in enumerate(pattern_types):
        for j, p2 in enumerate(pattern_types):
            if i != j:
                prob = np.random.uniform(0.1, 0.8)
                if prob > 0.3:  # Only include significant transitions
                    transition_matrix_data[(p1, p2)] = prob
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Price Series with Patterns',
            'Pattern Sequence Timeline',
            'Pattern Transition Matrix',
            'Pattern Flow Diagram',
            'Pattern Duration Analysis',
            'Confidence Timeline'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "sankey"}],
            [{"type": "box"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Price series with pattern highlights
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='lightgray', width=1)
        ),
        row=1, col=1
    )
    
    # Highlight patterns on price chart
    colors = list(visualizer.color_palette.values())
    for i, pattern in enumerate(patterns[:8]):  # Show first 8 patterns
        pattern_df = df.iloc[pattern['start_idx']:pattern['end_idx']]
        fig.add_trace(
            go.Scatter(
                x=pattern_df['timestamp'],
                y=pattern_df['price'],
                mode='lines',
                name=f"{pattern['type']} ({pattern['id']})",
                line=dict(color=colors[i % len(colors)], width=3),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Pattern sequence timeline
    timeline_fig = visualizer.create_pattern_timeline(
        patterns, 
        transitions=transitions,
        show_probabilities=True
    )
    
    # Add timeline traces to subplot
    for trace in timeline_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # 3. Transition matrix
    matrix_fig = visualizer.create_transition_matrix(
        transition_matrix_data,
        min_probability=0.3
    )
    
    # Add heatmap to subplot
    fig.add_trace(matrix_fig.data[0], row=2, col=1)
    
    # 4. Pattern flow diagram
    flow_transitions = {}
    for i, pattern in enumerate(patterns[:5]):
        if i < len(patterns) - 1:
            flow_transitions[pattern['id']] = [
                {
                    'to_pattern': patterns[i+1]['type'],
                    'probability': transitions.get(
                        f"{pattern['id']}_to_{patterns[i+1]['id']}", 
                        {}
                    ).get('probability', 0.5)
                }
            ]
    
    flow_fig = visualizer.create_pattern_flow_diagram(
        patterns[:5],
        flow_transitions,
        max_depth=3
    )
    
    # Add Sankey to subplot
    fig.add_trace(flow_fig.data[0], row=2, col=2)
    
    # 5. Pattern duration analysis
    duration_fig = visualizer.create_pattern_duration_analysis(patterns)
    
    # Add box plots to subplot
    for trace in duration_fig.data:
        fig.add_trace(trace, row=3, col=1)
    
    # 6. Confidence timeline
    confidence_fig = visualizer.create_confidence_timeline(patterns)
    
    # Add confidence traces to subplot
    for trace in confidence_fig.data:
        fig.add_trace(trace, row=3, col=2)
    
    # Update layout
    fig.update_layout(
        height=1800,
        title_text="Pattern Sequence Visualization Demo",
        title_font_size=24,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    # Save and display
    output_file = "pattern_sequence_visualization_demo.html"
    fig.write_html(output_file)
    print(f"\n5. Visualization saved to: {output_file}")
    
    # Create individual visualizations
    print("\n6. Creating individual visualizations...")
    
    # Pattern timeline with all features
    timeline_full = visualizer.create_pattern_timeline(
        patterns,
        transitions=transitions,
        show_probabilities=True,
        height=600
    )
    timeline_full.update_layout(title="Complete Pattern Sequence Timeline")
    timeline_full.write_html("pattern_timeline_full.html")
    
    # Transition matrix
    matrix_full = visualizer.create_transition_matrix(
        transition_matrix_data,
        min_probability=0.2
    )
    matrix_full.write_html("pattern_transition_matrix.html")
    
    # Pattern flow
    flow_full = visualizer.create_pattern_flow_diagram(
        patterns[:10],
        flow_transitions,
        max_depth=4
    )
    flow_full.write_html("pattern_flow_diagram.html")
    
    print("\nIndividual visualizations saved:")
    print("  - pattern_timeline_full.html")
    print("  - pattern_transition_matrix.html")
    print("  - pattern_flow_diagram.html")
    
    # Print summary statistics
    print("\n7. Pattern Sequence Statistics:")
    print(f"   - Total patterns detected: {len(patterns)}")
    print(f"   - Pattern types: {len(set(p['type'] for p in patterns))}")
    print(f"   - Average confidence: {np.mean([p['confidence'] for p in patterns]):.2%}")
    print(f"   - Transitions analyzed: {len(transitions)}")
    
    # Calculate pattern type distribution
    type_counts = {}
    for pattern in patterns:
        ptype = pattern['type']
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    
    print("\n   Pattern Type Distribution:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {ptype}: {count} ({count/len(patterns)*100:.1f}%)")
    
    # Open main visualization
    webbrowser.open(f'file://{os.path.abspath(output_file)}')
    
    return patterns, transitions


if __name__ == "__main__":
    patterns, transitions = create_pattern_sequence_demo()
    print("\nDemo completed successfully!")
    print("\nPattern sequence visualization is now integrated and ready for use in the dashboard.")
