"""
Demo script for Pattern Gallery visualization component
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.visualizations.pattern_gallery import PatternGallery
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def demo_pattern_gallery():
    """Demonstrate pattern gallery functionality"""
    print("=== Pattern Gallery Demo ===\n")
    
    # Create gallery instance
    gallery = PatternGallery()
    
    # 1. Basic gallery view
    print("1. Creating basic gallery view...")
    fig_basic = gallery.create_gallery_layout()
    fig_basic.write_html('pattern_gallery_basic.html')
    print("   Saved to pattern_gallery_basic.html")
    
    # 2. Gallery sorted by frequency
    print("\n2. Creating gallery sorted by frequency...")
    fig_frequency = gallery.create_gallery_layout(sort_by='frequency')
    fig_frequency.write_html('pattern_gallery_frequency.html')
    print("   Saved to pattern_gallery_frequency.html")
    
    # 3. Gallery sorted by recency
    print("\n3. Creating gallery sorted by recency...")
    fig_recency = gallery.create_gallery_layout(sort_by='recency')
    fig_recency.write_html('pattern_gallery_recency.html')
    print("   Saved to pattern_gallery_recency.html")
    
    # 4. Filtered gallery - specific pattern type
    print("\n4. Creating filtered gallery (Head and Shoulders patterns)...")
    fig_filtered_type = gallery.create_gallery_layout(
        filter_type='Head and Shoulders'
    )
    fig_filtered_type.write_html('pattern_gallery_filtered_type.html')
    print("   Saved to pattern_gallery_filtered_type.html")
    
    # 5. Filtered gallery - specific ticker
    print("\n5. Creating filtered gallery (BTC/USD patterns)...")
    fig_filtered_ticker = gallery.create_gallery_layout(
        filter_ticker='BTC/USD'
    )
    fig_filtered_ticker.write_html('pattern_gallery_filtered_ticker.html')
    print("   Saved to pattern_gallery_filtered_ticker.html")
    
    # 6. Combined filter and sort
    print("\n6. Creating gallery with combined filters...")
    fig_combined = gallery.create_gallery_layout(
        sort_by='quality',
        filter_ticker='ETH/USD',
        cards_per_row=3
    )
    fig_combined.write_html('pattern_gallery_combined.html')
    print("   Saved to pattern_gallery_combined.html")
    
    # 7. Detailed pattern views
    print("\n7. Creating detailed pattern views...")
    patterns = gallery.pattern_data['patterns'][:3]  # First 3 patterns
    for i, pattern in enumerate(patterns):
        detail_fig = gallery.create_detailed_view(pattern['id'])
        filename = f'pattern_detail_{i+1}.html'
        detail_fig.write_html(filename)
        print(f"   Pattern {i+1} ({pattern['type']}) saved to {filename}")
    
    # 8. Get available filters
    print("\n8. Available filters:")
    print(f"   Pattern types: {gallery.get_pattern_types()}")
    print(f"   Tickers: {gallery.get_tickers()}")
    
    # 9. Demonstrate pattern selection for highlighting
    print("\n9. Pattern selection demo:")
    selected_pattern = gallery.get_pattern_for_highlight('pattern_0')
    if selected_pattern:
        print(f"   Selected pattern: {selected_pattern['type']} on {selected_pattern['ticker']}")
        print(f"   Quality: {selected_pattern['quality_score']:.2%}")
        print(f"   Occurrences: {selected_pattern['occurrence_count']}")
        print(f"   Time range: {selected_pattern['duration_hours']} hours")
    
    # 10. Create a comprehensive demo with multiple views
    print("\n10. Creating comprehensive demo...")
    fig_comprehensive = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Quality Distribution',
            'Frequency Distribution',
            'Pattern Types',
            'Ticker Distribution'
        ),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'domain'}, {'type': 'xy'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Note: For the comprehensive view, we'll create a summary instead
    # of actual subplots since the gallery creates its own subplot structure
    
    # Create summary statistics
    patterns = gallery.pattern_data['patterns']
    
    # Quality distribution
    quality_scores = [p['quality_score'] for p in patterns]
    fig_comprehensive.add_trace(
        go.Histogram(
            x=quality_scores,
            nbinsx=10,
            name='Quality Distribution',
            marker_color='#00D9FF'
        ),
        row=1, col=1
    )
    
    # Frequency distribution
    frequencies = [p['occurrence_count'] for p in patterns]
    fig_comprehensive.add_trace(
        go.Histogram(
            x=frequencies,
            nbinsx=10,
            name='Frequency Distribution',
            marker_color='#FF6B6B'
        ),
        row=1, col=2
    )
    
    # Pattern types pie chart
    type_counts = {}
    for p in patterns:
        type_counts[p['type']] = type_counts.get(p['type'], 0) + 1
    
    fig_comprehensive.add_trace(
        go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            name='Pattern Types'
        ),
        row=2, col=1
    )
    
    # Ticker distribution
    ticker_counts = {}
    for p in patterns:
        ticker_counts[p['ticker']] = ticker_counts.get(p['ticker'], 0) + 1
    
    fig_comprehensive.add_trace(
        go.Bar(
            x=list(ticker_counts.keys()),
            y=list(ticker_counts.values()),
            name='Ticker Distribution',
            marker_color='#4ECDC4'
        ),
        row=2, col=2
    )
    
    fig_comprehensive.update_layout(
        title='Pattern Gallery Statistics',
        height=800,
        showlegend=False,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white')
    )
    
    fig_comprehensive.write_html('pattern_gallery_statistics.html')
    print("   Saved comprehensive statistics to pattern_gallery_statistics.html")
    
    print("\n=== Demo Complete ===")
    print("\nGenerated files:")
    print("- pattern_gallery_basic.html")
    print("- pattern_gallery_frequency.html")
    print("- pattern_gallery_recency.html")
    print("- pattern_gallery_filtered_type.html")
    print("- pattern_gallery_filtered_ticker.html")
    print("- pattern_gallery_combined.html")
    print("- pattern_detail_1.html")
    print("- pattern_detail_2.html")
    print("- pattern_detail_3.html")
    print("- pattern_gallery_statistics.html")
    
    return gallery


def demo_integration_with_main_chart():
    """Demonstrate integration with main chart highlighting"""
    print("\n=== Pattern Gallery Integration Demo ===\n")
    
    gallery = PatternGallery()
    
    # Simulate pattern selection and highlighting
    print("Simulating pattern selection workflow:")
    
    # Get first pattern
    pattern = gallery.pattern_data['patterns'][0]
    pattern_id = pattern['id']
    
    print(f"1. User clicks on pattern: {pattern['type']} ({pattern['ticker']})")
    
    # Get pattern data for highlighting
    highlight_data = gallery.get_pattern_for_highlight(pattern_id)
    
    if highlight_data:
        print(f"2. Pattern data retrieved:")
        print(f"   - Start time: {highlight_data['start_time']}")
        print(f"   - End time: {highlight_data['end_time']}")
        print(f"   - Data points: {len(highlight_data['data'])}")
        
        # Create a mock main chart with pattern highlight
        fig = go.Figure()
        
        # Mock main chart data
        main_data_length = 200
        main_x = list(range(main_data_length))
        main_y = np.cumsum(np.random.randn(main_data_length)) + 100
        
        fig.add_trace(go.Scatter(
            x=main_x,
            y=main_y,
            mode='lines',
            name='Main Chart',
            line=dict(color='#666', width=1)
        ))
        
        # Add pattern highlight
        pattern_start = 50  # Mock position
        pattern_x = list(range(pattern_start, pattern_start + len(highlight_data['data'])))
        pattern_y = [main_y[pattern_start] + (d - 0.5) * 10 for d in highlight_data['data']]
        
        fig.add_trace(go.Scatter(
            x=pattern_x,
            y=pattern_y,
            mode='lines',
            name='Selected Pattern',
            line=dict(color='#00D9FF', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.2)'
        ))
        
        # Add pattern region highlight
        fig.add_vrect(
            x0=pattern_start,
            x1=pattern_start + len(highlight_data['data']),
            fillcolor='rgba(0, 217, 255, 0.1)',
            layer='below',
            line_width=0
        )
        
        fig.update_layout(
            title=f"Main Chart with Highlighted Pattern: {pattern['type']}",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white'),
            hovermode='x unified'
        )
        
        fig.write_html('pattern_gallery_integration.html')
        print("3. Main chart updated with pattern highlight")
        print("   Saved to pattern_gallery_integration.html")
    
    print("\n=== Integration Demo Complete ===")


if __name__ == "__main__":
    # Run main demo
    gallery = demo_pattern_gallery()
    
    # Run integration demo
    demo_integration_with_main_chart()
