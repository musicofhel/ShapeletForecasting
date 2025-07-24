"""
Demo script for Pattern Analytics Module

Shows comprehensive analytics including:
- Pattern frequency analysis
- Quality distribution
- Prediction accuracy
- Trading signals
- Risk metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.dashboard.visualizations.analytics import PatternAnalytics, create_sample_data
import webbrowser
import os


def main():
    """Run pattern analytics demo"""
    
    print("Pattern Analytics Demo")
    print("=" * 50)
    
    # Create sample data with more patterns for better visualization
    print("\nGenerating sample pattern data...")
    patterns = create_sample_data(n_patterns=1000)
    
    # Initialize analytics
    analytics = PatternAnalytics()
    
    # 1. Generate Analytics Report
    print("\nGenerating analytics report...")
    report = analytics.generate_analytics_report(patterns)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    
    summary = report['summary']
    print(f"\nTotal Patterns Detected: {summary['total_patterns']:,}")
    print(f"Unique Tickers Analyzed: {summary['unique_tickers']}")
    print(f"Pattern Types Identified: {summary['pattern_types']}")
    print(f"Analysis Period: {summary['date_range']['start']} to {summary['date_range']['end']} ({summary['date_range']['days']} days)")
    
    print(f"\nQuality Metrics:")
    print(f"  - Average Quality Score: {summary['quality_metrics']['avg_quality']:.3f}")
    print(f"  - High Quality Patterns (>0.7): {summary['quality_metrics']['high_quality_patterns']} ({summary['quality_metrics']['high_quality_patterns']/summary['total_patterns']*100:.1f}%)")
    print(f"  - Low Quality Patterns (<0.3): {summary['quality_metrics']['low_quality_patterns']} ({summary['quality_metrics']['low_quality_patterns']/summary['total_patterns']*100:.1f}%)")
    
    if summary['performance_metrics']:
        print(f"\nPrediction Performance:")
        print(f"  - Predictions Made: {summary['performance_metrics']['predictions_made']}")
        print(f"  - Overall Accuracy: {summary['performance_metrics']['accuracy']:.1%}")
        print(f"  - Average Return: {summary['performance_metrics']['avg_return']:.2%}")
        print(f"  - Total Return: {summary['performance_metrics']['total_return']:.2%}")
    
    # 2. Pattern Frequency Analysis
    print("\n" + "="*50)
    print("PATTERN FREQUENCY ANALYSIS")
    print("="*50)
    
    freq_analysis = report['frequency_analysis']
    
    print("\nPattern Frequency by Type:")
    for pattern_type, count in sorted(freq_analysis['type_frequency'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {pattern_type}: {count} ({count/summary['total_patterns']*100:.1f}%)")
    
    print("\nPattern Frequency by Ticker:")
    for ticker, count in sorted(freq_analysis['ticker_frequency'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {ticker}: {count} patterns")
    
    # 3. Quality Analysis
    print("\n" + "="*50)
    print("QUALITY ANALYSIS")
    print("="*50)
    
    quality_analysis = report['quality_analysis']
    quality_stats = quality_analysis['statistics']
    
    print(f"\nQuality Score Statistics:")
    print(f"  - Mean: {quality_stats['mean']:.3f}")
    print(f"  - Std Dev: {quality_stats['std']:.3f}")
    print(f"  - Median: {quality_stats['median']:.3f}")
    print(f"  - Q1: {quality_stats['q1']:.3f}")
    print(f"  - Q3: {quality_stats['q3']:.3f}")
    
    print("\nAverage Quality by Pattern Type:")
    quality_by_type = quality_analysis['by_type'].sort_values('mean', ascending=False)
    for pattern_type, row in quality_by_type.iterrows():
        print(f"  - {pattern_type}: {row['mean']:.3f} (±{row['std']:.3f}, n={int(row['count'])})")
    
    # 4. Risk Analysis
    print("\n" + "="*50)
    print("RISK METRICS BY PATTERN TYPE")
    print("="*50)
    
    risk_analysis = report['risk_analysis']
    
    # Create risk summary table
    risk_summary = []
    for pattern_type, metrics in risk_analysis.items():
        risk_summary.append({
            'Pattern': pattern_type,
            'Avg Return': f"{metrics['avg_return']:.2%}",
            'Volatility': f"{metrics['volatility']:.2%}",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'Profit Factor': f"{metrics['profit_factor']:.2f}",
            'VaR 95%': f"{metrics['var_95']:.2%}"
        })
    
    risk_df = pd.DataFrame(risk_summary)
    print("\n" + risk_df.to_string(index=False))
    
    # 5. Trading Signals Summary
    print("\n" + "="*50)
    print("TRADING SIGNALS SUMMARY")
    print("="*50)
    
    signals = pd.DataFrame(report['signals'])
    signal_counts = signals['signal_type'].value_counts()
    
    print("\nSignal Distribution:")
    for signal_type, count in signal_counts.items():
        print(f"  - {signal_type}: {count} ({count/len(signals)*100:.1f}%)")
    
    # Average risk-reward by signal type
    print("\nAverage Risk-Reward Ratio by Signal Type:")
    for signal_type in signal_counts.index:
        avg_rr = signals[signals['signal_type'] == signal_type]['risk_reward_ratio'].mean()
        print(f"  - {signal_type}: {avg_rr:.2f}")
    
    # 6. Create Visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    dashboard = analytics.create_comprehensive_dashboard(patterns)
    
    # Save all visualizations
    for name, fig in dashboard.items():
        filename = f"pattern_analytics_{name}.html"
        fig.write_html(filename)
        print(f"\n✓ Created {name} visualization")
        print(f"  Saved to: {filename}")
    
    # 7. Create combined dashboard
    print("\n" + "="*50)
    print("CREATING COMBINED DASHBOARD")
    print("="*50)
    
    create_combined_dashboard(dashboard)
    print("\n✓ Created combined analytics dashboard")
    print("  Saved to: pattern_analytics_dashboard.html")
    
    # Open the combined dashboard
    print("\nOpening combined dashboard in browser...")
    webbrowser.open('file://' + os.path.realpath('pattern_analytics_dashboard.html'))
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("="*50)


def create_combined_dashboard(dashboard: dict):
    """Create a combined HTML dashboard with all visualizations"""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pattern Analytics Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header h1 {
                color: #333;
                margin: 0 0 10px 0;
            }
            .header p {
                color: #666;
                margin: 0;
            }
            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .chart-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chart-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }
            .nav-menu {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .nav-menu a {
                display: inline-block;
                margin: 0 10px;
                padding: 8px 16px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .nav-menu a:hover {
                background-color: #0056b3;
            }
            .timestamp {
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Pattern Analytics Dashboard</h1>
            <p>Comprehensive analysis of trading patterns including frequency, quality, predictions, and risk metrics</p>
        </div>
        
        <div class="nav-menu">
            <a href="#frequency">Frequency Timeline</a>
            <a href="#quality">Quality Trends</a>
            <a href="#predictions">Predictions</a>
            <a href="#correlation">Correlations</a>
            <a href="#risk">Risk Metrics</a>
            <a href="#signals">Trading Signals</a>
        </div>
        
        <div class="dashboard-grid">
            <div id="frequency" class="chart-container">
                <div class="chart-title">Pattern Frequency Timeline</div>
                <div id="frequency_timeline"></div>
            </div>
            
            <div id="quality" class="chart-container">
                <div class="chart-title">Quality Score Analysis</div>
                <div id="quality_trends"></div>
            </div>
            
            <div id="predictions" class="chart-container">
                <div class="chart-title">Prediction Performance</div>
                <div id="prediction_performance"></div>
            </div>
            
            <div id="correlation" class="chart-container">
                <div class="chart-title">Pattern Correlations</div>
                <div id="correlation_matrix"></div>
            </div>
            
            <div id="risk" class="chart-container">
                <div class="chart-title">Risk Metrics</div>
                <div id="risk_metrics"></div>
            </div>
            
            <div id="signals" class="chart-container">
                <div class="chart-title">Trading Signals</div>
                <div id="signal_dashboard"></div>
            </div>
        </div>
        
        <div class="timestamp">
            Generated on: {timestamp}
        </div>
        
        <script>
            {scripts}
        </script>
    </body>
    </html>
    """
    
    # Generate scripts for each chart
    scripts = []
    for name, fig in dashboard.items():
        # Get the plotly JSON
        fig_json = fig.to_json()
        
        # Create script to render the chart
        script = f"""
            var {name}_data = {fig_json};
            Plotly.newPlot('{name}', {name}_data.data, {name}_data.layout);
        """
        scripts.append(script)
    
    # Combine all scripts
    all_scripts = '\n'.join(scripts)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create final HTML
    html_content = html_template.format(
        scripts=all_scripts,
        timestamp=timestamp
    )
    
    # Save to file
    with open('pattern_analytics_dashboard.html', 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
