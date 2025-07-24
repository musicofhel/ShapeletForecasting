"""
Demo script for Forecast and Accuracy Visualizations.

This script demonstrates the combined functionality of:
1. Forecast Visualization (forecast_view.py)
2. Accuracy Metrics Dashboard (accuracy_metrics.py)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# Import our visualization components
from src.dashboard.visualizations.forecast_view import ForecastVisualization
from src.dashboard.visualizations.accuracy_metrics import AccuracyMetricsDashboard
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.pattern_predictor import PatternPredictor


def generate_demo_data():
    """Generate comprehensive demo data for visualizations."""
    
    # Generate price data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Realistic price movement
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 200)
    prices = base_price * np.exp(np.cumsum(returns))
    
    price_data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
        'low': prices * (1 + np.random.uniform(-0.02, 0, 200)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)
    
    # Generate pattern history
    pattern_types = [
        'Head and Shoulders', 'Double Top', 'Double Bottom',
        'Ascending Triangle', 'Descending Triangle',
        'Bull Flag', 'Bear Flag', 'Cup and Handle'
    ]
    
    patterns = []
    for i in range(30):  # Reduced to ensure indices stay within bounds
        start_idx = i * 5
        if start_idx >= 180:  # Ensure we don't exceed data length
            break
        duration = np.random.randint(10, 20)
        end_idx = min(start_idx + duration, 199)  # Cap at max index
        patterns.append({
            'pattern_type': np.random.choice(pattern_types),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_idx,
            'duration': end_idx - start_idx,
            'confidence': np.random.uniform(0.6, 0.95)
        })
    
    # Generate predictions
    predictions = []
    for pattern in pattern_types[:5]:
        predictions.append({
            'pattern_type': pattern,
            'probability': np.random.uniform(0.3, 0.9)
        })
    
    # Generate confidence bands
    time_steps = 20
    confidence_bands = {}
    for pred in predictions[:3]:
        pattern = pred['pattern_type']
        prob = pred['probability']
        confidence_bands[pattern] = {
            'upper': [prob * (1 - t/time_steps) + 0.1 * (1 - t/time_steps) for t in range(time_steps)],
            'lower': [prob * (1 - t/time_steps) - 0.1 * (1 - t/time_steps) for t in range(time_steps)]
        }
    
    # Generate accuracy data
    models = ['LSTM', 'Transformer', 'Markov', 'Ensemble']
    accuracy_data = []
    
    for date in dates[-100:]:
        for model in models:
            base_acc = {'LSTM': 0.75, 'Transformer': 0.78, 'Markov': 0.65, 'Ensemble': 0.80}
            accuracy_data.append({
                'date': date,
                'model': model,
                'accuracy': np.clip(base_acc[model] + np.random.normal(0, 0.05), 0, 1)
            })
    
    # Generate pattern-specific accuracy
    pattern_accuracy = []
    for pattern in pattern_types:
        n_samples = np.random.randint(50, 200)
        accuracy = np.random.uniform(0.65, 0.85)
        pattern_accuracy.append({
            'pattern_type': pattern,
            'accuracy': accuracy,
            'precision': accuracy + np.random.uniform(-0.05, 0.05),
            'recall': accuracy + np.random.uniform(-0.08, 0.02),
            'f1_score': accuracy + np.random.uniform(-0.03, 0.03),
            'support': n_samples
        })
    
    # Generate prediction history for calibration
    n_predictions = 1000
    prediction_history = pd.DataFrame({
        'pattern_type': np.random.choice(pattern_types, n_predictions),
        'probability': np.random.uniform(0.3, 0.95, n_predictions),
        'model': np.random.choice(models, n_predictions),
        'correct': np.random.choice([True, False], n_predictions, p=[0.72, 0.28])
    }, index=pd.date_range(start='2023-01-01', periods=n_predictions, freq='H'))
    
    # Generate error data
    error_data = pd.DataFrame({
        'error': np.random.normal(0, 0.1, 500),
        'confidence': np.random.uniform(0.3, 0.9, 500),
        'hour': np.random.randint(0, 24, 500)
    })
    
    # Generate model metrics
    model_metrics = []
    for model in models + ['Baseline']:
        perf = {
            'LSTM': {'acc': 0.75, 'time': 50},
            'Transformer': {'acc': 0.78, 'time': 100},
            'Markov': {'acc': 0.65, 'time': 10},
            'Ensemble': {'acc': 0.80, 'time': 150},
            'Baseline': {'acc': 0.55, 'time': 5}
        }
        
        model_metrics.append({
            'model': model,
            'accuracy': perf[model]['acc'],
            'precision': perf[model]['acc'] + np.random.uniform(-0.02, 0.02),
            'recall': perf[model]['acc'] + np.random.uniform(-0.05, 0.02),
            'f1_score': perf[model]['acc'] + np.random.uniform(-0.03, 0.03),
            'inference_time': perf[model]['time'],
            'model_size': np.random.randint(10, 100)
        })
    
    return {
        'price_data': price_data,
        'patterns': patterns,
        'predictions': predictions,
        'confidence_bands': confidence_bands,
        'accuracy_data': pd.DataFrame(accuracy_data),
        'pattern_accuracy': pd.DataFrame(pattern_accuracy),
        'prediction_history': prediction_history,
        'error_data': error_data,
        'model_metrics': pd.DataFrame(model_metrics)
    }


def create_forecast_demo(data, forecast_viz):
    """Create forecast visualization demos."""
    figures = []
    
    # 1. Current Context View
    current_pattern = data['patterns'][-1]
    historical_patterns = data['patterns'][:-1]
    
    fig1 = forecast_viz.create_current_context_view(
        current_pattern=current_pattern,
        historical_patterns=historical_patterns,
        price_data=data['price_data']
    )
    figures.append(('forecast_context', fig1))
    
    # 2. Prediction Visualization
    fig2 = forecast_viz.create_prediction_visualization(
        predictions=data['predictions'],
        confidence_bands=data['confidence_bands'],
        time_horizon=20
    )
    figures.append(('forecast_predictions', fig2))
    
    # 3. Scenario Analysis
    scenarios = [
        {
            'name': 'Bullish Breakout',
            'pattern_sequence': ['Bull Flag', 'Ascending Triangle', 'Cup and Handle'],
            'price_path': [100, 102, 105, 108, 112, 118, 125],
            'probability': 0.65
        },
        {
            'name': 'Consolidation',
            'pattern_sequence': ['Triangle', 'Range', 'Triangle'],
            'price_path': [100, 101, 99, 100, 98, 101, 100],
            'probability': 0.25
        },
        {
            'name': 'Bearish Reversal',
            'pattern_sequence': ['Double Top', 'Bear Flag', 'Head and Shoulders'],
            'price_path': [100, 98, 95, 92, 88, 85, 80],
            'probability': 0.10
        }
    ]
    
    fig3 = forecast_viz.create_scenario_analysis(scenarios, current_price=100)
    figures.append(('forecast_scenarios', fig3))
    
    # 4. Historical Accuracy Overlay
    actual_patterns = data['prediction_history'][data['prediction_history']['correct'] == True][['pattern_type']]
    
    fig4 = forecast_viz.create_historical_accuracy_overlay(
        predictions_history=data['prediction_history'],
        actual_patterns=actual_patterns
    )
    figures.append(('forecast_accuracy_overlay', fig4))
    
    # 5. Confidence Calibration
    fig5 = forecast_viz.create_confidence_calibration_plot(data['prediction_history'])
    figures.append(('forecast_calibration', fig5))
    
    return figures


def create_accuracy_demo(data, metrics_dashboard):
    """Create accuracy metrics dashboard demos."""
    figures = []
    
    # 1. Accuracy Over Time
    fig1 = metrics_dashboard.create_accuracy_over_time(
        accuracy_data=data['accuracy_data'],
        window_sizes=[7, 30, 90]
    )
    figures.append(('accuracy_over_time', fig1))
    
    # 2. Pattern Type Accuracy
    fig2 = metrics_dashboard.create_pattern_type_accuracy(
        pattern_accuracy=data['pattern_accuracy']
    )
    figures.append(('pattern_accuracy', fig2))
    
    # 3. Confidence Calibration
    fig3 = metrics_dashboard.create_confidence_calibration(
        predictions=data['prediction_history']
    )
    figures.append(('confidence_calibration', fig3))
    
    # 4. Error Distribution
    fig4 = metrics_dashboard.create_error_distribution(
        errors=data['error_data']
    )
    figures.append(('error_distribution', fig4))
    
    # 5. Model Comparison
    fig5 = metrics_dashboard.create_model_comparison(
        model_metrics=data['model_metrics']
    )
    figures.append(('model_comparison', fig5))
    
    # 6. Summary Metrics Card
    summary_metrics = {
        'overall_accuracy': 0.756,
        'best_model': 'Ensemble',
        'avg_confidence': 0.682,
        'total_predictions': 12345
    }
    fig6 = metrics_dashboard.create_summary_metrics_card(summary_metrics)
    figures.append(('summary_metrics', fig6))
    
    return figures


def create_combined_dashboard(forecast_figures, accuracy_figures):
    """Create a combined dashboard with all visualizations."""
    
    # Create HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wavelet Pattern Forecasting Dashboard - Visualizations Demo</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section {{
                margin-bottom: 40px;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .plot-container {{
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                border: 1px solid #e9ecef;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }}
            .stat-label {{
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Wavelet Pattern Forecasting Dashboard</h1>
            <p>Comprehensive Visualization Demo - Forecast Views & Accuracy Metrics</p>
        </div>
        
        <div class="section">
            <h2>📊 Forecast Visualizations</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">5</div>
                    <div class="stat-label">Forecast Views</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">8</div>
                    <div class="stat-label">Pattern Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">20</div>
                    <div class="stat-label">Time Horizon</div>
                </div>
            </div>
            {forecast_plots}
        </div>
        
        <div class="section">
            <h2>📈 Accuracy Metrics Dashboard</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">75.6%</div>
                    <div class="stat-label">Overall Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">4</div>
                    <div class="stat-label">Models Compared</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">1000+</div>
                    <div class="stat-label">Predictions Analyzed</div>
                </div>
            </div>
            {accuracy_plots}
        </div>
        
        <div class="section">
            <h2>✅ Implementation Status</h2>
            <p><strong>Completed Components:</strong></p>
            <ul>
                <li>✅ Forecast Visualization (forecast_view.py)</li>
                <li>✅ Accuracy Metrics Dashboard (accuracy_metrics.py)</li>
                <li>✅ Comprehensive test suites for both components</li>
                <li>✅ Integration with Pattern Classifier and Predictor</li>
                <li>✅ Performance optimizations</li>
            </ul>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>Current pattern context with price overlay</li>
                <li>Multi-scenario predictions with confidence bands</li>
                <li>Historical accuracy tracking</li>
                <li>Model performance comparison</li>
                <li>Error distribution analysis</li>
                <li>Confidence calibration plots</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Generate plot HTML
    forecast_html = ""
    for name, fig in forecast_figures:
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        forecast_html += f'<div class="plot-container"><h3>{name.replace("_", " ").title()}</h3>{plot_html}</div>'
    
    accuracy_html = ""
    for name, fig in accuracy_figures:
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        accuracy_html += f'<div class="plot-container"><h3>{name.replace("_", " ").title()}</h3>{plot_html}</div>'
    
    # Fill template
    final_html = html_template.format(
        forecast_plots=forecast_html,
        accuracy_plots=accuracy_html
    )
    
    return final_html


def main():
    """Run the complete visualization demo."""
    print("🚀 Starting Forecast & Accuracy Visualizations Demo...")
    
    # Initialize components
    print("📦 Initializing visualization components...")
    forecast_viz = ForecastVisualization()
    metrics_dashboard = AccuracyMetricsDashboard()
    
    # Generate demo data
    print("📊 Generating demo data...")
    data = generate_demo_data()
    
    # Create visualizations
    print("🎨 Creating forecast visualizations...")
    forecast_figures = create_forecast_demo(data, forecast_viz)
    
    print("📈 Creating accuracy metrics visualizations...")
    accuracy_figures = create_accuracy_demo(data, metrics_dashboard)
    
    # Create combined dashboard
    print("🔧 Building combined dashboard...")
    dashboard_html = create_combined_dashboard(forecast_figures, accuracy_figures)
    
    # Save and display
    output_file = "forecast_accuracy_dashboard_demo.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"✅ Dashboard saved to {output_file}")
    
    # Open in browser
    abs_path = os.path.abspath(output_file)
    webbrowser.open(f'file://{abs_path}')
    print("🌐 Opening dashboard in browser...")
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION DEMO SUMMARY")
    print("="*60)
    print(f"✅ Forecast Visualizations: {len(forecast_figures)} created")
    print(f"✅ Accuracy Visualizations: {len(accuracy_figures)} created")
    print(f"✅ Total Visualizations: {len(forecast_figures) + len(accuracy_figures)}")
    print("\nKey Features Demonstrated:")
    print("- Pattern context visualization with price overlay")
    print("- Multi-step predictions with confidence bands")
    print("- Scenario analysis with probability distributions")
    print("- Historical accuracy tracking")
    print("- Model performance comparison")
    print("- Error distribution analysis")
    print("- Confidence calibration plots")
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
