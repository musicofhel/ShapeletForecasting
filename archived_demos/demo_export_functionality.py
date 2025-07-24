"""Demo script for pattern export functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.dashboard.export.report_generator import (
    ReportGenerator,
    ExportFormat,
    ReportConfig,
    export_patterns_to_csv,
    export_patterns_to_json,
    generate_pdf_report,
    create_dashboard_snapshot,
    export_pattern_templates,
    batch_export_patterns
)


def generate_sample_patterns(ticker: str, count: int = 10) -> list:
    """Generate sample pattern data for testing."""
    patterns = []
    pattern_types = ["head_shoulders", "double_top", "triangle", "flag", "wedge"]
    
    for i in range(count):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
        duration = np.random.randint(5, 30)
        
        pattern = {
            "id": f"{ticker}_{i:04d}",
            "ticker": ticker,
            "type": np.random.choice(pattern_types),
            "start_date": start_date.isoformat(),
            "end_date": (start_date + timedelta(days=duration)).isoformat(),
            "duration": duration,
            "scale": np.random.randint(1, 10),
            "strength": np.random.uniform(0.5, 1.0),
            "confidence": np.random.uniform(0.6, 0.95),
            "frequency": np.random.uniform(0.1, 0.9),
            "min_correlation": 0.7,
            "scale_range": [1, 10],
            "performance": {
                "accuracy": np.random.uniform(0.6, 0.9),
                "profit_factor": np.random.uniform(1.2, 2.5),
                "win_rate": np.random.uniform(0.5, 0.7)
            },
            "statistics": {
                "mean_return": np.random.uniform(-0.02, 0.05),
                "volatility": np.random.uniform(0.1, 0.3),
                "sharpe_ratio": np.random.uniform(0.5, 2.0)
            }
        }
        patterns.append(pattern)
        
    return patterns


def demo_csv_export():
    """Demo CSV export functionality."""
    print("\n=== CSV Export Demo ===")
    
    patterns = generate_sample_patterns("AAPL", 20)
    filepath = export_patterns_to_csv(patterns, "demo_patterns_csv")
    
    print(f"✓ Exported {len(patterns)} patterns to CSV")
    print(f"  File: {filepath}")
    
    # Read and display first few rows
    df = pd.read_csv(filepath)
    print(f"\nFirst 5 rows of exported CSV:")
    print(df.head())
    

def demo_json_export():
    """Demo JSON export functionality."""
    print("\n=== JSON Export Demo ===")
    
    patterns = generate_sample_patterns("GOOGL", 15)
    filepath = export_patterns_to_json(patterns, "demo_patterns_json")
    
    print(f"✓ Exported {len(patterns)} patterns to JSON")
    print(f"  File: {filepath}")
    
    # Read and display structure
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    print(f"\nJSON structure:")
    print(f"  - Metadata: {list(data['metadata'].keys())}")
    print(f"  - Pattern count: {data['metadata']['pattern_count']}")
    print(f"  - First pattern keys: {list(data['patterns'][0].keys())}")


def demo_pdf_export():
    """Demo PDF export functionality."""
    print("\n=== PDF Export Demo ===")
    
    patterns = generate_sample_patterns("MSFT", 50)
    
    # Configure report
    config = ReportConfig(
        title="Microsoft Pattern Analysis Report",
        author="Pattern Dashboard System",
        include_charts=True,
        include_statistics=True,
        include_patterns=True,
        chart_width=6,
        chart_height=4
    )
    
    try:
        filepath = generate_pdf_report(patterns, "demo_patterns_report", config)
        print(f"✓ Generated PDF report with {len(patterns)} patterns")
        print(f"  File: {filepath}")
        print(f"  Includes: Charts, Statistics, Pattern Table")
    except ImportError as e:
        print(f"⚠ PDF generation requires reportlab: {e}")


def demo_excel_export():
    """Demo Excel export functionality."""
    print("\n=== Excel Export Demo ===")
    
    patterns = generate_sample_patterns("TSLA", 30)
    
    generator = ReportGenerator()
    filepath = generator.export_patterns(patterns, ExportFormat.EXCEL, "demo_patterns_excel")
    
    print(f"✓ Exported {len(patterns)} patterns to Excel")
    print(f"  File: {filepath}")
    print(f"  Sheets: Patterns, Summary, Ticker Analysis")


def demo_template_export():
    """Demo pattern template export."""
    print("\n=== Pattern Template Export Demo ===")
    
    patterns = generate_sample_patterns("BTC", 10)
    filepath = export_pattern_templates(patterns, "demo_bitcoin_templates")
    
    print(f"✓ Exported {len(patterns)} patterns as templates")
    print(f"  File: {filepath}")
    
    # Read and display template structure
    import json
    with open(filepath, 'r') as f:
        templates = json.load(f)
        
    print(f"\nTemplate structure:")
    print(f"  - Version: {templates['version']}")
    print(f"  - Template count: {templates['template_count']}")
    print(f"  - First template:")
    first_template = templates['templates'][0]
    print(f"    - Name: {first_template['name']}")
    print(f"    - Type: {first_template['type']}")
    print(f"    - Parameters: {list(first_template['parameters'].keys())}")


def demo_dashboard_snapshot():
    """Demo dashboard snapshot creation."""
    print("\n=== Dashboard Snapshot Demo ===")
    
    # Create sample dashboard state
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    values = 100 + np.cumsum(np.random.randn(100) * 2)
    
    patterns = generate_sample_patterns("SPY", 5)
    
    dashboard_state = {
        "title": "S&P 500 Pattern Analysis",
        "ticker": "SPY",
        "timeframe": "Daily",
        "pattern_count": len(patterns),
        "accuracy": "87.5%",
        "timeseries": {
            "dates": dates.strftime('%Y-%m-%d').tolist(),
            "values": values.tolist()
        },
        "patterns": [
            {
                "dates": dates[i:i+20].strftime('%Y-%m-%d').tolist(),
                "values": (values[i:i+20] + np.random.randn(20) * 5).tolist()
            }
            for i in range(0, 80, 20)
        ]
    }
    
    filepath = create_dashboard_snapshot(dashboard_state, "demo_dashboard_snapshot")
    
    print(f"✓ Created dashboard snapshot")
    print(f"  File: {filepath}")
    print(f"  Features: Interactive chart, Metrics, Pattern overlays")


def demo_batch_export():
    """Demo batch export for multiple tickers."""
    print("\n=== Batch Export Demo ===")
    
    # Generate patterns for multiple tickers
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    ticker_patterns = {}
    
    for ticker in tickers:
        ticker_patterns[ticker] = generate_sample_patterns(ticker, np.random.randint(10, 30))
    
    # Batch export in JSON format
    export_paths = batch_export_patterns(ticker_patterns, ExportFormat.JSON)
    
    print(f"✓ Batch exported patterns for {len(tickers)} tickers")
    print(f"\nExport summary:")
    for ticker, path in export_paths.items():
        if ticker != "_summary":
            pattern_count = len(ticker_patterns.get(ticker, []))
            print(f"  - {ticker}: {pattern_count} patterns → {path}")
    
    print(f"\n  Summary file: {export_paths['_summary']}")


def demo_custom_export():
    """Demo custom export with specific configuration."""
    print("\n=== Custom Export Demo ===")
    
    # Generate patterns
    patterns = generate_sample_patterns("ETH", 25)
    
    # Create custom report generator
    generator = ReportGenerator(output_dir="custom_exports")
    
    # Export with custom configuration
    config = ReportConfig(
        title="Ethereum Pattern Analysis - Q1 2024",
        author="Crypto Analysis Team",
        include_charts=True,
        include_statistics=True,
        include_patterns=True,
        include_predictions=False,
        page_size="A4",
        chart_width=7,
        chart_height=5
    )
    
    # Export in multiple formats
    formats = [ExportFormat.CSV, ExportFormat.JSON, ExportFormat.EXCEL]
    
    for format in formats:
        try:
            filepath = generator.export_patterns(
                patterns, 
                format, 
                f"ethereum_patterns_{format.value}",
                config
            )
            print(f"✓ Exported to {format.value}: {filepath}")
        except Exception as e:
            print(f"⚠ Error exporting to {format.value}: {e}")


def main():
    """Run all export demos."""
    print("Pattern Export Functionality Demo")
    print("=" * 50)
    
    # Create exports directory
    os.makedirs("exports", exist_ok=True)
    os.makedirs("custom_exports", exist_ok=True)
    
    # Run demos
    demo_csv_export()
    demo_json_export()
    demo_pdf_export()
    demo_excel_export()
    demo_template_export()
    demo_dashboard_snapshot()
    demo_batch_export()
    demo_custom_export()
    
    print("\n" + "=" * 50)
    print("✓ Export functionality demo completed!")
    print("\nCheck the 'exports' and 'custom_exports' directories for generated files.")


if __name__ == "__main__":
    main()
