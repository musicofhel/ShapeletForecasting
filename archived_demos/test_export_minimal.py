"""Minimal test for export functionality."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.dashboard.export.report_generator import ReportGenerator
    print("✓ Successfully imported ReportGenerator")
    
    # Create a simple pattern
    patterns = [{"id": "1", "ticker": "TEST", "type": "pattern", "strength": 0.8}]
    
    # Create generator
    generator = ReportGenerator()
    print(f"✓ Created ReportGenerator with output_dir: {generator.output_dir}")
    
    # Test CSV export
    from src.dashboard.export.report_generator import ExportFormat
    filepath = generator.export_patterns(patterns, ExportFormat.CSV, "test")
    print(f"✓ CSV export successful: {filepath}")
    
    # Check if file exists
    if os.path.exists(filepath):
        print(f"✓ File exists: {filepath}")
        with open(filepath, 'r') as f:
            content = f.read()
            print(f"✓ File content:\n{content}")
    else:
        print(f"✗ File not found: {filepath}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
