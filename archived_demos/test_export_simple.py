"""Simple test for export functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from src.dashboard.export.report_generator import (
    export_patterns_to_csv,
    export_patterns_to_json,
    ExportFormat,
    ReportGenerator
)

# Create test pattern data
patterns = [
    {
        "id": "TEST_001",
        "ticker": "AAPL",
        "type": "head_shoulders",
        "strength": 0.85,
        "duration": 15,
        "start_date": "2024-01-01",
        "end_date": "2024-01-15"
    },
    {
        "id": "TEST_002",
        "ticker": "GOOGL",
        "type": "double_top",
        "strength": 0.72,
        "duration": 10,
        "start_date": "2024-02-01",
        "end_date": "2024-02-10"
    }
]

print("Testing Export Functionality")
print("=" * 40)

# Test CSV export
try:
    csv_path = export_patterns_to_csv(patterns, "test_export")
    print(f"✓ CSV export successful: {csv_path}")
except Exception as e:
    print(f"✗ CSV export failed: {e}")

# Test JSON export
try:
    json_path = export_patterns_to_json(patterns, "test_export")
    print(f"✓ JSON export successful: {json_path}")
except Exception as e:
    print(f"✗ JSON export failed: {e}")

# Test Excel export
try:
    generator = ReportGenerator()
    excel_path = generator.export_patterns(patterns, ExportFormat.EXCEL, "test_export")
    print(f"✓ Excel export successful: {excel_path}")
except Exception as e:
    print(f"✗ Excel export failed: {e}")

print("\nExport test completed!")
