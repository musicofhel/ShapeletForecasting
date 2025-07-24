"""Report generator for exporting pattern analysis data."""

import json
import csv
import base64
from io import BytesIO, StringIO
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Plotting imports for charts in PDF
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly for interactive snapshots
import plotly.graph_objects as go
import plotly.io as pio


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    TEMPLATE = "template"


class ReportConfig:
    """Configuration for report generation."""
    
    def __init__(
        self,
        title: str = "Pattern Analysis Report",
        author: str = "Pattern Dashboard",
        include_charts: bool = True,
        include_statistics: bool = True,
        include_patterns: bool = True,
        include_predictions: bool = True,
        page_size: str = "letter",
        chart_width: int = 6,
        chart_height: int = 4
    ):
        self.title = title
        self.author = author
        self.include_charts = include_charts
        self.include_statistics = include_statistics
        self.include_patterns = include_patterns
        self.include_predictions = include_predictions
        self.page_size = letter if page_size == "letter" else A4
        self.chart_width = chart_width
        self.chart_height = chart_height
        self.timestamp = datetime.now()


class ReportGenerator:
    """Main class for generating reports and exports."""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_patterns(
        self,
        patterns: List[Dict[str, Any]],
        format: ExportFormat,
        filename: Optional[str] = None,
        config: Optional[ReportConfig] = None
    ) -> str:
        """Export patterns in specified format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patterns_export_{timestamp}"
            
        if format == ExportFormat.CSV:
            return self._export_csv(patterns, filename)
        elif format == ExportFormat.JSON:
            return self._export_json(patterns, filename)
        elif format == ExportFormat.PDF:
            return self._export_pdf(patterns, filename, config or ReportConfig())
        elif format == ExportFormat.EXCEL:
            return self._export_excel(patterns, filename)
        elif format == ExportFormat.TEMPLATE:
            return self._export_template(patterns, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _export_csv(self, patterns: List[Dict[str, Any]], filename: str) -> str:
        """Export patterns to CSV format."""
        filepath = self.output_dir / f"{filename}.csv"
        
        # Flatten nested dictionaries for CSV export
        flattened_patterns = []
        for pattern in patterns:
            flat_pattern = self._flatten_dict(pattern)
            flattened_patterns.append(flat_pattern)
            
        # Create DataFrame and export
        df = pd.DataFrame(flattened_patterns)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
        
    def _export_json(self, patterns: List[Dict[str, Any]], filename: str) -> str:
        """Export patterns to JSON format."""
        filepath = self.output_dir / f"{filename}.json"
        
        # Add metadata
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "pattern_count": len(patterns),
                "version": "1.0"
            },
            "patterns": patterns
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        return str(filepath)
        
    def _export_pdf(self, patterns: List[Dict[str, Any]], filename: str, config: ReportConfig) -> str:
        """Export patterns to PDF report."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
            
        filepath = self.output_dir / f"{filename}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=config.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story
        story = []
        styles = getSampleStyleSheet()
        
        # Title page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(config.title, title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated on: {config.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Author: {config.author}", styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary
        if config.include_statistics:
            story.append(Paragraph("Executive Summary", styles['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            summary_data = self._generate_summary_statistics(patterns)
            for key, value in summary_data.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
            story.append(PageBreak())
            
        # Pattern Analysis
        if config.include_patterns:
            story.append(Paragraph("Pattern Analysis", styles['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add pattern details table
            pattern_table_data = self._create_pattern_table(patterns[:10])  # First 10 patterns
            pattern_table = Table(pattern_table_data)
            pattern_table.setStyle(self._get_table_style())
            story.append(pattern_table)
            story.append(PageBreak())
            
        # Charts
        if config.include_charts:
            story.append(Paragraph("Visual Analysis", styles['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            # Generate and add charts
            chart_paths = self._generate_charts(patterns, config)
            for chart_path in chart_paths:
                img = Image(chart_path, width=config.chart_width*inch, height=config.chart_height*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
                
        # Build PDF
        doc.build(story)
        
        # Clean up temporary chart files
        for chart_path in chart_paths:
            Path(chart_path).unlink()
            
        return str(filepath)
        
    def _export_excel(self, patterns: List[Dict[str, Any]], filename: str) -> str:
        """Export patterns to Excel format with multiple sheets."""
        filepath = self.output_dir / f"{filename}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Main patterns sheet
            df_patterns = pd.DataFrame(self._flatten_patterns(patterns))
            df_patterns.to_excel(writer, sheet_name='Patterns', index=False)
            
            # Summary statistics sheet
            summary_stats = self._generate_summary_statistics(patterns)
            df_summary = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Pattern statistics by ticker
            ticker_stats = self._calculate_ticker_statistics(patterns)
            df_ticker = pd.DataFrame(ticker_stats)
            df_ticker.to_excel(writer, sheet_name='Ticker Analysis', index=False)
            
        return str(filepath)
        
    def _export_template(self, patterns: List[Dict[str, Any]], filename: str) -> str:
        """Export patterns as reusable templates."""
        filepath = self.output_dir / f"{filename}_template.json"
        
        templates = []
        for pattern in patterns:
            template = {
                "id": pattern.get("id", ""),
                "name": pattern.get("name", f"Pattern_{len(templates)+1}"),
                "type": pattern.get("type", "unknown"),
                "parameters": {
                    "scale": pattern.get("scale", 0),
                    "strength": pattern.get("strength", 0),
                    "duration": pattern.get("duration", 0),
                    "frequency": pattern.get("frequency", 0)
                },
                "detection_criteria": {
                    "min_correlation": pattern.get("min_correlation", 0.7),
                    "min_strength": pattern.get("min_strength", 0.5),
                    "scale_range": pattern.get("scale_range", [1, 10])
                },
                "metadata": {
                    "created_date": datetime.now().isoformat(),
                    "source_ticker": pattern.get("ticker", ""),
                    "performance_metrics": pattern.get("performance", {})
                }
            }
            templates.append(template)
            
        template_data = {
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "template_count": len(templates),
            "templates": templates
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=2)
            
        return str(filepath)
        
    def create_dashboard_snapshot(
        self,
        dashboard_state: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Create a shareable HTML snapshot of the dashboard."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_snapshot_{timestamp}"
            
        filepath = self.output_dir / f"{filename}.html"
        
        # Create interactive Plotly dashboard
        fig = go.Figure()
        
        # Add main time series
        if "timeseries" in dashboard_state:
            ts_data = dashboard_state["timeseries"]
            fig.add_trace(go.Scatter(
                x=ts_data.get("dates", []),
                y=ts_data.get("values", []),
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
        # Add pattern overlays
        if "patterns" in dashboard_state:
            for i, pattern in enumerate(dashboard_state["patterns"][:5]):  # Limit to 5 patterns
                fig.add_trace(go.Scatter(
                    x=pattern.get("dates", []),
                    y=pattern.get("values", []),
                    mode='lines',
                    name=f'Pattern {i+1}',
                    line=dict(width=3, dash='dash'),
                    opacity=0.7
                ))
                
        # Update layout
        fig.update_layout(
            title=dashboard_state.get("title", "Pattern Analysis Dashboard"),
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        # Generate HTML with embedded data
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_state.get("title", "Pattern Dashboard Snapshot")}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #1f77b4;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{dashboard_state.get("title", "Pattern Analysis Dashboard")}</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_state.get("pattern_count", 0)}</div>
                        <div class="metric-label">Patterns Detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_state.get("accuracy", "N/A")}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_state.get("ticker", "N/A")}</div>
                        <div class="metric-label">Ticker</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_state.get("timeframe", "N/A")}</div>
                        <div class="metric-label">Timeframe</div>
                    </div>
                </div>
                
                <div id="plotly-chart"></div>
                
                <script>
                    var data = {pio.to_json(fig)};
                    Plotly.newPlot('plotly-chart', data.data, data.layout);
                </script>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
            
        return str(filepath)
        
    def batch_export(
        self,
        ticker_patterns: Dict[str, List[Dict[str, Any]]],
        format: ExportFormat,
        config: Optional[ReportConfig] = None
    ) -> Dict[str, str]:
        """Batch export patterns for multiple tickers."""
        export_paths = {}
        
        for ticker, patterns in ticker_patterns.items():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_patterns_{timestamp}"
            
            try:
                path = self.export_patterns(patterns, format, filename, config)
                export_paths[ticker] = path
            except Exception as e:
                export_paths[ticker] = f"Error: {str(e)}"
                
        # Create summary file
        summary_path = self._create_batch_summary(export_paths, format)
        export_paths["_summary"] = summary_path
        
        return export_paths
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _flatten_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten list of pattern dictionaries."""
        return [self._flatten_dict(pattern) for pattern in patterns]
        
    def _generate_summary_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from patterns."""
        if not patterns:
            return {"Total Patterns": 0}
            
        stats = {
            "Total Patterns": len(patterns),
            "Unique Tickers": len(set(p.get("ticker", "") for p in patterns)),
            "Pattern Types": len(set(p.get("type", "") for p in patterns)),
            "Average Strength": np.mean([p.get("strength", 0) for p in patterns]),
            "Average Duration": np.mean([p.get("duration", 0) for p in patterns]),
            "Date Range": f"{min(p.get('start_date', '') for p in patterns)} to {max(p.get('end_date', '') for p in patterns)}"
        }
        
        return stats
        
    def _create_pattern_table(self, patterns: List[Dict[str, Any]]) -> List[List[str]]:
        """Create table data for pattern display."""
        headers = ["ID", "Type", "Ticker", "Strength", "Duration", "Start Date"]
        
        data = [headers]
        for pattern in patterns:
            row = [
                str(pattern.get("id", ""))[:8],
                pattern.get("type", ""),
                pattern.get("ticker", ""),
                f"{pattern.get('strength', 0):.2f}",
                f"{pattern.get('duration', 0)} days",
                pattern.get("start_date", "")[:10]
            ]
            data.append(row)
            
        return data
        
    def _get_table_style(self) -> TableStyle:
        """Get table style for PDF reports."""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        
    def _generate_charts(self, patterns: List[Dict[str, Any]], config: ReportConfig) -> List[str]:
        """Generate charts for PDF report."""
        chart_paths = []
        
        # Pattern distribution chart
        plt.figure(figsize=(config.chart_width, config.chart_height))
        pattern_types = [p.get("type", "unknown") for p in patterns]
        type_counts = pd.Series(pattern_types).value_counts()
        
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        plt.title("Pattern Type Distribution")
        
        chart_path = self.output_dir / f"temp_chart_dist_{datetime.now().timestamp()}.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=150)
        plt.close()
        chart_paths.append(str(chart_path))
        
        # Strength histogram
        plt.figure(figsize=(config.chart_width, config.chart_height))
        strengths = [p.get("strength", 0) for p in patterns]
        plt.hist(strengths, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Pattern Strength")
        plt.ylabel("Frequency")
        plt.title("Pattern Strength Distribution")
        
        chart_path = self.output_dir / f"temp_chart_strength_{datetime.now().timestamp()}.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=150)
        plt.close()
        chart_paths.append(str(chart_path))
        
        # Timeline chart
        if len(patterns) > 0 and all("start_date" in p for p in patterns):
            plt.figure(figsize=(config.chart_width, config.chart_height))
            
            dates = pd.to_datetime([p["start_date"] for p in patterns])
            plt.hist(dates, bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel("Date")
            plt.ylabel("Pattern Count")
            plt.title("Pattern Detection Timeline")
            plt.xticks(rotation=45)
            
            chart_path = self.output_dir / f"temp_chart_timeline_{datetime.now().timestamp()}.png"
            plt.savefig(chart_path, bbox_inches='tight', dpi=150)
            plt.close()
            chart_paths.append(str(chart_path))
            
        return chart_paths
        
    def _calculate_ticker_statistics(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate statistics by ticker."""
        ticker_stats = {}
        
        for pattern in patterns:
            ticker = pattern.get("ticker", "unknown")
            if ticker not in ticker_stats:
                ticker_stats[ticker] = {
                    "ticker": ticker,
                    "pattern_count": 0,
                    "avg_strength": [],
                    "types": set()
                }
                
            ticker_stats[ticker]["pattern_count"] += 1
            ticker_stats[ticker]["avg_strength"].append(pattern.get("strength", 0))
            ticker_stats[ticker]["types"].add(pattern.get("type", ""))
            
        # Calculate averages
        result = []
        for ticker, stats in ticker_stats.items():
            result.append({
                "ticker": ticker,
                "pattern_count": stats["pattern_count"],
                "avg_strength": np.mean(stats["avg_strength"]),
                "unique_types": len(stats["types"])
            })
            
        return result
        
    def _create_batch_summary(self, export_paths: Dict[str, str], format: ExportFormat) -> str:
        """Create summary file for batch exports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"batch_export_summary_{timestamp}.json"
        
        summary = {
            "export_date": datetime.now().isoformat(),
            "format": format.value,
            "total_tickers": len(export_paths) - 1,  # Exclude summary itself
            "exports": export_paths
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return str(summary_path)


# Convenience functions
def export_patterns_to_csv(patterns: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    """Export patterns to CSV format."""
    generator = ReportGenerator()
    return generator.export_patterns(patterns, ExportFormat.CSV, filename)


def export_patterns_to_json(patterns: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    """Export patterns to JSON format."""
    generator = ReportGenerator()
    return generator.export_patterns(patterns, ExportFormat.JSON, filename)


def generate_pdf_report(
    patterns: List[Dict[str, Any]],
    filename: Optional[str] = None,
    config: Optional[ReportConfig] = None
) -> str:
    """Generate PDF report from patterns."""
    generator = ReportGenerator()
    return generator.export_patterns(patterns, ExportFormat.PDF, filename, config)


def create_dashboard_snapshot(dashboard_state: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Create shareable dashboard snapshot."""
    generator = ReportGenerator()
    return generator.create_dashboard_snapshot(dashboard_state, filename)


def export_pattern_templates(patterns: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    """Export patterns as reusable templates."""
    generator = ReportGenerator()
    return generator.export_patterns(patterns, ExportFormat.TEMPLATE, filename)


def batch_export_patterns(
    ticker_patterns: Dict[str, List[Dict[str, Any]]],
    format: ExportFormat = ExportFormat.JSON,
    config: Optional[ReportConfig] = None
) -> Dict[str, str]:
    """Batch export patterns for multiple tickers."""
    generator = ReportGenerator()
    return generator.batch_export(ticker_patterns, format, config)
