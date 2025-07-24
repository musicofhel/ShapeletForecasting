# Interactive Financial Analysis Dashboard Guide

## Overview
The enhanced demo at http://localhost:8051 provides a complete financial analysis dashboard with real-time data visualizations that update based on your sidebar selections.

## Dashboard Layout

### Left Side: Interactive Sidebar (Width: 25%)
The sidebar contains all the controls from the previous demo:
- **Ticker Selection** with categories and multi-select
- **Date Range Picker** with presets
- **Pattern Type Filters** (Wavelets, Shapelets, Motifs)
- **Quality Thresholds** with sliders
- **Real-time Data Toggle**
- **Advanced Settings**
- **Apply Settings & Analyze Button** (Green GO button)

### Right Side: Data Visualizations (Width: 75%)

#### Top Row:
1. **Price Chart (Left, 60% width)**
   - Shows line charts for selected tickers
   - Different colors for each ticker
   - Interactive hover tooltips
   - Updates when you change ticker selection
   - Shows "Select tickers and click 'Apply Settings & Analyze' to view data" when no data

2. **Pattern Detection Results (Right, 40% width)**
   - Bar chart showing detected patterns
   - Color-coded by confidence level
   - Shows count of each pattern type
   - Updates based on selected patterns and thresholds

#### Bottom Row:
1. **Quality Thresholds Gauges (Left, 65% width)**
   - Four gauge charts: Confidence, Significance, Stability, Robustness
   - Color-coded (Red < 50%, Yellow 50-70%, Green > 70%)
   - Real-time updates as you adjust sliders

2. **Current Configuration Table (Right, 35% width)**
   - Summary table showing:
     - Number of tickers selected
     - Date range
     - Time granularity
     - Number of patterns selected
     - Data mode (Historical/Real-time)

## How to Use:

### Step 1: Select Your Tickers
1. Click on category tabs (Crypto, Stocks, ETFs, etc.)
2. Use the multi-select dropdown to choose tickers
3. Or click "Popular" for quick selection

### Step 2: Set Date Range
1. Use preset buttons (1W, 1M, 3M, etc.) for quick selection
2. Or use the date picker for custom range
3. Select time granularity (1h, 4h, 1d, etc.)

### Step 3: Choose Patterns
1. Expand pattern categories in the accordion
2. Check the patterns you want to analyze
3. Or click "Recommended" for suggested patterns

### Step 4: Adjust Quality Thresholds
1. Move sliders to set quality levels
2. Or use preset buttons (Conservative, Balanced, Aggressive)
3. Watch the gauges update in real-time

### Step 5: Apply and Analyze
1. Click the green "Apply Settings & Analyze" button
2. Watch the progress bar show analysis stages
3. See time estimates update in real-time
4. View results in the visualizations

## What You'll See:

### Initial State:
- Default 3 tickers selected (BTC-USD, ETH-USD, SPY)
- 90-day date range
- No patterns selected initially
- Quality thresholds at default values
- Empty price chart with instruction message

### After Clicking GO:
1. Progress bar appears showing:
   - "📊 Loading data for BTC-USD..."
   - "📈 Analyzing patterns for ETH-USD..."
   - "🔍 Extracting features for SPY..."
   - Time remaining (e.g., "~15s remaining")

2. When complete:
   - Success message: "✅ Analysis completed successfully in X.X seconds!"
   - Price chart populates with colored lines
   - Pattern analysis shows detected patterns
   - All visualizations update

### Interactive Features:
- Hover over price lines to see values
- Click legend items to show/hide tickers
- Hover over pattern bars to see details
- All charts update when you change selections

## Sample Data:
The demo uses synthetic data that simulates:
- Price trends with seasonal patterns
- Random pattern detection results
- Realistic confidence scores
- Volume and high/low data

## Tips:
1. Try different ticker combinations to see color variations
2. Adjust quality thresholds and watch pattern counts change
3. Select many patterns to see the progress bar in action
4. Use date presets for quick time period changes
5. Toggle between Historical and Real-time modes

## Technical Details:
- Built with Dash and Plotly
- Real-time state management
- Responsive design with Bootstrap
- Simulated data for demonstration
- Production-ready component architecture

Visit http://localhost:8051 to explore the full interactive experience!
