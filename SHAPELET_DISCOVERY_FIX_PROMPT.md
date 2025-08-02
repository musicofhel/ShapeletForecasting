# Shapelet Discovery Dashboard Fix - Complete Task Description

## Task: Fix Shapelet Discovery Dashboard Implementation

### Goal
Implement a working shapelet discovery system that:
1. Discovers discriminative shapelets from time series data using sliding window approach
2. Labels shapelets using SAX (Symbolic Aggregate approXimation) with alphabetic labels
3. Stores shapelets in a library tagged by ticker and timeframe
4. Integrates with the existing dashboard to visualize discovered shapelets
5. Fixes current duplicate ID errors preventing the dashboard from loading

### Current Issues
1. **Duplicate ID Error**: The dashboard has duplicate component IDs:
   - `progress-panel-container` appears in both `run_dashboard.py` and `src/dashboard/layouts/forecast_layout.py`
   - `progress-task-name` appears in both files as well
   - These need unique IDs to prevent Dash from crashing

2. **Dashboard won't load** due to the duplicate ID conflicts

### Key Files and Their Interconnections

#### Core Shapelet Discovery Files:
1. **`src/shapelet_discovery/__init__.py`** - Module initialization, exports ShapeletDiscoverer
2. **`src/shapelet_discovery/shapelet_discoverer.py`** - Main shapelet discovery implementation
   - Uses SAX from `src/advanced/time_series_integration.py`
   - Implements sliding window shapelet extraction
   - Stores shapelets with SAX labels, statistics, and metadata

3. **`src/dashboard/visualizations/shapelet_visualization.py`** - Visualization components
   - Creates overlay visualizations
   - Shows shapelet library
   - Displays distribution analysis
   - Timeline visualization

#### Dashboard Integration Files:
4. **`run_dashboard.py`** - Main dashboard application
   - Imports ShapeletDiscoverer and ShapeletVisualizer
   - Has duplicate IDs that need fixing
   - Contains shapelet discovery callback
   - Integrates with data_manager for pattern detection

5. **`src/dashboard/layouts/forecast_layout.py`** - Dashboard layout
   - Contains duplicate IDs that conflict with run_dashboard.py
   - Defines UI structure including shapelet analysis panel

6. **`src/dashboard/data_utils_yfinance.py`** - Data management
   - Fetches real-time data from YFinance
   - Used by DataManager in run_dashboard.py

#### Supporting Files:
7. **`src/advanced/time_series_integration.py`** - SAX implementation
   - Provides SAXTransformer and SAXConfig
   - Used for converting shapelets to alphabetic labels

8. **`src/wavelet_analysis/advanced_pattern_detector.py`** - Pattern detection
   - Works alongside shapelet discovery
   - Provides advanced pattern analysis

### File Dependencies:
```
run_dashboard.py
├── imports ShapeletDiscoverer from src/shapelet_discovery/__init__.py
├── imports ShapeletVisualizer from src/dashboard/visualizations/shapelet_visualization.py
├── imports create_forecast_layout from src/dashboard/layouts/forecast_layout.py
├── uses YFinanceDataManager from src/dashboard/data_utils_yfinance.py
└── uses SAXTransformer from src/advanced/time_series_integration.py

src/shapelet_discovery/shapelet_discoverer.py
├── imports SAXTransformer from src/advanced/time_series_integration.py
└── defines Shapelet dataclass and ShapeletDiscoverer class

src/dashboard/visualizations/shapelet_visualization.py
└── uses Shapelet objects from shapelet_discoverer.py
```

### Implementation Details:
1. **Shapelet Discovery Process**:
   - Sliding window extraction (min_length=10, max_length=50)
   - SAX transformation for each candidate
   - Group by SAX representation
   - Select representative shapelets
   - Calculate statistics (frequency, returns, confidence)

2. **SAX Labeling**:
   - Uses 20 segments and alphabet size of 5
   - Creates alphabetic labels like "abcde", "aabcd", etc.
   - Groups similar patterns by SAX representation

3. **Dashboard Integration**:
   - "Discover Shapelets" button triggers discovery
   - Results displayed in multiple visualizations
   - Shapelets cached for performance

### Fix Required:
1. Rename duplicate IDs in either `run_dashboard.py` or `src/dashboard/layouts/forecast_layout.py`:
   - Change `progress-panel-container` to unique IDs
   - Change `progress-task-name` to unique IDs
   - Ensure all component IDs are unique across the application

2. Test the dashboard loads successfully after fixing IDs

3. Verify shapelet discovery works:
   - Click "Discover Shapelets" button
   - Check that shapelets are discovered and displayed
   - Verify SAX labels are shown
   - Confirm visualizations render correctly

### Current Working Directory:
`c:/Users/aaron/AppData/Local/Programs/AppData/Local/Programs/Microsoft VS Code/financial_wavelet_prediction`

### To Run Dashboard:
```bash
python run_dashboard.py
```
Then navigate to http://localhost:8050

The dashboard should load without errors and allow shapelet discovery from real YFinance data.

### Recent Changes Made:
1. Fixed one duplicate ID issue: renamed `progress-task-name` to `floating-progress-task-name` in run_dashboard.py
2. Dashboard still has errors - need to fix remaining duplicate IDs

### Next Steps:
1. Find and fix the `progress-panel-container` duplicate ID
2. Check for any other duplicate IDs
3. Ensure dashboard loads successfully
4. Test shapelet discovery functionality
