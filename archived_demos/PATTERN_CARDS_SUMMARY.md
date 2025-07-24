# Pattern Information Cards Component Summary

## Overview
The Pattern Information Cards component provides comprehensive, detailed views of discovered patterns in financial time series data. Each card displays extensive information about a pattern including visualizations, statistics, quality metrics, occurrences, and predictions.

## Key Features

### 1. **Pattern Visualization**
- Thumbnail pattern plots with trend lines
- Interactive hover information
- Visual quality indicators
- Compact representation for quick scanning

### 2. **Statistical Properties**
- Mean, standard deviation, trend, energy
- Entropy, skewness, kurtosis
- Autocorrelation analysis
- Power spectrum visualization
- Distribution histograms

### 3. **Quality Metrics**
- Overall quality score with color coding
- Confidence score
- Significance score
- Stability score
- Robustness score
- Visual badges for quick assessment

### 4. **Occurrence Tracking**
- Complete list of all pattern occurrences
- Timeline visualization of occurrences
- Quality scores for each occurrence
- Outcome tracking and performance
- Duration information
- Interactive occurrence details

### 5. **Prediction Analysis**
- Prediction vs actual scatter plots
- Error distribution analysis
- Rolling accuracy tracking
- Hit rate gauge visualization
- Mean absolute error metrics
- Performance summary statistics

### 6. **Market Analysis**
- Volatility conditions during pattern
- Trend strength assessment
- Volume ratio analysis
- Trading implications
- Actionable insights and recommendations

### 7. **Interactive Features**
- Expand/collapse individual cards
- Expand/collapse all cards at once
- Pattern comparison selection
- Export individual patterns (JSON format)
- Locate patterns on main chart
- View detailed occurrence information

### 8. **Filtering and Sorting**
- Sort by quality, recency, occurrences, or accuracy
- Filter by pattern type
- Minimum quality threshold
- Adjustable number of cards displayed

## Component Structure

### PatternInfo Dataclass
```python
@dataclass
class PatternInfo:
    pattern_id: str
    pattern_type: str
    ticker: str
    discovery_timestamp: datetime
    duration_hours: float
    start_time: datetime
    end_time: datetime
    pattern_data: List[float]
    normalized_data: List[float]
    timestamps: List[datetime]
    # Statistical properties
    mean: float
    std: float
    trend: float
    energy: float
    entropy: float
    skewness: float
    kurtosis: float
    # Quality metrics
    quality_score: float
    confidence_score: float
    significance_score: float
    stability_score: float
    robustness_score: float
    # Occurrences and predictions
    occurrences: List[Dict[str, Any]]
    total_occurrences: int
    predictions: List[Dict[str, Any]]
    prediction_accuracy: float
    mean_absolute_error: float
    hit_rate: float
    metadata: Dict[str, Any]
```

### Main Methods
- `create_pattern_card()`: Creates individual pattern card
- `create_pattern_cards_layout()`: Creates full layout with multiple cards
- `register_callbacks()`: Registers all interactive callbacks

## Visual Design

### Color Scheme
- **Primary**: #00D9FF (Cyan) for highlights and active elements
- **Success**: Green shades for high quality/accuracy
- **Warning**: Yellow/orange for medium quality
- **Danger**: Red shades for low quality
- **Background**: Dark theme (#0E1117, #1E1E1E, #262626)
- **Text**: #FAFAFA (Light gray) for readability

### Layout Structure
1. **Card Header**: Pattern name, ticker, quality badge, action buttons
2. **Basic Info** (Always visible):
   - Pattern thumbnail visualization
   - Key metrics grid
   - Discovery information
3. **Expanded Details** (Collapsible):
   - Statistics tab
   - Occurrences tab
   - Predictions tab
   - Analysis tab

## Usage Example

```python
from src.dashboard.components.pattern_cards import PatternCards, PatternInfo

# Initialize component
pattern_cards = PatternCards()

# Create pattern info objects
patterns = [PatternInfo(...), PatternInfo(...), ...]

# Create layout
layout = pattern_cards.create_pattern_cards_layout(
    patterns=patterns,
    max_cards=10
)

# Register callbacks in Dash app
PatternCards.register_callbacks(app)
```

## Integration Points

### Input Requirements
- Pattern data and metadata
- Statistical calculations
- Quality scores
- Occurrence history
- Prediction results

### Output Capabilities
- Pattern export (JSON format)
- Chart highlighting coordinates
- Comparison pattern lists
- Filtered/sorted pattern displays

### Callback Integration
- Expand/collapse controls
- Export functionality
- Pattern comparison
- Chart location highlighting
- Occurrence detail viewing
- Filtering and sorting

## Performance Considerations

1. **Lazy Loading**: Detailed tabs only render when expanded
2. **Efficient Updates**: Uses pattern matching callbacks
3. **Data Optimization**: Limits displayed occurrences/predictions
4. **Responsive Design**: Cards adapt to screen size

## Future Enhancements

1. **Additional Visualizations**
   - 3D pattern representations
   - Animated pattern evolution
   - Heatmap comparisons

2. **Advanced Analytics**
   - Machine learning insights
   - Pattern clustering visualization
   - Anomaly detection highlights

3. **Export Options**
   - CSV format support
   - PDF report generation
   - Batch export functionality

4. **Real-time Updates**
   - Live pattern detection
   - Streaming occurrence updates
   - Dynamic quality recalculation

## Demo Script
Run `python demo_pattern_cards.py` to see:
- 8 demo patterns with varying quality
- All interactive features
- Dark theme styling
- Responsive layout
- Full callback functionality

## Dependencies
- dash
- dash-bootstrap-components
- plotly
- pandas
- numpy
- dataclasses

## File Locations
- Component: `src/dashboard/components/pattern_cards.py`
- Demo: `demo_pattern_cards.py`
- Styles: Embedded in demo script
