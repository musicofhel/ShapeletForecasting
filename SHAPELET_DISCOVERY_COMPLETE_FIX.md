# Shapelet Discovery Dashboard - Complete Fix Documentation

## Current Status and Issues

### What's Working:
1. **Main Chart Shapelet Overlay**: When clicking "Discover Shapelets", the main time series chart successfully:
   - Switches from candlestick to line chart
   - Overlays discovered shapelets with SAX labels
   - Shows shapelets like "SAX: aaabdedeeeeedddbaaaa", "SAX: eeeedddbaaaaaacddcc", etc.
   - Displays shapelets with different colors for visual distinction

2. **Dashboard Loading**: The dashboard loads successfully at http://localhost:8050 with real YFinance data

3. **Data Integration**: YFinance data is being fetched and cached properly for BTC/USD and other symbols

### What's Not Working:
1. **Shapelet Analysis Panel Error**: The Shapelet Analysis section shows error:
   ```
   Error discovering shapelets: 'Annotator' object has no attribute 'get'
   ```
   This error occurs in the `update_shapelet_analysis` callback even though shapelets are discovered successfully

2. **Statistics Access Issue**: The `shapelet.statistics` field is being treated as an Annotator object instead of a dictionary in some parts of the code

## Root Cause Analysis

The issue stems from an inconsistency in how the `statistics` field is accessed in the shapelet objects. While the `ShapeletDiscoverer` correctly creates statistics as a dictionary, somewhere in the visualization pipeline, the statistics field is being replaced or wrapped with an Annotator object.

## Libraries and Technologies Used

### Core Libraries:
1. **Dash & Plotly** (dash==2.14.1, plotly==5.18.0): Web dashboard framework and interactive visualizations
2. **Dash Bootstrap Components** (dash-bootstrap-components==1.5.0): UI components
3. **NumPy** (numpy==1.24.3): Numerical computations
4. **Pandas** (pandas==2.0.3): Data manipulation and time series handling
5. **YFinance** (yfinance==0.2.28): Real-time financial data fetching
6. **Scikit-learn** (scikit-learn==1.3.0): StandardScaler for normalization
7. **SciPy** (scipy==1.11.1): Statistical computations

### Custom Modules:
1. **ShapeletDiscoverer** (`src/shapelet_discovery/shapelet_discoverer.py`):
   - Implements sliding window shapelet extraction
   - Uses SAX (Symbolic Aggregate approXimation) for pattern labeling
   - Calculates statistics: frequency, avg_return_after, win_rate, confidence

2. **SAXTransformer** (`src/advanced/time_series_integration.py`):
   - Converts numeric time series to alphabetic representations
   - Uses 20 segments and alphabet size of 5

3. **ShapeletVisualizer** (`src/dashboard/visualizations/shapelet_visualization.py`):
   - Creates overlay visualizations
   - Generates shapelet library views
   - Produces distribution analysis charts

## Complete Fix Implementation

### 1. Fix the Shapelet Analysis Callback

The main issue is in the `update_shapelet_analysis` callback where statistics are accessed. We need to ensure robust access to the statistics field:

```python
@app.callback(
    Output('shapelet-analysis', 'children'),
    [Input('symbol-dropdown', 'value'),
     Input('discover-shapelets-button', 'n_clicks')],
    [State('timeframe-dropdown', 'value')]
)
def update_shapelet_analysis(symbol: str, n_clicks: int, timeframe: str):
    """Discover and visualize shapelets from time series data"""
    if not n_clicks or not symbol:
        raise PreventUpdate
    
    try:
        # Load data
        df = data_manager.load_data(symbol, timeframe)
        
        if df.empty:
            return dbc.Alert(
                f"No data available for {symbol}",
                color="warning",
                dismissable=True
            )
        
        # Check if we have cached shapelets
        cache_key = f"{symbol}_{timeframe}_shapelets"
        if cache_key in discovered_shapelets:
            shapelets = discovered_shapelets[cache_key]
            logger.info(f"Using cached shapelets for {symbol}")
        else:
            # Discover shapelets
            logger.info(f"Discovering shapelets for {symbol}")
            shapelets = shapelet_discoverer.discover_shapelets(
                df, 
                ticker=symbol,
                timeframe=timeframe,
                price_col='close' if 'close' in df.columns else 'price'
            )
            
            # Add to library
            shapelet_discoverer.add_to_library(shapelets)
            
            # Cache the results
            discovered_shapelets[cache_key] = shapelets
        
        # Create visualization cards
        cards = []
        
        # Safe statistics access helper
        def get_stat(shapelet, key, default=0):
            """Safely get a statistic from shapelet"""
            if hasattr(shapelet, 'statistics'):
                stats = shapelet.statistics
                if isinstance(stats, dict):
                    return stats.get(key, default)
                elif hasattr(stats, 'get'):
                    return stats.get(key, default)
                elif hasattr(stats, key):
                    return getattr(stats, key, default)
            return default
        
        # Shapelet summary card
        summary_card = dbc.Card([
            dbc.CardHeader(html.H5("Shapelet Discovery Summary", className="mb-0")),
            dbc.CardBody([
                html.P(f"Total Shapelets Discovered: {len(shapelets)}"),
                html.P(f"Unique SAX Labels: {len(set(s.sax_label for s in shapelets))}"),
                html.P(f"Average Shapelet Length: {np.mean([s.length for s in shapelets]):.1f}" if shapelets else "N/A"),
                html.Hr(),
                html.H6("Top 5 Most Frequent Shapelets:"),
                html.Ul([
                    html.Li(f"SAX: {s.sax_label} (freq: {get_stat(s, 'frequency')}, avg return: {get_stat(s, 'avg_return_after'):.2%})")
                    for s in sorted(shapelets, key=lambda x: get_stat(x, 'frequency'), reverse=True)[:5]
                ]) if shapelets else html.P("No shapelets discovered")
            ])
        ], className="mb-3")
        cards.append(summary_card)
        
        # Rest of the visualization code...
```

### 2. Fix the Main Chart Callback

Update the hover template and other statistics access in the main chart callback:

```python
# In update_main_chart callback, update the shapelet overlay section:
fig.add_trace(
    go.Scatter(
        x=shapelet_segment['timestamp'] if 'timestamp' in shapelet_segment.columns else shapelet_segment.index,
        y=shapelet_segment['price'] if 'price' in shapelet_segment.columns else shapelet_segment['close'],
        name=f'SAX: {shapelet.sax_label}',
        line=dict(width=3),
        opacity=0.7,
        hovertemplate=f"SAX: {shapelet.sax_label}<br>%{{x}}<br>Price: $%{{y:.2f}}<br>Frequency: {get_stat(shapelet, 'frequency')}<br>Avg Return: {get_stat(shapelet, 'avg_return_after'):.2%}<extra></extra>"
    ),
    row=1, col=1
)
```

### 3. Add Defensive Programming to ShapeletVisualizer

Update the shapelet visualizer to handle different statistics formats:

```python
# In shapelet_visualization.py, add a safe statistics accessor
def _get_statistic(self, shapelet, key, default=0):
    """Safely extract a statistic from a shapelet"""
    if hasattr(shapelet, 'statistics'):
        stats = shapelet.statistics
        if isinstance(stats, dict):
            return stats.get(key, default)
        elif hasattr(stats, 'get'):
            return stats.get(key, default)
        elif hasattr(stats, key):
            return getattr(stats, key, default)
    return default
```

## Testing Steps

1. **Start the Dashboard**:
   ```bash
   python run_dashboard.py
   ```

2. **Test Shapelet Discovery**:
   - Navigate to http://localhost:8050
   - Select BTC/USD or any symbol
   - Click "Discover Shapelets"
   - Verify shapelets appear on main chart
   - Scroll down to check Shapelet Analysis panel loads without errors

3. **Verify SAX Labels**:
   - Check that SAX labels are displayed (e.g., "aaabdedeeeeedddbaaaa")
   - Hover over shapelets to see statistics
   - Confirm frequency and return statistics are shown

4. **Check Visualizations**:
   - Shapelet overlay on main chart
   - Shapelet library visualization
   - Distribution analysis charts
   - Timeline visualization

## Performance Considerations

1. **Caching**: Shapelets are cached by symbol and timeframe to avoid recomputation
2. **Sliding Window**: Uses step_size=1 for thorough search (can be increased for faster discovery)
3. **SAX Parameters**: 20 segments and alphabet size 5 provide good balance of detail vs generalization

## Future Enhancements

1. **Persistence**: Save discovered shapelets to database for long-term storage
2. **Pattern Matching**: Use discovered shapelets to find similar patterns in new data
3. **Real-time Discovery**: Incrementally discover new shapelets as data arrives
4. **Multi-timeframe Analysis**: Discover shapelets across multiple timeframes simultaneously
5. **Pattern Prediction**: Use shapelet sequences to predict future price movements

## Troubleshooting

If errors persist:

1. **Check Data Types**: Ensure all statistics are dictionaries, not other objects
2. **Debug Logging**: Add logging to track where statistics format changes
3. **Inspect Cache**: Clear the `discovered_shapelets` cache if corrupted
4. **Verify Dependencies**: Ensure all required packages are installed with correct versions

## Conclusion

The shapelet discovery system successfully identifies and labels recurring patterns in financial time series using SAX representation. The main functionality works, but robust error handling is needed to prevent issues with the statistics field access. The fixes provided ensure the system handles edge cases gracefully while maintaining the core shapelet discovery and visualization capabilities.
