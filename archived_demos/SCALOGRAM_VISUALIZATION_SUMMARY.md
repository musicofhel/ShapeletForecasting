# Scalogram Visualization Module Summary

## Overview
The scalogram visualization module provides interactive wavelet scalogram visualizations with advanced features including heatmap representation, ridge detection, and time-series synchronization.

## Key Features

### 1. **Interactive Scalogram Heatmap**
- Time-scale representation of wavelet coefficients
- Logarithmic frequency scale for better visualization
- Customizable colorscales (Viridis, Plasma, Inferno, Jet, Hot)
- Hover tooltips showing time, frequency, and magnitude

### 2. **Ridge Detection**
- Automatic detection of significant ridges in the scalogram
- Visual overlay of detected ridges on the heatmap
- Ridge strength and frequency information
- Configurable detection parameters (min_length, min_snr, gap_threshold)

### 3. **Synchronized Time Series**
- Main time series plot synchronized with scalogram
- Shared x-axis for coordinated zooming and panning
- Range slider for navigation
- Click interactions to highlight corresponding time periods

### 4. **Multiple View Modes**
- **2D Scalogram**: Traditional heatmap view with time series
- **3D Surface**: 3D surface plot for enhanced visualization
- **Ridge Analysis**: Detailed breakdown of top ridges with individual components

### 5. **Time-Scale Feature Extraction**
- Extract features at any time point:
  - Dominant frequency and scale
  - Energy and entropy measures
  - Phase coherence
  - Ridge membership information

## Module Structure

### Core Class: `ScalogramVisualizer`
```python
class ScalogramVisualizer:
    def __init__(self, wavelet='morl', sampling_rate=1.0)
    def compute_cwt(data, scales=None, min_scale=1, max_scale=128, num_scales=100)
    def detect_ridges(min_length=10, min_snr=1.0, gap_threshold=3)
    def create_scalogram_plot(data, time_index=None, show_ridges=True, colorscale='Viridis', height=800)
    def create_3d_scalogram(data, time_index=None, colorscale='Viridis')
    def create_ridge_analysis_plot(data, time_index=None, top_n_ridges=5)
    def get_time_scale_features(time_point)
```

### Supported Wavelets
- Morlet ('morl')
- Mexican Hat ('mexh')
- Gaussian ('gaus8')
- Complex Morlet ('cmor1.5-1.0')

## Usage Examples

### Basic Scalogram Creation
```python
from src.dashboard.visualizations.scalogram import ScalogramVisualizer

# Initialize visualizer
viz = ScalogramVisualizer(wavelet='morl', sampling_rate=100)

# Create scalogram with ridge detection
fig = viz.create_scalogram_plot(data, time_index, show_ridges=True)
fig.show()
```

### Interactive Dashboard
```python
# Run the interactive Dash application
python demo_scalogram_visualization.py --dash
```

### Standalone Visualizations
```python
# Create HTML files
python demo_scalogram_visualization.py
```

## Generated Visualizations

1. **scalogram_interactive.html**: Main interactive scalogram with ridges
2. **scalogram_3d_view.html**: 3D surface visualization
3. **scalogram_ridge_analysis.html**: Detailed ridge component analysis
4. **scalogram_wavelet_comparison.html**: Comparison of different wavelet types

## Integration with Dashboard

The scalogram module integrates seamlessly with the pattern dashboard:
- Can be embedded as a component in larger dashboards
- Supports click callbacks for interaction with other visualizations
- Provides feature extraction for pattern analysis
- Compatible with the sidebar navigation system

## Technical Details

### Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- plotly: Interactive visualizations
- pywt: Wavelet transforms
- scipy: Signal processing and ridge detection
- dash: Interactive web applications

### Performance Considerations
- Efficient CWT computation using PyWavelets
- Logarithmic scale generation for better frequency resolution
- Optimized ridge detection algorithm
- Client-side rendering for smooth interactions

## Future Enhancements
1. Support for more wavelet families
2. Advanced ridge tracking algorithms
3. Integration with pattern matching
4. Real-time streaming data support
5. Export functionality for analysis results
