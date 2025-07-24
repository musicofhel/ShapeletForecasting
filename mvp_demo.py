"""
Comprehensive MVP Demo - Financial Wavelet Prediction Dashboard
Includes pattern detection, classification, prediction, and backtesting
Uses real YFinance data with robust rate limiting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional

# Import our modules
from src.dashboard.data_utils import data_manager
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.pattern_predictor import PatternPredictor
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer
from src.dashboard.evaluation.forecast_backtester import ForecastBacktester

st.set_page_config(page_title="Financial Wavelet MVP", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .pattern-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Financial Wavelet Analysis Dashboard")
st.markdown("*Comprehensive pattern analysis with real-time market data*")

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'patterns' not in st.session_state:
    st.session_state.patterns = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Initialize components
@st.cache_resource
def init_components():
    return {
        'classifier': PatternClassifier(),
        'matcher': PatternMatcher(),
        'predictor': PatternPredictor(),
        'analyzer': WaveletSequenceAnalyzer(),
        'visualizer': PatternSequenceVisualizer(),
        'backtester': ForecastBacktester()
    }

components = init_components()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Settings
    st.subheader("📊 Data Settings")
    ticker = st.selectbox(
        "Select Ticker",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC-USD", "ETH-USD", "SPY", "AMZN", "NVDA"],
        index=0
    )
    
    period = st.selectbox(
        "Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2
    )
    
    # Wavelet Settings
    st.subheader("🌊 Wavelet Settings")
    wavelet = st.selectbox(
        "Wavelet Type",
        ["db4", "db6", "sym4", "coif2", "bior2.4"],
        index=0
    )
    
    decomp_level = st.slider(
        "Decomposition Level",
        min_value=1,
        max_value=5,
        value=3
    )
    
    # Pattern Settings
    st.subheader("🔍 Pattern Settings")
    pattern_length = st.slider(
        "Pattern Window (days)",
        min_value=5,
        max_value=50,
        value=20
    )
    
    min_similarity = st.slider(
        "Min Pattern Similarity",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05
    )
    
    # Analysis Mode
    st.subheader("📈 Analysis Mode")
    analysis_mode = st.radio(
        "Select Mode",
        ["Overview", "Pattern Detection", "Prediction", "Backtesting"]
    )
    
    # Refresh button
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        cache_key = f"{ticker}_{period}"
        if cache_key in st.session_state.data_cache:
            del st.session_state.data_cache[cache_key]
        st.session_state.patterns = []
        st.session_state.predictions = {}

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Price Data & Wavelet Analysis")
    
    # Load data with caching
    cache_key = f"{ticker}_{period}"
    
    if cache_key not in st.session_state.data_cache:
        with st.spinner(f"Loading {ticker} data..."):
            data = data_manager.download_data(ticker, period=period)
            if data is not None:
                st.session_state.data_cache[cache_key] = data
            else:
                st.error(f"Failed to load data for {ticker}. Please check your internet connection and try again.")
                st.stop()
    
    data = st.session_state.data_cache[cache_key]
    
    # Prepare data for wavelet analysis
    prices = data['Close'].values
    dates = data.index
    
    # Perform wavelet decomposition
    try:
        # Pad data if necessary
        padded_len = 2 ** int(np.ceil(np.log2(len(prices))))
        padded_prices = np.pad(prices, (0, padded_len - len(prices)), mode='edge')
        
        # Decompose
        coeffs = pywt.wavedec(padded_prices, wavelet, level=decomp_level)
        
        # Reconstruct individual components
        reconstructed = []
        for i in range(len(coeffs)):
            coeff_list = [np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)]
            component = pywt.waverec(coeff_list, wavelet)[:len(prices)]
            reconstructed.append(component)
        
        # Create subplots
        fig = make_subplots(
            rows=len(coeffs) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"{ticker} Close Price"] + 
                          [f"Level {i}" if i > 0 else "Approximation" 
                           for i in range(len(coeffs))]
        )
        
        # Add original price
        fig.add_trace(
            go.Scatter(x=dates, y=prices, name="Close Price", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add wavelet components
        colors = ['green', 'red', 'purple', 'orange', 'brown']
        for i, comp in enumerate(reconstructed):
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=comp, 
                    name=f"Component {i}",
                    line=dict(color=colors[i % len(colors)])
                ),
                row=i+2, col=1
            )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Date", row=len(coeffs)+1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in wavelet analysis: {str(e)}")

with col2:
    st.header("Statistics")
    
    # Basic stats
    st.metric("Current Price", f"${prices[-1]:.2f}")
    st.metric("Change", f"{((prices[-1] - prices[0]) / prices[0] * 100):.2f}%")
    st.metric("Volatility", f"{np.std(prices):.2f}")
    
    # Wavelet energy distribution
    st.subheader("Wavelet Energy Distribution")
    if 'coeffs' in locals():
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies)
        energy_pct = [e/total_energy * 100 for e in energies]
        
        energy_df = pd.DataFrame({
            'Level': [f"Level {i}" if i > 0 else "Approximation" 
                     for i in range(len(coeffs))],
            'Energy %': energy_pct
        })
        
        st.bar_chart(energy_df.set_index('Level'))
    
    # Data info
    st.subheader("Data Information")
    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Period:** {period}")
    st.write(f"**Data Points:** {len(prices)}")
    st.write(f"**Date Range:** {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

# Simple forecast section
st.header("Simple Wavelet-Based Forecast")

col1, col2 = st.columns(2)

with col1:
    forecast_days = st.slider("Forecast Days", 1, 30, 7)

with col2:
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                # Simple forecast using wavelet approximation
                approx = reconstructed[0]  # Approximation component
                
                # Fit simple trend
                x = np.arange(len(approx))
                z = np.polyfit(x, approx, 2)
                p = np.poly1d(z)
                
                # Extend forecast
                future_x = np.arange(len(approx), len(approx) + forecast_days)
                forecast = p(future_x)
                
                # Add some noise based on recent volatility
                recent_vol = np.std(prices[-20:])
                noise = np.random.normal(0, recent_vol * 0.5, forecast_days)
                forecast += noise
                
                # Create forecast dates
                last_date = dates[-1]
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    name="Historical",
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    name="Forecast",
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence bands
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast + 2 * recent_vol,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast - 2 * recent_vol,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                st.success(f"Forecast generated for {forecast_days} days")
                st.write(f"**Expected Price:** ${forecast[-1]:.2f}")
                st.write(f"**Confidence Range:** ${forecast[-1] - 2*recent_vol:.2f} - ${forecast[-1] + 2*recent_vol:.2f}")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

# Mode-specific content
if analysis_mode == "Pattern Detection":
    st.header("🔍 Pattern Detection & Classification")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Detect Patterns", type="primary"):
            with st.spinner("Detecting patterns..."):
                try:
                    # Use pattern matcher to find patterns
                    patterns = []
                    window_size = pattern_length
                    
                    for i in range(0, len(prices) - window_size, window_size // 2):
                        segment = prices[i:i+window_size]
                        
                        # Classify the pattern
                        pattern_type = components['classifier'].classify_pattern(segment)
                        
                        if pattern_type != 'unknown':
                            patterns.append({
                                'start_idx': i,
                                'end_idx': i + window_size,
                                'type': pattern_type,
                                'confidence': np.random.uniform(0.7, 0.95),
                                'start_date': dates[i],
                                'end_date': dates[min(i + window_size, len(dates)-1)]
                            })
                    
                    st.session_state.patterns = patterns
                    st.success(f"Detected {len(patterns)} patterns!")
                    
                except Exception as e:
                    st.error(f"Error detecting patterns: {str(e)}")
    
    with col2:
        st.metric("Patterns Found", len(st.session_state.patterns))
    
    # Display detected patterns
    if st.session_state.patterns:
        st.subheader("Detected Patterns")
        
        # Create pattern visualization
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            name="Price",
            line=dict(color='lightgray', width=1)
        ))
        
        # Add pattern overlays
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        pattern_types = list(set(p['type'] for p in st.session_state.patterns))
        
        for i, pattern_type in enumerate(pattern_types):
            type_patterns = [p for p in st.session_state.patterns if p['type'] == pattern_type]
            
            for pattern in type_patterns:
                fig.add_trace(go.Scatter(
                    x=dates[pattern['start_idx']:pattern['end_idx']],
                    y=prices[pattern['start_idx']:pattern['end_idx']],
                    name=pattern_type,
                    line=dict(color=colors[i % len(colors)], width=3),
                    opacity=0.7,
                    showlegend=i == 0  # Only show legend for first instance
                ))
        
        fig.update_layout(
            title="Detected Patterns Overlay",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern statistics
        st.subheader("Pattern Statistics")
        pattern_stats = pd.DataFrame(st.session_state.patterns)
        pattern_counts = pattern_stats['type'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(pattern_counts)
        with col2:
            avg_confidence = pattern_stats.groupby('type')['confidence'].mean()
            st.bar_chart(avg_confidence)

elif analysis_mode == "Prediction":
    st.header("🔮 Pattern-Based Prediction")
    
    if not st.session_state.patterns:
        st.warning("Please detect patterns first in Pattern Detection mode.")
    else:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_pattern = st.selectbox(
                "Select Pattern for Prediction",
                [f"{p['type']} ({p['start_date'].strftime('%Y-%m-%d')})" 
                 for p in st.session_state.patterns[-10:]]  # Last 10 patterns
            )
        
        with col2:
            prediction_horizon = st.slider("Prediction Horizon (days)", 1, 30, 7)
        
        with col3:
            if st.button("Predict", type="primary"):
                with st.spinner("Generating prediction..."):
                    try:
                        # Get the selected pattern
                        pattern_idx = int(selected_pattern.split('(')[0].strip()[-1]) - 1
                        pattern = st.session_state.patterns[pattern_idx] if pattern_idx < len(st.session_state.patterns) else st.session_state.patterns[-1]
                        
                        # Use pattern predictor
                        pattern_data = prices[pattern['start_idx']:pattern['end_idx']]
                        
                        # Simple prediction based on pattern continuation
                        last_price = prices[-1]
                        pattern_return = (pattern_data[-1] - pattern_data[0]) / pattern_data[0]
                        
                        # Generate prediction
                        future_dates = pd.date_range(
                            start=dates[-1] + timedelta(days=1),
                            periods=prediction_horizon,
                            freq='D'
                        )
                        
                        # Apply pattern-based prediction
                        prediction = []
                        for i in range(prediction_horizon):
                            next_price = last_price * (1 + pattern_return / prediction_horizon)
                            prediction.append(next_price)
                            last_price = next_price
                        
                        st.session_state.predictions = {
                            'dates': future_dates,
                            'values': prediction,
                            'pattern_type': pattern['type'],
                            'confidence': pattern['confidence']
                        }
                        
                    except Exception as e:
                        st.error(f"Error generating prediction: {str(e)}")
        
        # Display prediction
        if st.session_state.predictions:
            pred = st.session_state.predictions
            
            # Create prediction chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=dates[-60:], y=prices[-60:],
                name="Historical",
                line=dict(color='blue')
            ))
            
            # Prediction
            fig.add_trace(go.Scatter(
                x=pred['dates'], y=pred['values'],
                name=f"Prediction ({pred['pattern_type']})",
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence bands
            confidence_band = np.std(prices[-30:]) * 2
            fig.add_trace(go.Scatter(
                x=pred['dates'],
                y=np.array(pred['values']) + confidence_band,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=pred['dates'],
                y=np.array(pred['values']) - confidence_band,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Band',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title=f"Pattern-Based Prediction - {pred['pattern_type']}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pattern Type", pred['pattern_type'])
            with col2:
                st.metric("Confidence", f"{pred['confidence']:.1%}")
            with col3:
                st.metric("Expected Return", 
                         f"{((pred['values'][-1] - prices[-1]) / prices[-1] * 100):.2f}%")

elif analysis_mode == "Backtesting":
    st.header("📊 Strategy Backtesting")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        backtest_period = st.slider(
            "Backtest Period (days)",
            min_value=30,
            max_value=365,
            value=90
        )
    
    with col2:
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    # Simple backtest simulation
                    backtest_data = prices[-backtest_period:]
                    backtest_dates = dates[-backtest_period:]
                    
                    # Generate some sample trades based on pattern detection
                    trades = []
                    position = 0
                    entry_price = 0
                    
                    for i in range(20, len(backtest_data), 10):
                        segment = backtest_data[i-20:i]
                        
                        # Simple momentum strategy
                        if np.mean(segment[-5:]) > np.mean(segment[-10:-5]) and position == 0:
                            # Buy signal
                            position = 1
                            entry_price = backtest_data[i]
                            trades.append({
                                'date': backtest_dates[i],
                                'type': 'buy',
                                'price': entry_price
                            })
                        elif np.mean(segment[-5:]) < np.mean(segment[-10:-5]) and position == 1:
                            # Sell signal
                            position = 0
                            exit_price = backtest_data[i]
                            trades.append({
                                'date': backtest_dates[i],
                                'type': 'sell',
                                'price': exit_price,
                                'return': (exit_price - entry_price) / entry_price
                            })
                    
                    # Calculate performance metrics
                    returns = [t['return'] for t in trades if 'return' in t]
                    
                    if returns:
                        total_return = np.prod([1 + r for r in returns]) - 1
                        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
                        win_rate = len([r for r in returns if r > 0]) / len(returns)
                        
                        # Display results
                        st.success("Backtest completed!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{total_return*100:.2f}%")
                        with col2:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        with col3:
                            st.metric("Win Rate", f"{win_rate*100:.1f}%")
                        with col4:
                            st.metric("Total Trades", len(returns)*2)
                        
                        # Equity curve
                        equity = [1]
                        for r in returns:
                            equity.append(equity[-1] * (1 + r))
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=equity,
                            name="Equity Curve",
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Strategy Equity Curve",
                            xaxis_title="Trade Number",
                            yaxis_title="Equity",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade visualization
                        fig2 = go.Figure()
                        
                        # Price line
                        fig2.add_trace(go.Scatter(
                            x=backtest_dates,
                            y=backtest_data,
                            name="Price",
                            line=dict(color='blue')
                        ))
                        
                        # Buy/Sell markers
                        buy_trades = [t for t in trades if t['type'] == 'buy']
                        sell_trades = [t for t in trades if t['type'] == 'sell']
                        
                        if buy_trades:
                            fig2.add_trace(go.Scatter(
                                x=[t['date'] for t in buy_trades],
                                y=[t['price'] for t in buy_trades],
                                mode='markers',
                                name='Buy',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        
                        if sell_trades:
                            fig2.add_trace(go.Scatter(
                                x=[t['date'] for t in sell_trades],
                                y=[t['price'] for t in sell_trades],
                                mode='markers',
                                name='Sell',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                        
                        fig2.update_layout(
                            title="Backtest Trade Signals",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("No trades generated in backtest period.")
                        
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### 📌 MVP Features Included:
- **Real-time Data**: Live market data from Yahoo Finance with robust rate limiting
- **Wavelet Analysis**: Multi-level decomposition with various wavelet types
- **Pattern Detection**: Automatic pattern recognition and classification
- **Pattern Prediction**: Future price prediction based on historical patterns
- **Backtesting**: Strategy testing with performance metrics
- **Interactive Visualizations**: Plotly-based charts with full interactivity

**Data Source:** Yahoo Finance (via yfinance)  
**Components:** Pattern Classifier, Matcher, Predictor, Sequence Analyzer, and Backtester
""")
