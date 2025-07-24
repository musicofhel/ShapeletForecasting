"""
Comprehensive Test Suite for Forecast Dashboard Components

This module contains all tests for the forecasting dashboard including:
- Component rendering tests
- Callback function tests
- Layout responsiveness tests
- Performance tests
- User interaction tests
- Error handling tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from unittest.mock import Mock, patch, MagicMock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dash.testing.application_runners import import_app
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Import dashboard modules
from src.dashboard.forecast_app import (
    app, DataManager, PerformanceMonitor, 
    compute_pattern_predictions, update_main_chart,
    update_pattern_sequence, update_predictions,
    update_accuracy_metrics
)
from src.dashboard.layouts.forecast_layout import (
    create_forecast_layout, create_header, create_control_panel,
    create_main_chart_section, create_pattern_sequence_section,
    create_prediction_panel, create_metrics_panel,
    get_responsive_layout_config
)
from src.dashboard.callbacks.prediction_callbacks import (
    register_prediction_callbacks, create_pattern_detail_chart,
    create_pattern_stats_table, create_pattern_history_chart,
    perform_pattern_matching, create_match_results_display
)

# Test fixtures
@pytest.fixture
def dash_app():
    """Create test dash app instance"""
    return app

@pytest.fixture
def data_manager():
    """Create test data manager instance"""
    return DataManager()

@pytest.fixture
def performance_monitor():
    """Create test performance monitor instance"""
    return PerformanceMonitor()

@pytest.fixture
def sample_data():
    """Generate sample test data"""
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
    return pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'volume': np.random.randint(1000, 10000, 1000)
    })

@pytest.fixture
def selenium_driver():
    """Create Selenium WebDriver for browser tests"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

# Component Rendering Tests
class TestComponentRendering:
    """Test component rendering functionality"""
    
    def test_layout_creation(self):
        """Test that layout is created successfully"""
        layout = create_forecast_layout()
        assert layout is not None
        assert isinstance(layout, html.Div)
        assert layout.id == "app-container"
    
    def test_header_rendering(self):
        """Test header component rendering"""
        header = create_header()
        assert isinstance(header, dbc.Navbar)
        assert header.color == "primary"
        assert header.dark is True
    
    def test_control_panel_rendering(self):
        """Test control panel rendering"""
        panel = create_control_panel()
        assert isinstance(panel, dbc.Card)
        
        # Check for required components
        body = panel.children[1]
        assert isinstance(body, dbc.CardBody)
        
        # Verify dropdowns exist
        components = str(body)
        assert "symbol-dropdown" in components
        assert "timeframe-dropdown" in components
        assert "pattern-toggle" in components
    
    def test_main_chart_section(self):
        """Test main chart section rendering"""
        chart_section = create_main_chart_section()
        assert isinstance(chart_section, dbc.Card)
        
        # Check for graph component
        body = chart_section.children[1]
        assert isinstance(body, dbc.CardBody)
    
    def test_pattern_sequence_section(self):
        """Test pattern sequence section rendering"""
        sequence_section = create_pattern_sequence_section()
        assert isinstance(sequence_section, dbc.Card)
        
        # Check for slider
        header = sequence_section.children[0]
        assert "sequence-length-slider" in str(header)
    
    def test_prediction_panel(self):
        """Test prediction panel rendering"""
        panel = create_prediction_panel()
        assert isinstance(panel, dbc.Card)
        
        # Check for prediction display div
        body = panel.children[1]
        assert "prediction-display" in str(body)
    
    def test_metrics_panel(self):
        """Test metrics panel rendering"""
        panel = create_metrics_panel()
        assert isinstance(panel, dbc.Card)
        
        # Check for dropdown and graph
        header = panel.children[0]
        assert "metric-type-dropdown" in str(header)

# Callback Function Tests
class TestCallbacks:
    """Test callback functionality"""
    
    def test_main_chart_update(self, data_manager, performance_monitor):
        """Test main chart update callback"""
        # Test callback execution
        fig, pattern_data = update_main_chart(
            "BTCUSD", "1H", ["patterns"], 0
        )
        
        assert fig is not None
        assert isinstance(pattern_data, list)
        assert len(fig.data) > 0
    
    def test_pattern_sequence_update(self):
        """Test pattern sequence visualization update"""
        pattern_data = [
            {'pattern_id': 'pattern_0', 'start': 0, 'end': 50, 'confidence': 0.8},
            {'pattern_id': 'pattern_1', 'start': 60, 'end': 110, 'confidence': 0.9}
        ]
        
        fig = update_pattern_sequence(pattern_data, 5)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_prediction_generation(self):
        """Test prediction generation callback"""
        # Mock button click
        cards = update_predictions(1, "BTCUSD", 10)
        
        assert cards is not None
        assert len(cards) > 0
        assert isinstance(cards[0], dbc.Card)
    
    def test_accuracy_metrics_update(self):
        """Test accuracy metrics update"""
        # Test different metric types
        for metric_type in ['accuracy', 'returns', 'confusion']:
            fig = update_accuracy_metrics(0, metric_type)
            assert fig is not None
            assert len(fig.data) > 0
    
    def test_pattern_detail_chart(self):
        """Test pattern detail chart creation"""
        fig = create_pattern_detail_chart("pattern_0")
        assert fig is not None
        assert len(fig.data) >= 2  # Pattern and confidence band
    
    def test_pattern_stats_table(self):
        """Test pattern statistics table creation"""
        table = create_pattern_stats_table("pattern_0")
        assert table is not None
        assert isinstance(table, html.Div)
    
    def test_pattern_history_chart(self):
        """Test pattern history chart creation"""
        fig = create_pattern_history_chart()
        assert fig is not None
        assert len(fig.data) >= 2  # Occurrences and performance

# Performance Tests
class TestPerformance:
    """Test performance criteria"""
    
    def test_callback_execution_time(self, performance_monitor):
        """Test that callbacks execute within 500ms"""
        # Simulate multiple callback executions
        for _ in range(10):
            start_time = time.time()
            
            # Execute callback
            update_main_chart("BTCUSD", "1H", ["patterns"], 0)
            
            duration = time.time() - start_time
            performance_monitor.record_callback_time(duration)
        
        avg_time = performance_monitor.get_average_callback_time()
        assert avg_time < 0.5  # Less than 500ms
    
    def test_large_dataset_handling(self, data_manager):
        """Test handling of 100k+ data points"""
        # Generate large dataset
        large_dates = pd.date_range(end=datetime.now(), periods=100000, freq='1min')
        large_data = pd.DataFrame({
            'timestamp': large_dates,
            'price': 100 + np.cumsum(np.random.randn(100000) * 0.1),
            'volume': np.random.randint(1000, 10000, 100000)
        })
        
        # Cache the data
        data_manager.data_cache['TEST_LARGE'] = large_data
        
        # Test loading
        start_time = time.time()
        loaded_data = data_manager.load_data('TEST', 'LARGE')
        load_time = time.time() - start_time
        
        assert len(loaded_data) == 100000
        assert load_time < 3.0  # Should load in less than 3 seconds
    
    def test_memory_usage(self, performance_monitor):
        """Test memory usage remains stable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended usage
        for i in range(100):
            update_main_chart("BTCUSD", "1H", ["patterns"], i)
            
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (less than 100MB)
                assert memory_increase < 100
    
    def test_concurrent_requests(self, data_manager):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request(symbol):
            return data_manager.load_data(symbol, "1H")
        
        # Simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, f"SYMBOL_{i}")
                for i in range(20)
            ]
            
            results = [f.result() for f in futures]
        
        assert len(results) == 20
        assert all(isinstance(r, pd.DataFrame) for r in results)

# Layout Responsiveness Tests
class TestResponsiveness:
    """Test responsive design across devices"""
    
    def test_responsive_breakpoints(self):
        """Test responsive breakpoint configurations"""
        config = get_responsive_layout_config()
        
        assert 'mobile' in config
        assert 'tablet' in config
        assert 'desktop' in config
        
        # Verify chart heights
        assert config['mobile']['chart_height'] < config['desktop']['chart_height']
    
    @pytest.mark.parametrize("viewport_size", [
        (375, 667),   # iPhone SE
        (768, 1024),  # iPad
        (1920, 1080), # Desktop
    ])
    def test_viewport_rendering(self, dash_app, selenium_driver, viewport_size):
        """Test rendering at different viewport sizes"""
        width, height = viewport_size
        selenium_driver.set_window_size(width, height)
        
        # Navigate to app
        selenium_driver.get("http://localhost:8050")
        
        # Wait for app to load
        WebDriverWait(selenium_driver, 10).until(
            EC.presence_of_element_located((By.ID, "app-container"))
        )
        
        # Check that main components are visible
        assert selenium_driver.find_element(By.CLASS_NAME, "navbar")
        assert selenium_driver.find_element(By.ID, "main-time-series")
        
        # Verify no horizontal scroll on mobile
        if width < 768:
            # Check that viewport meta tag is present
            viewport_meta = selenium_driver.find_element(By.CSS_SELECTOR, "meta[name='viewport']")
            assert viewport_meta is not None
            
            # Check that main container doesn't exceed viewport
            container = selenium_driver.find_element(By.CLASS_NAME, "container-fluid")
            # Allow small margin for scrollbar
            assert container.size['width'] <= width + 20

# User Interaction Tests
class TestUserInteractions:
    """Test user interaction flows"""
    
    def test_symbol_selection(self, dash_duo):
        """Test symbol dropdown interaction"""
        app = import_app("src.dashboard.forecast_app")
        dash_duo.start_server(app)
        
        # Select different symbol
        dropdown = dash_duo.find_element("#symbol-dropdown")
        dropdown.click()
        
        # Select ETH/USD option
        option = dash_duo.find_element("div[id*='option-1']")
        option.click()
        
        # Verify chart updates
        dash_duo.wait_for_element("#main-time-series", timeout=5)
    
    def test_prediction_generation_flow(self, dash_duo):
        """Test prediction generation user flow"""
        app = import_app("src.dashboard.forecast_app")
        dash_duo.start_server(app)
        
        # Adjust prediction horizon
        slider = dash_duo.find_element("#prediction-horizon-slider")
        # Simulate slider interaction
        
        # Click predict button
        predict_btn = dash_duo.find_element("#predict-button")
        predict_btn.click()
        
        # Wait for predictions
        dash_duo.wait_for_element("#prediction-display .card", timeout=5)
        
        # Verify predictions displayed
        predictions = dash_duo.find_elements(".card")
        assert len(predictions) > 0
    
    def test_pattern_explorer_navigation(self, dash_duo):
        """Test pattern explorer tab navigation"""
        app = import_app("src.dashboard.forecast_app")
        dash_duo.start_server(app)
        
        # Click on pattern details tab
        details_tab = dash_duo.find_element("a[data-rb-event-key='details']")
        details_tab.click()
        
        # Verify content changes
        dash_duo.wait_for_element("#pattern-select", timeout=5)
        
        # Click on history tab
        history_tab = dash_duo.find_element("a[data-rb-event-key='history']")
        history_tab.click()
        
        # Verify history chart appears
        dash_duo.wait_for_element("#pattern-history-chart", timeout=5)

# Error Handling Tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_data_handling(self, data_manager):
        """Test handling of invalid data"""
        # Test with empty symbol
        with patch.object(data_manager, 'load_data', side_effect=Exception("Data not found")):
            fig, pattern_data = update_main_chart("INVALID", "1H", [], 0)
            
            assert isinstance(fig, go.Figure)
            assert pattern_data == []
    
    def test_network_error_handling(self, dash_duo):
        """Test handling of network errors"""
        app = import_app("src.dashboard.forecast_app")
        
        # Mock network error
        with patch('requests.get', side_effect=Exception("Network error")):
            dash_duo.start_server(app)
            
            # App should still load
            dash_duo.wait_for_element("#app-container", timeout=10)
    
    def test_callback_error_recovery(self):
        """Test callback error recovery"""
        # Test with invalid inputs
        cards = update_predictions(1, None, None)
        
        # Should return error alert
        assert isinstance(cards, dbc.Alert)
        assert cards.color == "warning"  # Warning for missing inputs
    
    def test_concurrent_callback_handling(self, dash_duo):
        """Test handling of concurrent callback executions"""
        app = import_app("src.dashboard.forecast_app")
        dash_duo.start_server(app)
        
        # Trigger multiple callbacks simultaneously
        for _ in range(5):
            dash_duo.find_element("#refresh-chart").click()
        
        # App should remain responsive
        dash_duo.wait_for_element("#main-time-series", timeout=10)

# Cross-browser Compatibility Tests
class TestCrossBrowser:
    """Test cross-browser compatibility"""
    
    @pytest.mark.parametrize("browser", ["chrome", "firefox"])
    def test_browser_compatibility(self, browser):
        """Test dashboard in different browsers"""
        if browser == "chrome":
            driver = webdriver.Chrome()
        elif browser == "firefox":
            driver = webdriver.Firefox()
        # Skip Edge due to driver issues
        # elif browser == "edge":
        #     driver = webdriver.Edge()
        
        try:
            driver.get("http://localhost:8050")
            
            # Wait for app to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "app-container"))
            )
            
            # Test basic functionality
            assert driver.find_element(By.ID, "main-time-series")
            assert driver.find_element(By.ID, "pattern-sequence-viz")
            assert driver.find_element(By.ID, "accuracy-metrics")
            
        finally:
            driver.quit()

# Integration Tests
class TestIntegration:
    """Test integration between components"""
    
    def test_data_flow_integration(self, data_manager, performance_monitor):
        """Test data flow between components"""
        # Load data
        data = data_manager.load_data("BTCUSD", "1H")
        
        # Generate patterns
        patterns = data_manager.get_patterns("BTCUSD")
        
        # Create visualizations
        fig, pattern_data = update_main_chart("BTCUSD", "1H", ["patterns"], 0)
        
        # Generate predictions
        predictions = compute_pattern_predictions("BTCUSD", 10)
        
        # Verify data consistency
        assert len(data) > 0
        assert len(patterns) > 0
        assert len(pattern_data) > 0
        assert predictions['confidence'] > 0
    
    def test_performance_monitoring_integration(self, performance_monitor):
        """Test performance monitoring integration"""
        # Execute multiple operations
        for _ in range(20):
            start = time.time()
            update_main_chart("BTCUSD", "1H", ["patterns"], 0)
            performance_monitor.record_callback_time(time.time() - start)
        
        # Check performance status
        perf_status = performance_monitor.check_performance()
        assert 'meets_criteria' in perf_status
        assert 'avg_callback_time' in perf_status

# Accessibility Tests
class TestAccessibility:
    """Test accessibility features"""
    
    def test_keyboard_navigation(self, selenium_driver):
        """Test keyboard navigation"""
        selenium_driver.get("http://localhost:8050")
        
        # Tab through interactive elements
        body = selenium_driver.find_element(By.TAG_NAME, "body")
        
        # Simulate tab key presses
        for _ in range(10):
            body.send_keys(Keys.TAB)
            
            # Check that focused element is visible
            focused = selenium_driver.switch_to.active_element
            assert focused.is_displayed()
    
    def test_aria_labels(self, selenium_driver):
        """Test ARIA labels and roles"""
        selenium_driver.get("http://localhost:8050")
        
        # Check for proper ARIA attributes
        buttons = selenium_driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            # Buttons should have accessible text or aria-label
            text = button.text or button.get_attribute("aria-label")
            assert text
    
    def test_color_contrast(self):
        """Test color contrast ratios"""
        # This would use a tool like axe-core
        # For now, we'll verify CSS variables are defined
        with open("assets/forecast_dashboard.css", "r") as f:
            css_content = f.read()
            
        assert "--primary-color" in css_content
        assert "--success-color" in css_content
        assert "--danger-color" in css_content

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
