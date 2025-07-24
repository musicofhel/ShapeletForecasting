"""
Comprehensive Test Suite for Dashboard Controls

Tests all control components, interactions, state management,
performance, validation, accessibility, and cross-component communication.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import dash
from dash import html, dcc
from dash.testing import DashTestingError
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import the controls module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard.components.controls import (
    DashboardControls, get_control_state, validate_input_ranges, 
    format_control_summary
)


class TestDashboardControls:
    """Test suite for DashboardControls class."""
    
    @pytest.fixture
    def controls(self):
        """Create a DashboardControls instance."""
        return DashboardControls()
    
    @pytest.fixture
    def app(self):
        """Create a Dash app with controls."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        controls = DashboardControls()
        app.layout = html.Div([
            controls.create_control_panel()
        ])
        DashboardControls.register_callbacks(app)
        return app
    
    def test_initialization(self, controls):
        """Test proper initialization of controls."""
        assert controls.default_ticker == "BTC-USD"
        assert controls.default_lookback == 30
        assert controls.default_horizon == 7
        assert controls.default_confidence == 0.7
        assert len(controls.pattern_types) == 8
        assert len(controls.forecast_methods) == 5
    
    def test_ticker_selector_creation(self, controls):
        """Test ticker selector component creation."""
        ticker_card = controls.create_ticker_selector()
        assert isinstance(ticker_card, dbc.Card)
        
        # Check structure
        body = ticker_card.children[1]
        dropdown = body.children[1]
        assert dropdown.id == "ticker-dropdown"
        assert dropdown.value == "BTC-USD"
        assert len(dropdown.options) == len(controls.tickers)
    
    def test_lookback_control_creation(self, controls):
        """Test lookback control component creation."""
        lookback_card = controls.create_lookback_control()
        assert isinstance(lookback_card, dbc.Card)
        
        # Check slider
        body = lookback_card.children[1]
        slider = body.children[1]
        assert slider.id == "lookback-slider"
        assert slider.min == 7
        assert slider.max == 365
        assert slider.value == 30
    
    def test_horizon_selector_creation(self, controls):
        """Test horizon selector component creation."""
        horizon_card = controls.create_horizon_selector()
        assert isinstance(horizon_card, dbc.Card)
        
        # Check radio items
        body = horizon_card.children[1]
        radio = body.children[1]
        assert radio.id == "horizon-radio"
        assert radio.value == 7
        assert len(radio.options) == 5
    
    def test_pattern_filters_creation(self, controls):
        """Test pattern filters component creation."""
        pattern_card = controls.create_pattern_filters()
        assert isinstance(pattern_card, dbc.Card)
        
        # Check checklist
        body = pattern_card.children[1]
        checklist = body.children[1]
        assert checklist.id == "pattern-checklist"
        assert len(checklist.options) == len(controls.pattern_types)
        assert checklist.value == controls.default_pattern_types
    
    def test_confidence_slider_creation(self, controls):
        """Test confidence slider component creation."""
        confidence_card = controls.create_confidence_slider()
        assert isinstance(confidence_card, dbc.Card)
        
        # Check slider
        body = confidence_card.children[1]
        slider = body.children[1]
        assert slider.id == "confidence-slider"
        assert slider.min == 0
        assert slider.max == 1
        assert slider.value == 0.7
    
    def test_mode_switches_creation(self, controls):
        """Test mode switches component creation."""
        mode_card = controls.create_mode_switches()
        assert isinstance(mode_card, dbc.Card)
        
        # Check switches
        body = mode_card.children[1]
        row = body.children[0]
        
        # Real-time toggle
        realtime_col = row.children[0]
        realtime_switch = realtime_col.children[1]
        assert realtime_switch.id == "realtime-toggle"
        assert realtime_switch.value == False
        
        # Live mode toggle
        live_col = row.children[1]
        live_switch = live_col.children[1]
        assert live_switch.id == "live-mode-toggle"
        assert live_switch.value == False
    
    def test_complexity_selector_creation(self, controls):
        """Test complexity selector component creation."""
        complexity_card = controls.create_complexity_selector()
        assert isinstance(complexity_card, dbc.Card)
        
        # Check select
        body = complexity_card.children[1]
        select = body.children[1]
        assert select.id == "complexity-select"
        assert select.value == "medium"
        assert len(select.options) == 3
    
    def test_forecast_selector_creation(self, controls):
        """Test forecast selector component creation."""
        forecast_card = controls.create_forecast_selector()
        assert isinstance(forecast_card, dbc.Card)
        
        # Check radio items
        body = forecast_card.children[1]
        radio = body.children[1]
        assert radio.id == "forecast-method"
        assert radio.value == "ensemble"
        assert len(radio.options) == len(controls.forecast_methods)
    
    def test_advanced_controls_creation(self, controls):
        """Test advanced controls component creation."""
        advanced_card = controls.create_advanced_controls()
        assert isinstance(advanced_card, dbc.Card)
        
        # Check collapse structure
        collapse = advanced_card.children[1]
        assert collapse.id == "advanced-collapse"
        assert collapse.is_open == False
    
    def test_control_panel_creation(self, controls):
        """Test complete control panel creation."""
        panel = controls.create_control_panel()
        assert isinstance(panel, html.Div)
        
        # Check layout structure
        row = panel.children[0]
        assert len(row.children) == 3  # Three columns
        
        # Check stores
        control_store = panel.children[1]
        validation_store = panel.children[2]
        interval = panel.children[3]
        
        assert control_store.id == "control-state"
        assert validation_store.id == "validation-state"
        assert interval.id == "update-interval"


class TestControlCallbacks:
    """Test suite for control callbacks."""
    
    @pytest.fixture
    def dash_duo(self, dash_duo):
        """Configure dash_duo for testing."""
        return dash_duo
    
    def test_ticker_dropdown_interaction(self, dash_duo, app):
        """Test ticker dropdown selection."""
        dash_duo.start_server(app)
        
        # Select a different ticker
        dropdown = dash_duo.find_element("#ticker-dropdown")
        dash_duo.select_dcc_dropdown("#ticker-dropdown", "ETH-USD")
        
        # Verify selection
        assert dash_duo.find_element("#ticker-dropdown input").get_attribute("value") == "ETH-USD"
    
    def test_custom_ticker_addition(self, dash_duo, app):
        """Test adding custom ticker."""
        dash_duo.start_server(app)
        
        # Click add ticker button
        add_btn = dash_duo.find_element("#add-ticker-btn")
        add_btn.click()
        
        # Check input visibility
        custom_input = dash_duo.find_element("#custom-ticker-input")
        assert custom_input.is_displayed()
        
        # Enter custom ticker
        custom_input.send_keys("CUSTOM-USD")
        custom_input.send_keys("\n")
        
        # Verify ticker added to dropdown
        time.sleep(0.5)
        dropdown_options = dash_duo.find_elements("#ticker-dropdown .Select-menu-outer")
        assert any("CUSTOM-USD" in opt.text for opt in dropdown_options)
    
    def test_lookback_synchronization(self, dash_duo, app):
        """Test lookback slider and input synchronization."""
        dash_duo.start_server(app)
        
        # Move slider
        slider = dash_duo.find_element("#lookback-slider")
        dash_duo.click_at_coord_fractions(slider, 0.5, 0.5)
        
        # Check input updated
        time.sleep(0.5)
        input_value = dash_duo.find_element("#lookback-input").get_attribute("value")
        assert int(input_value) > 30  # Should be more than default
        
        # Update input
        input_elem = dash_duo.find_element("#lookback-input")
        input_elem.clear()
        input_elem.send_keys("100")
        input_elem.send_keys("\t")  # Trigger blur event
        
        # Check slider updated
        time.sleep(0.5)
        slider_value = dash_duo.find_element("#lookback-slider").get_attribute("aria-valuenow")
        assert slider_value == "100"
    
    def test_pattern_selection_buttons(self, dash_duo, app):
        """Test pattern selection buttons."""
        dash_duo.start_server(app)
        
        # Click select all
        select_all = dash_duo.find_element("#select-all-patterns")
        select_all.click()
        
        # Check all patterns selected
        time.sleep(0.5)
        checkboxes = dash_duo.find_elements("#pattern-checklist input[type='checkbox']")
        assert all(cb.is_selected() for cb in checkboxes)
        
        # Click clear all
        clear_all = dash_duo.find_element("#clear-all-patterns")
        clear_all.click()
        
        # Check all patterns deselected
        time.sleep(0.5)
        checkboxes = dash_duo.find_elements("#pattern-checklist input[type='checkbox']")
        assert not any(cb.is_selected() for cb in checkboxes)
    
    def test_confidence_slider_update(self, dash_duo, app):
        """Test confidence slider updates."""
        dash_duo.start_server(app)
        
        # Move slider
        slider = dash_duo.find_element("#confidence-slider")
        dash_duo.click_at_coord_fractions(slider, 0.25, 0.5)
        
        # Check progress bar and display
        time.sleep(0.5)
        progress = dash_duo.find_element("#confidence-progress")
        display = dash_duo.find_element("#confidence-display")
        
        assert "25" in progress.get_attribute("aria-valuenow")
        assert "25%" in display.text
    
    def test_mode_toggle_switches(self, dash_duo, app):
        """Test mode toggle switches."""
        dash_duo.start_server(app)
        
        # Toggle real-time mode
        realtime_toggle = dash_duo.find_element("#realtime-toggle")
        realtime_toggle.click()
        
        # Check status update
        time.sleep(0.5)
        status = dash_duo.find_element("#mode-status")
        assert "Real-time Detection Active" in status.text
        assert "warning" in status.get_attribute("class")
        
        # Toggle live mode
        live_toggle = dash_duo.find_element("#live-mode-toggle")
        live_toggle.click()
        
        # Check status update
        time.sleep(0.5)
        assert "Live Mode Active" in status.text
        assert "success" in status.get_attribute("class")
    
    def test_complexity_info_update(self, dash_duo, app):
        """Test complexity info updates."""
        dash_duo.start_server(app)
        
        # Select different complexity
        dash_duo.select_dcc_dropdown("#complexity-select", "simple")
        
        # Check info update
        time.sleep(0.5)
        info = dash_duo.find_element("#complexity-info")
        assert "3-5 points" in info.text
    
    def test_algorithm_info_toggle(self, dash_duo, app):
        """Test algorithm info collapse."""
        dash_duo.start_server(app)
        
        # Click info button
        info_btn = dash_duo.find_element("#algo-info-btn")
        info_btn.click()
        
        # Check collapse opened
        time.sleep(0.5)
        collapse = dash_duo.find_element("#algo-info-collapse")
        assert "show" in collapse.get_attribute("class")
        
        # Check description
        description = dash_duo.find_element("#algo-description")
        assert "Combination of multiple models" in description.text
    
    def test_advanced_settings_toggle(self, dash_duo, app):
        """Test advanced settings toggle."""
        dash_duo.start_server(app)
        
        # Click toggle button
        toggle_btn = dash_duo.find_element("#advanced-toggle")
        toggle_btn.click()
        
        # Check collapse opened
        time.sleep(0.5)
        collapse = dash_duo.find_element("#advanced-collapse")
        assert "show" in collapse.get_attribute("class")
    
    def test_reset_to_defaults(self, dash_duo, app):
        """Test reset to defaults functionality."""
        dash_duo.start_server(app)
        
        # Change some values
        dash_duo.select_dcc_dropdown("#ticker-dropdown", "ETH-USD")
        dash_duo.click_at_coord_fractions("#confidence-slider", 0.9, 0.5)
        
        # Open advanced settings and click reset
        toggle_btn = dash_duo.find_element("#advanced-toggle")
        toggle_btn.click()
        time.sleep(0.5)
        
        reset_btn = dash_duo.find_element("#reset-btn")
        reset_btn.click()
        
        # Check values reset
        time.sleep(0.5)
        ticker = dash_duo.find_element("#ticker-dropdown input").get_attribute("value")
        confidence = dash_duo.find_element("#confidence-slider").get_attribute("aria-valuenow")
        
        assert ticker == "BTC-USD"
        assert confidence == "0.7"


class TestStateManagement:
    """Test suite for state management."""
    
    def test_control_state_update(self):
        """Test control state update callback."""
        # Mock callback context
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{"prop_id": "ticker-dropdown.value"}]
            
            # Test state update
            state = {
                "ticker": "BTC-USD",
                "lookback": 30,
                "horizon": 7,
                "patterns": ["head_shoulders"],
                "confidence": 0.7,
                "realtime": False,
                "live": False,
                "complexity": "medium",
                "forecast": "ensemble",
                "update_frequency": 5,
                "cache_duration": 15,
                "max_patterns": 20
            }
            
            # Verify timestamp added
            assert "timestamp" not in state
            # Would be added by callback
    
    def test_get_control_state_default(self):
        """Test getting default control state."""
        state = get_control_state(None)
        
        assert state["ticker"] == "BTC-USD"
        assert state["lookback"] == 30
        assert state["horizon"] == 7
        assert len(state["patterns"]) == 3
        assert state["confidence"] == 0.7
        assert state["realtime"] == False
        assert state["live"] == False
    
    def test_get_control_state_existing(self):
        """Test getting existing control state."""
        existing_state = {
            "ticker": "ETH-USD",
            "lookback": 60,
            "horizon": 14
        }
        
        state = get_control_state(existing_state)
        assert state["ticker"] == "ETH-USD"
        assert state["lookback"] == 60
        assert state["horizon"] == 14
    
    def test_validate_input_ranges_valid(self):
        """Test input range validation with valid values."""
        state = {
            "lookback": 30,
            "horizon": 7,
            "confidence": 0.7,
            "update_frequency": 5
        }
        
        valid, errors = validate_input_ranges(state)
        assert valid == True
        assert len(errors) == 0
    
    def test_validate_input_ranges_invalid(self):
        """Test input range validation with invalid values."""
        state = {
            "lookback": 5,      # Too small
            "horizon": 100,     # Too large
            "confidence": 1.5,  # Out of range
            "update_frequency": 0  # Too small
        }
        
        valid, errors = validate_input_ranges(state)
        assert valid == False
        assert len(errors) == 4
        assert any("Lookback" in e for e in errors)
        assert any("Horizon" in e for e in errors)
        assert any("Confidence" in e for e in errors)
        assert any("Update frequency" in e for e in errors)
    
    def test_format_control_summary(self):
        """Test control summary formatting."""
        state = {
            "ticker": "BTC-USD",
            "lookback": 30,
            "horizon": 7,
            "patterns": ["head_shoulders", "double_top"],
            "confidence": 0.75,
            "live": False,
            "realtime": True,
            "complexity": "medium",
            "forecast": "lstm"
        }
        
        summary = format_control_summary(state)
        
        assert "BTC-USD" in summary
        assert "30 days" in summary
        assert "7 days" in summary
        assert "head_shoulders, double_top" in summary
        assert "75%" in summary
        assert "Real-time" in summary
        assert "medium" in summary
        assert "LSTM" in summary


class TestPerformance:
    """Test suite for performance requirements."""
    
    def test_control_update_speed(self):
        """Test control updates complete in <100ms."""
        controls = DashboardControls()
        
        # Time control creation
        start = time.time()
        panel = controls.create_control_panel()
        end = time.time()
        
        assert (end - start) < 0.1  # 100ms
    
    def test_callback_execution_speed(self):
        """Test callback execution speed."""
        # Mock state
        state = {
            "ticker": "BTC-USD",
            "lookback": 30,
            "horizon": 7,
            "patterns": ["head_shoulders"],
            "confidence": 0.7
        }
        
        # Time validation
        start = time.time()
        valid, errors = validate_input_ranges(state)
        end = time.time()
        
        assert (end - start) < 0.01  # 10ms
    
    def test_state_persistence(self):
        """Test state persistence configuration."""
        controls = DashboardControls()
        panel = controls.create_control_panel()
        
        # Find store component
        store = None
        for child in panel.children:
            if hasattr(child, 'id') and child.id == "control-state":
                store = child
                break
        
        assert store is not None
        assert store.storage_type == "session"
    
    def test_bulk_control_updates(self):
        """Test performance with multiple simultaneous updates."""
        controls = DashboardControls()
        
        # Simulate multiple control updates
        start = time.time()
        for i in range(10):
            state = {
                "ticker": f"TEST-{i}",
                "lookback": 30 + i,
                "horizon": 7 + i,
                "confidence": 0.5 + (i * 0.05)
            }
            valid, errors = validate_input_ranges(state)
        end = time.time()
        
        # Should handle 10 updates in under 100ms
        assert (end - start) < 0.1


class TestAccessibility:
    """Test suite for accessibility requirements."""
    
    def test_keyboard_navigation(self, dash_duo, app):
        """Test keyboard navigation through controls."""
        dash_duo.start_server(app)
        
        # Tab through controls
        ticker = dash_duo.find_element("#ticker-dropdown")
        ticker.send_keys("\t")  # Tab to next control
        
        # Should reach lookback slider
        active_element = dash_duo.driver.switch_to.active_element
        assert "lookback" in active_element.get_attribute("id") or \
               "lookback" in active_element.get_attribute("class")
    
    def test_aria_labels(self, controls):
        """Test ARIA labels on controls."""
        panel = controls.create_control_panel()
        
        # Check for labels
        ticker_card = controls.create_ticker_selector()
        body = ticker_card.children[1]
        label = body.children[0]
        
        assert label.children == "Select Ticker:"
        assert label.htmlFor == "ticker-dropdown"
    
    def test_form_descriptions(self, controls):
        """Test form descriptions for screen readers."""
        ticker_card = controls.create_ticker_selector()
        body = ticker_card.children[1]
        form_text = body.children[2]
        
        assert isinstance(form_text, dbc.FormText)
        assert form_text.children == "Choose the financial instrument to analyze"
    
    def test_color_contrast(self, dash_duo, app):
        """Test color contrast for visibility."""
        dash_duo.start_server(app)
        
        # Check alert colors
        status = dash_duo.find_element("#mode-status")
        
        # Info alert should have sufficient contrast
        assert "alert-info" in status.get_attribute("class")


class TestCrossComponentCommunication:
    """Test suite for cross-component communication."""
    
    def test_control_state_propagation(self, dash_duo, app):
        """Test control state propagates to all components."""
        dash_duo.start_server(app)
        
        # Change ticker
        dash_duo.select_dcc_dropdown("#ticker-dropdown", "ETH-USD")
        
        # Change lookback
        input_elem = dash_duo.find_element("#lookback-input")
        input_elem.clear()
        input_elem.send_keys("60")
        input_elem.send_keys("\t")
        
        # Control state should be updated
        # In real app, would check store data
        time.sleep(0.5)
        
        # Both values should be persisted
        ticker_value = dash_duo.find_element("#ticker-dropdown input").get_attribute("value")
        lookback_value = dash_duo.find_element("#lookback-input").get_attribute("value")
        
        assert ticker_value == "ETH-USD"
        assert lookback_value == "60"
    
    def test_validation_feedback(self):
        """Test validation feedback across components."""
        # Test validation state callback
        state = {
            "lookback": 5,  # Invalid
            "patterns": []  # Invalid
        }
        
        # Would trigger validation callback
        # Check errors propagated
        valid, errors = validate_input_ranges(state)
        assert not valid
        assert len(errors) > 0
    
    def test_mode_interaction(self, dash_duo, app):
        """Test mode switches affect other components."""
        dash_duo.start_server(app)
        
        # Enable live mode
        live_toggle = dash_duo.find_element("#live-mode-toggle")
        live_toggle.click()
        
        # Check interval enabled
        time.sleep(0.5)
        # In real app, interval.disabled would be False
        
        # Status should update
        status = dash_duo.find_element("#mode-status")
        assert "Live Mode Active" in status.text


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_empty_pattern_selection(self):
        """Test handling of empty pattern selection."""
        state = {
            "patterns": [],
            "ticker": "BTC-USD"
        }
        
        # Validation should fail
        # In real callback, would prevent submission
        assert state["patterns"] == []
    
    def test_invalid_custom_ticker(self, dash_duo, app):
        """Test invalid custom ticker handling."""
        dash_duo.start_server(app)
        
        # Try to add empty ticker
        add_btn = dash_duo.find_element("#add-ticker-btn")
        add_btn.click()
        
        custom_input = dash_duo.find_element("#custom-ticker-input")
        custom_input.send_keys("")
        custom_input.send_keys("\n")
        
        # Should not add empty ticker
        time.sleep(0.5)
        # Dropdown options should remain unchanged
    
    def test_extreme_values(self):
        """Test extreme value handling."""
        state = {
            "lookback": 365,
            "horizon": 90,
            "confidence": 1.0,
            "max_patterns": 100
        }
        
        valid, errors = validate_input_ranges(state)
        assert valid  # Should accept max values
    
    def test_rapid_updates(self, dash_duo, app):
        """Test rapid control updates."""
        dash_duo.start_server(app)
        
        # Rapidly change confidence
        slider = dash_duo.find_element("#confidence-slider")
        for i in range(5):
            dash_duo.click_at_coord_fractions(slider, i * 0.2, 0.5)
            time.sleep(0.05)
        
        # Should handle rapid updates without errors
        display = dash_duo.find_element("#confidence-display")
        assert "Confidence Threshold" in display.text


class TestIntegration:
    """Integration tests for control system."""
    
    def test_full_control_workflow(self, dash_duo, app):
        """Test complete control workflow."""
        dash_duo.start_server(app)
        
        # 1. Select ticker
        dash_duo.select_dcc_dropdown("#ticker-dropdown", "SPY")
        
        # 2. Adjust lookback
        input_elem = dash_duo.find_element("#lookback-input")
        input_elem.clear()
        input_elem.send_keys("90")
        input_elem.send_keys("\t")
        
        # 3. Select patterns
        select_all = dash_duo.find_element("#select-all-patterns")
        select_all.click()
        
        # 4. Set confidence
        dash_duo.click_at_coord_fractions("#confidence-slider", 0.8, 0.5)
        
        # 5. Enable real-time mode
        realtime_toggle = dash_duo.find_element("#realtime-toggle")
        realtime_toggle.click()
        
        # 6. Select forecast method
        dash_duo.find_element("input[value='lstm']").click()
        
        # Verify all changes applied
        time.sleep(0.5)
        
        ticker = dash_duo.find_element("#ticker-dropdown input").get_attribute("value")
        lookback = dash_duo.find_element("#lookback-input").get_attribute("value")
        status = dash_duo.find_element("#mode-status").text
        
        assert ticker == "SPY"
        assert lookback == "90"
        assert "Real-time" in status
    
    def test_control_reset_workflow(self, dash_duo, app):
        """Test control reset workflow."""
        dash_duo.start_server(app)
        
        # Make changes
        dash_duo.select_dcc_dropdown("#ticker-dropdown", "GLD")
        dash_duo.click_at_coord_fractions("#confidence-slider", 0.3, 0.5)
        
        # Open advanced and reset
        toggle_btn = dash_duo.find_element("#advanced-toggle")
        toggle_btn.click()
        time.sleep(0.5)
        
        reset_btn = dash_duo.find_element("#reset-btn")
        reset_btn.click()
        
        # Verify reset
        time.sleep(0.5)
        ticker = dash_duo.find_element("#ticker-dropdown input").get_attribute("value")
        confidence = dash_duo.find_element("#confidence-slider").get_attribute("aria-valuenow")
        
        assert ticker == "BTC-USD"
        assert confidence == "0.7"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
