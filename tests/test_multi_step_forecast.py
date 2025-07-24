"""
Tests for Multi-Step Forecasting Engine
"""

import pytest
import numpy as np
import json
from datetime import datetime
from src.dashboard.advanced.multi_step_forecast import (
    MultiStepForecastEngine, 
    ForecastNode, 
    ForecastPath
)


class TestForecastNode:
    """Test suite for ForecastNode dataclass"""
    
    def test_forecast_node_creation(self):
        """Test basic ForecastNode creation"""
        node = ForecastNode(
            pattern_id="test_pattern",
            pattern_type="continuation",
            confidence=0.85,
            probability=0.75,
            expected_return=0.05,
            time_horizon=1
        )
        
        assert node.pattern_id == "test_pattern"
        assert node.pattern_type == "continuation"
        assert node.confidence == 0.85
        assert node.children == []
    
    def test_forecast_node_with_children(self):
        """Test ForecastNode with children"""
        child = ForecastNode(
            pattern_id="child_pattern",
            pattern_type="reversal",
            confidence=0.70,
            probability=0.60,
            expected_return=0.03,
            time_horizon=2
        )
        
        parent = ForecastNode(
            pattern_id="parent_pattern",
            pattern_type="continuation",
            confidence=0.90,
            probability=0.80,
            expected_return=0.04,
            time_horizon=1,
            children=[child]
        )
        
        assert len(parent.children) == 1
        assert parent.children[0].pattern_id == "child_pattern"


class TestForecastPath:
    """Test suite for ForecastPath dataclass"""
    
    def test_forecast_path_creation(self):
        """Test basic ForecastPath creation"""
        path = ForecastPath(
            patterns=["pattern1", "pattern2", "pattern3"],
            probabilities=[0.8, 0.7, 0.6],
            cumulative_probability=0.336,
            expected_return=0.15,
            confidence_decay=0.729,
            path_length=3
        )
        
        assert len(path.patterns) == 3
        assert path.cumulative_probability == 0.336
        assert path.path_length == 3


class TestMultiStepForecastEngine:
    """Test suite for MultiStepForecastEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create MultiStepForecastEngine instance"""
        return MultiStepForecastEngine(max_steps=3, confidence_decay=0.9)
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.max_steps == 3
        assert engine.confidence_decay == 0.9
        assert engine.classifier is not None
        assert engine.predictor is not None
        assert engine.analyzer is not None
    
    def test_build_forecast_tree(self, engine):
        """Test forecast tree building"""
        current_patterns = ["ascending_triangle", "bull_flag"]
        market_context = {
            'trend': 'upward',
            'volatility': 0.15,
            'volume': 'increasing'
        }
        
        tree = engine.build_forecast_tree(current_patterns, market_context)
        
        assert tree is not None
        assert tree.pattern_id == "root"
        assert tree.confidence == 1.0
        assert isinstance(tree.children, list)
    
    def test_calculate_path_probabilities(self, engine):
        """Test path probability calculation"""
        # Create a simple tree
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        child = ForecastNode(
            pattern_id="child",
            pattern_type="continuation",
            confidence=0.8,
            probability=0.7,
            expected_return=0.05,
            time_horizon=1
        )
        
        root.children = [child]
        
        paths = engine.calculate_path_probabilities(root)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        path = paths[0]
        assert isinstance(path, ForecastPath)
        assert path.path_length > 0
    
    def test_visualize_forecast_tree(self, engine):
        """Test tree visualization"""
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        fig = engine.visualize_forecast_tree(root)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_scenario_analysis(self, engine):
        """Test scenario analysis creation"""
        paths = [
            ForecastPath(
                patterns=["p1", "p2", "p3"],
                probabilities=[0.8, 0.7, 0.6],
                cumulative_probability=0.336,
                expected_return=0.15,
                confidence_decay=0.729,
                path_length=3
            ),
            ForecastPath(
                patterns=["p1", "p4", "p5"],
                probabilities=[0.8, 0.6, 0.5],
                cumulative_probability=0.24,
                expected_return=0.10,
                confidence_decay=0.729,
                path_length=3
            )
        ]
        
        scenarios = engine.create_scenario_analysis(paths, num_scenarios=2)
        
        assert isinstance(scenarios, dict)
        assert 'scenarios' in scenarios
        assert 'summary' in scenarios
        assert len(scenarios['scenarios']) == 2
    
    def test_create_confidence_decay_chart(self, engine):
        """Test confidence decay chart creation"""
        paths = [
            ForecastPath(
                patterns=["p1", "p2"],
                probabilities=[0.8, 0.7],
                cumulative_probability=0.56,
                expected_return=0.10,
                confidence_decay=0.81,
                path_length=2
            )
        ]
        
        fig = engine.create_confidence_decay_chart(paths)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_generate_forecast_report(self, engine):
        """Test forecast report generation"""
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        paths = [
            ForecastPath(
                patterns=["p1", "p2"],
                probabilities=[0.8, 0.7],
                cumulative_probability=0.56,
                expected_return=0.10,
                confidence_decay=0.81,
                path_length=2
            )
        ]
        
        report = engine.generate_forecast_report(root, paths)
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'scenarios' in report
        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)
    
    def test_calculate_risk_score(self, engine):
        """Test risk score calculation"""
        path = ForecastPath(
            patterns=["p1", "p2", "p3"],
            probabilities=[0.8, 0.7, 0.6],
            cumulative_probability=0.336,
            expected_return=0.15,
            confidence_decay=0.729,
            path_length=3
        )
        
        risk_score = engine._calculate_risk_score(path)
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 1
    
    def test_estimate_volatility(self, engine):
        """Test volatility estimation"""
        path = ForecastPath(
            patterns=["p1", "p2"],
            probabilities=[0.8, 0.7],
            cumulative_probability=0.56,
            expected_return=0.10,
            confidence_decay=0.81,
            path_length=2
        )
        
        volatility = engine._estimate_volatility(path)
        
        assert isinstance(volatility, float)
        assert volatility > 0
    
    def test_empty_tree(self, engine):
        """Test handling of empty tree"""
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        paths = engine.calculate_path_probabilities(root)
        
        assert isinstance(paths, list)
        assert len(paths) == 1  # Should have root path
    
    def test_max_steps_limit(self, engine):
        """Test max steps limit enforcement"""
        # Create a deep tree manually
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        # Add many levels of children
        current = root
        for i in range(5):  # More than max_steps
            child = ForecastNode(
                pattern_id=f"level_{i+1}",
                pattern_type="continuation",
                confidence=0.9 - i * 0.1,
                probability=0.8,
                expected_return=0.02,
                time_horizon=i + 1
            )
            current.children = [child]
            current = child
        
        paths = engine.calculate_path_probabilities(root)
        
        # Should respect max_steps limit
        max_path_length = max([p.path_length for p in paths])
        assert max_path_length <= engine.max_steps + 1  # +1 for root
    
    def test_confidence_decay_calculation(self, engine):
        """Test confidence decay calculation"""
        path = ForecastPath(
            patterns=["p1", "p2", "p3"],
            probabilities=[0.8, 0.7, 0.6],
            cumulative_probability=0.336,
            expected_return=0.15,
            confidence_decay=0.729,  # 0.9^3
            path_length=3
        )
        
        expected_decay = 0.9 ** 3
        assert abs(path.confidence_decay - expected_decay) < 0.001


class TestMultiStepForecastIntegration:
    """Integration tests for MultiStepForecastEngine"""
    
    def test_full_workflow(self):
        """Test complete multi-step forecasting workflow"""
        engine = MultiStepForecastEngine(max_steps=2)
        
        current_patterns = ["ascending_triangle"]
        market_context = {
            'trend': 'upward',
            'volatility': 0.15,
            'volume': 'increasing'
        }
        
        # Build tree
        tree = engine.build_forecast_tree(current_patterns, market_context)
        assert tree is not None
        
        # Calculate paths
        paths = engine.calculate_path_probabilities(tree)
        assert len(paths) > 0
        
        # Create scenarios
        scenarios = engine.create_scenario_analysis(paths)
        assert 'scenarios' in scenarios
        
        # Generate report
        report = engine.generate_forecast_report(tree, paths)
        assert 'recommendations' in report
        
        # Create visualizations
        fig_tree = engine.visualize_forecast_tree(tree)
        assert fig_tree is not None
        
        fig_decay = engine.create_confidence_decay_chart(paths)
        assert fig_decay is not None
    
    def test_recommendation_generation(self):
        """Test recommendation generation logic"""
        engine = MultiStepForecastEngine()
        
        # Test high return scenario
        scenarios = {
            'scenarios': [
                {'expected_return': 0.08, 'risk_score': 0.3},
                {'expected_return': 0.12, 'risk_score': 0.4}
            ],
            'summary': {
                'best_scenario': {'expected_return': 0.12},
                'worst_scenario': {'expected_return': 0.05},
                'max_risk': 0.4
            }
        }
        
        recommendations = engine._generate_recommendations(scenarios)
        assert len(recommendations) > 0
        assert any("long position" in rec.lower() for rec in recommendations)
    
    def test_risk_assessment(self):
        """Test risk assessment calculation"""
        engine = MultiStepForecastEngine()
        
        scenarios = {
            'scenarios': [
                {'expected_return': 0.05, 'risk_score': 0.3, 'cumulative_probability': 0.5},
                {'expected_return': -0.02, 'risk_score': 0.7, 'cumulative_probability': 0.3}
            ]
        }
        
        risk_assessment = engine._assess_risk(scenarios)
        
        assert isinstance(risk_assessment, dict)
        assert 'max_drawdown_risk' in risk_assessment
        assert 'probability_weighted_risk' in risk_assessment
        assert risk_assessment['max_drawdown_risk'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
