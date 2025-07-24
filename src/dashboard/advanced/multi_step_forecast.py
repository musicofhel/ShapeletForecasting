"""
Multi-Step Forecasting Engine for Wavelet Pattern Forecasting Dashboard
Provides advanced forecasting capabilities including sequence prediction and scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.preprocessing import StandardScaler
import json

from ..pattern_classifier import PatternClassifier
from ..pattern_predictor import PatternPredictor
from ..wavelet_sequence_analyzer import WaveletSequenceAnalyzer


@dataclass
class ForecastNode:
    """Represents a node in the forecast tree"""
    pattern_id: str
    pattern_type: str
    confidence: float
    probability: float
    expected_return: float
    time_horizon: int
    children: List['ForecastNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class ForecastPath:
    """Represents a complete forecast path"""
    patterns: List[str]
    probabilities: List[float]
    cumulative_probability: float
    expected_return: float
    confidence_decay: float
    path_length: int


class MultiStepForecastEngine:
    """Advanced multi-step forecasting engine"""
    
    def __init__(self, max_steps: int = 5, confidence_decay: float = 0.9):
        self.max_steps = max_steps
        self.confidence_decay = confidence_decay
        self.classifier = PatternClassifier()
        self.predictor = PatternPredictor()
        self.analyzer = WaveletSequenceAnalyzer()
        
    def build_forecast_tree(self, 
                          current_patterns: List[str], 
                          market_context: Dict[str, Any]) -> ForecastNode:
        """Build a tree of possible future pattern sequences"""
        
        root = ForecastNode(
            pattern_id="root",
            pattern_type="current",
            confidence=1.0,
            probability=1.0,
            expected_return=0.0,
            time_horizon=0
        )
        
        self._build_tree_recursive(
            node=root,
            current_patterns=current_patterns,
            market_context=market_context,
            depth=0
        )
        
        return root
    
    def _build_tree_recursive(self,
                            node: ForecastNode,
                            current_patterns: List[str],
                            market_context: Dict[str, Any],
                            depth: int):
        """Recursively build the forecast tree"""
        
        if depth >= self.max_steps:
            return
        
        # Get next pattern predictions
        predictions = self.predictor.predict_next_patterns(
            current_patterns[-3:],  # Use last 3 patterns for context
            market_context
        )
        
        for pred in predictions[:3]:  # Limit to top 3 predictions
            child_node = ForecastNode(
                pattern_id=pred['pattern_id'],
                pattern_type=pred['pattern_type'],
                confidence=pred['confidence'] * (self.confidence_decay ** depth),
                probability=pred['probability'],
                expected_return=pred['expected_return'],
                time_horizon=depth + 1
            )
            
            new_patterns = current_patterns + [pred['pattern_id']]
            self._build_tree_recursive(
                child_node,
                new_patterns,
                market_context,
                depth + 1
            )
            
            node.children.append(child_node)
    
    def calculate_path_probabilities(self, root: ForecastNode) -> List[ForecastPath]:
        """Calculate probabilities for all possible paths"""
        paths = []
        self._collect_paths(root, [], [], 1.0, 0.0, 1.0, paths)
        return paths
    
    def _collect_paths(self,
                      node: ForecastNode,
                      current_patterns: List[str],
                      current_probs: List[float],
                      cumulative_prob: float,
                      expected_return: float,
                      confidence: float,
                      paths: List[ForecastPath]):
        """Collect all paths from the tree"""
        
        if not node.children:
            # Leaf node - create path
            path = ForecastPath(
                patterns=current_patterns + [node.pattern_id],
                probabilities=current_probs + [node.probability],
                cumulative_probability=cumulative_prob * node.probability,
                expected_return=expected_return + node.expected_return,
                confidence_decay=confidence * self.confidence_decay,
                path_length=len(current_patterns) + 1
            )
            paths.append(path)
            return
        
        for child in node.children:
            self._collect_paths(
                child,
                current_patterns + [node.pattern_id],
                current_probs + [node.probability],
                cumulative_prob * node.probability,
                expected_return + node.expected_return,
                confidence * self.confidence_decay,
                paths
            )
    
    def visualize_forecast_tree(self, root: ForecastNode) -> go.Figure:
        """Create interactive visualization of the forecast tree"""
        
        # Create network graph
        G = nx.DiGraph()
        
        def add_nodes_edges(node: ForecastNode, parent_id: str = None):
            node_id = f"{node.pattern_id}_{node.time_horizon}"
            G.add_node(node_id, 
                      pattern=node.pattern_id,
                      type=node.pattern_type,
                      confidence=node.confidence,
                      probability=node.probability,
                      return_=node.expected_return)
            
            if parent_id:
                G.add_edge(parent_id, node_id)
            
            for child in node.children:
                add_nodes_edges(child, node_id)
        
        add_nodes_edges(root)
        
        # Create plotly figure
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_text.append(f"{node_data['pattern']}<br>"
                           f"Conf: {node_data['confidence']:.2f}<br>"
                           f"Prob: {node_data['probability']:.2f}<br>"
                           f"Return: {node_data['return_']:.2%}")
            
            node_color.append(node_data['confidence'])
            node_size.append(10 + node_data['probability'] * 20)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n.split('_')[0] for n in G.nodes()],
            textposition="top center",
            marker=dict(
                color=node_color,
                size=node_size,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Multi-Step Pattern Forecast Tree',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def create_scenario_analysis(self, 
                               paths: List[ForecastPath],
                               num_scenarios: int = 5) -> Dict[str, Any]:
        """Create scenario analysis for top forecast paths"""
        
        # Sort paths by cumulative probability
        sorted_paths = sorted(paths, key=lambda x: x.cumulative_probability, reverse=True)
        top_paths = sorted_paths[:num_scenarios]
        
        scenarios = []
        for i, path in enumerate(top_paths):
            scenario = {
                'scenario_id': f"scenario_{i+1}",
                'patterns': path.patterns,
                'probabilities': path.probabilities,
                'cumulative_probability': path.cumulative_probability,
                'expected_return': path.expected_return,
                'confidence_decay': path.confidence_decay,
                'risk_score': self._calculate_risk_score(path),
                'volatility_estimate': self._estimate_volatility(path)
            }
            scenarios.append(scenario)
        
        return {
            'scenarios': scenarios,
            'summary': {
                'total_paths': len(paths),
                'top_paths_considered': len(top_paths),
                'avg_expected_return': np.mean([s['expected_return'] for s in scenarios]),
                'max_risk': max([s['risk_score'] for s in scenarios]),
                'best_scenario': max(scenarios, key=lambda x: x['expected_return']),
                'worst_scenario': min(scenarios, key=lambda x: x['expected_return'])
            }
        }
    
    def _calculate_risk_score(self, path: ForecastPath) -> float:
        """Calculate risk score for a forecast path"""
        # Simple risk calculation based on path length and confidence decay
        base_risk = 0.3
        length_risk = path.path_length * 0.1
        confidence_risk = (1 - path.confidence_decay) * 0.5
        
        return min(base_risk + length_risk + confidence_risk, 1.0)
    
    def _estimate_volatility(self, path: ForecastPath) -> float:
        """Estimate volatility for a forecast path"""
        # Use historical volatility as base
        base_vol = 0.15
        path_vol = base_vol * np.sqrt(path.path_length)
        
        return path_vol
    
    def create_confidence_decay_chart(self, paths: List[ForecastPath]) -> go.Figure:
        """Create visualization of confidence decay over time"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Decay Over Steps', 'Expected Return Distribution'),
            vertical_spacing=0.15
        )
        
        # Confidence decay
        max_steps = max([p.path_length for p in paths])
        for path in paths[:10]:  # Show top 10 paths
            steps = list(range(1, path.path_length + 1))
            confidence_values = [path.confidence_decay ** (i-1) for i in steps]
            
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=confidence_values,
                    mode='lines+markers',
                    name=f"Path {path.patterns[-1]}",
                    line=dict(width=2),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Expected return distribution
        returns = [p.expected_return for p in paths]
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=20,
                name='Expected Returns',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Steps", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)
        fig.update_xaxes(title_text="Expected Return", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="Multi-Step Forecast Analysis",
            showlegend=False
        )
        
        return fig
    
    def generate_forecast_report(self,
                               root: ForecastNode,
                               paths: List[ForecastPath]) -> Dict[str, Any]:
        """Generate comprehensive forecast report"""
        
        scenarios = self.create_scenario_analysis(paths)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'forecast_horizon': self.max_steps,
            'total_paths': len(paths),
            'scenarios': scenarios['scenarios'],
            'summary': scenarios['summary'],
            'recommendations': self._generate_recommendations(scenarios),
            'risk_assessment': self._assess_risk(scenarios),
            'confidence_metrics': {
                'avg_confidence': np.mean([p.confidence_decay for p in paths]),
                'min_confidence': min([p.confidence_decay for p in paths]),
                'max_confidence': max([p.confidence_decay for p in paths])
            }
        }
        
        return report
    
    def _generate_recommendations(self, scenarios: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on scenarios"""
        recommendations = []
        
        best_scenario = scenarios['summary']['best_scenario']
        worst_scenario = scenarios['summary']['worst_scenario']
        
        if best_scenario['expected_return'] > 0.05:
            recommendations.append("Consider long position based on favorable scenarios")
        
        if worst_scenario['expected_return'] < -0.05:
            recommendations.append("Consider risk management or short position")
        
        if scenarios['summary']['max_risk'] > 0.7:
            recommendations.append("High risk detected - reduce position size")
        
        recommendations.append("Monitor pattern development closely")
        
        return recommendations
    
    def _assess_risk(self, scenarios: Dict[str, Any]) -> Dict[str, float]:
        """Assess overall risk of the forecast"""
        scenarios_list = scenarios['scenarios']
        
        return {
            'max_drawdown_risk': max([s['risk_score'] for s in scenarios_list]),
            'probability_weighted_risk': np.mean([s['risk_score'] * s['cumulative_probability'] 
                                                for s in scenarios_list]),
            'tail_risk': len([s for s in scenarios_list if s['expected_return'] < -0.1]),
            'diversification_benefit': 1 - np.std([s['expected_return'] for s in scenarios_list])
        }


# Example usage and testing
if __name__ == "__main__":
    engine = MultiStepForecastEngine(max_steps=4)
    
    # Example usage
    current_patterns = ["ascending_triangle", "bull_flag", "fractal_pattern"]
    market_context = {
        'trend': 'upward',
        'volatility': 0.15,
        'volume': 'increasing'
    }
    
    # Build forecast tree
    tree = engine.build_forecast_tree(current_patterns, market_context)
    
    # Calculate paths
    paths = engine.calculate_path_probabilities(tree)
    
    # Create visualizations
    fig_tree = engine.visualize_forecast_tree(tree)
    fig_decay = engine.create_confidence_decay_chart(paths)
    
    # Generate report
    report = engine.generate_forecast_report(tree, paths)
    
    print("Multi-Step Forecast Report:")
    print(json.dumps(report, indent=2, default=str))
    
    fig_tree.show()
    fig_decay.show()
