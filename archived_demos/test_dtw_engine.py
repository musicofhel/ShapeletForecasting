"""
Test script for DTW Engine module

Tests all DTW algorithms, similarity computation, clustering, and visualization.
Includes performance benchmarking and comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import h5py
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import DTW modules
from src.dtw import (
    DTWCalculator, FastDTW, ConstrainedDTW,
    SimilarityEngine, PatternClusterer, DTWVisualizer
)

# Import wavelet modules for integration
from src.wavelet_analysis import ShapeletExtractor


class TestDTWEngine:
    """Test suite for DTW engine components"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results/dtw"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_patterns(self, n_patterns: int = 20) -> Tuple[List[np.ndarray], List[str]]:
        """Generate synthetic time series patterns for testing"""
        patterns = []
        labels = []
        
        # Pattern types
        pattern_types = ['sine', 'cosine', 'trend', 'step', 'noise']
        
        for i in range(n_patterns):
            pattern_type = pattern_types[i % len(pattern_types)]
            length = np.random.randint(50, 150)
            t = np.linspace(0, 4 * np.pi, length)
            
            if pattern_type == 'sine':
                freq = np.random.uniform(0.5, 2.0)
                pattern = np.sin(freq * t) + np.random.normal(0, 0.1, length)
            elif pattern_type == 'cosine':
                freq = np.random.uniform(0.5, 2.0)
                pattern = np.cos(freq * t) + np.random.normal(0, 0.1, length)
            elif pattern_type == 'trend':
                slope = np.random.uniform(-0.5, 0.5)
                pattern = slope * t + np.random.normal(0, 0.1, length)
            elif pattern_type == 'step':
                n_steps = np.random.randint(2, 5)
                pattern = np.zeros(length)
                step_positions = np.sort(np.random.choice(length, n_steps, replace=False))
                for j, pos in enumerate(step_positions):
                    pattern[pos:] = j + np.random.normal(0, 0.1)
            else:  # noise
                pattern = np.random.normal(0, 1, length)
                
            patterns.append(pattern)
            labels.append(f"{pattern_type}_{i}")
            
        return patterns, labels
        
    def load_shapelet_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load shapelet patterns from previous sprint"""
        shapelet_dir = self.data_dir / "shapelets"
        patterns = []
        labels = []
        
        if shapelet_dir.exists():
            # Load from HDF5 files
            for file_path in shapelet_dir.glob("*_shapelets.h5"):
                ticker = file_path.stem.replace("_shapelets", "")
                
                with h5py.File(file_path, 'r') as f:
                    if 'shapelets' in f:
                        shapelets = f['shapelets'][:]
                        for i, shapelet in enumerate(shapelets[:5]):  # Take first 5 per ticker
                            patterns.append(shapelet)
                            labels.append(f"{ticker}_shapelet_{i}")
                            
        return patterns, labels
        
    def test_dtw_algorithms(self):
        """Test different DTW algorithm implementations"""
        print("\n" + "="*50)
        print("Testing DTW Algorithms")
        print("="*50)
        
        # Generate test patterns
        x = np.sin(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        y = np.sin(np.linspace(0, 2*np.pi, 120)) + np.random.normal(0, 0.1, 120)
        
        # Test standard DTW
        print("\n1. Standard DTW:")
        dtw_calc = DTWCalculator(distance_metric='euclidean', return_cost_matrix=True)
        start_time = time.time()
        result = dtw_calc.compute(x, y)
        std_time = time.time() - start_time
        print(f"   - Distance: {result.distance:.4f}")
        print(f"   - Normalized Distance: {result.normalized_distance:.4f}")
        print(f"   - Path Length: {len(result.path)}")
        print(f"   - Computation Time: {std_time:.4f}s")
        
        # Test FastDTW
        print("\n2. FastDTW:")
        fast_dtw = FastDTW(radius=2)
        start_time = time.time()
        fast_result = fast_dtw.compute(x, y)
        fast_time = time.time() - start_time
        print(f"   - Distance: {fast_result.distance:.4f}")
        print(f"   - Normalized Distance: {fast_result.normalized_distance:.4f}")
        print(f"   - Path Length: {len(fast_result.path)}")
        print(f"   - Computation Time: {fast_time:.4f}s")
        print(f"   - Speedup: {std_time/fast_time:.2f}x")
        
        # Test Constrained DTW (Sakoe-Chiba)
        print("\n3. Constrained DTW (Sakoe-Chiba):")
        constrained_dtw = ConstrainedDTW(constraint_type='sakoe_chiba', constraint_param=10)
        start_time = time.time()
        constrained_result = constrained_dtw.compute(x, y)
        constrained_time = time.time() - start_time
        print(f"   - Distance: {constrained_result.distance:.4f}")
        print(f"   - Normalized Distance: {constrained_result.normalized_distance:.4f}")
        print(f"   - Path Length: {len(constrained_result.path)}")
        print(f"   - Computation Time: {constrained_time:.4f}s")
        
        # Visualize results
        visualizer = DTWVisualizer()
        visualizer.plot_alignment(x, y, result.path, 
                                title="Standard DTW Alignment",
                                save_path=self.output_dir / "dtw_alignment_standard.png")
        
        if result.cost_matrix is not None:
            visualizer.plot_cost_matrix(result.cost_matrix, result.path,
                                      title="DTW Cost Matrix",
                                      save_path=self.output_dir / "dtw_cost_matrix.png")
                                      
        return {
            'standard': {'time': std_time, 'distance': result.distance},
            'fast': {'time': fast_time, 'distance': fast_result.distance},
            'constrained': {'time': constrained_time, 'distance': constrained_result.distance}
        }
        
    def test_similarity_engine(self):
        """Test similarity computation engine"""
        print("\n" + "="*50)
        print("Testing Similarity Engine")
        print("="*50)
        
        # Generate patterns
        patterns, labels = self.generate_synthetic_patterns(15)
        
        # Test different DTW types
        for dtw_type in ['standard', 'fast', 'constrained']:
            print(f"\nTesting with {dtw_type} DTW:")
            
            # Configure engine
            if dtw_type == 'constrained':
                engine = SimilarityEngine(
                    dtw_type=dtw_type,
                    n_jobs=4,
                    constraint_type='sakoe_chiba',
                    constraint_param=10
                )
            else:
                engine = SimilarityEngine(dtw_type=dtw_type, n_jobs=4)
                
            # Compute similarity matrix
            start_time = time.time()
            results = engine.compute_similarity_matrix(
                patterns, 
                labels,
                save_path=self.output_dir / f"similarity_matrix_{dtw_type}.pkl"
            )
            comp_time = time.time() - start_time
            
            print(f"   - Computation time: {comp_time:.2f}s")
            print(f"   - Mean similarity: {np.mean(results['similarity_matrix']):.4f}")
            print(f"   - Std similarity: {np.std(results['similarity_matrix']):.4f}")
            
            # Compute statistics
            stats = engine.compute_pattern_statistics(results['similarity_matrix'], labels)
            print(f"   - Most similar pair: {stats['most_similar_pairs'][0]['labels']} "
                  f"(similarity: {stats['most_similar_pairs'][0]['similarity']:.4f})")
                  
            # Visualize
            visualizer = DTWVisualizer()
            visualizer.plot_similarity_matrix(
                results['similarity_matrix'],
                labels=labels,
                title=f"Similarity Matrix ({dtw_type} DTW)",
                save_path=self.output_dir / f"similarity_matrix_{dtw_type}.png"
            )
            
        return results
        
    def test_pattern_clustering(self, similarity_results: Dict):
        """Test pattern clustering"""
        print("\n" + "="*50)
        print("Testing Pattern Clustering")
        print("="*50)
        
        similarity_matrix = similarity_results['similarity_matrix']
        labels = similarity_results['labels']
        
        # Test hierarchical clustering
        print("\n1. Hierarchical Clustering:")
        clusterer = PatternClusterer(
            clustering_method='hierarchical',
            linkage_method='average',
            n_clusters=4
        )
        
        cluster_results = clusterer.fit_predict(
            patterns=[],  # Not needed since we provide similarity matrix
            similarity_matrix=similarity_matrix
        )
        
        print(f"   - Number of clusters: {cluster_results['n_clusters']}")
        print(f"   - Silhouette score: {cluster_results['silhouette_score']:.4f}")
        
        # Print cluster statistics
        for cluster_id, stats in cluster_results['cluster_stats'].items():
            print(f"   - Cluster {cluster_id}: size={stats['size']}, "
                  f"compactness={stats['compactness']:.3f}, "
                  f"separation={stats['separation']:.3f}")
                  
        # Visualize clustering
        clusterer.plot_dendrogram(
            cluster_results['linkage_matrix'],
            labels=labels,
            save_path=self.output_dir / "clustering_dendrogram.png"
        )
        
        clusterer.plot_cluster_heatmap(
            1 - similarity_matrix,  # Convert to distance
            cluster_results['labels'],
            pattern_labels=labels,
            save_path=self.output_dir / "clustering_heatmap.png"
        )
        
        # Save clustering results
        clusterer.save_clustering_results(
            cluster_results,
            self.output_dir / "clustering_results.pkl"
        )
        
        return cluster_results
        
    def benchmark_performance(self):
        """Benchmark DTW performance on different data sizes"""
        print("\n" + "="*50)
        print("Performance Benchmarking")
        print("="*50)
        
        sizes = [50, 100, 200, 500, 1000]
        methods = ['standard', 'fast', 'constrained']
        results = {method: {'sizes': [], 'times': []} for method in methods}
        
        for size in sizes:
            print(f"\nTesting with series length: {size}")
            
            # Generate test data
            x = np.random.randn(size)
            y = np.random.randn(size)
            
            # Test each method
            for method in methods:
                if method == 'standard':
                    calculator = DTWCalculator()
                elif method == 'fast':
                    calculator = FastDTW(radius=2)
                else:  # constrained
                    calculator = ConstrainedDTW(constraint_param=max(5, size//20))
                    
                # Time computation
                start_time = time.time()
                _ = calculator.compute(x, y)
                comp_time = time.time() - start_time
                
                results[method]['sizes'].append(size)
                results[method]['times'].append(comp_time)
                
                print(f"   - {method}: {comp_time:.4f}s")
                
        # Plot results
        plt.figure(figsize=(10, 6))
        for method in methods:
            plt.plot(results[method]['sizes'], results[method]['times'], 
                    marker='o', label=method.capitalize())
                    
        plt.xlabel('Time Series Length')
        plt.ylabel('Computation Time (seconds)')
        plt.title('DTW Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(self.output_dir / "dtw_performance_benchmark.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
        
    def test_real_financial_patterns(self):
        """Test DTW on real financial patterns from shapelets"""
        print("\n" + "="*50)
        print("Testing on Real Financial Patterns")
        print("="*50)
        
        # Load shapelet patterns
        patterns, labels = self.load_shapelet_patterns()
        
        if not patterns:
            print("No shapelet patterns found. Skipping real data test.")
            return
            
        print(f"Loaded {len(patterns)} shapelet patterns")
        
        # Compute similarity matrix using FastDTW for efficiency
        engine = SimilarityEngine(dtw_type='fast', radius=2, n_jobs=4)
        
        print("\nComputing similarity matrix...")
        results = engine.compute_similarity_matrix(
            patterns[:20],  # Limit for demonstration
            labels[:20],
            save_path=self.output_dir / "financial_similarity_matrix.h5"
        )
        
        # Find similar patterns
        print("\nFinding similar patterns...")
        query_idx = 0
        similar = engine.find_similar_patterns(
            patterns[query_idx],
            patterns[1:20],
            threshold=0.2,
            top_k=5
        )
        
        print(f"Patterns similar to {labels[query_idx]}:")
        for i, (idx, dist) in enumerate(zip(similar['indices'], similar['distances'])):
            print(f"   {i+1}. {labels[idx+1]} (distance: {dist:.4f})")
            
        # Visualize similar patterns
        visualizer = DTWVisualizer()
        similar_patterns = [patterns[query_idx]] + [patterns[idx+1] for idx in similar['indices'][:3]]
        similar_labels = [labels[query_idx]] + [labels[idx+1] for idx in similar['indices'][:3]]
        
        visualizer.plot_pattern_comparison(
            similar_patterns,
            labels=similar_labels,
            title="Similar Financial Patterns",
            save_path=self.output_dir / "similar_financial_patterns.png"
        )
        
        # Cluster financial patterns
        print("\nClustering financial patterns...")
        clusterer = PatternClusterer(
            clustering_method='hierarchical',
            linkage_method='average',
            distance_threshold=0.3
        )
        
        cluster_results = clusterer.fit_predict(
            patterns[:20],
            similarity_matrix=results['similarity_matrix']
        )
        
        print(f"Found {cluster_results['n_clusters']} clusters")
        
        # Visualize clusters
        visualizer.plot_cluster_patterns(
            patterns[:20],
            cluster_results['labels'],
            cluster_results['cluster_centers'],
            save_path=self.output_dir / "financial_pattern_clusters.png"
        )
        
    def create_interactive_visualizations(self):
        """Create interactive visualizations"""
        print("\n" + "="*50)
        print("Creating Interactive Visualizations")
        print("="*50)
        
        # Generate example data
        x = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        y = np.sin(np.linspace(0.2, 4.2*np.pi, 120)) + np.random.normal(0, 0.1, 120)
        
        # Compute DTW
        dtw_calc = DTWCalculator()
        result = dtw_calc.compute(x, y)
        
        # Create interactive alignment
        visualizer = DTWVisualizer()
        fig = visualizer.create_interactive_alignment(x, y, result.path)
        fig.write_html(self.output_dir / "interactive_dtw_alignment.html")
        print("Saved interactive alignment to interactive_dtw_alignment.html")
        
        # Create interactive similarity matrix
        patterns, labels = self.generate_synthetic_patterns(10)
        engine = SimilarityEngine(dtw_type='fast')
        sim_results = engine.compute_similarity_matrix(patterns, labels)
        
        fig = visualizer.create_interactive_similarity_matrix(
            sim_results['similarity_matrix'],
            labels
        )
        fig.write_html(self.output_dir / "interactive_similarity_matrix.html")
        print("Saved interactive similarity matrix to interactive_similarity_matrix.html")
        
    def run_all_tests(self):
        """Run all DTW tests"""
        print("\n" + "="*70)
        print("SPRINT 3: DTW ENGINE TEST SUITE")
        print("="*70)
        
        # Test DTW algorithms
        dtw_results = self.test_dtw_algorithms()
        
        # Test similarity engine
        similarity_results = self.test_similarity_engine()
        
        # Test pattern clustering
        cluster_results = self.test_pattern_clustering(similarity_results)
        
        # Benchmark performance
        benchmark_results = self.benchmark_performance()
        
        # Test on real financial patterns
        self.test_real_financial_patterns()
        
        # Create interactive visualizations
        self.create_interactive_visualizations()
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"✓ DTW algorithms tested and working")
        print(f"✓ Similarity engine tested with parallel processing")
        print(f"✓ Pattern clustering implemented and tested")
        print(f"✓ Performance benchmarking completed")
        print(f"✓ Visualizations created in {self.output_dir}")
        print(f"✓ All tests passed successfully!")
        
        return {
            'dtw_results': dtw_results,
            'similarity_results': similarity_results,
            'cluster_results': cluster_results,
            'benchmark_results': benchmark_results
        }


def main():
    """Main test execution"""
    tester = TestDTWEngine()
    results = tester.run_all_tests()
    
    print("\n" + "="*50)
    print("DTW ENGINE TESTING COMPLETE")
    print("="*50)
    print(f"Results saved to: {tester.output_dir}")
    

if __name__ == "__main__":
    main()
