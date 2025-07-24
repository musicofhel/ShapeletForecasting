"""
Performance Benchmarking Script
Comprehensive performance testing for pattern discovery system
"""

import time
import psutil
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import sys
from memory_profiler import profile
import cProfile
import pstats
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.pattern_predictor import PatternPredictor
from src.dashboard.pattern_features import PatternFeatureExtractor
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.wavelet_analysis.wavelet_analyzer import WaveletAnalyzer
from src.dtw.dtw_calculator import DTWCalculator
from src.dashboard.visualizations.timeseries import create_timeseries_plot
from src.dashboard.visualizations.scalogram import create_scalogram_plot
from src.dashboard.visualizations.pattern_gallery import create_pattern_gallery
from src.dashboard.realtime.pattern_monitor_simple_complete import PatternMonitor


class PerformanceBenchmark:
    """Main performance benchmarking class"""
    
    def __init__(self):
        self.results = {
            'pattern_discovery': {},
            'visualization': {},
            'real_time': {},
            'memory': {},
            'scalability': {}
        }
        self.process = psutil.Process(os.getpid())
    
    def generate_test_data(self, size='medium'):
        """Generate test data of various sizes"""
        sizes = {
            'small': 1000,
            'medium': 10000,
            'large': 100000,
            'xlarge': 1000000
        }
        
        n_points = sizes.get(size, 10000)
        
        # Generate realistic financial data
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='1min')
        
        # Create price series with trends and patterns
        t = np.linspace(0, 100, n_points)
        trend = 100 + 0.1 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 1000)
        noise = np.random.randn(n_points) * 2
        
        # Add some patterns
        patterns = np.zeros(n_points)
        pattern_locations = np.random.choice(n_points - 100, size=20, replace=False)
        for loc in pattern_locations:
            pattern_length = np.random.randint(20, 100)
            pattern_t = np.linspace(0, 2*np.pi, pattern_length)
            patterns[loc:loc+pattern_length] += 10 * np.sin(pattern_t) * np.exp(-pattern_t/np.pi)
        
        price = trend + seasonal + patterns + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': price,
            'volume': np.random.randint(1000, 10000, n_points)
        })
    
    def benchmark_pattern_discovery(self):
        """Benchmark pattern discovery algorithms"""
        print("\n=== Pattern Discovery Benchmarks ===")
        
        data_sizes = ['small', 'medium', 'large']
        results = {}
        
        for size in data_sizes:
            print(f"\nTesting with {size} dataset...")
            data = self.generate_test_data(size)
            
            # Test PatternMatcher
            matcher = PatternMatcher()
            start_time = time.time()
            patterns = matcher.find_patterns(data['price'].values)
            matcher_time = time.time() - start_time
            
            # Test WaveletAnalyzer
            analyzer = WaveletAnalyzer()
            start_time = time.time()
            coeffs = analyzer.decompose(data['price'].values)
            wavelet_patterns = analyzer.extract_patterns(coeffs)
            wavelet_time = time.time() - start_time
            
            # Test DTW
            if size != 'large':  # DTW is O(n²), skip for large
                dtw_calc = DTWCalculator()
                sample1 = data['price'].values[:100]
                sample2 = data['price'].values[100:200]
                
                start_time = time.time()
                distance, path = dtw_calc.calculate_dtw(sample1, sample2)
                dtw_time = time.time() - start_time
            else:
                dtw_time = None
            
            results[size] = {
                'data_points': len(data),
                'pattern_matcher_time': matcher_time,
                'patterns_found': len(patterns),
                'wavelet_time': wavelet_time,
                'wavelet_patterns': len(wavelet_patterns),
                'dtw_time': dtw_time,
                'throughput': len(data) / matcher_time
            }
            
            print(f"  Pattern Matcher: {matcher_time:.3f}s ({len(patterns)} patterns)")
            print(f"  Wavelet Analysis: {wavelet_time:.3f}s ({len(wavelet_patterns)} patterns)")
            if dtw_time:
                print(f"  DTW Calculation: {dtw_time:.3f}s")
            print(f"  Throughput: {results[size]['throughput']:.0f} points/second")
        
        self.results['pattern_discovery'] = results
        return results
    
    def benchmark_visualization(self):
        """Benchmark visualization performance"""
        print("\n=== Visualization Benchmarks ===")
        
        data_sizes = ['small', 'medium', 'large']
        results = {}
        
        for size in data_sizes:
            print(f"\nTesting {size} visualizations...")
            data = self.generate_test_data(size)
            
            # Benchmark timeseries plot
            start_time = time.time()
            fig = create_timeseries_plot(data)
            timeseries_time = time.time() - start_time
            
            # Benchmark timeseries with downsampling
            start_time = time.time()
            fig_downsampled = create_timeseries_plot(data, downsample=True, max_points=5000)
            downsampled_time = time.time() - start_time
            
            # Benchmark scalogram (smaller data)
            if size != 'large':
                analyzer = WaveletAnalyzer()
                coeffs = analyzer.decompose(data['price'].values[:1000])
                scales = np.arange(1, 65)
                
                start_time = time.time()
                scalogram_fig = create_scalogram_plot(coeffs, scales=scales)
                scalogram_time = time.time() - start_time
            else:
                scalogram_time = None
            
            # Benchmark pattern gallery
            patterns = []
            for i in range(20):
                patterns.append({
                    'id': f'pattern_{i}',
                    'data': np.random.randn(100),
                    'type': 'test',
                    'confidence': np.random.rand()
                })
            
            start_time = time.time()
            gallery_fig = create_pattern_gallery(patterns)
            gallery_time = time.time() - start_time
            
            # Measure export times
            start_time = time.time()
            html_output = fig.to_html()
            html_export_time = time.time() - start_time
            
            start_time = time.time()
            json_output = fig.to_json()
            json_export_time = time.time() - start_time
            
            results[size] = {
                'data_points': len(data),
                'timeseries_time': timeseries_time,
                'downsampled_time': downsampled_time,
                'scalogram_time': scalogram_time,
                'gallery_time': gallery_time,
                'html_export_time': html_export_time,
                'json_export_time': json_export_time,
                'html_size_kb': len(html_output) / 1024,
                'json_size_kb': len(json_output) / 1024
            }
            
            print(f"  Timeseries: {timeseries_time:.3f}s")
            print(f"  Downsampled: {downsampled_time:.3f}s")
            if scalogram_time:
                print(f"  Scalogram: {scalogram_time:.3f}s")
            print(f"  Pattern Gallery: {gallery_time:.3f}s")
            print(f"  HTML Export: {html_export_time:.3f}s ({results[size]['html_size_kb']:.1f} KB)")
            print(f"  JSON Export: {json_export_time:.3f}s ({results[size]['json_size_kb']:.1f} KB)")
        
        self.results['visualization'] = results
        return results
    
    def benchmark_real_time_processing(self):
        """Benchmark real-time pattern monitoring"""
        print("\n=== Real-Time Processing Benchmarks ===")
        
        # Create pattern monitor
        monitor = PatternMonitor()
        
        # Test different update frequencies
        frequencies = [10, 50, 100, 500]  # Updates per second
        results = {}
        
        for freq in frequencies:
            print(f"\nTesting {freq} updates/second...")
            
            update_times = []
            memory_usage = []
            
            # Run for 10 seconds
            duration = 10
            updates = freq * duration
            delay = 1.0 / freq
            
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            for i in range(updates):
                # Generate new data point
                new_point = {
                    'timestamp': datetime.now(),
                    'price': 100 + np.random.randn(),
                    'volume': np.random.randint(1000, 10000)
                }
                
                start_time = time.time()
                monitor.update(new_point)
                update_time = time.time() - start_time
                update_times.append(update_time)
                
                if i % 100 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_usage.append(current_memory - initial_memory)
                
                # Sleep to maintain frequency
                sleep_time = delay - update_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            results[freq] = {
                'target_frequency': freq,
                'actual_updates': len(update_times),
                'avg_update_time': np.mean(update_times),
                'max_update_time': np.max(update_times),
                'min_update_time': np.min(update_times),
                'std_update_time': np.std(update_times),
                'memory_increase_mb': np.mean(memory_usage) if memory_usage else 0,
                'can_sustain': np.mean(update_times) < delay
            }
            
            print(f"  Average update time: {results[freq]['avg_update_time']*1000:.2f}ms")
            print(f"  Max update time: {results[freq]['max_update_time']*1000:.2f}ms")
            print(f"  Memory increase: {results[freq]['memory_increase_mb']:.1f}MB")
            print(f"  Can sustain rate: {results[freq]['can_sustain']}")
        
        self.results['real_time'] = results
        return results
    
    @profile
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\n=== Memory Usage Benchmarks ===")
        
        results = {}
        
        # Test pattern discovery memory usage
        print("\nPattern Discovery Memory Usage:")
        data_sizes = ['small', 'medium', 'large']
        
        for size in data_sizes:
            data = self.generate_test_data(size)
            
            # Measure memory before
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Run pattern discovery
            matcher = PatternMatcher()
            patterns = matcher.find_patterns(data['price'].values)
            
            # Measure memory after
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            results[f'pattern_discovery_{size}'] = {
                'data_points': len(data),
                'memory_used_mb': memory_used,
                'memory_per_point': memory_used / len(data) * 1000000,  # bytes per point
                'patterns_found': len(patterns)
            }
            
            print(f"  {size}: {memory_used:.1f}MB for {len(data)} points")
        
        # Test visualization memory usage
        print("\nVisualization Memory Usage:")
        
        for size in ['small', 'medium']:
            data = self.generate_test_data(size)
            
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Create multiple visualizations
            figs = []
            for _ in range(5):
                fig = create_timeseries_plot(data)
                figs.append(fig)
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            results[f'visualization_{size}'] = {
                'data_points': len(data),
                'num_figures': len(figs),
                'memory_used_mb': memory_used,
                'memory_per_figure': memory_used / len(figs)
            }
            
            print(f"  {size}: {memory_used:.1f}MB for {len(figs)} figures")
        
        self.results['memory'] = results
        return results
    
    def benchmark_scalability(self):
        """Test scalability with parallel processing"""
        print("\n=== Scalability Benchmarks ===")
        
        # Generate multiple time series
        n_series = 20
        series_list = []
        for _ in range(n_series):
            series_list.append(self.generate_test_data('small')['price'].values)
        
        results = {}
        
        # Sequential processing
        print("\nSequential Processing:")
        matcher = PatternMatcher()
        
        start_time = time.time()
        sequential_results = []
        for series in series_list:
            patterns = matcher.find_patterns(series)
            sequential_results.append(patterns)
        sequential_time = time.time() - start_time
        
        print(f"  Time: {sequential_time:.2f}s")
        print(f"  Throughput: {n_series/sequential_time:.1f} series/second")
        
        # Thread-based parallel processing
        print("\nThread-based Parallel Processing:")
        
        def process_series(series):
            matcher = PatternMatcher()
            return matcher.find_patterns(series)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.time()
            thread_results = list(executor.map(process_series, series_list))
            thread_time = time.time() - start_time
        
        print(f"  Time: {thread_time:.2f}s")
        print(f"  Throughput: {n_series/thread_time:.1f} series/second")
        print(f"  Speedup: {sequential_time/thread_time:.2f}x")
        
        # Process-based parallel processing
        print("\nProcess-based Parallel Processing:")
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            start_time = time.time()
            process_results = list(executor.map(process_series, series_list))
            process_time = time.time() - start_time
        
        print(f"  Time: {process_time:.2f}s")
        print(f"  Throughput: {n_series/process_time:.1f} series/second")
        print(f"  Speedup: {sequential_time/process_time:.2f}x")
        
        results['scalability'] = {
            'n_series': n_series,
            'sequential_time': sequential_time,
            'thread_time': thread_time,
            'process_time': process_time,
            'thread_speedup': sequential_time / thread_time,
            'process_speedup': sequential_time / process_time,
            'thread_efficiency': (sequential_time / thread_time) / 4,  # 4 workers
            'process_efficiency': (sequential_time / process_time) / 4
        }
        
        self.results['scalability'] = results
        return results
    
    def profile_critical_functions(self):
        """Profile critical functions for optimization"""
        print("\n=== Function Profiling ===")
        
        # Generate test data
        data = self.generate_test_data('medium')
        
        # Profile pattern matching
        print("\nProfiling Pattern Matching:")
        profiler = cProfile.Profile()
        
        profiler.enable()
        matcher = PatternMatcher()
        patterns = matcher.find_patterns(data['price'].values)
        profiler.disable()
        
        # Print stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print(s.getvalue())
        
        # Profile wavelet analysis
        print("\nProfiling Wavelet Analysis:")
        profiler = cProfile.Profile()
        
        profiler.enable()
        analyzer = WaveletAnalyzer()
        coeffs = analyzer.decompose(data['price'].values[:1000])
        patterns = analyzer.extract_patterns(coeffs)
        profiler.disable()
        
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n=== Generating Performance Report ===")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Benchmark Results', fontsize=16)
        
        # Pattern discovery performance
        if 'pattern_discovery' in self.results:
            ax = axes[0, 0]
            sizes = list(self.results['pattern_discovery'].keys())
            times = [self.results['pattern_discovery'][s]['pattern_matcher_time'] for s in sizes]
            points = [self.results['pattern_discovery'][s]['data_points'] for s in sizes]
            
            ax.plot(points, times, 'o-', label='Pattern Matcher')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Pattern Discovery Performance')
            ax.set_xscale('log')
            ax.grid(True)
        
        # Visualization performance
        if 'visualization' in self.results:
            ax = axes[0, 1]
            sizes = list(self.results['visualization'].keys())
            times = [self.results['visualization'][s]['timeseries_time'] for s in sizes]
            times_downsampled = [self.results['visualization'][s]['downsampled_time'] for s in sizes]
            points = [self.results['visualization'][s]['data_points'] for s in sizes]
            
            ax.plot(points, times, 'o-', label='Full Resolution')
            ax.plot(points, times_downsampled, 's-', label='Downsampled')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Visualization Performance')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True)
        
        # Real-time processing
        if 'real_time' in self.results:
            ax = axes[0, 2]
            frequencies = list(self.results['real_time'].keys())
            avg_times = [self.results['real_time'][f]['avg_update_time'] * 1000 for f in frequencies]
            max_times = [self.results['real_time'][f]['max_update_time'] * 1000 for f in frequencies]
            
            ax.plot(frequencies, avg_times, 'o-', label='Average')
            ax.plot(frequencies, max_times, 's-', label='Maximum')
            ax.axhline(y=1000/max(frequencies), color='r', linestyle='--', label='Target')
            ax.set_xlabel('Update Frequency (Hz)')
            ax.set_ylabel('Update Time (ms)')
            ax.set_title('Real-time Processing')
            ax.legend()
            ax.grid(True)
        
        # Memory usage
        if 'memory' in self.results:
            ax = axes[1, 0]
            # Extract memory data
            pattern_sizes = ['small', 'medium', 'large']
            pattern_memory = []
            pattern_points = []
            
            for size in pattern_sizes:
                key = f'pattern_discovery_{size}'
                if key in self.results['memory']:
                    pattern_memory.append(self.results['memory'][key]['memory_used_mb'])
                    pattern_points.append(self.results['memory'][key]['data_points'])
            
            if pattern_memory:
                ax.plot(pattern_points, pattern_memory, 'o-')
                ax.set_xlabel('Data Points')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title('Memory Usage Scaling')
                ax.set_xscale('log')
                ax.grid(True)
        
        # Scalability
        if 'scalability' in self.results and 'scalability' in self.results['scalability']:
            ax = axes[1, 1]
            methods = ['Sequential', 'Threads', 'Processes']
            times = [
                self.results['scalability']['scalability']['sequential_time'],
                self.results['scalability']['scalability']['thread_time'],
                self.results['scalability']['scalability']['process_time']
            ]
            
            ax.bar(methods, times)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Parallel Processing Performance')
            ax.grid(True, axis='y')
        
        # Efficiency
        if 'scalability' in self.results and 'scalability' in self.results['scalability']:
            ax = axes[1, 2]
            methods = ['Threads', 'Processes']
            efficiency = [
                self.results['scalability']['scalability']['thread_efficiency'] * 100,
                self.results['scalability']['scalability']['process_efficiency'] * 100
            ]
            
            ax.bar(methods, efficiency)
            ax.set_ylabel('Efficiency (%)')
            ax.set_title('Parallel Processing Efficiency')
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('performance_benchmark_report.png', dpi=300, bbox_inches='tight')
        print("  Saved visualization to performance_benchmark_report.png")
        
        # Save detailed results to JSON
        with open('performance_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("  Saved detailed results to performance_benchmark_results.json")
        
        # Generate summary statistics
        print("\n=== Performance Summary ===")
        
        if 'pattern_discovery' in self.results and 'large' in self.results['pattern_discovery']:
            large_results = self.results['pattern_discovery']['large']
            print(f"Pattern Discovery (100k points):")
            print(f"  Time: {large_results['pattern_matcher_time']:.2f}s")
            print(f"  Throughput: {large_results['throughput']:.0f} points/second")
        
        if 'real_time' in self.results and 100 in self.results['real_time']:
            rt_results = self.results['real_time'][100]
            print(f"\nReal-time Processing (100 Hz):")
            print(f"  Average latency: {rt_results['avg_update_time']*1000:.2f}ms")
            print(f"  Can sustain: {rt_results['can_sustain']}")
        
        if 'scalability' in self.results and 'scalability' in self.results['scalability']:
            scale_results = self.results['scalability']['scalability']
            print(f"\nParallel Processing:")
            print(f"  Thread speedup: {scale_results['thread_speedup']:.2f}x")
            print(f"  Process speedup: {scale_results['process_speedup']:.2f}x")


def main():
    """Run all benchmarks"""
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_pattern_discovery()
    benchmark.benchmark_visualization()
    benchmark.benchmark_real_time_processing()
    benchmark.benchmark_memory_usage()
    benchmark.benchmark_scalability()
    benchmark.profile_critical_functions()
    
    # Generate report
    benchmark.generate_report()
    
    print("\n=== Benchmarking Complete ===")


if __name__ == "__main__":
    main()
