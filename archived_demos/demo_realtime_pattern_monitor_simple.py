#!/usr/bin/env python3
"""
Demo script for simplified real-time pattern monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.realtime.pattern_monitor_simple import (
    RealTimePatternMonitor, PatternAlert, create_demo_monitor
)
import time
import random
import argparse
from datetime import datetime


def simulate_live_data(monitor: RealTimePatternMonitor, 
                      ticker: str = "AAPL",
                      base_price: float = 150.0,
                      duration_seconds: int = 60):
    """
    Simulate live price data for testing
    
    Args:
        monitor: Pattern monitor instance
        ticker: Stock ticker
        base_price: Starting price
        duration_seconds: Simulation duration
    """
    import random
    
    print(f"Starting live data simulation for {ticker}")
    print(f"Base price: ${base_price:.2f}")
    print(f"Duration: {duration_seconds} seconds")
    print("-" * 60)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate data
    start_time = time.time()
    price = base_price
    
    # Pattern scenarios
    scenarios = [
        'normal',      # Random walk
        'breakout',    # Upward breakout
        'reversal',    # Trend reversal
        'triangle',    # Triangle formation
    ]
    
    current_scenario = random.choice(scenarios)
    scenario_start = time.time()
    
    print(f"Starting scenario: {current_scenario}")
    
    # Progress bar setup
    last_update = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            progress = elapsed / duration_seconds
            
            # Update progress bar every second
            if int(elapsed) > last_update:
                last_update = int(elapsed)
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                remaining = duration_seconds - elapsed
                print(f"\r[{bar}] {progress*100:.0f}% | Time remaining: {remaining:.0f}s", end='', flush=True)
            
            # Switch scenarios periodically
            if time.time() - scenario_start > 20:
                current_scenario = random.choice(scenarios)
                scenario_start = time.time()
                print(f"\n\nSwitching to scenario: {current_scenario}")
            
            # Generate price based on scenario
            if current_scenario == 'normal':
                # Random walk
                change = random.gauss(0, 0.002) * price
                volume = random.uniform(1000000, 2000000)
                
            elif current_scenario == 'breakout':
                # Gradual increase with volume surge
                change = random.gauss(0.001, 0.001) * price
                volume = random.uniform(2000000, 4000000)
                
            elif current_scenario == 'reversal':
                # Trend change
                elapsed_scenario = time.time() - scenario_start
                if elapsed_scenario < 10:
                    change = random.gauss(0.002, 0.001) * price
                else:
                    change = random.gauss(-0.002, 0.001) * price
                volume = random.uniform(1500000, 3000000)
                
            elif current_scenario == 'triangle':
                # Converging price action
                elapsed_scenario = time.time() - scenario_start
                volatility = max(0.001, 0.003 - elapsed_scenario * 0.0001)
                change = random.gauss(0, volatility) * price
                volume = random.uniform(800000, 1500000)
            
            # Update price
            price += change
            price = max(price, base_price * 0.9)  # Floor at -10%
            price = min(price, base_price * 1.1)  # Cap at +10%
            
            # Add data to monitor
            monitor.add_price_data(ticker, price, volume)
            
            # Display active patterns periodically
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                active_patterns = monitor.get_active_patterns(ticker)
                if active_patterns:
                    print(f"\n\nActive patterns for {ticker}:")
                    for pattern_id, progress in active_patterns.items():
                        print(f"  - {progress.pattern_type}: "
                              f"{progress.completion:.0f}% complete, "
                              f"stage: {progress.current_stage}")
            
            time.sleep(0.1)  # 10 updates per second
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Display statistics
        stats = monitor.get_pattern_statistics(ticker)
        print(f"\n\n{'='*60}")
        print(f"SIMULATION COMPLETE - Pattern Statistics for {ticker}")
        print(f"{'='*60}")
        print(f"Total patterns detected: {stats['total_patterns']}")
        print(f"Average confidence: {stats['avg_confidence']:.1%}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print("\nPattern counts:")
        for pattern_type, count in stats['pattern_counts'].items():
            print(f"  - {pattern_type}: {count}")
        print(f"{'='*60}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Real-time Pattern Monitor Demo')
    parser.add_argument('--mode', choices=['simple', 'advanced'], default='simple',
                       help='Demo mode')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration in seconds')
    parser.add_argument('--price', type=float, default=150.0,
                       help='Base price')
    
    args = parser.parse_args()
    
    print("="*60)
    print("REAL-TIME PATTERN MONITOR DEMO")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Ticker: {args.ticker}")
    print(f"Duration: {args.duration}s")
    print(f"Base Price: ${args.price:.2f}")
    print("="*60)
    print()
    
    if args.mode == 'simple':
        # Create monitor with alert handler
        monitor = create_demo_monitor()
        
        # Run simulation
        simulate_live_data(
            monitor,
            ticker=args.ticker,
            base_price=args.price,
            duration_seconds=args.duration
        )
    
    else:  # advanced mode
        print("Advanced mode with multiple tickers and strategies")
        
        # Create monitor
        monitor = create_demo_monitor()
        
        # Simulate multiple tickers
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        base_prices = [150.0, 2800.0, 400.0]
        
        print(f"Monitoring {len(tickers)} tickers simultaneously")
        print("-" * 60)
        
        # Start monitoring
        monitor.start_monitoring()
        
        start_time = time.time()
        prices = {ticker: price for ticker, price in zip(tickers, base_prices)}
        
        try:
            while time.time() - start_time < args.duration:
                # Update each ticker
                for ticker in tickers:
                    # Random price movement
                    change = random.gauss(0, 0.002) * prices[ticker]
                    prices[ticker] += change
                    
                    # Random volume
                    volume = random.uniform(1000000, 3000000)
                    
                    # Add data
                    monitor.add_price_data(ticker, prices[ticker], volume)
                
                # Show progress
                elapsed = time.time() - start_time
                progress = elapsed / args.duration
                print(f"\rProgress: {progress*100:.0f}%", end='', flush=True)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted")
            
        finally:
            monitor.stop_monitoring()
            
            # Show statistics for each ticker
            print("\n\nPattern Statistics by Ticker:")
            print("-" * 60)
            for ticker in tickers:
                stats = monitor.get_pattern_statistics(ticker)
                print(f"\n{ticker}:")
                print(f"  Total patterns: {stats['total_patterns']}")
                print(f"  Pattern types: {list(stats['pattern_counts'].keys())}")


if __name__ == "__main__":
    main()
