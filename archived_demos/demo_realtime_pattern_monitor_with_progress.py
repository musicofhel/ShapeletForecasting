"""
Demo script for Real-time Pattern Detection Monitor with Progress Bar

This script demonstrates:
1. Real-time pattern detection with progress tracking
2. Live alerts generation
3. Pattern formation tracking
4. Dynamic prediction updates
5. Pattern statistics and history
"""

import sys
import time
import threading
from datetime import datetime, timedelta
import numpy as np
import random

# Add src to path
sys.path.append('src')

from dashboard.realtime.pattern_monitor import (
    RealTimePatternMonitor,
    PatternAlert,
    create_demo_monitor,
    simulate_live_data
)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    # Calculate time remaining
    if iteration > 0:
        elapsed = time.time() - print_progress_bar.start_time
        time_per_iteration = elapsed / iteration
        remaining_iterations = total - iteration
        time_remaining = time_per_iteration * remaining_iterations
        
        # Format time remaining
        if time_remaining < 60:
            time_str = f"{int(time_remaining)}s"
        else:
            minutes = int(time_remaining // 60)
            seconds = int(time_remaining % 60)
            time_str = f"{minutes}m {seconds}s"
        
        suffix_with_time = f"{suffix} - Time remaining: {time_str}"
    else:
        suffix_with_time = suffix
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix_with_time}', end='\r')
    
    # Print New Line on Complete
    if iteration == total: 
        print()


def simple_demo_with_progress():
    """Run simple demo with progress bar"""
    print("="*80)
    print("REAL-TIME PATTERN DETECTION MONITOR - PROGRESS DEMO")
    print("="*80)
    print()
    
    # Create monitor with custom alert handler
    alerts_received = []
    patterns_detected = []
    
    def custom_alert_handler(alert: PatternAlert):
        """Custom alert handler that stores alerts"""
        alerts_received.append(alert)
        patterns_detected.append({
            'time': alert.timestamp,
            'type': alert.pattern_type,
            'ticker': alert.ticker,
            'confidence': alert.confidence,
            'price': alert.price_level
        })
        
        # Don't print during progress bar updates
        # Store for later display
    
    # Create monitor
    monitor = RealTimePatternMonitor(
        window_size=100,
        update_interval=0.5,
        min_confidence=0.65,
        alert_callback=custom_alert_handler
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulation parameters
    ticker = "AAPL"
    base_price = 150.0
    duration_seconds = 60
    update_interval = 0.1
    total_iterations = int(duration_seconds / update_interval)
    
    print(f"Starting simulation for {ticker} at ${base_price:.2f}")
    print(f"Duration: {duration_seconds} seconds")
    print()
    
    # Initialize progress bar
    print_progress_bar.start_time = time.time()
    
    # Pattern scenarios
    pattern_scenarios = [
        ('Consolidation', 0, 20),
        ('Breakout Formation', 20, 35),
        ('Trend Continuation', 35, 50),
        ('Reversal Signal', 50, 60)
    ]
    
    # Simulate data with progress tracking
    price = base_price
    start_time = time.time()
    
    for i in range(total_iterations):
        current_time = time.time() - start_time
        
        # Determine current scenario
        current_scenario = None
        for scenario, start, end in pattern_scenarios:
            if start <= current_time < end:
                current_scenario = scenario
                break
        
        if not current_scenario:
            current_scenario = "Random Walk"
        
        # Generate price based on scenario
        if current_scenario == 'Consolidation':
            change = random.gauss(0, 0.001) * price
            volume = random.uniform(1000000, 1500000)
            
        elif current_scenario == 'Breakout Formation':
            # Building pressure
            elapsed_in_phase = current_time - 20
            volatility = 0.001 + elapsed_in_phase * 0.0001
            change = random.gauss(0.0005, volatility) * price
            volume = random.uniform(1500000, 2500000) * (1 + elapsed_in_phase/15)
            
        elif current_scenario == 'Trend Continuation':
            # Strong upward movement
            change = random.gauss(0.002, 0.001) * price
            volume = random.uniform(2000000, 3500000)
            
        elif current_scenario == 'Reversal Signal':
            # Topping pattern
            elapsed_in_phase = current_time - 50
            if elapsed_in_phase < 5:
                change = random.gauss(0.001, 0.001) * price
            else:
                change = random.gauss(-0.002, 0.001) * price
            volume = random.uniform(2500000, 4000000)
            
        else:
            change = random.gauss(0, 0.002) * price
            volume = random.uniform(1000000, 2000000)
        
        # Update price
        price += change
        price = max(price, base_price * 0.95)
        price = min(price, base_price * 1.05)
        
        # Add to monitor
        monitor.add_price_data(ticker, price, volume)
        
        # Update progress bar
        print_progress_bar(
            i + 1, 
            total_iterations,
            prefix=f'Analyzing {current_scenario}:',
            suffix=f'Price: ${price:.2f}',
            length=40
        )
        
        time.sleep(update_interval)
    
    # Stop monitor
    monitor.stop_monitoring()
    
    # Clear progress bar line
    print("\n")
    
    # Display results
    print("="*80)
    print("SIMULATION COMPLETE - PATTERN DETECTION RESULTS")
    print("="*80)
    
    # Get statistics
    stats = monitor.get_pattern_statistics(ticker)
    history = monitor.get_pattern_history(ticker, hours=1)
    
    print(f"\nOverall Statistics:")
    print(f"  Total patterns detected: {stats['total_patterns']}")
    print(f"  Average confidence: {stats['avg_confidence']:.1%}")
    
    if stats['pattern_counts']:
        print(f"\nPattern Breakdown:")
        for ptype, count in stats['pattern_counts'].items():
            print(f"  - {ptype}: {count} occurrences")
    
    # Display detected patterns timeline
    if patterns_detected:
        print(f"\nPattern Detection Timeline:")
        print("-" * 60)
        print(f"{'Time':>8} | {'Pattern Type':^20} | {'Confidence':^12} | {'Price':^10}")
        print("-" * 60)
        
        for pattern in patterns_detected:
            time_offset = (pattern['time'] - patterns_detected[0]['time']).total_seconds()
            print(f"{time_offset:>7.1f}s | {pattern['type']:^20} | {pattern['confidence']:^11.1%} | ${pattern['price']:>8.2f}")
    
    # Display alerts summary
    if alerts_received:
        print(f"\nAlerts Summary:")
        high_risk = sum(1 for a in alerts_received if a.risk_level == 'high')
        medium_risk = sum(1 for a in alerts_received if a.risk_level == 'medium')
        low_risk = sum(1 for a in alerts_received if a.risk_level == 'low')
        
        print(f"  High Risk: {high_risk}")
        print(f"  Medium Risk: {medium_risk}")
        print(f"  Low Risk: {low_risk}")
        
        # Show most recent high-risk alert
        high_risk_alerts = [a for a in alerts_received if a.risk_level == 'high']
        if high_risk_alerts:
            latest_high = high_risk_alerts[-1]
            print(f"\nLatest High-Risk Alert:")
            print(f"  Pattern: {latest_high.pattern_type}")
            print(f"  Confidence: {latest_high.confidence:.1%}")
            print(f"  Expected Move: ${latest_high.expected_move:.2f} ({latest_high.expected_move/latest_high.price_level*100:+.1f}%)")
    
    print("\n" + "="*80)


def advanced_demo_with_progress():
    """Run advanced demo with multiple progress bars"""
    print("="*80)
    print("REAL-TIME PATTERN DETECTION MONITOR - MULTI-TICKER PROGRESS")
    print("="*80)
    print()
    
    # Create monitor
    monitor = create_demo_monitor()
    monitor.start_monitoring()
    
    # Simulation parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    base_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 380.0}
    duration_seconds = 90
    
    print(f"Starting multi-ticker simulation for {duration_seconds} seconds")
    print("Monitoring: " + ", ".join(tickers))
    print()
    
    # Thread control
    stop_event = threading.Event()
    pattern_counts = {ticker: 0 for ticker in tickers}
    lock = threading.Lock()
    
    def update_pattern_count(ticker):
        with lock:
            pattern_counts[ticker] += 1
    
    # Custom alert handler
    def alert_handler(alert: PatternAlert):
        update_pattern_count(alert.ticker)
    
    monitor.alert_callback = alert_handler
    
    # Simulate tickers in threads
    threads = []
    
    def simulate_ticker(ticker, base_price):
        price = base_price
        while not stop_event.is_set():
            change = random.gauss(0, 0.002) * price
            price += change
            volume = random.uniform(1000000, 3000000)
            monitor.add_price_data(ticker, price, volume)
            time.sleep(0.1)
    
    for ticker in tickers:
        thread = threading.Thread(
            target=simulate_ticker,
            args=(ticker, base_prices[ticker])
        )
        thread.start()
        threads.append(thread)
    
    # Progress tracking
    start_time = time.time()
    print_progress_bar.start_time = start_time
    
    try:
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            
            # Build status string
            status_parts = []
            with lock:
                for ticker in tickers:
                    count = pattern_counts[ticker]
                    status_parts.append(f"{ticker}:{count}")
            
            status = " | ".join(status_parts)
            
            # Update progress
            print_progress_bar(
                int(elapsed),
                duration_seconds,
                prefix='Monitoring:',
                suffix=f'Patterns: {status}',
                length=30
            )
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted")
    
    finally:
        # Stop everything
        stop_event.set()
        for thread in threads:
            thread.join()
        monitor.stop_monitoring()
        
        print("\n\nFinal Results:")
        print("-" * 40)
        with lock:
            for ticker in tickers:
                print(f"{ticker}: {pattern_counts[ticker]} patterns detected")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Pattern Monitor Demo with Progress')
    parser.add_argument('--mode', choices=['simple', 'multi'], 
                       default='simple',
                       help='Demo mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        simple_demo_with_progress()
    else:
        advanced_demo_with_progress()


if __name__ == "__main__":
    main()
