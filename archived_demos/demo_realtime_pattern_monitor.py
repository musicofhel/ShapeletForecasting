"""
Demo script for Real-time Pattern Detection Monitor

This script demonstrates:
1. Real-time pattern detection
2. Live alerts generation
3. Pattern formation tracking
4. Dynamic prediction updates
5. Pattern statistics and history
"""

import sys
import time
import threading
from datetime import datetime
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


def advanced_demo():
    """Run advanced demo with multiple tickers and custom scenarios"""
    
    print("="*80)
    print("REAL-TIME PATTERN DETECTION MONITOR - ADVANCED DEMO")
    print("="*80)
    print()
    
    # Create monitor with custom alert handler
    alerts_received = []
    
    def custom_alert_handler(alert: PatternAlert):
        """Custom alert handler that stores alerts"""
        alerts_received.append(alert)
        
        # Color coding based on risk level
        risk_colors = {
            'low': '\033[92m',      # Green
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m'      # Red
        }
        color = risk_colors.get(alert.risk_level, '\033[0m')
        reset = '\033[0m'
        
        print(f"\n{color}{'='*60}")
        print(f"🚨 PATTERN ALERT - {alert.pattern_type.upper()}")
        print(f"{'='*60}{reset}")
        print(f"Ticker: {alert.ticker}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Confidence: {alert.confidence:.1%}")
        print(f"Price: ${alert.price_level:.2f}")
        print(f"Expected Move: ${alert.expected_move:.2f} ({alert.expected_move/alert.price_level*100:+.1f}%)")
        print(f"Risk Level: {color}{alert.risk_level}{reset}")
        
        if alert.metadata.get('completion'):
            print(f"Status: ✅ Pattern Completed")
        else:
            print(f"Status: 🔄 Pattern Forming")
            
        print(f"{color}{'='*60}{reset}\n")
    
    # Create monitor
    monitor = RealTimePatternMonitor(
        window_size=100,
        update_interval=0.5,  # Faster updates
        min_confidence=0.65,  # Lower threshold for demo
        alert_callback=custom_alert_handler
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate multiple tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    base_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 380.0}
    
    # Create threads for each ticker
    threads = []
    stop_event = threading.Event()
    
    def simulate_ticker(ticker, base_price):
        """Simulate data for a single ticker"""
        price = base_price
        scenario_patterns = {
            'AAPL': ['breakout', 'triangle', 'continuation'],
            'GOOGL': ['reversal', 'head_shoulders', 'breakout'],
            'MSFT': ['triangle', 'continuation', 'reversal']
        }
        
        patterns = scenario_patterns[ticker]
        current_pattern_idx = 0
        pattern_start = time.time()
        
        print(f"Starting simulation for {ticker} at ${base_price:.2f}")
        
        while not stop_event.is_set():
            # Switch patterns every 30 seconds
            if time.time() - pattern_start > 30:
                current_pattern_idx = (current_pattern_idx + 1) % len(patterns)
                pattern_start = time.time()
                print(f"\n{ticker}: Switching to {patterns[current_pattern_idx]} scenario")
            
            current_pattern = patterns[current_pattern_idx]
            
            # Generate price based on pattern
            if current_pattern == 'breakout':
                # Consolidation then breakout
                elapsed = time.time() - pattern_start
                if elapsed < 15:
                    change = random.gauss(0, 0.001) * price
                    volume = random.uniform(1000000, 1500000)
                else:
                    change = random.gauss(0.002, 0.001) * price
                    volume = random.uniform(2500000, 4000000)
                    
            elif current_pattern == 'reversal':
                # Strong trend then reversal
                elapsed = time.time() - pattern_start
                if elapsed < 15:
                    change = random.gauss(0.003, 0.001) * price
                else:
                    change = random.gauss(-0.003, 0.001) * price
                volume = random.uniform(1800000, 3000000)
                
            elif current_pattern == 'triangle':
                # Converging volatility
                elapsed = time.time() - pattern_start
                volatility = max(0.0005, 0.003 - elapsed * 0.0001)
                change = random.gauss(0, volatility) * price
                volume = random.uniform(800000, 1200000) * (1 - elapsed/30)
                
            elif current_pattern == 'continuation':
                # Flag pattern
                change = random.gauss(0.001, 0.0005) * price
                volume = random.uniform(900000, 1100000)
                
            elif current_pattern == 'head_shoulders':
                # Complex pattern
                elapsed = time.time() - pattern_start
                phase = int(elapsed / 10) % 3
                if phase == 0:  # Left shoulder
                    change = random.gauss(0.002, 0.001) * price
                elif phase == 1:  # Head
                    change = random.gauss(0.003, 0.001) * price
                else:  # Right shoulder
                    change = random.gauss(-0.002, 0.001) * price
                volume = random.uniform(1500000, 2500000)
            
            else:
                # Default random walk
                change = random.gauss(0, 0.002) * price
                volume = random.uniform(1000000, 2000000)
            
            # Update price
            price += change
            price = max(price, base_price * 0.85)
            price = min(price, base_price * 1.15)
            
            # Add to monitor
            monitor.add_price_data(ticker, price, volume)
            
            time.sleep(0.1)
    
    # Start threads
    for ticker in tickers:
        thread = threading.Thread(
            target=simulate_ticker,
            args=(ticker, base_prices[ticker])
        )
        thread.start()
        threads.append(thread)
    
    # Run for specified duration
    duration = 90  # 90 seconds
    print(f"\nRunning simulation for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        # Monitor active patterns
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < duration:
            # Show status every 15 seconds
            if time.time() - last_status_time > 15:
                print("\n" + "="*80)
                print("ACTIVE PATTERNS STATUS")
                print("="*80)
                
                for ticker in tickers:
                    active = monitor.get_active_patterns(ticker)
                    if active:
                        print(f"\n{ticker}:")
                        for pattern_id, progress in active.items():
                            print(f"  📊 {progress.pattern_type.upper()}")
                            print(f"     Completion: {progress.completion:.0f}%")
                            print(f"     Confidence: {progress.confidence:.1%}")
                            print(f"     Stage: {progress.current_stage}")
                            print(f"     Key Levels: {[f'${l:.2f}' for l in progress.key_levels[:3]]}")
                    else:
                        print(f"\n{ticker}: No active patterns")
                
                print("="*80)
                last_status_time = time.time()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        # Stop threads
        stop_event.set()
        for thread in threads:
            thread.join()
        
        # Stop monitor
        monitor.stop_monitoring()
        
        # Display final statistics
        print("\n" + "="*80)
        print("SIMULATION COMPLETE - FINAL STATISTICS")
        print("="*80)
        
        for ticker in tickers:
            stats = monitor.get_pattern_statistics(ticker)
            history = monitor.get_pattern_history(ticker, hours=1)
            
            print(f"\n{ticker}:")
            print(f"  Total patterns detected: {stats['total_patterns']}")
            print(f"  Average confidence: {stats['avg_confidence']:.1%}")
            
            if stats['pattern_counts']:
                print(f"  Pattern breakdown:")
                for ptype, count in stats['pattern_counts'].items():
                    print(f"    - {ptype}: {count}")
            
            if history:
                print(f"  Recent patterns:")
                for pattern in history[-3:]:  # Last 3 patterns
                    duration = (pattern['end_time'] - pattern['start_time']).total_seconds()
                    print(f"    - {pattern['pattern_type']} "
                          f"({pattern['confidence']:.1%} confidence, "
                          f"{duration:.0f}s duration)")
        
        print(f"\nTotal alerts received: {len(alerts_received)}")
        
        # Alert summary
        if alerts_received:
            print("\nAlert Summary:")
            alert_types = {}
            for alert in alerts_received:
                key = f"{alert.ticker}-{alert.pattern_type}"
                alert_types[key] = alert_types.get(key, 0) + 1
            
            for key, count in sorted(alert_types.items()):
                ticker, ptype = key.split('-')
                print(f"  {ticker} {ptype}: {count} alerts")
        
        print("="*80)


def simple_demo():
    """Run simple demo with single ticker"""
    print("="*80)
    print("REAL-TIME PATTERN DETECTION MONITOR - SIMPLE DEMO")
    print("="*80)
    print()
    
    # Create monitor
    monitor = create_demo_monitor()
    
    # Run simulation
    simulate_live_data(
        monitor,
        ticker="AAPL",
        base_price=150.0,
        duration_seconds=60  # 1 minute
    )


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Pattern Monitor Demo')
    parser.add_argument('--mode', choices=['simple', 'advanced'], 
                       default='advanced',
                       help='Demo mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        simple_demo()
    else:
        advanced_demo()


if __name__ == "__main__":
    main()
