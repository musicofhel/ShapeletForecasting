#!/usr/bin/env python3
"""
Simple test for progress bar functionality
"""

import time
import sys

def test_progress_bar():
    """Test progress bar display"""
    duration = 10  # 10 seconds
    start_time = time.time()
    last_update = 0
    
    print("Testing progress bar for 10 seconds...")
    print("-" * 60)
    
    try:
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress = elapsed / duration
            
            # Update progress bar every 0.1 seconds
            if elapsed - last_update >= 0.1:
                last_update = elapsed
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                remaining = duration - elapsed
                
                # Use carriage return to update the same line
                sys.stdout.write(f"\r[{bar}] {progress*100:.0f}% | Time remaining: {remaining:.0f}s")
                sys.stdout.flush()
            
            time.sleep(0.05)  # Small sleep to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    print("\n\nProgress bar test complete!")

if __name__ == "__main__":
    test_progress_bar()
