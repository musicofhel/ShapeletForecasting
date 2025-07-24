"""Test analytics with file output"""

import sys

# Redirect output to file
output_file = open('test_analytics_output.txt', 'w')
sys.stdout = output_file
sys.stderr = output_file

try:
    print("Testing analytics module...")
    
    # Test imports
    print("1. Testing imports...")
    from src.dashboard.visualizations.analytics import PatternAnalytics, create_sample_data
    print("   ✓ Imports successful")
    
    # Test sample data creation
    print("\n2. Creating sample data...")
    patterns = create_sample_data(100)
    print(f"   ✓ Created {len(patterns)} patterns")
    print(f"   Columns: {list(patterns.columns)}")
    
    # Test analytics initialization
    print("\n3. Initializing analytics...")
    analytics = PatternAnalytics()
    print("   ✓ Analytics initialized")
    
    # Test frequency analysis
    print("\n4. Testing frequency analysis...")
    freq_analysis = analytics.analyze_pattern_frequency(patterns)
    print(f"   ✓ Frequency analysis complete")
    print(f"   Pattern types: {list(freq_analysis['type_frequency'].keys())}")
    
    # Test visualization creation
    print("\n5. Creating frequency timeline...")
    fig = analytics.create_frequency_timeline(patterns)
    fig.write_html("test_frequency_timeline.html")
    print("   ✓ Created test_frequency_timeline.html")
    
    # Create all visualizations
    print("\n6. Creating all visualizations...")
    dashboard = analytics.create_comprehensive_dashboard(patterns)
    
    for name, fig in dashboard.items():
        filename = f"pattern_analytics_{name}.html"
        fig.write_html(filename)
        print(f"   ✓ Created {filename}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    output_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # Print to console that test is complete
    print("Test complete. Check test_analytics_output.txt for results.")
