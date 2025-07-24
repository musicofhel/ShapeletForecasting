"""Minimal test to check imports"""

print("Starting minimal test...")

try:
    # Check if sklearn is available
    print("1. Checking sklearn...")
    import sklearn
    print("   sklearn version:", sklearn.__version__)
    
    # Check if scipy is available
    print("\n2. Checking scipy...")
    import scipy
    print("   scipy version:", scipy.__version__)
    
    # Check if seaborn is available
    print("\n3. Checking seaborn...")
    import seaborn
    print("   seaborn version:", seaborn.__version__)
    
    # Try importing the analytics module
    print("\n4. Importing analytics module...")
    from src.dashboard.visualizations.analytics import PatternAnalytics
    print("   ✓ Analytics module imported successfully")
    
    # Try creating sample data
    print("\n5. Testing create_sample_data...")
    from src.dashboard.visualizations.analytics import create_sample_data
    data = create_sample_data(10)
    print(f"   ✓ Created {len(data)} sample patterns")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMissing dependency. You may need to install:")
    if 'sklearn' in str(e):
        print("   pip install scikit-learn")
    elif 'scipy' in str(e):
        print("   pip install scipy")
    elif 'seaborn' in str(e):
        print("   pip install seaborn")
        
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
