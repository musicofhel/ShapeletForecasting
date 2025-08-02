"""
Script to generate all architecture diagrams
"""

import subprocess
import os
import sys

def generate_diagram(script_name):
    """Generate a single diagram"""
    print(f"Generating {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✓ {script_name} generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating {script_name}: {e}")
        return False
    return True

def main():
    """Generate all architecture diagrams"""
    # Change to the architecture_diagrams directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # List of diagram scripts
    diagrams = [
        "1_high_level_overview.py",
        "2_data_flow.py",
        "3_dashboard_architecture.py",
        "4_ml_pipeline.py",
        "5_wavelet_analysis.py",
        "6_deployment_architecture.py"
    ]
    
    print("Financial Wavelet Prediction - Architecture Diagram Generator")
    print("=" * 60)
    print()
    
    # Check if Graphviz is installed
    try:
        subprocess.run(["dot", "-V"], capture_output=True, check=True)
        print("✓ Graphviz is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Graphviz is not installed!")
        print("Please install Graphviz from https://graphviz.org/download/")
        print("On Windows: choco install graphviz")
        print("On Mac: brew install graphviz")
        print("On Linux: sudo apt-get install graphviz")
        return
    
    print()
    print("Generating diagrams...")
    print("-" * 60)
    
    success_count = 0
    for diagram in diagrams:
        if generate_diagram(diagram):
            success_count += 1
    
    print()
    print("-" * 60)
    print(f"Generated {success_count}/{len(diagrams)} diagrams successfully")
    
    if success_count == len(diagrams):
        print()
        print("All diagrams generated successfully!")
        print("Check the .png files in the architecture_diagrams folder")

if __name__ == "__main__":
    main()
