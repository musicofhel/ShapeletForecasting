"""
Run the wavelet pattern extraction pipeline to generate pattern_sequences.pkl
"""

from wavelet_pattern_pipeline import WaveletPatternPipeline

if __name__ == "__main__":
    # Initialize pipeline with minimal tickers for faster execution
    pipeline = WaveletPatternPipeline(
        n_clusters=10,
        min_pattern_length=10,
        max_pattern_length=40
    )
    
    # Use fewer tickers for quick generation
    tickers = ["AAPL", "MSFT", "SPY", "BTC-USD"]
    
    print("Starting pattern extraction...")
    
    # Extract patterns
    results = pipeline.extract_patterns_from_multiple_tickers(tickers, period_days=90)
    
    # Display summary
    pipeline.display_summary(results)
    
    # Save results - this creates pattern_sequences.pkl
    output_path = pipeline.save_results()
    
    print(f"\n✅ Pattern extraction complete!")
    print(f"✅ Data saved to: {output_path}")
    print(f"✅ Window 1 objectives completed!")
