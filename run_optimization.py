"""
Run hyperparameter optimization and model compression
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from src.optimization.model_compressor import ModelCompressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_optimization_pipeline():
    """Run the complete optimization pipeline"""
    
    # Create output directories
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    optimized_dir = models_dir / "optimized"
    optimized_dir.mkdir(exist_ok=True)
    
    # Initialize optimizer
    logger.info("Initializing hyperparameter optimizer...")
    optimizer = HyperparameterOptimizer(
        n_trials=50,  # Adjust based on available time
        n_jobs=4,     # Parallel trials
        seed=42
    )
    
    # Initialize compressor
    logger.info("Initializing model compressor...")
    compressor = ModelCompressor()
    
    # Results storage
    optimization_results = {}
    compression_results = {}
    
    # 1. Optimize XGBoost
    logger.info("\n" + "="*50)
    logger.info("Optimizing XGBoost model...")
    try:
        # Note: This would need actual training data
        # For demonstration, we'll show the structure
        logger.info("Would optimize XGBoost with following parameter space:")
        logger.info("- n_estimators: [50, 500]")
        logger.info("- max_depth: [3, 10]")
        logger.info("- learning_rate: [0.01, 0.3]")
        logger.info("- subsample: [0.6, 1.0]")
        
        # In practice:
        # best_params, best_score = optimizer.optimize_xgboost(X_train, y_train)
        # optimization_results['xgboost'] = {
        #     'best_params': best_params,
        #     'best_score': best_score
        # }
        
        # Compress if model exists
        xgb_path = models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            logger.info("Compressing XGBoost model...")
            compressed_path = optimized_dir / "xgboost_optimized.pkl"
            metrics = compressor.compress_xgboost(str(xgb_path), str(compressed_path))
            compression_results['xgboost'] = metrics
            logger.info(f"Compression metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error optimizing XGBoost: {e}")
    
    # 2. Optimize LSTM
    logger.info("\n" + "="*50)
    logger.info("Optimizing LSTM model...")
    try:
        logger.info("Would optimize LSTM with following parameter space:")
        logger.info("- hidden_size: [32, 256]")
        logger.info("- num_layers: [1, 3]")
        logger.info("- dropout: [0.0, 0.5]")
        logger.info("- learning_rate: [0.0001, 0.01]")
        
        # Compress if model exists
        lstm_path = models_dir / "lstm_model.pth"
        if lstm_path.exists():
            logger.info("Compressing LSTM model...")
            compressed_path = optimized_dir / "lstm_optimized.pth"
            metrics = compressor.compress_neural_network(
                str(lstm_path), 
                str(compressed_path),
                model_type='lstm'
            )
            compression_results['lstm'] = metrics
            logger.info(f"Compression metrics: {metrics}")
            
    except Exception as e:
        logger.error(f"Error optimizing LSTM: {e}")
    
    # 3. Optimize Transformer
    logger.info("\n" + "="*50)
    logger.info("Optimizing Transformer model...")
    try:
        logger.info("Would optimize Transformer with following parameter space:")
        logger.info("- d_model: [64, 256]")
        logger.info("- nhead: [2, 8]")
        logger.info("- num_layers: [2, 6]")
        logger.info("- dropout: [0.0, 0.3]")
        
        # Compress if model exists
        transformer_path = models_dir / "transformer_model.pth"
        if transformer_path.exists():
            logger.info("Compressing Transformer model...")
            compressed_path = optimized_dir / "transformer_optimized.pth"
            metrics = compressor.compress_neural_network(
                str(transformer_path), 
                str(compressed_path),
                model_type='transformer'
            )
            compression_results['transformer'] = metrics
            logger.info(f"Compression metrics: {metrics}")
            
    except Exception as e:
        logger.error(f"Error optimizing Transformer: {e}")
    
    # 4. Optimize Ensemble
    logger.info("\n" + "="*50)
    logger.info("Optimizing Ensemble model...")
    try:
        logger.info("Would optimize Ensemble weights and model selection")
        
        # Compress if model exists
        ensemble_path = models_dir / "ensemble_model.pkl"
        if ensemble_path.exists():
            logger.info("Compressing Ensemble model...")
            compressed_path = optimized_dir / "ensemble_optimized.pkl"
            metrics = compressor.compress_ensemble(
                str(ensemble_path), 
                str(compressed_path)
            )
            compression_results['ensemble'] = metrics
            logger.info(f"Compression metrics: {metrics}")
            
    except Exception as e:
        logger.error(f"Error optimizing Ensemble: {e}")
    
    # Save results
    logger.info("\n" + "="*50)
    logger.info("Saving optimization results...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization': optimization_results,
        'compression': compression_results
    }
    
    results_path = output_dir / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results


def generate_summary_report(results, output_dir):
    """Generate a summary report of optimization results"""
    
    report_path = output_dir / "optimization_summary.md"
    
    with open(report_path, 'w') as f:
        f.write("# Optimization Results Summary\n\n")
        f.write(f"Generated: {results['timestamp']}\n\n")
        
        f.write("## Model Compression Results\n\n")
        
        if results['compression']:
            f.write("| Model | Original Size | Compressed Size | Reduction | Performance Impact |\n")
            f.write("|-------|--------------|-----------------|-----------|-------------------|\n")
            
            for model, metrics in results['compression'].items():
                orig_mb = metrics['original_size_mb']
                comp_mb = metrics['compressed_size_mb']
                reduction = metrics['size_reduction_percent']
                perf_impact = metrics.get('performance_impact', 'N/A')
                
                f.write(f"| {model.upper()} | {orig_mb:.2f} MB | {comp_mb:.2f} MB | "
                       f"{reduction:.1f}% | {perf_impact} |\n")
        else:
            f.write("No compression results available.\n")
        
        f.write("\n## Hyperparameter Optimization Results\n\n")
        
        if results['optimization']:
            for model, opt_results in results['optimization'].items():
                f.write(f"### {model.upper()}\n\n")
                f.write(f"Best Score: {opt_results['best_score']:.4f}\n\n")
                f.write("Best Parameters:\n")
                for param, value in opt_results['best_params'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
        else:
            f.write("No optimization results available.\n")
            f.write("\nNote: To run actual optimization, training data is required.\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. Use compressed models for production deployment\n")
        f.write("2. Monitor model performance after compression\n")
        f.write("3. Consider A/B testing optimized vs original models\n")
        f.write("4. Implement gradual rollout for safety\n")
    
    logger.info(f"Summary report saved to {report_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model optimization pipeline")
    parser.add_argument("--models-only", action="store_true", 
                       help="Only compress existing models without optimization")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials per model")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    logger.info("Starting optimization pipeline...")
    logger.info(f"Configuration: n_trials={args.n_trials}, n_jobs={args.n_jobs}")
    
    if args.models_only:
        logger.info("Running in compression-only mode")
    
    try:
        results = run_optimization_pipeline()
        logger.info("\nOptimization pipeline completed successfully!")
        
        # Print summary
        if results['compression']:
            logger.info("\nCompression Summary:")
            for model, metrics in results['compression'].items():
                logger.info(f"  {model}: {metrics['size_reduction_percent']:.1f}% size reduction")
        
    except Exception as e:
        logger.error(f"Optimization pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
