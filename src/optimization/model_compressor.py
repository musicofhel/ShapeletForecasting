"""
Model compression and quantization for deployment optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.nn.utils import prune
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import joblib
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelCompressor(ABC):
    """Base class for model compression"""
    
    @abstractmethod
    def compress(self, model: Any, compression_ratio: float = 0.5) -> Any:
        """Compress model"""
        pass
    
    @abstractmethod
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        pass


class QuantizationCompressor(ModelCompressor):
    """Quantization-based model compression for PyTorch models"""
    
    def __init__(self, backend: str = 'fbgemm'):
        """
        Initialize quantization compressor
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        self.backend = backend
        self.original_size = None
        self.compressed_size = None
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
    
    def compress(self, model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
        """
        Compress PyTorch model using quantization
        
        Args:
            model: PyTorch model to compress
            compression_ratio: Not used for quantization
            
        Returns:
            Quantized model
        """
        # Store original size
        self.original_size = self._get_model_size(model)
        
        # Prepare model for quantization
        model.eval()
        
        # Dynamic quantization (easier, works for most models)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
            dtype=torch.qint8
        )
        
        # Store compressed size
        self.compressed_size = self._get_model_size(quantized_model)
        
        logger.info(f"Model quantized: {self.original_size:.2f}MB -> {self.compressed_size:.2f}MB")
        
        return quantized_model
    
    def compress_static(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """
        Static quantization (more complex but better performance)
        
        Args:
            model: PyTorch model
            calibration_data: Data for calibration
            
        Returns:
            Quantized model
        """
        # Prepare model
        model.eval()
        
        # Fuse modules (Conv-BN-ReLU, Linear-ReLU, etc.)
        model = self._fuse_modules(model)
        
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with data
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=True)
        
        self.compressed_size = self._get_model_size(quantized_model)
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for better quantization"""
        # This is model-specific, override for custom models
        return model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        if self.original_size is None or self.compressed_size is None:
            return {}
        
        return {
            'original_size_mb': self.original_size,
            'compressed_size_mb': self.compressed_size,
            'compression_ratio': self.compressed_size / self.original_size,
            'size_reduction': 1 - (self.compressed_size / self.original_size)
        }


class PruningCompressor(ModelCompressor):
    """Pruning-based model compression"""
    
    def __init__(self, pruning_method: str = 'l1_unstructured'):
        """
        Initialize pruning compressor
        
        Args:
            pruning_method: Pruning method to use
        """
        self.pruning_method = pruning_method
        self.original_params = None
        self.pruned_params = None
    
    def compress(self, model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
        """
        Compress model using pruning
        
        Args:
            model: PyTorch model
            compression_ratio: Fraction of parameters to prune
            
        Returns:
            Pruned model
        """
        # Count original parameters
        self.original_params = sum(p.numel() for p in model.parameters())
        
        # Get pruning method
        if self.pruning_method == 'l1_unstructured':
            prune_method = prune.l1_unstructured
        elif self.pruning_method == 'random_unstructured':
            prune_method = prune.random_unstructured
        elif self.pruning_method == 'ln_structured':
            prune_method = prune.ln_structured
        else:
            raise ValueError(f"Unknown pruning method: {self.pruning_method}")
        
        # Prune layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune_method(module, name='weight', amount=compression_ratio)
        
        # Count pruned parameters
        self.pruned_params = sum(
            (p != 0).sum().item() for p in model.parameters()
        )
        
        logger.info(f"Model pruned: {self.original_params} -> {self.pruned_params} parameters")
        
        return model
    
    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning reparameterization"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        return model
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        if self.original_params is None or self.pruned_params is None:
            return {}
        
        return {
            'original_params': self.original_params,
            'pruned_params': self.pruned_params,
            'compression_ratio': self.pruned_params / self.original_params,
            'sparsity': 1 - (self.pruned_params / self.original_params)
        }


class XGBoostCompressor(ModelCompressor):
    """Compression for XGBoost models"""
    
    def __init__(self):
        """Initialize XGBoost compressor"""
        self.original_trees = None
        self.pruned_trees = None
        self.original_size = None
        self.compressed_size = None
    
    def compress(self, model: xgb.XGBRegressor, compression_ratio: float = 0.5) -> xgb.XGBRegressor:
        """
        Compress XGBoost model by reducing number of trees
        
        Args:
            model: XGBoost model
            compression_ratio: Fraction of trees to keep
            
        Returns:
            Compressed model
        """
        # Get original number of trees
        self.original_trees = model.n_estimators
        
        # Calculate new number of trees
        new_n_trees = int(self.original_trees * (1 - compression_ratio))
        new_n_trees = max(1, new_n_trees)  # Keep at least 1 tree
        
        # Create new model with fewer trees
        compressed_model = xgb.XGBRegressor(**model.get_params())
        compressed_model.n_estimators = new_n_trees
        
        # Copy the first n trees from original model
        booster = model.get_booster()
        
        # Save and reload with limited trees
        temp_path = 'temp_xgb_model.json'
        booster.save_model(temp_path)
        
        # Load with limited trees
        compressed_booster = xgb.Booster()
        compressed_booster.load_model(temp_path)
        compressed_model._Booster = compressed_booster
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        self.pruned_trees = new_n_trees
        
        logger.info(f"XGBoost compressed: {self.original_trees} -> {self.pruned_trees} trees")
        
        return compressed_model
    
    def compress_by_importance(self, model: xgb.XGBRegressor, 
                             importance_threshold: float = 0.01) -> xgb.XGBRegressor:
        """
        Compress by removing less important features
        
        Args:
            model: XGBoost model
            importance_threshold: Minimum feature importance to keep
            
        Returns:
            Compressed model
        """
        # Get feature importance
        importance = model.feature_importances_
        
        # Select important features
        important_features = np.where(importance > importance_threshold)[0]
        
        logger.info(f"Keeping {len(important_features)} out of {len(importance)} features")
        
        # Note: This returns feature indices, actual retraining needed
        return model, important_features
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        if self.original_trees is None or self.pruned_trees is None:
            return {}
        
        return {
            'original_trees': self.original_trees,
            'compressed_trees': self.pruned_trees,
            'compression_ratio': self.pruned_trees / self.original_trees,
            'tree_reduction': 1 - (self.pruned_trees / self.original_trees)
        }


class ModelOptimizer:
    """Combined optimization: compression + quantization"""
    
    def __init__(self):
        """Initialize model optimizer"""
        self.quantizer = QuantizationCompressor()
        self.pruner = PruningCompressor()
    
    def optimize_pytorch_model(self, model: nn.Module, 
                             pruning_ratio: float = 0.3,
                             quantize: bool = True) -> nn.Module:
        """
        Optimize PyTorch model with pruning and quantization
        
        Args:
            model: PyTorch model
            pruning_ratio: Fraction to prune
            quantize: Whether to quantize after pruning
            
        Returns:
            Optimized model
        """
        optimized_model = model
        stats = {}
        
        # Step 1: Pruning
        if pruning_ratio > 0:
            optimized_model = self.pruner.compress(optimized_model, pruning_ratio)
            stats['pruning'] = self.pruner.get_compression_stats()
        
        # Step 2: Quantization
        if quantize:
            optimized_model = self.quantizer.compress(optimized_model)
            stats['quantization'] = self.quantizer.get_compression_stats()
        
        # Log results
        logger.info("Model optimization complete:")
        for method, method_stats in stats.items():
            logger.info(f"{method}: {method_stats}")
        
        return optimized_model
    
    def benchmark_model(self, original_model: nn.Module, 
                       optimized_model: nn.Module,
                       test_data: torch.Tensor,
                       n_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark original vs optimized model
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
            test_data: Test data for benchmarking
            n_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        # Warm up
        for _ in range(10):
            _ = original_model(test_data)
            _ = optimized_model(test_data)
        
        # Benchmark original
        original_times = []
        for _ in range(n_runs):
            start = time.time()
            with torch.no_grad():
                _ = original_model(test_data)
            original_times.append(time.time() - start)
        
        # Benchmark optimized
        optimized_times = []
        for _ in range(n_runs):
            start = time.time()
            with torch.no_grad():
                _ = optimized_model(test_data)
            optimized_times.append(time.time() - start)
        
        # Calculate statistics
        results = {
            'original_mean_time': np.mean(original_times),
            'optimized_mean_time': np.mean(optimized_times),
            'speedup': np.mean(original_times) / np.mean(optimized_times),
            'original_std': np.std(original_times),
            'optimized_std': np.std(optimized_times)
        }
        
        return results


def save_compressed_model(model: Any, filepath: str, compression_stats: Dict[str, Any]):
    """
    Save compressed model with metadata
    
    Args:
        model: Compressed model
        filepath: Save path
        compression_stats: Compression statistics
    """
    # Save model
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), filepath)
    elif isinstance(model, xgb.XGBRegressor):
        model.save_model(filepath)
    else:
        joblib.dump(model, filepath)
    
    # Save compression stats
    stats_path = filepath.replace('.pkl', '_compression_stats.json')
    stats_path = stats_path.replace('.pth', '_compression_stats.json')
    stats_path = stats_path.replace('.json', '_compression_stats.json')
    
    with open(stats_path, 'w') as f:
        json.dump(compression_stats, f, indent=2)
    
    logger.info(f"Compressed model saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    import torch.nn.functional as F
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Create model
    model = SimpleModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test data
    test_data = torch.randn(32, 10)
    
    # Test pruning
    print("\n--- Testing Pruning ---")
    pruner = PruningCompressor()
    pruned_model = pruner.compress(model, compression_ratio=0.5)
    print(f"Pruning stats: {pruner.get_compression_stats()}")
    
    # Test quantization
    print("\n--- Testing Quantization ---")
    quantizer = QuantizationCompressor()
    quantized_model = quantizer.compress(model)
    print(f"Quantization stats: {quantizer.get_compression_stats()}")
    
    # Test combined optimization
    print("\n--- Testing Combined Optimization ---")
    optimizer = ModelOptimizer()
    optimized_model = optimizer.optimize_pytorch_model(
        model, 
        pruning_ratio=0.3,
        quantize=True
    )
    
    # Benchmark
    print("\n--- Benchmarking ---")
    benchmark_results = optimizer.benchmark_model(
        model, optimized_model, test_data, n_runs=50
    )
    print(f"Benchmark results: {benchmark_results}")
