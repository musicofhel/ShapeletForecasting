"""
Model evaluation metrics and visualization tools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize evaluator
        
        Args:
            model_name: Name of the model for reporting
        """
        self.model_name = model_name
        self.results = {}
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps for time-based analysis
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        
        # Additional metrics
        residuals = y_true - y_pred
        metrics.update({
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skew_residual': stats.skew(residuals),
            'kurtosis_residual': stats.kurtosis(residuals),
            'max_error': np.max(np.abs(residuals)),
            'percentile_95_error': np.percentile(np.abs(residuals), 95)
        })
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'residuals': residuals,
            'timestamps': timestamps
        }
        
        return metrics
    
    def plot_results(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 12)):
        """
        Create comprehensive evaluation plots
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate() first.")
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'{self.model_name} Evaluation Results', fontsize=16)
        
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        residuals = self.results['residuals']
        
        # 1. Predictions vs Actual
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs Actual')
        ax.text(0.05, 0.95, f"R² = {self.results['metrics']['r2']:.4f}",
                transform=ax.transAxes, verticalalignment='top')
        
        # 2. Residual Plot
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        
        # 3. Residual Distribution
        ax = axes[0, 2]
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.text(0.05, 0.95, f"Skew: {self.results['metrics']['skew_residual']:.3f}",
                transform=ax.transAxes, verticalalignment='top')
        
        # 4. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        # 5. Time Series Plot (if timestamps available)
        ax = axes[1, 1]
        if self.results['timestamps'] is not None:
            ax.plot(self.results['timestamps'], y_true, label='Actual', alpha=0.7)
            ax.plot(self.results['timestamps'], y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Time')
        else:
            indices = np.arange(len(y_true))
            ax.plot(indices, y_true, label='Actual', alpha=0.7)
            ax.plot(indices, y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Comparison')
        ax.legend()
        
        # 6. Error over Time
        ax = axes[1, 2]
        abs_errors = np.abs(residuals)
        if self.results['timestamps'] is not None:
            ax.plot(self.results['timestamps'], abs_errors, alpha=0.7)
            ax.set_xlabel('Time')
        else:
            ax.plot(abs_errors, alpha=0.7)
            ax.set_xlabel('Index')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error over Time')
        
        # 7. Actual vs Predicted Scatter with Density
        ax = axes[2, 0]
        from scipy.stats import gaussian_kde
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = y_true[idx], y_pred[idx], z[idx]
        scatter = ax.scatter(x, y, c=z, s=10, cmap='viridis', alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions with Density')
        plt.colorbar(scatter, ax=ax)
        
        # 8. Cumulative Error
        ax = axes[2, 1]
        cumulative_error = np.cumsum(residuals)
        if self.results['timestamps'] is not None:
            ax.plot(self.results['timestamps'], cumulative_error)
            ax.set_xlabel('Time')
        else:
            ax.plot(cumulative_error)
            ax.set_xlabel('Index')
        ax.set_ylabel('Cumulative Error')
        ax.set_title('Cumulative Error')
        ax.axhline(y=0, color='r', linestyle='--')
        
        # 9. Metrics Summary
        ax = axes[2, 2]
        ax.axis('off')
        metrics_text = f"""
        Metrics Summary:
        
        RMSE: {self.results['metrics']['rmse']:.4f}
        MAE: {self.results['metrics']['mae']:.4f}
        MAPE: {self.results['metrics']['mape']:.4f}
        R²: {self.results['metrics']['r2']:.4f}
        
        Max Error: {self.results['metrics']['max_error']:.4f}
        95% Error: {self.results['metrics']['percentile_95_error']:.4f}
        
        Direction Acc: {self.results['metrics'].get('directional_accuracy', 'N/A')}
        """
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plot saved to {save_path}")
        
        return fig
    
    def plot_error_analysis(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Create detailed error analysis plots
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{self.model_name} Error Analysis', fontsize=16)
        
        residuals = self.results['residuals']
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # 1. Error Distribution by Percentile
        ax = axes[0, 0]
        percentiles = np.arange(0, 101, 5)
        error_percentiles = np.percentile(np.abs(residuals), percentiles)
        ax.plot(percentiles, error_percentiles)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution by Percentile')
        ax.grid(True, alpha=0.3)
        
        # 2. Error vs Actual Value
        ax = axes[0, 1]
        ax.scatter(y_true, np.abs(residuals), alpha=0.5, s=10)
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Actual Value')
        
        # Add trend line
        z = np.polyfit(y_true, np.abs(residuals), 1)
        p = np.poly1d(z)
        ax.plot(sorted(y_true), p(sorted(y_true)), "r--", alpha=0.8)
        
        # 3. Standardized Residuals
        ax = axes[0, 2]
        std_residuals = residuals / np.std(residuals)
        ax.hist(std_residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Standardized Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Standardized Residual Distribution')
        
        # Add normal distribution overlay
        x = np.linspace(-4, 4, 100)
        ax2 = ax.twinx()
        ax2.plot(x, stats.norm.pdf(x, 0, 1), 'r-', alpha=0.8, label='Normal')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 4. ACF of Residuals
        ax = axes[1, 0]
        from statsmodels.tsa.stattools import acf
        lags = min(40, len(residuals) // 4)
        acf_values = acf(residuals, nlags=lags)
        ax.bar(range(len(acf_values)), acf_values)
        ax.axhline(y=0, color='black', linestyle='-')
        ax.axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('Autocorrelation of Residuals')
        
        # 5. Partial ACF of Residuals
        ax = axes[1, 1]
        from statsmodels.tsa.stattools import pacf
        pacf_values = pacf(residuals, nlags=lags)
        ax.bar(range(len(pacf_values)), pacf_values)
        ax.axhline(y=0, color='black', linestyle='-')
        ax.axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('PACF')
        ax.set_title('Partial Autocorrelation of Residuals')
        
        # 6. Error Heatmap by Time (if applicable)
        ax = axes[1, 2]
        if len(residuals) > 100:
            # Reshape residuals into a 2D array for heatmap
            n_rows = 20
            n_cols = len(residuals) // n_rows
            if n_cols > 0:
                reshaped_errors = np.abs(residuals[:n_rows*n_cols]).reshape(n_rows, n_cols)
                im = ax.imshow(reshaped_errors, aspect='auto', cmap='YlOrRd')
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('Periods')
                ax.set_title('Error Heatmap Over Time')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                       transform=ax.transAxes, ha='center')
                ax.set_title('Error Heatmap Over Time')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   transform=ax.transAxes, ha='center')
            ax.set_title('Error Heatmap Over Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")
        
        return fig
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading-specific metrics
        
        Args:
            y_true: True values (returns)
            y_pred: Predicted values (returns)
            transaction_cost: Cost per transaction as fraction
            
        Returns:
            Dictionary of trading metrics
        """
        # Generate trading signals
        true_signals = np.sign(y_true)
        pred_signals = np.sign(y_pred)
        
        # Calculate returns
        strategy_returns = pred_signals[:-1] * y_true[1:]
        
        # Account for transaction costs
        signal_changes = np.diff(pred_signals)
        n_trades = np.sum(signal_changes != 0)
        total_cost = n_trades * transaction_cost
        
        # Calculate metrics
        metrics = {
            'total_return': np.sum(strategy_returns) - total_cost,
            'sharpe_ratio': np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(np.cumsum(strategy_returns)),
            'win_rate': np.mean(strategy_returns > 0),
            'profit_factor': np.sum(strategy_returns[strategy_returns > 0]) / (np.abs(np.sum(strategy_returns[strategy_returns < 0])) + 1e-6),
            'number_of_trades': n_trades,
            'avg_return_per_trade': (np.sum(strategy_returns) - total_cost) / (n_trades + 1e-6)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-6)
        return np.min(drawdown)
    
    def compare_models(self, models: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {model_name: (y_true, y_pred)}
            save_path: Path to save comparison plot
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = []
        
        for model_name, (y_true, y_pred) in models.items():
            evaluator = ModelEvaluator(model_name)
            metrics = evaluator.evaluate(y_true, y_pred)
            metrics['model'] = model_name
            comparison_results.append(metrics)
        
        df_comparison = pd.DataFrame(comparison_results)
        
        # Create comparison plot
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model Comparison', fontsize=16)
            
            # Metrics to compare
            metrics_to_plot = ['rmse', 'mae', 'r2', 'directional_accuracy']
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]
                if metric in df_comparison.columns:
                    df_comparison.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
                    ax.set_title(metric.upper())
                    ax.set_xlabel('')
                    ax.set_ylabel(metric.upper())
                    
                    # Add value labels on bars
                    for i, v in enumerate(df_comparison[metric]):
                        ax.text(i, v + 0.01 * max(df_comparison[metric]), 
                               f'{v:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return df_comparison


def create_evaluation_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                           save_dir: str, timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create comprehensive evaluation report
    
    Args:
        model_name: Name of the model
        y_true: True values
        y_pred: Predicted values
        save_dir: Directory to save report
        timestamps: Optional timestamps
        
    Returns:
        Dictionary with all evaluation results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_name)
    
    # Calculate metrics
    metrics = evaluator.evaluate(y_true, y_pred, timestamps)
    
    # Create plots
    evaluator.plot_results(os.path.join(save_dir, 'evaluation_results.png'))
    evaluator.plot_error_analysis(os.path.join(save_dir, 'error_analysis.png'))
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    # Create text report
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model Evaluation Report: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric:.<25} {value:.6f}\n")
        
        f.write("\n\nStatistical Summary:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of predictions: {len(y_true)}\n")
        f.write(f"Mean actual value: {np.mean(y_true):.6f}\n")
        f.write(f"Mean predicted value: {np.mean(y_pred):.6f}\n")
        f.write(f"Std actual value: {np.std(y_true):.6f}\n")
        f.write(f"Std predicted value: {np.std(y_pred):.6f}\n")
    
    logger.info(f"Evaluation report saved to {save_dir}")
    
    return {
        'metrics': metrics,
        'evaluator': evaluator,
        'report_path': report_path
    }


if __name__ == "__main__":
    # Example usage
    print("Model evaluator module loaded successfully")
    print("Use ModelEvaluator class to evaluate model predictions")
