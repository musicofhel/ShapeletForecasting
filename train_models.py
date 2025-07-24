"""
Main training script for financial wavelet prediction models
"""

import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.models.sequence_predictor import LSTMModel, GRUModel
from src.models.transformer_predictor import TransformerPredictor
from src.models.xgboost_predictor import XGBoostPredictor
from src.models.ensemble_model import EnsembleModel
from src.models.model_trainer import ModelTrainer, WalkForwardValidator
from src.models.model_evaluator import ModelEvaluator, create_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> tuple:
    """
    Load processed features and targets
    
    Args:
        data_path: Path to processed data directory
        
    Returns:
        Tuple of (features, targets, feature_names, timestamps)
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load features
    features_df = pd.read_csv(os.path.join(data_path, 'selected_features.csv'))
    
    # Assume last column is target
    feature_cols = features_df.columns[:-1]
    target_col = features_df.columns[-1]
    
    X = features_df[feature_cols].values
    y = features_df[target_col].values
    
    # Generate timestamps (or load if available)
    timestamps = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    return X, y, list(feature_cols), timestamps


def prepare_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int = 20) -> tuple:
    """
    Prepare sequences for LSTM/GRU/Transformer models
    
    Args:
        X: Features
        y: Targets
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X_seq, y_seq)
    """
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def train_lstm_model(X_train, y_train, X_val, y_val, config):
    """Train LSTM model"""
    logger.info("Training LSTM model...")
    
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, config['sequence_length'])
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, config['sequence_length'])
    
    # Model configuration
    model_config = {
        'input_size': X_train.shape[1],
        'hidden_size': config.get('hidden_size', 128),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.2)
    }
    
    training_config = {
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'sequence_length': config['sequence_length']
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='lstm',
        model_config=model_config,
        training_config=training_config,
        use_mlflow=False
    )
    
    # Train model
    model = trainer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        optimize_hyperparams=config.get('optimize_hyperparams', False)
    )
    
    return model, trainer


def train_gru_model(X_train, y_train, X_val, y_val, config):
    """Train GRU model"""
    logger.info("Training GRU model...")
    
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, config['sequence_length'])
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, config['sequence_length'])
    
    # Model configuration
    model_config = {
        'input_size': X_train.shape[1],
        'hidden_size': config.get('hidden_size', 128),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.2)
    }
    
    training_config = {
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'sequence_length': config['sequence_length']
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='gru',
        model_config=model_config,
        training_config=training_config,
        use_mlflow=False
    )
    
    # Train model
    model = trainer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        optimize_hyperparams=config.get('optimize_hyperparams', False)
    )
    
    return model, trainer


def train_transformer_model(X_train, y_train, X_val, y_val, config):
    """Train Transformer model"""
    logger.info("Training Transformer model...")
    
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, config['sequence_length'])
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, config['sequence_length'])
    
    # Model configuration
    model_config = {
        'input_size': X_train.shape[1],
        'd_model': config.get('d_model', 128),
        'n_heads': config.get('n_heads', 8),
        'n_layers': config.get('n_layers', 4),
        'd_ff': config.get('d_ff', 512),
        'dropout': config.get('dropout', 0.1),
        'max_seq_len': config['sequence_length']
    }
    
    training_config = {
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'sequence_length': config['sequence_length']
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='transformer',
        model_config=model_config,
        training_config=training_config,
        use_mlflow=False
    )
    
    # Train model
    model = trainer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        optimize_hyperparams=config.get('optimize_hyperparams', False)
    )
    
    return model, trainer


def train_xgboost_model(X_train, y_train, X_val, y_val, config):
    """Train XGBoost model"""
    logger.info("Training XGBoost model...")
    
    # Model configuration
    model_config = {
        'n_estimators': config.get('n_estimators', 100),
        'max_depth': config.get('max_depth', 6),
        'learning_rate': config.get('learning_rate', 0.1),
        'subsample': config.get('subsample', 0.8),
        'colsample_bytree': config.get('colsample_bytree', 0.8)
    }
    
    training_config = {
        'cv_splits': config.get('cv_splits', 5),
        'n_trials': config.get('n_trials', 50) if config.get('optimize_hyperparams', False) else 0
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='xgboost',
        model_config=model_config,
        training_config=training_config,
        use_mlflow=False
    )
    
    # Train model
    model = trainer.train(
        X_train, y_train,
        X_val, y_val,
        optimize_hyperparams=config.get('optimize_hyperparams', False)
    )
    
    return model, trainer


def train_ensemble_model(models_dict, X_train, y_train, X_val, y_val, config):
    """Train ensemble model"""
    logger.info("Training Ensemble model...")
    
    # Ensemble configuration
    from sklearn.linear_model import Ridge
    
    model_config = {
        'models': models_dict,
        'strategy': config.get('strategy', 'stacking'),
        'meta_learner': Ridge(alpha=0.1) if config.get('strategy') == 'stacking' else None
    }
    
    training_config = {}
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='ensemble',
        model_config=model_config,
        training_config=training_config,
        use_mlflow=False
    )
    
    # Train model
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    return model, trainer


def evaluate_models(models_dict, X_test, y_test, save_dir):
    """Evaluate all models"""
    logger.info("Evaluating models...")
    
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for model_name, model in models_dict.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # Handle neural network models
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_tensor).numpy().flatten()
        
        # Create evaluation report
        model_save_dir = os.path.join(save_dir, model_name.lower().replace(' ', '_'))
        report = create_evaluation_report(
            model_name, y_test, y_pred,
            save_dir=model_save_dir
        )
        
        results[model_name] = report['metrics']
    
    # Compare models
    evaluator = ModelEvaluator()
    models_comparison = {
        name: (y_test, model.predict(X_test) if hasattr(model, 'predict') else 
               model(torch.FloatTensor(X_test)).detach().numpy().flatten())
        for name, model in models_dict.items()
    }
    
    comparison_df = evaluator.compare_models(
        models_comparison,
        save_path=os.path.join(save_dir, 'model_comparison.png')
    )
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    return results, comparison_df


def main(config_path: str):
    """Main training pipeline"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = config.get('output_dir', 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, y, feature_names, timestamps = load_data(config['data_path'])
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train individual models
    trained_models = {}
    
    if config.get('train_lstm', True):
        lstm_model, lstm_trainer = train_lstm_model(
            X_train, y_train, X_val, y_val,
            config.get('lstm_config', {})
        )
        trained_models['LSTM'] = lstm_model
        lstm_trainer.save_training_results(os.path.join(output_dir, 'lstm'))
    
    if config.get('train_gru', True):
        gru_model, gru_trainer = train_gru_model(
            X_train, y_train, X_val, y_val,
            config.get('gru_config', {})
        )
        trained_models['GRU'] = gru_model
        gru_trainer.save_training_results(os.path.join(output_dir, 'gru'))
    
    if config.get('train_transformer', True):
        transformer_model, transformer_trainer = train_transformer_model(
            X_train, y_train, X_val, y_val,
            config.get('transformer_config', {})
        )
        trained_models['Transformer'] = transformer_model
        transformer_trainer.save_training_results(os.path.join(output_dir, 'transformer'))
    
    if config.get('train_xgboost', True):
        xgboost_model, xgboost_trainer = train_xgboost_model(
            X_train, y_train, X_val, y_val,
            config.get('xgboost_config', {})
        )
        trained_models['XGBoost'] = xgboost_model
        xgboost_trainer.save_training_results(os.path.join(output_dir, 'xgboost'))
    
    # Train ensemble if requested
    if config.get('train_ensemble', True) and len(trained_models) > 1:
        # For ensemble, we need sklearn-compatible models
        # Create wrapper for neural networks if needed
        ensemble_models = {}
        for name, model in trained_models.items():
            if name == 'XGBoost':
                ensemble_models[name] = model
            # Skip neural networks for now in ensemble
        
        if len(ensemble_models) > 0:
            ensemble_model, ensemble_trainer = train_ensemble_model(
                ensemble_models, X_train, y_train, X_val, y_val,
                config.get('ensemble_config', {})
            )
            trained_models['Ensemble'] = ensemble_model
            ensemble_trainer.save_training_results(os.path.join(output_dir, 'ensemble'))
    
    # Evaluate all models
    # For sequence models, we need to prepare test sequences
    eval_models = {}
    for name, model in trained_models.items():
        if name in ['LSTM', 'GRU', 'Transformer']:
            X_test_seq, y_test_seq = prepare_sequences(
                X_test, y_test, 
                config.get(f'{name.lower()}_config', {}).get('sequence_length', 20)
            )
            eval_models[name] = (model, X_test_seq, y_test_seq)
        else:
            eval_models[name] = (model, X_test, y_test)
    
    # Evaluate each model with its appropriate data
    results = {}
    for model_name, (model, X_eval, y_eval) in eval_models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_eval)
        else:
            # Handle neural network models
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_eval)
                y_pred = model(X_tensor).numpy().flatten()
        
        # Create evaluation report
        model_save_dir = os.path.join(output_dir, 'evaluation', model_name.lower().replace(' ', '_'))
        report = create_evaluation_report(
            model_name, y_eval, y_pred,
            save_dir=model_save_dir
        )
        
        results[model_name] = report['metrics']
    
    # Save final results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'final_results.csv'))
    
    logger.info("Training complete!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train financial prediction models')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        default_config = {
            'data_path': 'data/processed',
            'output_dir': 'models',
            'train_lstm': True,
            'train_gru': True,
            'train_transformer': True,
            'train_xgboost': True,
            'train_ensemble': True,
            'lstm_config': {
                'sequence_length': 20,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimize_hyperparams': False
            },
            'gru_config': {
                'sequence_length': 20,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimize_hyperparams': False
            },
            'transformer_config': {
                'sequence_length': 20,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 4,
                'd_ff': 512,
                'dropout': 0.1,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimize_hyperparams': False
            },
            'xgboost_config': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'optimize_hyperparams': False
            },
            'ensemble_config': {
                'strategy': 'stacking'
            }
        }
        
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default configuration at {args.config}")
    
    main(args.config)
