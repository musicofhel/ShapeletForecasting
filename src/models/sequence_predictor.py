"""
Sequence prediction models using LSTM and GRU architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 20, prediction_horizon: int = 1):
        """
        Initialize dataset
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Calculate valid indices
        self.valid_indices = len(features) - sequence_length - prediction_horizon + 1
        
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Get sequence
        seq_start = idx
        seq_end = idx + self.sequence_length
        sequence = self.features[seq_start:seq_end]
        
        # Get target
        target_idx = seq_end + self.prediction_horizon - 1
        target = self.targets[target_idx]
        
        return sequence, target


class SequencePredictor(nn.Module, ABC):
    """Base class for sequence prediction models"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int = 1, dropout: float = 0.2):
        """
        Initialize sequence predictor
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        
        # To be defined in subclasses
        self.rnn = None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    @abstractmethod
    def init_hidden(self, batch_size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden state"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        
        # RNN forward pass
        if isinstance(hidden, tuple):
            out, hidden = self.rnn(x, hidden)
        else:
            out, hidden = self.rnn(x, hidden)
        
        # Take the last output
        out = out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
    def predict(self, features: np.ndarray, sequence_length: int = 20) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features: Feature matrix
            sequence_length: Length of input sequences
            
        Returns:
            Predictions
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(features) - sequence_length + 1):
                # Get sequence
                sequence = features[i:i+sequence_length]
                sequence = torch.FloatTensor(sequence).unsqueeze(0)
                
                # Make prediction
                pred = self.forward(sequence)
                predictions.append(pred.numpy()[0])
        
        return np.array(predictions)


class LSTMModel(SequencePredictor):
    """LSTM model for sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1, 
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout)
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Adjust FC layer for bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        
        logger.info(f"Initialized LSTM model with {self.count_parameters():,} parameters")
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states"""
        device = next(self.parameters()).device
        
        h0 = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size
        ).to(device)
        
        c0 = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size
        ).to(device)
        
        return h0, c0
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUModel(SequencePredictor):
    """GRU model for sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1, 
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Initialize GRU model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of GRU layers
            output_size: Number of output features
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout)
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Adjust FC layer for bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        
        logger.info(f"Initialized GRU model with {self.count_parameters():,} parameters")
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state"""
        device = next(self.parameters()).device
        
        h0 = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size
        ).to(device)
        
        return h0
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequenceModelFactory:
    """Factory for creating sequence models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict) -> SequencePredictor:
        """
        Create a sequence model
        
        Args:
            model_type: Type of model ('lstm' or 'gru')
            config: Model configuration
            
        Returns:
            Sequence model instance
        """
        if model_type.lower() == 'lstm':
            return LSTMModel(**config)
        elif model_type.lower() == 'gru':
            return GRUModel(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def train_sequence_model(
    model: SequencePredictor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience: int = 10,
    checkpoint_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Train a sequence model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        early_stopping_patience: Patience for early stopping
        checkpoint_path: Path to save best model
        
    Returns:
        Training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                if checkpoint_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        else:
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )
    
    return history


if __name__ == "__main__":
    # Example usage
    print("Sequence predictor module loaded successfully")
    print("Use LSTMModel or GRUModel classes for sequence prediction")
