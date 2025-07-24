"""
Transformer architecture for financial time series prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor (seq_len, batch_size, d_model)
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            
        Returns:
            Output tensor and attention weights
        """
        seq_len, batch_size, _ = query.size()
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(seq_len, batch_size, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (seq_len, batch_size, d_model)
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerPredictor(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 512, max_seq_len: int = 100,
                 output_size: int = 1, dropout: float = 0.1):
        """
        Initialize transformer predictor
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            output_size: Number of output features
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized Transformer model with {self.count_parameters():,} parameters")
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create padding mask for variable length sequences
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            lengths: Actual lengths of sequences
            
        Returns:
            Padding mask
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        if lengths is None:
            return None
        
        mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        return mask.to(x.device)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            mask: Padding mask
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        # Transpose to (seq_len, batch_size, input_size)
        x = x.transpose(0, 1)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Take the last output
        x = x[-1, :, :]  # (batch_size, d_model)
        
        # Output projection
        x = self.dropout(x)
        output = self.output_projection(x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all layers
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            List of attention weight tensors
        """
        x = x.transpose(0, 1)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        attention_weights = []
        
        for transformer in self.transformer_blocks:
            attn_output, attn_weights = transformer.attention(x, x, x)
            attention_weights.append(attn_weights)
            x = transformer.norm1(x + transformer.dropout(attn_output))
            ff_output = transformer.feed_forward(x)
            x = transformer.norm2(x + transformer.dropout(ff_output))
        
        return attention_weights
    
    def predict(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features: Feature matrix (n_samples, seq_len, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                batch = torch.FloatTensor(batch)
                
                pred = self.forward(batch)
                predictions.extend(pred.numpy())
        
        return np.array(predictions)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for interpretable predictions"""
    
    def __init__(self, input_size: int, hidden_size: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        """
        Initialize TFT model
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Variable selection networks
        self.static_vsn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        
        self.temporal_vsn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.GLU(dim=-1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Predictions and interpretability outputs
        """
        batch_size, seq_len, _ = x.size()
        
        # Variable selection
        static_weights = self.static_vsn(x[:, -1, :])  # Use last timestep for static
        temporal_weights = self.temporal_vsn(x)
        
        # Apply variable selection
        x_selected = x * temporal_weights
        
        # LSTM encoding
        encoded, _ = self.encoder(x_selected)
        
        # Self-attention
        attn_output, attn_weights = self.attention(encoded, encoded, encoded)
        
        # Gated residual connection
        grn_output = self.grn(attn_output[:, -1, :])
        
        # Final prediction
        output = self.output_layer(grn_output)
        
        # Interpretability outputs
        interpretability = {
            'static_weights': static_weights,
            'temporal_weights': temporal_weights,
            'attention_weights': attn_weights
        }
        
        return output, interpretability


def create_transformer_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Create a transformer model
    
    Args:
        model_type: Type of transformer ('standard' or 'tft')
        **kwargs: Model configuration
        
    Returns:
        Transformer model instance
    """
    if model_type == 'standard':
        return TransformerPredictor(**kwargs)
    elif model_type == 'tft':
        return TemporalFusionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown transformer type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("Transformer predictor module loaded successfully")
    print("Use TransformerPredictor or TemporalFusionTransformer classes for sequence prediction")
