"""
Pattern Predictor Module

This module implements multiple models for predicting the next pattern in a sequence:
1. LSTM/GRU model for pattern sequence prediction
2. Transformer model for pattern attention
3. Markov chain for pattern transitions
4. Ensemble approach combining multiple methods

The predictor provides confidence intervals and supports multiple prediction horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PatternDataset(Dataset):
    """PyTorch dataset for pattern sequences"""
    
    def __init__(self, sequences: List[List[Dict]], seq_length: int = 10):
        self.sequences = sequences
        self.seq_length = seq_length
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Prepare data
        self.X, self.y = self._prepare_sequences()
        
    def _prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        X, y = [], []
        
        # Extract all pattern types for encoding
        all_types = []
        for seq in self.sequences:
            for pattern in seq:
                all_types.append(pattern['type'])
        
        # Fit label encoder
        self.label_encoder.fit(all_types)
        
        # Create training sequences
        for seq in self.sequences:
            if len(seq) < self.seq_length + 1:
                continue
                
            for i in range(len(seq) - self.seq_length):
                # Input sequence
                input_seq = seq[i:i + self.seq_length]
                # Target pattern
                target = seq[i + self.seq_length]
                
                # Extract features
                seq_features = []
                for pattern in input_seq:
                    features = self._extract_pattern_features(pattern)
                    seq_features.append(features)
                
                X.append(seq_features)
                y.append(self.label_encoder.transform([target['type']])[0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit and transform features
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.feature_scaler.fit(X_reshaped)
        X_scaled = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
        
        return X_scaled, y
    
    def _extract_pattern_features(self, pattern: Dict) -> np.ndarray:
        """Extract numerical features from pattern"""
        features = [
            self.label_encoder.transform([pattern['type']])[0],
            pattern.get('scale', 0),
            pattern.get('amplitude', 0),
            pattern.get('duration', 0),
            pattern.get('energy', 0),
            pattern.get('entropy', 0),
            pattern.get('skewness', 0),
            pattern.get('kurtosis', 0)
        ]
        return np.array(features)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])


class LSTMPredictor(nn.Module):
    """LSTM model for pattern sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRUPredictor(nn.Module):
    """GRU model for pattern sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        # Take the last output
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class TransformerPredictor(nn.Module):
    """Transformer model for pattern attention"""
    
    def __init__(self, input_size: int, num_classes: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.2):
        super(TransformerPredictor, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 
                                                   dim_feedforward=512,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Transformer encoding
        output = self.transformer_encoder(x)
        
        # Take the last output and transpose back
        output = output[-1, :, :]
        
        # Final classification
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MarkovChainPredictor:
    """Markov chain model for pattern transitions"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transition_matrix = {}
        self.state_counts = {}
        self.pattern_types = set()
        
    def fit(self, sequences: List[List[str]]):
        """Fit the Markov chain model"""
        # Build transition matrix
        for seq in sequences:
            for i in range(len(seq) - self.order):
                # Create state from order previous patterns
                state = tuple(seq[i:i + self.order])
                next_pattern = seq[i + self.order]
                
                # Update counts
                if state not in self.transition_matrix:
                    self.transition_matrix[state] = {}
                    self.state_counts[state] = 0
                
                if next_pattern not in self.transition_matrix[state]:
                    self.transition_matrix[state][next_pattern] = 0
                
                self.transition_matrix[state][next_pattern] += 1
                self.state_counts[state] += 1
                self.pattern_types.add(next_pattern)
        
        # Convert counts to probabilities
        for state in self.transition_matrix:
            total = self.state_counts[state]
            for next_pattern in self.transition_matrix[state]:
                self.transition_matrix[state][next_pattern] /= total
    
    def predict_proba(self, sequence: List[str]) -> Dict[str, float]:
        """Predict probabilities for next pattern"""
        if len(sequence) < self.order:
            # Return uniform distribution if sequence too short
            uniform_prob = 1.0 / len(self.pattern_types)
            return {p: uniform_prob for p in self.pattern_types}
        
        # Get current state
        state = tuple(sequence[-self.order:])
        
        if state in self.transition_matrix:
            # Return learned probabilities
            probs = self.transition_matrix[state].copy()
            # Add small probability for unseen patterns
            for p in self.pattern_types:
                if p not in probs:
                    probs[p] = 1e-6
            # Normalize
            total = sum(probs.values())
            return {p: prob/total for p, prob in probs.items()}
        else:
            # Return uniform distribution for unseen state
            uniform_prob = 1.0 / len(self.pattern_types)
            return {p: uniform_prob for p in self.pattern_types}


class PatternPredictor:
    """Main pattern predictor with ensemble approach"""
    
    def __init__(self, seq_length: int = 10, device: str = None):
        self.seq_length = seq_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.lstm_model = None
        self.gru_model = None
        self.transformer_model = None
        self.markov_model = MarkovChainPredictor(order=2)
        
        # Ensemble weights
        self.ensemble_weights = {
            'lstm': 0.3,
            'gru': 0.3,
            'transformer': 0.3,
            'markov': 0.1
        }
        
        # Data processors
        self.label_encoder = None
        self.feature_scaler = None
        
        # Training history
        self.training_history = {
            'lstm': [],
            'gru': [],
            'transformer': [],
            'ensemble': []
        }
        
    def train(self, sequences: List[List[Dict]], epochs: int = 50, 
              batch_size: int = 32, learning_rate: float = 0.001,
              validation_split: float = 0.2):
        """Train all models"""
        print("Preparing dataset...")
        dataset = PatternDataset(sequences, self.seq_length)
        self.label_encoder = dataset.label_encoder
        self.feature_scaler = dataset.feature_scaler
        
        # Split data
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Get dimensions
        input_size = dataset.X.shape[-1]
        num_classes = len(self.label_encoder.classes_)
        
        # Initialize models
        self.lstm_model = LSTMPredictor(input_size, 128, num_classes).to(self.device)
        self.gru_model = GRUPredictor(input_size, 128, num_classes).to(self.device)
        self.transformer_model = TransformerPredictor(input_size, num_classes).to(self.device)
        
        # Train neural models
        print("\nTraining LSTM model...")
        self._train_model(self.lstm_model, train_loader, val_loader, epochs, 
                         learning_rate, 'lstm')
        
        print("\nTraining GRU model...")
        self._train_model(self.gru_model, train_loader, val_loader, epochs,
                         learning_rate, 'gru')
        
        print("\nTraining Transformer model...")
        self._train_model(self.transformer_model, train_loader, val_loader, epochs,
                         learning_rate, 'transformer')
        
        # Train Markov model
        print("\nTraining Markov chain model...")
        pattern_sequences = []
        for seq in sequences:
            pattern_types = [p['type'] for p in seq]
            pattern_sequences.append(pattern_types)
        self.markov_model.fit(pattern_sequences)
        
        # Optimize ensemble weights
        print("\nOptimizing ensemble weights...")
        self._optimize_ensemble_weights(val_loader)
        
    def _train_model(self, model: nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, epochs: int, learning_rate: float,
                    model_name: str):
        """Train a single neural network model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.squeeze().to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Store history
            self.training_history[model_name].append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
    
    def _optimize_ensemble_weights(self, val_loader: DataLoader):
        """Optimize ensemble weights using validation data"""
        # Get predictions from all models
        all_predictions = {
            'lstm': [],
            'gru': [],
            'transformer': [],
            'markov': []
        }
        true_labels = []
        
        self.lstm_model.eval()
        self.gru_model.eval()
        self.transformer_model.eval()
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.squeeze().cpu().numpy()
                
                # Get neural network predictions
                lstm_probs = F.softmax(self.lstm_model(batch_x), dim=1).cpu().numpy()
                gru_probs = F.softmax(self.gru_model(batch_x), dim=1).cpu().numpy()
                transformer_probs = F.softmax(self.transformer_model(batch_x), dim=1).cpu().numpy()
                
                all_predictions['lstm'].append(lstm_probs)
                all_predictions['gru'].append(gru_probs)
                all_predictions['transformer'].append(transformer_probs)
                
                # Get Markov predictions (simplified for batch)
                markov_probs = np.zeros_like(lstm_probs)
                for i in range(len(batch_y)):
                    # Use uniform distribution for Markov in batch setting
                    markov_probs[i] = 1.0 / len(self.label_encoder.classes_)
                
                all_predictions['markov'].append(markov_probs)
                true_labels.extend(batch_y)
        
        # Concatenate predictions
        for model in all_predictions:
            all_predictions[model] = np.vstack(all_predictions[model])
        true_labels = np.array(true_labels)
        
        # Grid search for optimal weights
        best_acc = 0
        best_weights = self.ensemble_weights.copy()
        
        for w1 in np.arange(0.1, 0.5, 0.1):
            for w2 in np.arange(0.1, 0.5, 0.1):
                for w3 in np.arange(0.1, 0.5, 0.1):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0 or w4 > 0.3:
                        continue
                    
                    # Calculate ensemble predictions
                    ensemble_probs = (w1 * all_predictions['lstm'] +
                                    w2 * all_predictions['gru'] +
                                    w3 * all_predictions['transformer'] +
                                    w4 * all_predictions['markov'])
                    
                    # Calculate accuracy
                    ensemble_preds = np.argmax(ensemble_probs, axis=1)
                    acc = np.mean(ensemble_preds == true_labels)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = {
                            'lstm': w1,
                            'gru': w2,
                            'transformer': w3,
                            'markov': w4
                        }
        
        self.ensemble_weights = best_weights
        print(f"Optimized ensemble weights: {self.ensemble_weights}")
        print(f"Ensemble validation accuracy: {best_acc*100:.2f}%")
    
    def predict(self, sequence: List[Dict], horizon: int = 1,
                return_confidence: bool = True) -> Dict[str, Any]:
        """Predict next pattern(s) with confidence intervals"""
        if len(sequence) < self.seq_length:
            raise ValueError(f"Sequence must have at least {self.seq_length} patterns")
        
        predictions = []
        
        for h in range(horizon):
            # Use last seq_length patterns
            input_seq = sequence[-(self.seq_length):]
            
            # Extract features
            features = []
            for pattern in input_seq:
                feat = self._extract_pattern_features(pattern)
                features.append(feat)
            
            features = np.array(features)
            features_scaled = self.feature_scaler.transform(
                features.reshape(-1, features.shape[-1])
            ).reshape(1, self.seq_length, -1)
            
            # Get predictions from each model
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            with torch.no_grad():
                lstm_probs = F.softmax(self.lstm_model(features_tensor), dim=1).cpu().numpy()[0]
                gru_probs = F.softmax(self.gru_model(features_tensor), dim=1).cpu().numpy()[0]
                transformer_probs = F.softmax(self.transformer_model(features_tensor), dim=1).cpu().numpy()[0]
            
            # Get Markov predictions
            pattern_types = [p['type'] for p in input_seq]
            markov_probs_dict = self.markov_model.predict_proba(pattern_types)
            markov_probs = np.zeros(len(self.label_encoder.classes_))
            for i, class_name in enumerate(self.label_encoder.classes_):
                markov_probs[i] = markov_probs_dict.get(class_name, 1e-6)
            
            # Ensemble predictions
            ensemble_probs = (self.ensemble_weights['lstm'] * lstm_probs +
                            self.ensemble_weights['gru'] * gru_probs +
                            self.ensemble_weights['transformer'] * transformer_probs +
                            self.ensemble_weights['markov'] * markov_probs)
            
            # Get prediction
            pred_idx = np.argmax(ensemble_probs)
            pred_type = self.label_encoder.inverse_transform([pred_idx])[0]
            
            # Calculate confidence
            confidence = float(ensemble_probs[pred_idx])
            
            # Calculate confidence interval (using ensemble variance)
            model_preds = np.array([
                lstm_probs[pred_idx],
                gru_probs[pred_idx],
                transformer_probs[pred_idx],
                markov_probs[pred_idx]
            ])
            std_dev = np.std(model_preds)
            confidence_interval = (
                max(0, confidence - 1.96 * std_dev),
                min(1, confidence + 1.96 * std_dev)
            )
            
            prediction = {
                'horizon': h + 1,
                'pattern_type': pred_type,
                'confidence': confidence,
                'confidence_interval': confidence_interval,
                'probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(ensemble_probs)
                }
            }
            
            if return_confidence:
                prediction['model_confidences'] = {
                    'lstm': float(lstm_probs[pred_idx]),
                    'gru': float(gru_probs[pred_idx]),
                    'transformer': float(transformer_probs[pred_idx]),
                    'markov': float(markov_probs[pred_idx])
                }
            
            predictions.append(prediction)
            
            # Add predicted pattern to sequence for multi-horizon prediction
            predicted_pattern = {
                'type': pred_type,
                'scale': np.mean([p['scale'] for p in input_seq[-3:]]),
                'amplitude': np.mean([p['amplitude'] for p in input_seq[-3:]]),
                'duration': np.mean([p['duration'] for p in input_seq[-3:]]),
                'energy': np.mean([p['energy'] for p in input_seq[-3:]]),
                'entropy': np.mean([p['entropy'] for p in input_seq[-3:]]),
                'skewness': np.mean([p['skewness'] for p in input_seq[-3:]]),
                'kurtosis': np.mean([p['kurtosis'] for p in input_seq[-3:]])
            }
            sequence.append(predicted_pattern)
        
        return {
            'predictions': predictions,
            'ensemble_weights': self.ensemble_weights,
            'prediction_time_ms': 0  # Will be updated by timing wrapper
        }
    
    def _extract_pattern_features(self, pattern: Dict) -> np.ndarray:
        """Extract numerical features from pattern"""
        features = [
            self.label_encoder.transform([pattern['type']])[0],
            pattern.get('scale', 0),
            pattern.get('amplitude', 0),
            pattern.get('duration', 0),
            pattern.get('energy', 0),
            pattern.get('entropy', 0),
            pattern.get('skewness', 0),
            pattern.get('kurtosis', 0)
        ]
        return np.array(features)
    
    def evaluate_calibration(self, test_sequences: List[List[Dict]]) -> Dict[str, Any]:
        """Evaluate confidence calibration"""
        all_confidences = []
        all_accuracies = []
        
        for seq in test_sequences:
            if len(seq) < self.seq_length + 1:
                continue
            
            for i in range(len(seq) - self.seq_length):
                input_seq = seq[i:i + self.seq_length]
                true_pattern = seq[i + self.seq_length]
                
                # Get prediction
                result = self.predict(input_seq, horizon=1, return_confidence=True)
                pred = result['predictions'][0]
                
                # Check if prediction is correct
                is_correct = pred['pattern_type'] == true_pattern['type']
                
                all_confidences.append(pred['confidence'])
                all_accuracies.append(int(is_correct))
        
        # Calculate calibration metrics
        confidences = np.array(all_confidences)
        accuracies = np.array(all_accuracies)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=10
        )
        
        return {
            'ece': float(ece),
            'mean_confidence': float(np.mean(confidences)),
            'mean_accuracy': float(np.mean(accuracies)),
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def save(self, path: str):
        """Save all models and configurations"""
        os.makedirs(path, exist_ok=True)
        
        # Save neural network models
        torch.save(self.lstm_model.state_dict(), 
                  os.path.join(path, 'lstm_model.pth'))
        torch.save(self.gru_model.state_dict(), 
                  os.path.join(path, 'gru_model.pth'))
        torch.save(self.transformer_model.state_dict(), 
                  os.path.join(path, 'transformer_model.pth'))
        
        # Save Markov model (convert tuple keys to strings for JSON)
        transition_matrix_serializable = {}
        for state, transitions in self.markov_model.transition_matrix.items():
            state_str = '|'.join(state)  # Convert tuple to string
            transition_matrix_serializable[state_str] = transitions
        
        state_counts_serializable = {}
        for state, count in self.markov_model.state_counts.items():
            state_str = '|'.join(state)
            state_counts_serializable[state_str] = count
        
        markov_data = {
            'transition_matrix': transition_matrix_serializable,
            'state_counts': state_counts_serializable,
            'pattern_types': list(self.markov_model.pattern_types),
            'order': self.markov_model.order
        }
        with open(os.path.join(path, 'markov_model.json'), 'w') as f:
            json.dump(markov_data, f)
        
        # Save encoders and scalers
        joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(path, 'feature_scaler.pkl'))
        
        # Save configuration
        config = {
            'seq_length': self.seq_length,
            'ensemble_weights': self.ensemble_weights,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        """Load all models and configurations"""
        # Load configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.seq_length = config['seq_length']
        self.ensemble_weights = config['ensemble_weights']
        self.training_history = config['training_history']
        
        # Load encoders and scalers
        self.label_encoder = joblib.load(os.path.join(path, 'label_encoder.pkl'))
        self.feature_scaler = joblib.load(os.path.join(path, 'feature_scaler.pkl'))
        
        # Load Markov model (convert string keys back to tuples)
        with open(os.path.join(path, 'markov_model.json'), 'r') as f:
            markov_data = json.load(f)
        
        # Convert string keys back to tuples
        transition_matrix = {}
        for state_str, transitions in markov_data['transition_matrix'].items():
            state = tuple(state_str.split('|'))
            transition_matrix[state] = transitions
        
        state_counts = {}
        for state_str, count in markov_data['state_counts'].items():
            state = tuple(state_str.split('|'))
            state_counts[state] = count
        
        self.markov_model.transition_matrix = transition_matrix
        self.markov_model.state_counts = state_counts
        self.markov_model.pattern_types = set(markov_data['pattern_types'])
        self.markov_model.order = markov_data['order']
        
        # Get dimensions from saved config
        input_size = 8  # Number of features per pattern
        num_classes = len(self.label_encoder.classes_)
        
        # Initialize neural network models
        self.lstm_model = LSTMPredictor(input_size, 128, num_classes).to(self.device)
        self.gru_model = GRUPredictor(input_size, 128, num_classes).to(self.device)
        self.transformer_model = TransformerPredictor(input_size, num_classes).to(self.device)
        
        # Load model weights
        self.lstm_model.load_state_dict(
            torch.load(os.path.join(path, 'lstm_model.pth'), 
                      map_location=self.device)
        )
        self.gru_model.load_state_dict(
            torch.load(os.path.join(path, 'gru_model.pth'),
                      map_location=self.device)
        )
        self.transformer_model.load_state_dict(
            torch.load(os.path.join(path, 'transformer_model.pth'),
                      map_location=self.device)
        )
        
        # Set models to evaluation mode
        self.lstm_model.eval()
        self.gru_model.eval()
        self.transformer_model.eval()
        
        print(f"Models loaded from {path}")
