"""
Market regime detection using Hidden Markov Models and clustering.
Identifies different market states (trending, ranging, volatile, etc.).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Represents a market regime with its characteristics."""
    name: str
    volatility_level: str  # 'low', 'medium', 'high'
    trend_direction: str   # 'bullish', 'bearish', 'neutral'
    market_state: str      # 'trending', 'ranging', 'breakout', 'reversal'
    confidence: float
    features: Dict[str, float]


class MarketRegimeDetector:
    """
    Detects and classifies market regimes using multiple techniques
    including HMM, clustering, and rule-based approaches.
    """
    
    def __init__(self, n_regimes: int = 4, lookback_period: int = 100):
        """
        Initialize market regime detector.
        
        Args:
            n_regimes: Number of market regimes to identify
            lookback_period: Historical period for regime detection
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        
        # Initialize models
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100
        )
        
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        
        self.regime_labels = {
            0: "Low Volatility Trending",
            1: "High Volatility Trending", 
            2: "Range-Bound",
            3: "Volatile Reversal"
        }
        
        self.current_regime = None
        self.regime_history = []
        
    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Feature matrix for regime detection
        """
        features = []
        
        # Price-based features
        returns = data['close'].pct_change().fillna(0)
        features.append(returns.rolling(20).mean())  # Trend
        features.append(returns.rolling(20).std())   # Volatility
        
        # Volume features
        if 'volume' in data.columns:
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            features.append(volume_ratio)
        
        # Technical indicators for regime detection
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
        
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        features.append(atr / data['close'])  # Normalized ATR
        
        # Market structure
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        features.append((sma_20 - sma_50) / sma_50)  # Trend strength
        
        # Combine features
        feature_matrix = pd.concat(features, axis=1).fillna(0)
        return feature_matrix.values
    
    def train_hmm(self, data: pd.DataFrame):
        """
        Train Hidden Markov Model for regime detection.
        
        Args:
            data: Historical OHLCV data
        """
        logger.info("Training HMM for regime detection...")
        
        # Extract features
        features = self.extract_regime_features(data)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train HMM
        self.hmm_model.fit(features_scaled)
        
        # Get regime predictions
        regimes = self.hmm_model.predict(features_scaled)
        
        # Analyze regime characteristics
        self._analyze_regime_characteristics(data, regimes)
        
    def _analyze_regime_characteristics(self, data: pd.DataFrame, regimes: np.ndarray):
        """
        Analyze characteristics of each detected regime.
        
        Args:
            data: Original data
            regimes: Regime labels
        """
        returns = data['close'].pct_change().fillna(0)
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            # Calculate regime statistics
            avg_return = regime_returns.mean()
            volatility = regime_returns.std()
            
            # Determine regime characteristics
            if avg_return > 0.001:
                trend = "bullish"
            elif avg_return < -0.001:
                trend = "bearish"
            else:
                trend = "neutral"
                
            if volatility < returns.std() * 0.5:
                vol_level = "low"
            elif volatility > returns.std() * 1.5:
                vol_level = "high"
            else:
                vol_level = "medium"
                
            logger.info(f"Regime {regime}: {trend} trend, {vol_level} volatility")
    
    def detect_current_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            data: Recent market data
            
        Returns:
            Current market regime
        """
        # Extract features
        features = self.extract_regime_features(data)
        features_scaled = self.scaler.transform(features[-1:])
        
        # Get HMM prediction
        hmm_regime = self.hmm_model.predict(features_scaled)[0]
        
        # Get probability distribution
        probs = self.hmm_model.predict_proba(features_scaled)[0]
        confidence = probs[hmm_regime]
        
        # Calculate additional regime characteristics
        recent_data = data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().fillna(0)
        
        # Volatility analysis
        current_vol = returns.tail(20).std()
        avg_vol = returns.std()
        
        if current_vol < avg_vol * 0.7:
            vol_level = "low"
        elif current_vol > avg_vol * 1.3:
            vol_level = "high"
        else:
            vol_level = "medium"
            
        # Trend analysis
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = recent_data['close'].rolling(50).mean().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            trend_direction = "bullish"
        elif current_price < sma_20 < sma_50:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
            
        # Market state determination
        price_range = recent_data['high'].max() - recent_data['low'].min()
        avg_range = (recent_data['high'] - recent_data['low']).mean()
        
        if price_range < avg_range * 2:
            market_state = "ranging"
        elif abs(returns.tail(5).mean()) > returns.std() * 2:
            market_state = "breakout"
        elif (returns.tail(10).mean() * returns.tail(5).mean()) < 0:
            market_state = "reversal"
        else:
            market_state = "trending"
            
        # Create regime object
        regime = MarketRegime(
            name=self.regime_labels.get(hmm_regime, f"Regime {hmm_regime}"),
            volatility_level=vol_level,
            trend_direction=trend_direction,
            market_state=market_state,
            confidence=confidence,
            features={
                'current_volatility': float(current_vol),
                'volatility_ratio': float(current_vol / avg_vol),
                'trend_strength': float((sma_20 - sma_50) / sma_50),
                'price_position': float((current_price - sma_50) / sma_50),
                'regime_id': int(hmm_regime)
            }
        )
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
    
    def get_regime_transition_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.
        
        Args:
            data: Historical data
            
        Returns:
            Transition probability matrix
        """
        features = self.extract_regime_features(data)
        features_scaled = self.scaler.transform(features)
        regimes = self.hmm_model.predict(features_scaled)
        
        # Calculate transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_matrix[current_regime, next_regime] += 1
            
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Convert to DataFrame
        regime_names = [self.regime_labels.get(i, f"Regime {i}") for i in range(self.n_regimes)]
        transition_df = pd.DataFrame(
            transition_matrix,
            index=regime_names,
            columns=regime_names
        )
        
        return transition_df
    
    def predict_regime_change(self, data: pd.DataFrame, horizon: int = 5) -> Dict[str, float]:
        """
        Predict probability of regime change.
        
        Args:
            data: Recent market data
            horizon: Prediction horizon in periods
            
        Returns:
            Probabilities of transitioning to each regime
        """
        current_regime = self.detect_current_regime(data)
        current_id = current_regime.features['regime_id']
        
        # Get transition matrix
        trans_matrix = self.get_regime_transition_matrix(data)
        
        # Calculate multi-step transition probabilities
        trans_array = trans_matrix.values
        multi_step_trans = np.linalg.matrix_power(trans_array, horizon)
        
        # Get probabilities for current regime
        regime_probs = multi_step_trans[current_id]
        
        # Create probability dictionary
        prob_dict = {}
        for i, prob in enumerate(regime_probs):
            regime_name = self.regime_labels.get(i, f"Regime {i}")
            prob_dict[regime_name] = float(prob)
            
        return prob_dict
    
    def get_regime_specific_strategy(self, regime: MarketRegime) -> Dict[str, any]:
        """
        Get trading strategy recommendations based on current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Strategy recommendations
        """
        strategy = {
            'position_size': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'indicators': [],
            'timeframe': 'medium'
        }
        
        # Adjust strategy based on regime
        if regime.volatility_level == "high":
            strategy['position_size'] = 0.5
            strategy['stop_loss'] = 0.03
            strategy['indicators'].append('ATR')
            
        if regime.market_state == "trending":
            strategy['indicators'].extend(['MA_crossover', 'ADX'])
            strategy['timeframe'] = 'long'
            
        elif regime.market_state == "ranging":
            strategy['indicators'].extend(['RSI', 'Bollinger_Bands'])
            strategy['take_profit'] = 0.02
            
        elif regime.market_state == "breakout":
            strategy['position_size'] = 1.5
            strategy['indicators'].extend(['Volume', 'Momentum'])
            
        elif regime.market_state == "reversal":
            strategy['position_size'] = 0.3
            strategy['stop_loss'] = 0.015
            strategy['indicators'].extend(['Divergence', 'Support_Resistance'])
            
        return strategy
    
    def analyze_regime_performance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze historical performance in each regime.
        
        Args:
            data: Historical data with returns
            
        Returns:
            Performance statistics by regime
        """
        features = self.extract_regime_features(data)
        features_scaled = self.scaler.transform(features)
        regimes = self.hmm_model.predict(features_scaled)
        
        returns = data['close'].pct_change().fillna(0)
        
        # Calculate performance metrics for each regime
        performance_stats = []
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            stats = {
                'regime': self.regime_labels.get(regime, f"Regime {regime}"),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'max_drawdown': (regime_returns + 1).cumprod().expanding().max().diff().min(),
                'win_rate': (regime_returns > 0).mean(),
                'avg_win': regime_returns[regime_returns > 0].mean() if (regime_returns > 0).any() else 0,
                'avg_loss': regime_returns[regime_returns < 0].mean() if (regime_returns < 0).any() else 0,
                'duration_pct': mask.mean()
            }
            
            performance_stats.append(stats)
            
        return pd.DataFrame(performance_stats)
