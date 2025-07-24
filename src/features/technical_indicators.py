"""
Technical Indicators for Financial Time Series

This module implements various technical indicators commonly used in
financial analysis and trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes various technical indicators for financial time series.
    
    Indicators include:
    - Moving averages (SMA, EMA, WMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR, Standard Deviation)
    - Volume indicators (OBV, VWAP, MFI)
    - Trend indicators (ADX, Aroon, Ichimoku)
    """
    
    def __init__(self, 
                 price_col: str = 'close',
                 high_col: str = 'high',
                 low_col: str = 'low',
                 volume_col: str = 'volume'):
        """
        Initialize technical indicators calculator.
        
        Parameters:
        -----------
        price_col : str
            Column name for closing price
        high_col : str
            Column name for high price
        low_col : str
            Column name for low price
        volume_col : str
            Column name for volume
        """
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        
    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all available technical indicators.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        indicators : pd.DataFrame
            DataFrame with all computed indicators
        """
        indicators = pd.DataFrame(index=df.index)
        
        # Price data
        close = df[self.price_col]
        high = df.get(self.high_col, close)
        low = df.get(self.low_col, close)
        volume = df.get(self.volume_col, pd.Series(1, index=df.index))
        
        # Moving Averages
        ma_indicators = self.compute_moving_averages(close)
        indicators = pd.concat([indicators, ma_indicators], axis=1)
        
        # Momentum Indicators
        momentum_indicators = self.compute_momentum_indicators(close, high, low, volume)
        indicators = pd.concat([indicators, momentum_indicators], axis=1)
        
        # Volatility Indicators
        volatility_indicators = self.compute_volatility_indicators(close, high, low)
        indicators = pd.concat([indicators, volatility_indicators], axis=1)
        
        # Volume Indicators
        if self.volume_col in df.columns:
            volume_indicators = self.compute_volume_indicators(close, high, low, volume)
            indicators = pd.concat([indicators, volume_indicators], axis=1)
        
        # Trend Indicators
        trend_indicators = self.compute_trend_indicators(close, high, low)
        indicators = pd.concat([indicators, trend_indicators], axis=1)
        
        return indicators
        
    def compute_moving_averages(self, close: pd.Series) -> pd.DataFrame:
        """Compute various moving averages."""
        ma_df = pd.DataFrame(index=close.index)
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            ma_df[f'sma_{period}'] = close.rolling(window=period).mean()
            
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            ma_df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            
        # Weighted Moving Average
        for period in [10, 20]:
            weights = np.arange(1, period + 1)
            ma_df[f'wma_{period}'] = close.rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            
        # Hull Moving Average
        for period in [20]:
            wma_half = close.rolling(period // 2).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            wma_full = close.rolling(period).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            hull_raw = 2 * wma_half - wma_full
            sqrt_period = int(np.sqrt(period))
            ma_df[f'hma_{period}'] = hull_raw.rolling(sqrt_period).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            
        return ma_df
        
    def compute_momentum_indicators(self, close: pd.Series, high: pd.Series, 
                                  low: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """Compute momentum-based indicators."""
        momentum_df = pd.DataFrame(index=close.index)
        
        # RSI (Relative Strength Index)
        for period in [14, 21]:
            momentum_df[f'rsi_{period}'] = self._calculate_rsi(close, period)
            
        # MACD (Moving Average Convergence Divergence)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        momentum_df['macd'] = ema_12 - ema_26
        momentum_df['macd_signal'] = momentum_df['macd'].ewm(span=9, adjust=False).mean()
        momentum_df['macd_histogram'] = momentum_df['macd'] - momentum_df['macd_signal']
        
        # Stochastic Oscillator
        for period in [14]:
            low_min = low.rolling(window=period).min()
            high_max = high.rolling(window=period).max()
            momentum_df[f'stoch_k_{period}'] = 100 * ((close - low_min) / (high_max - low_min))
            momentum_df[f'stoch_d_{period}'] = momentum_df[f'stoch_k_{period}'].rolling(window=3).mean()
            
        # Williams %R
        for period in [14]:
            high_max = high.rolling(window=period).max()
            low_min = low.rolling(window=period).min()
            momentum_df[f'williams_r_{period}'] = -100 * ((high_max - close) / (high_max - low_min))
            
        # Rate of Change (ROC)
        for period in [10, 20]:
            momentum_df[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
            
        # Commodity Channel Index (CCI)
        for period in [20]:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            momentum_df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
        # Money Flow Index (MFI)
        if volume is not None and len(volume.unique()) > 1:
            for period in [14]:
                momentum_df[f'mfi_{period}'] = self._calculate_mfi(high, low, close, volume, period)
                
        return momentum_df
        
    def compute_volatility_indicators(self, close: pd.Series, high: pd.Series, 
                                    low: pd.Series) -> pd.DataFrame:
        """Compute volatility-based indicators."""
        volatility_df = pd.DataFrame(index=close.index)
        
        # Bollinger Bands
        for period in [20]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            volatility_df[f'bb_upper_{period}'] = sma + (2 * std)
            volatility_df[f'bb_middle_{period}'] = sma
            volatility_df[f'bb_lower_{period}'] = sma - (2 * std)
            volatility_df[f'bb_width_{period}'] = volatility_df[f'bb_upper_{period}'] - volatility_df[f'bb_lower_{period}']
            volatility_df[f'bb_percent_{period}'] = (close - volatility_df[f'bb_lower_{period}']) / volatility_df[f'bb_width_{period}']
            
        # Average True Range (ATR)
        for period in [14]:
            volatility_df[f'atr_{period}'] = self._calculate_atr(high, low, close, period)
            
        # Standard Deviation
        for period in [20, 50]:
            volatility_df[f'std_{period}'] = close.rolling(window=period).std()
            
        # Keltner Channels
        for period in [20]:
            ema = close.ewm(span=period, adjust=False).mean()
            atr = self._calculate_atr(high, low, close, period)
            volatility_df[f'kc_upper_{period}'] = ema + (2 * atr)
            volatility_df[f'kc_middle_{period}'] = ema
            volatility_df[f'kc_lower_{period}'] = ema - (2 * atr)
            
        # Donchian Channels
        for period in [20]:
            volatility_df[f'dc_upper_{period}'] = high.rolling(window=period).max()
            volatility_df[f'dc_lower_{period}'] = low.rolling(window=period).min()
            volatility_df[f'dc_middle_{period}'] = (volatility_df[f'dc_upper_{period}'] + 
                                                    volatility_df[f'dc_lower_{period}']) / 2
            
        return volatility_df
        
    def compute_volume_indicators(self, close: pd.Series, high: pd.Series,
                                low: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """Compute volume-based indicators."""
        volume_df = pd.DataFrame(index=close.index)
        
        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        volume_df['obv'] = obv
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (high + low + close) / 3
        volume_df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        volume_df['ad_line'] = (clv * volume).cumsum()
        
        # Chaikin Money Flow
        for period in [20]:
            money_flow_volume = clv * volume
            volume_df[f'cmf_{period}'] = (money_flow_volume.rolling(window=period).sum() / 
                                         volume.rolling(window=period).sum())
            
        # Volume Rate of Change
        for period in [10]:
            volume_df[f'vroc_{period}'] = ((volume - volume.shift(period)) / 
                                          volume.shift(period)) * 100
            
        # Price Volume Trend
        volume_df['pvt'] = ((close.pct_change() * volume).fillna(0).cumsum())
        
        return volume_df
        
    def compute_trend_indicators(self, close: pd.Series, high: pd.Series,
                               low: pd.Series) -> pd.DataFrame:
        """Compute trend-based indicators."""
        trend_df = pd.DataFrame(index=close.index)
        
        # Average Directional Index (ADX)
        for period in [14]:
            adx, plus_di, minus_di = self._calculate_adx(high, low, close, period)
            trend_df[f'adx_{period}'] = adx
            trend_df[f'plus_di_{period}'] = plus_di
            trend_df[f'minus_di_{period}'] = minus_di
            
        # Aroon Indicator
        for period in [25]:
            aroon_up, aroon_down = self._calculate_aroon(high, low, period)
            trend_df[f'aroon_up_{period}'] = aroon_up
            trend_df[f'aroon_down_{period}'] = aroon_down
            trend_df[f'aroon_osc_{period}'] = aroon_up - aroon_down
            
        # Parabolic SAR
        trend_df['psar'] = self._calculate_psar(high, low, close)
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(high, low, close)
        for key, value in ichimoku.items():
            trend_df[f'ichimoku_{key}'] = value
            
        # Supertrend
        for period in [10]:
            for multiplier in [3]:
                trend_df[f'supertrend_{period}_{multiplier}'] = self._calculate_supertrend(
                    high, low, close, period, multiplier
                )
                
        return trend_df
        
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
        
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
        
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index and DI+/DI-."""
        # Calculate directional movements
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self._calculate_atr(high, low, close, period)
        
        # Calculate DI+ and DI-
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
        
    def _calculate_aroon(self, high: pd.Series, low: pd.Series, 
                        period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Up and Aroon Down."""
        aroon_up = high.rolling(window=period + 1).apply(
            lambda x: 100 * (period - x.argmax()) / period
        )
        aroon_down = low.rolling(window=period + 1).apply(
            lambda x: 100 * (period - x.argmin()) / period
        )
        return aroon_up, aroon_down
        
    def _calculate_psar(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        psar = close.copy()
        bull = True
        af_current = af
        ep = high.iloc[0] if bull else low.iloc[0]
        
        for i in range(1, len(close)):
            psar.iloc[i] = psar.iloc[i-1] + af_current * (ep - psar.iloc[i-1])
            
            if bull:
                if low.iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af_current = af
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af_current = min(af_current + af, max_af)
            else:
                if high.iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af_current = af
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af_current = min(af_current + af, max_af)
                        
        return psar
        
    def _calculate_ichimoku(self, high: pd.Series, low: pd.Series, 
                           close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line)
        tenkan_period = 9
        tenkan = (high.rolling(window=tenkan_period).max() + 
                 low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_period = 26
        kijun = (high.rolling(window=kijun_period).max() + 
                low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_b_period = 52
        senkou_b = ((high.rolling(window=senkou_b_period).max() + 
                    low.rolling(window=senkou_b_period).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou = close.shift(-26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
        
    def _calculate_supertrend(self, high: pd.Series, low: pd.Series,
                             close: pd.Series, period: int, multiplier: float) -> pd.Series:
        """Calculate Supertrend indicator."""
        # Calculate basic bands
        hl_avg = (high + low) / 2
        atr = self._calculate_atr(high, low, close, period)
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=close.index, dtype=float)
        in_uptrend = True
        
        for i in range(period, len(close)):
            if close.iloc[i] <= upper_band.iloc[i]:
                if in_uptrend:
                    if close.iloc[i] <= lower_band.iloc[i]:
                        in_uptrend = False
                        supertrend.iloc[i] = upper_band.iloc[i]
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
            else:
                if not in_uptrend:
                    in_uptrend = True
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    
        return supertrend
        
    def get_indicator_names(self) -> List[str]:
        """Get list of all indicator names that will be computed."""
        names = []
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            names.append(f'sma_{period}')
        for period in [12, 26, 50]:
            names.append(f'ema_{period}')
        for period in [10, 20]:
            names.append(f'wma_{period}')
        for period in [20]:
            names.append(f'hma_{period}')
            
        # Momentum indicators
        for period in [14, 21]:
            names.append(f'rsi_{period}')
        names.extend(['macd', 'macd_signal', 'macd_histogram'])
        for period in [14]:
            names.extend([f'stoch_k_{period}', f'stoch_d_{period}'])
            names.append(f'williams_r_{period}')
            names.append(f'mfi_{period}')
        for period in [10, 20]:
            names.append(f'roc_{period}')
        for period in [20]:
            names.append(f'cci_{period}')
            
        # Volatility indicators
        for period in [20]:
            names.extend([f'bb_upper_{period}', f'bb_middle_{period}', 
                         f'bb_lower_{period}', f'bb_width_{period}', f'bb_percent_{period}'])
            names.extend([f'kc_upper_{period}', f'kc_middle_{period}', f'kc_lower_{period}'])
            names.extend([f'dc_upper_{period}', f'dc_middle_{period}', f'dc_lower_{period}'])
        for period in [14]:
            names.append(f'atr_{period}')
        for period in [20, 50]:
            names.append(f'std_{period}')
            
        # Volume indicators
        names.extend(['obv', 'vwap', 'ad_line', 'pvt'])
        for period in [20]:
            names.append(f'cmf_{period}')
        for period in [10]:
            names.append(f'vroc_{period}')
            
        # Trend indicators
        for period in [14]:
            names.extend([f'adx_{period}', f'plus_di_{period}', f'minus_di_{period}'])
        for period in [25]:
            names.extend([f'aroon_up_{period}', f'aroon_down_{period}', f'aroon_osc_{period}'])
        names.append('psar')
        names.extend(['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a',
                     'ichimoku_senkou_b', 'ichimoku_chikou'])
        names.append('supertrend_10_3')
        
        return names
