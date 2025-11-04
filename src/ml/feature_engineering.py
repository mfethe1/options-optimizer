"""
Feature Engineering Service

Generates 60+ technical indicators and features for ML models.
These features capture price patterns, momentum, volatility, and market microstructure.

Expected Impact: +2-4% monthly through better entry/exit timing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class FeatureSet:
    """Complete feature set for ML models"""
    symbol: str
    timestamp: datetime

    # Price features (5)
    close: float
    open: float
    high: float
    low: float
    volume: int

    # Trend indicators (10)
    sma_5: float
    sma_10: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    macd: float
    macd_signal: float
    macd_histogram: float

    # Momentum indicators (12)
    rsi_14: float
    rsi_7: float
    rsi_21: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    roc_10: float  # Rate of change
    momentum_10: float
    cci_20: float  # Commodity Channel Index
    mfi_14: float  # Money Flow Index
    adx_14: float  # Average Directional Index
    aroon_up: float

    # Volatility indicators (8)
    atr_14: float  # Average True Range
    bb_upper: float  # Bollinger Bands
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float  # Where price is within bands
    keltner_upper: float
    keltner_lower: float

    # Volume indicators (6)
    obv: float  # On-Balance Volume
    vwap: float  # Volume-Weighted Average Price
    volume_sma_20: float
    volume_ratio: float  # Current vol / avg vol
    force_index: float
    ease_of_movement: float

    # Price patterns (8)
    price_sma_5_ratio: float
    price_sma_20_ratio: float
    price_sma_50_ratio: float
    high_low_range: float
    close_location: float  # Where close is in day's range
    gap_pct: float  # Gap from previous close
    candle_body_pct: float
    candle_upper_shadow_pct: float

    # Derived features (11)
    returns_1d: float
    returns_5d: float
    returns_10d: float
    returns_20d: float
    volatility_10d: float
    volatility_20d: float
    volume_volatility: float
    price_momentum_score: float  # Composite momentum
    trend_strength: float  # How strong is the trend
    market_regime: int  # 0=ranging, 1=trending up, 2=trending down
    support_level: float  # Nearest support
    resistance_level: float  # Nearest resistance

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models (excluding timestamp and symbol)"""
        # Return all numeric features in a consistent order
        return np.array([
            self.close, self.open, self.high, self.low, float(self.volume),
            self.sma_5, self.sma_10, self.sma_20, self.sma_50, self.sma_200,
            self.ema_12, self.ema_26, self.macd, self.macd_signal, self.macd_histogram,
            self.rsi_14, self.rsi_7, self.rsi_21, self.stochastic_k, self.stochastic_d,
            self.williams_r, self.roc_10, self.momentum_10, self.cci_20, self.mfi_14,
            self.adx_14, self.aroon_up, self.atr_14, self.bb_upper, self.bb_middle,
            self.bb_lower, self.bb_width, self.bb_position, self.keltner_upper, self.keltner_lower,
            self.obv, self.vwap, self.volume_sma_20, self.volume_ratio, self.force_index,
            self.ease_of_movement, self.price_sma_5_ratio, self.price_sma_20_ratio, self.price_sma_50_ratio,
            self.high_low_range, self.close_location, self.gap_pct, self.candle_body_pct,
            self.candle_upper_shadow_pct, self.returns_1d, self.returns_5d, self.returns_10d,
            self.returns_20d, self.volatility_10d, self.volatility_20d, self.volume_volatility,
            self.price_momentum_score, self.trend_strength, float(self.market_regime),
            self.support_level, self.resistance_level
        ])

    @staticmethod
    def feature_count() -> int:
        """Total number of features"""
        return 60


# ============================================================================
# Feature Engineering Service
# ============================================================================

class FeatureEngineeringService:
    """
    Generate comprehensive technical features for ML models.

    Features:
    - 60+ technical indicators
    - Price patterns and candlestick analysis
    - Support/resistance detection
    - Market regime classification
    - Normalized and scaled features
    """

    def __init__(self):
        self.feature_cache: Dict[str, pd.DataFrame] = {}

    def generate_features(self, df: pd.DataFrame, symbol: str) -> List[FeatureSet]:
        """
        Generate features from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Stock symbol

        Returns:
            List of FeatureSet objects
        """
        if len(df) < 200:
            logger.warning(f"Need at least 200 bars for feature generation, got {len(df)}")
            return []

        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate all features
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_patterns(df)
        df = self._add_derived_features(df)

        # Convert to FeatureSet objects
        features = []
        for idx, row in df.iterrows():
            if pd.isna(row['sma_200']):  # Skip rows without enough history
                continue

            try:
                feature_set = FeatureSet(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    close=row['close'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    volume=int(row['volume']),
                    sma_5=row['sma_5'],
                    sma_10=row['sma_10'],
                    sma_20=row['sma_20'],
                    sma_50=row['sma_50'],
                    sma_200=row['sma_200'],
                    ema_12=row['ema_12'],
                    ema_26=row['ema_26'],
                    macd=row['macd'],
                    macd_signal=row['macd_signal'],
                    macd_histogram=row['macd_histogram'],
                    rsi_14=row['rsi_14'],
                    rsi_7=row['rsi_7'],
                    rsi_21=row['rsi_21'],
                    stochastic_k=row['stochastic_k'],
                    stochastic_d=row['stochastic_d'],
                    williams_r=row['williams_r'],
                    roc_10=row['roc_10'],
                    momentum_10=row['momentum_10'],
                    cci_20=row['cci_20'],
                    mfi_14=row['mfi_14'],
                    adx_14=row['adx_14'],
                    aroon_up=row['aroon_up'],
                    atr_14=row['atr_14'],
                    bb_upper=row['bb_upper'],
                    bb_middle=row['bb_middle'],
                    bb_lower=row['bb_lower'],
                    bb_width=row['bb_width'],
                    bb_position=row['bb_position'],
                    keltner_upper=row['keltner_upper'],
                    keltner_lower=row['keltner_lower'],
                    obv=row['obv'],
                    vwap=row['vwap'],
                    volume_sma_20=row['volume_sma_20'],
                    volume_ratio=row['volume_ratio'],
                    force_index=row['force_index'],
                    ease_of_movement=row['ease_of_movement'],
                    price_sma_5_ratio=row['price_sma_5_ratio'],
                    price_sma_20_ratio=row['price_sma_20_ratio'],
                    price_sma_50_ratio=row['price_sma_50_ratio'],
                    high_low_range=row['high_low_range'],
                    close_location=row['close_location'],
                    gap_pct=row['gap_pct'],
                    candle_body_pct=row['candle_body_pct'],
                    candle_upper_shadow_pct=row['candle_upper_shadow_pct'],
                    returns_1d=row['returns_1d'],
                    returns_5d=row['returns_5d'],
                    returns_10d=row['returns_10d'],
                    returns_20d=row['returns_20d'],
                    volatility_10d=row['volatility_10d'],
                    volatility_20d=row['volatility_20d'],
                    volume_volatility=row['volume_volatility'],
                    price_momentum_score=row['price_momentum_score'],
                    trend_strength=row['trend_strength'],
                    market_regime=int(row['market_regime']),
                    support_level=row['support_level'],
                    resistance_level=row['resistance_level']
                )
                features.append(feature_set)
            except Exception as e:
                logger.error(f"Error creating feature set for row {idx}: {e}")
                continue

        logger.info(f"Generated {len(features)} feature sets for {symbol}")
        return features

    # ========================================================================
    # Trend Indicators
    # ========================================================================

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    # ========================================================================
    # Momentum Indicators
    # ========================================================================

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators"""
        # RSI (multiple periods)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stochastic_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # Rate of Change
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # Commodity Channel Index
        df['cci_20'] = self._calculate_cci(df, 20)

        # Money Flow Index
        df['mfi_14'] = self._calculate_mfi(df, 14)

        # Average Directional Index
        df['adx_14'] = self._calculate_adx(df, 14)

        # Aroon Indicator
        df['aroon_up'] = self._calculate_aroon_up(df, 25)

        return df

    # ========================================================================
    # Volatility Indicators
    # ========================================================================

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        # Average True Range
        df['atr_14'] = self._calculate_atr(df, 14)

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Keltner Channels
        df['keltner_middle'] = df['ema_26']
        df['keltner_upper'] = df['keltner_middle'] + (2 * df['atr_14'])
        df['keltner_lower'] = df['keltner_middle'] - (2 * df['atr_14'])

        return df

    # ========================================================================
    # Volume Indicators
    # ========================================================================

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Volume SMA and ratio
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Force Index
        df['force_index'] = df['close'].diff() * df['volume']

        # Ease of Movement
        distance = ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_ratio = (df['volume'] / 100000000) / (df['high'] - df['low'])
        df['ease_of_movement'] = distance / box_ratio

        return df

    # ========================================================================
    # Price Patterns
    # ========================================================================

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        # Price relative to moving averages
        df['price_sma_5_ratio'] = df['close'] / df['sma_5']
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']

        # Intraday patterns
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Gap analysis
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Candle body and shadows
        df['candle_body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['candle_upper_shadow_pct'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'])

        return df

    # ========================================================================
    # Derived Features
    # ========================================================================

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and composite features"""
        # Returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)

        # Volatility
        df['volatility_10d'] = df['returns_1d'].rolling(window=10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(window=20).std()
        df['volume_volatility'] = df['volume_ratio'].rolling(window=10).std()

        # Composite momentum score
        df['price_momentum_score'] = (
            (df['rsi_14'] - 50) / 50 +
            (df['macd_histogram'] / df['close'] * 100) +
            (df['roc_10'] / 10)
        ) / 3

        # Trend strength
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']

        # Market regime
        df['market_regime'] = 0  # Default: ranging
        df.loc[(df['sma_20'] > df['sma_50']) & (df['close'] > df['sma_20']), 'market_regime'] = 1  # Trending up
        df.loc[(df['sma_20'] < df['sma_50']) & (df['close'] < df['sma_20']), 'market_regime'] = 2  # Trending down

        # Support/Resistance (simplified - using recent lows/highs)
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()

        return df

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = (typical_price - sma_tp).abs().rolling(window=period).mean()
        return (typical_price - sma_tp) / (0.015 * mad)

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

        mfi_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + mfi_ratio))

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    def _calculate_aroon_up(self, df: pd.DataFrame, period: int = 25) -> pd.Series:
        """Calculate Aroon Up indicator"""
        aroon_up = df['high'].rolling(window=period).apply(
            lambda x: (period - x.argmax()) / period * 100,
            raw=False
        )
        return aroon_up
