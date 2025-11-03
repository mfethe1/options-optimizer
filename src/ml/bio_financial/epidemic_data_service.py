"""
Epidemic Volatility Data Collection Service

Collects and processes data for epidemic volatility forecasting:
1. VIX (fear index)
2. Realized volatility
3. Market sentiment
4. Trading volume
5. Event indicators (Fed decisions, earnings, news)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EpidemicMarketData:
    """Market data for epidemic volatility modeling"""
    timestamp: datetime
    vix: float
    realized_vol_5d: float
    realized_vol_20d: float
    sentiment: float  # [-1, 1]
    volume_ratio: float  # Volume / average volume
    spx_return: float
    news_count: int
    is_fed_event: bool = False
    is_earnings_season: bool = False


class EpidemicDataService:
    """Service for collecting epidemic volatility data"""

    def __init__(self):
        self.vix_cache = {}
        self.spx_cache = {}

    async def get_current_market_features(self) -> Dict:
        """
        Get current market features for epidemic prediction

        Returns:
            {
                'vix': float,
                'realized_vol': float,
                'sentiment': float,
                'volume': float,
                'features_array': np.ndarray
            }
        """
        try:
            # Get VIX
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            current_vix = vix_data['Close'].iloc[-1] if len(vix_data) > 0 else 15.0

            # Get S&P 500 for realized vol
            spx_ticker = yf.Ticker("^GSPC")
            spx_data = spx_ticker.history(period="1mo")

            if len(spx_data) > 0:
                # Calculate realized volatility (20-day)
                returns = spx_data['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252) * 100

                # Volume ratio
                current_volume = spx_data['Volume'].iloc[-1]
                avg_volume = spx_data['Volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                realized_vol = 15.0
                volume_ratio = 1.0

            # Sentiment (placeholder - in production, use FinBERT)
            # For now, derive from VIX level
            sentiment = self._vix_to_sentiment(current_vix)

            # Create feature array
            features_array = np.array([
                current_vix / 100.0,
                realized_vol / 100.0,
                (sentiment + 1) / 2,  # Map [-1,1] to [0,1]
                min(volume_ratio, 3.0) / 3.0  # Normalize, cap at 3x
            ])

            return {
                'vix': current_vix,
                'realized_vol': realized_vol,
                'sentiment': sentiment,
                'volume': volume_ratio,
                'features_array': features_array
            }

        except Exception as e:
            logger.error(f"Error getting market features: {e}")
            # Return default values
            return {
                'vix': 15.0,
                'realized_vol': 15.0,
                'sentiment': 0.0,
                'volume': 1.0,
                'features_array': np.array([0.15, 0.15, 0.5, 0.33])
            }

    async def get_historical_data(self,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """
        Get historical data for training epidemic model

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: [date, vix, realized_vol, sentiment, volume, ...]
        """
        try:
            # Get VIX history
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(start=start_date, end=end_date)

            # Get S&P 500 history
            spx_ticker = yf.Ticker("^GSPC")
            spx_data = spx_ticker.history(start=start_date, end=end_date)

            if len(vix_data) == 0 or len(spx_data) == 0:
                logger.warning("No historical data available")
                return pd.DataFrame()

            # Merge on date
            df = pd.DataFrame(index=vix_data.index)
            df['vix'] = vix_data['Close']
            df['spx_close'] = spx_data['Close']
            df['volume'] = spx_data['Volume']

            # Calculate realized volatility (20-day rolling)
            df['spx_return'] = df['spx_close'].pct_change()
            df['realized_vol_20d'] = df['spx_return'].rolling(20).std() * np.sqrt(252) * 100

            # Calculate realized volatility (5-day rolling)
            df['realized_vol_5d'] = df['spx_return'].rolling(5).std() * np.sqrt(252) * 100

            # Volume ratio (vs 20-day average)
            df['volume_avg_20d'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg_20d']

            # Sentiment from VIX
            df['sentiment'] = df['vix'].apply(self._vix_to_sentiment)

            # Detect Fed events (simplified - check for VIX spikes)
            df['vix_change'] = df['vix'].pct_change()
            df['is_fed_event'] = (df['vix_change'].abs() > 0.15)  # 15%+ VIX change

            # Earnings season (rough approximation - Jan, Apr, Jul, Oct)
            df['month'] = df.index.month
            df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10])

            # Drop NaN rows
            df = df.dropna()

            logger.info(f"Collected {len(df)} days of historical epidemic data")
            return df

        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return pd.DataFrame()

    def _vix_to_sentiment(self, vix: float) -> float:
        """
        Convert VIX level to sentiment score [-1, 1]

        VIX 10-15: Positive sentiment (0.5 to 1.0)
        VIX 15-25: Neutral (0.0 to 0.5)
        VIX 25-40: Negative (-0.5 to 0.0)
        VIX >40: Very negative (-1.0 to -0.5)
        """
        if vix < 15:
            return 1.0 - (vix - 10) / 5 * 0.5  # Map 10-15 to 1.0-0.5
        elif vix < 25:
            return 0.5 - (vix - 15) / 10 * 0.5  # Map 15-25 to 0.5-0.0
        elif vix < 40:
            return 0.0 - (vix - 25) / 15 * 0.5  # Map 25-40 to 0.0--0.5
        else:
            return max(-1.0, -0.5 - (vix - 40) / 20 * 0.5)  # Map 40-60 to -0.5--1.0

    async def prepare_training_data(self,
                                   lookback_days: int = 30,
                                   forecast_days: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for epidemic model

        Args:
            lookback_days: Historical window
            forecast_days: Forecast horizon

        Returns:
            (X_train, y_train) where:
                X_train: [n_samples, 4] features
                y_train: [n_samples, forecast_days] VIX targets
        """
        # Get 2 years of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)

        df = await self.get_historical_data(start_date, end_date)

        if len(df) < lookback_days + forecast_days:
            logger.warning("Insufficient historical data for training")
            return np.array([]), np.array([])

        X_samples = []
        y_samples = []

        # Create sliding windows
        for i in range(len(df) - lookback_days - forecast_days):
            window = df.iloc[i:i + lookback_days]
            future = df.iloc[i + lookback_days:i + lookback_days + forecast_days]

            # Features: current state
            current_vix = window['vix'].iloc[-1]
            current_realized_vol = window['realized_vol_20d'].iloc[-1]
            current_sentiment = window['sentiment'].iloc[-1]
            current_volume = window['volume_ratio'].iloc[-1]

            features = np.array([
                current_vix / 100.0,
                current_realized_vol / 100.0,
                (current_sentiment + 1) / 2,
                min(current_volume, 3.0) / 3.0
            ])

            # Target: future VIX values
            target = future['vix'].values

            X_samples.append(features)
            y_samples.append(target)

        X_train = np.array(X_samples)
        y_train = np.array(y_samples)

        logger.info(f"Prepared {len(X_train)} training samples")
        return X_train, y_train

    async def detect_epidemic_events(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect "epidemic" events in historical data

        These are volatility spikes that resemble disease outbreaks:
        - Rapid infection (VIX spike)
        - Peak
        - Recovery

        Returns:
            List of epidemic episodes with metadata
        """
        episodes = []

        # Find VIX spikes (infection starts)
        df['vix_spike'] = (df['vix'] > df['vix'].rolling(20).mean() + 2 * df['vix'].rolling(20).std())

        in_episode = False
        episode_start = None
        episode_peak_vix = 0

        for idx, row in df.iterrows():
            if row['vix_spike'] and not in_episode:
                # Episode starts
                in_episode = True
                episode_start = idx
                episode_peak_vix = row['vix']

            elif in_episode:
                if row['vix'] > episode_peak_vix:
                    episode_peak_vix = row['vix']

                # Check for recovery (VIX drops below 20-day average)
                if row['vix'] < df['vix'].rolling(20).mean().loc[idx]:
                    # Episode ends
                    episode_end = idx
                    duration = (episode_end - episode_start).days

                    episodes.append({
                        'start_date': episode_start,
                        'end_date': episode_end,
                        'duration_days': duration,
                        'peak_vix': episode_peak_vix,
                        'start_vix': df.loc[episode_start, 'vix'],
                        'end_vix': row['vix']
                    })

                    in_episode = False

        logger.info(f"Detected {len(episodes)} epidemic episodes in historical data")
        return episodes
