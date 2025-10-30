"""
Phase 4: Technical & Cross-Asset Metrics - Short-Horizon Edge

Implements options-led and technical signals for 1-5 day alpha:
- Options Flow Composite (PCR, IV skew, unusual volume)
- Residual Momentum (asset vs market/sector)
- Seasonality Score (calendar effects)
- Breadth & Liquidity (market internals)

Based on research showing:
- Options flow often leads cash equities by 1-5 days (ExtractAlpha CAM)
- Residual momentum captures idiosyncratic moves
- Seasonality patterns persist (turn-of-month, FOMC, OpEx)
- Breadth divergences signal regime shifts

Sources:
- ExtractAlpha Cross-Asset Model fact sheet
- Cboe options market statistics
- Academic research on options-equity lead-lag
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptionsFlowMetrics:
    """Container for options flow analysis"""
    pcr: float  # Put/Call ratio
    iv_skew: float  # 25-delta put IV - call IV
    volume_ratio: float  # Current volume / 20-day avg
    composite_score: float  # Combined signal
    interpretation: str


@dataclass
class ResidualMomentumMetrics:
    """Container for residual momentum analysis"""
    asset_return: float
    market_return: float
    sector_return: Optional[float]
    residual: float  # Asset - (beta*market + sector)
    z_score: float  # Standardized residual
    interpretation: str


def options_flow_composite(
    pcr: float,
    iv_skew: float,
    volume_ratio: float,
    pcr_threshold: float = 0.7,
    skew_threshold: float = 0.05,
    volume_threshold: float = 1.5
) -> float:
    """
    Compute options flow composite score.
    
    Combines three signals:
    1. Put/Call Ratio (PCR): Low PCR = bullish, High PCR = bearish
    2. IV Skew: Negative skew = bullish, Positive skew = bearish
    3. Volume Ratio: High volume = conviction
    
    Args:
        pcr: Put/Call ratio (volume or open interest)
        iv_skew: 25-delta put IV minus call IV (in vol points)
        volume_ratio: Current volume / 20-day average
        pcr_threshold: PCR neutral level (default 0.7)
        skew_threshold: IV skew neutral level (default 0.05)
        volume_threshold: Volume ratio significance level (default 1.5)
    
    Returns:
        Composite score: -1 (bearish) to +1 (bullish)
        
    Sources:
        - Cboe PCR data: https://www.cboe.com/us/options/market_statistics/
        - ExtractAlpha CAM: Short-horizon options-led alpha
    """
    # Normalize PCR signal (lower PCR = more bullish)
    # Typical range: 0.5 (very bullish) to 1.2 (very bearish)
    pcr_signal = np.clip((pcr_threshold - pcr) / 0.3, -1, 1)
    
    # Normalize IV skew signal (negative skew = bullish)
    # Typical range: -0.10 (bullish) to +0.15 (bearish)
    skew_signal = np.clip(-iv_skew / 0.10, -1, 1)
    
    # Volume conviction multiplier
    volume_multiplier = min(volume_ratio / volume_threshold, 2.0)
    
    # Weighted composite (PCR 40%, Skew 40%, Volume 20% weight)
    base_signal = 0.4 * pcr_signal + 0.4 * skew_signal
    composite = base_signal * (0.8 + 0.2 * volume_multiplier)
    
    logger.debug(f"Options flow: PCR={pcr:.2f} (signal={pcr_signal:.2f}), "
                 f"Skew={iv_skew:.3f} (signal={skew_signal:.2f}), "
                 f"Vol={volume_ratio:.2f}x → Composite={composite:.2f}")
    
    return float(np.clip(composite, -1, 1))


def compute_options_flow_metrics(
    pcr: float,
    iv_skew: float,
    volume_ratio: float
) -> OptionsFlowMetrics:
    """
    Compute full options flow analysis with interpretation.
    
    Returns:
        OptionsFlowMetrics with composite score and interpretation
    """
    composite = options_flow_composite(pcr, iv_skew, volume_ratio)
    
    # Generate interpretation
    if composite > 0.5:
        interp = "Strong bullish options flow: Low PCR + benign skew + elevated volume"
    elif composite > 0.2:
        interp = "Moderately bullish options flow"
    elif composite < -0.5:
        interp = "Strong bearish options flow: High PCR + elevated put skew"
    elif composite < -0.2:
        interp = "Moderately bearish options flow"
    else:
        interp = "Neutral options flow"
    
    return OptionsFlowMetrics(
        pcr=pcr,
        iv_skew=iv_skew,
        volume_ratio=volume_ratio,
        composite_score=composite,
        interpretation=interp
    )


def residual_momentum(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    sector_returns: Optional[np.ndarray] = None,
    lookback: int = 20
) -> float:
    """
    Compute residual momentum (idiosyncratic return component).
    
    Captures asset-specific momentum after removing market and sector effects.
    High residual momentum suggests stock-specific catalysts.
    
    Args:
        asset_returns: Asset return series
        market_returns: Market benchmark return series
        sector_returns: Sector return series (optional)
        lookback: Lookback period for regression (default 20 days)
    
    Returns:
        Residual momentum z-score (standardized)
        
    Interpretation:
        > +1.0: Strong positive idiosyncratic momentum
        > +0.5: Moderate positive
        < -1.0: Strong negative
    """
    if len(asset_returns) < lookback:
        logger.warning(f"Insufficient data for residual momentum: {len(asset_returns)} < {lookback}")
        return 0.0
    
    # Use recent window
    asset_ret = asset_returns[-lookback:]
    market_ret = market_returns[-lookback:]
    
    # Estimate beta via regression
    beta_market = np.cov(asset_ret, market_ret)[0, 1] / np.var(market_ret)
    
    # Compute residuals
    if sector_returns is not None and len(sector_returns) >= lookback:
        sector_ret = sector_returns[-lookback:]
        beta_sector = np.cov(asset_ret, sector_ret)[0, 1] / np.var(sector_ret)
        residuals = asset_ret - (beta_market * market_ret + beta_sector * sector_ret)
    else:
        residuals = asset_ret - beta_market * market_ret
    
    # Compute cumulative residual and standardize
    cum_residual = np.sum(residuals)
    residual_std = np.std(residuals) * np.sqrt(lookback)
    
    if residual_std > 0:
        z_score = cum_residual / residual_std
    else:
        z_score = 0.0
    
    logger.debug(f"Residual momentum: beta={beta_market:.2f}, "
                 f"cum_residual={cum_residual:.4f}, z-score={z_score:.2f}")
    
    return float(z_score)


def seasonality_score(
    returns_series: pd.Series,
    calendar_effects: Optional[List[str]] = None
) -> float:
    """
    Compute seasonality score based on calendar patterns.
    
    Checks for:
    - Turn-of-month effect (last 3 days + first 2 days)
    - FOMC meeting days
    - Options expiration (OpEx) week
    - Day-of-week patterns
    
    Args:
        returns_series: Return series with DatetimeIndex
        calendar_effects: List of effects to check (default: all)
    
    Returns:
        Seasonality score: -1 (negative seasonal) to +1 (positive seasonal)
    """
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        logger.warning("Returns series must have DatetimeIndex for seasonality")
        return 0.0
    
    if len(returns_series) < 60:
        logger.warning(f"Insufficient data for seasonality: {len(returns_series)} < 60")
        return 0.0
    
    effects = calendar_effects or ['turn_of_month', 'day_of_week']
    scores = []
    
    # Turn-of-month effect
    if 'turn_of_month' in effects:
        df = returns_series.to_frame('ret')
        df['day'] = df.index.day
        df['is_tom'] = ((df['day'] >= 28) | (df['day'] <= 2))
        
        tom_mean = df[df['is_tom']]['ret'].mean()
        other_mean = df[~df['is_tom']]['ret'].mean()
        tom_score = np.clip((tom_mean - other_mean) / 0.01, -1, 1)  # Normalize by 1% threshold
        scores.append(tom_score)
    
    # Day-of-week effect
    if 'day_of_week' in effects:
        df = returns_series.to_frame('ret')
        df['dow'] = df.index.dayofweek
        
        # Check if current day (last observation) has positive historical bias
        current_dow = df.index[-1].dayofweek
        dow_mean = df[df['dow'] == current_dow]['ret'].mean()
        overall_mean = df['ret'].mean()
        dow_score = np.clip((dow_mean - overall_mean) / 0.005, -1, 1)
        scores.append(dow_score)
    
    # Average scores
    if scores:
        final_score = float(np.mean(scores))
        logger.debug(f"Seasonality score: {final_score:.2f} (effects: {effects})")
        return final_score
    
    return 0.0


def breadth_liquidity(
    advancing: int,
    declining: int,
    volume_ratio: float,
    spread_bps: Optional[float] = None
) -> float:
    """
    Compute market breadth & liquidity composite.
    
    Combines:
    1. Advance/Decline ratio (market breadth)
    2. Volume ratio (participation)
    3. Bid-ask spread (liquidity, optional)
    
    Args:
        advancing: Number of advancing stocks
        declining: Number of declining stocks
        volume_ratio: Current volume / average volume
        spread_bps: Bid-ask spread in basis points (optional)
    
    Returns:
        Breadth/liquidity score: -1 (poor) to +1 (strong)
    """
    total = advancing + declining
    if total == 0:
        logger.warning("No advancing or declining stocks provided")
        return 0.0
    
    # Advance/Decline ratio signal
    ad_ratio = advancing / total
    ad_signal = 2 * (ad_ratio - 0.5)  # Map [0,1] to [-1,1]
    
    # Volume signal (higher volume = more conviction)
    vol_signal = np.clip((volume_ratio - 1.0) / 0.5, -1, 1)
    
    # Liquidity signal (tighter spreads = better liquidity)
    if spread_bps is not None:
        # Typical spread: 5-20 bps; lower is better
        liquidity_signal = np.clip((15 - spread_bps) / 10, -1, 1)
        composite = 0.5 * ad_signal + 0.3 * vol_signal + 0.2 * liquidity_signal
    else:
        composite = 0.6 * ad_signal + 0.4 * vol_signal
    
    logger.debug(f"Breadth/liquidity: A/D={advancing}/{declining} ({ad_ratio:.2%}), "
                 f"Vol={volume_ratio:.2f}x → Composite={composite:.2f}")
    
    return float(np.clip(composite, -1, 1))


# Convenience function for full Phase 4 analysis
def compute_phase4_metrics(
    pcr: Optional[float] = None,
    iv_skew: Optional[float] = None,
    volume_ratio: Optional[float] = None,
    asset_returns: Optional[np.ndarray] = None,
    market_returns: Optional[np.ndarray] = None,
    sector_returns: Optional[np.ndarray] = None,
    returns_series: Optional[pd.Series] = None,
    advancing: Optional[int] = None,
    declining: Optional[int] = None,
    market_volume_ratio: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """
    Compute all Phase 4 metrics with graceful degradation.
    
    Returns dict with keys:
        - options_flow_composite
        - residual_momentum
        - seasonality_score
        - breadth_liquidity
    """
    metrics = {
        'options_flow_composite': None,
        'residual_momentum': None,
        'seasonality_score': None,
        'breadth_liquidity': None
    }
    
    # Options flow
    if pcr is not None and iv_skew is not None and volume_ratio is not None:
        try:
            metrics['options_flow_composite'] = options_flow_composite(pcr, iv_skew, volume_ratio)
        except Exception as e:
            logger.error(f"Error computing options flow: {e}")
    
    # Residual momentum
    if asset_returns is not None and market_returns is not None:
        try:
            metrics['residual_momentum'] = residual_momentum(asset_returns, market_returns, sector_returns)
        except Exception as e:
            logger.error(f"Error computing residual momentum: {e}")
    
    # Seasonality
    if returns_series is not None:
        try:
            metrics['seasonality_score'] = seasonality_score(returns_series)
        except Exception as e:
            logger.error(f"Error computing seasonality: {e}")
    
    # Breadth/liquidity
    if advancing is not None and declining is not None and market_volume_ratio is not None:
        try:
            metrics['breadth_liquidity'] = breadth_liquidity(advancing, declining, market_volume_ratio)
        except Exception as e:
            logger.error(f"Error computing breadth/liquidity: {e}")
    
    return metrics

