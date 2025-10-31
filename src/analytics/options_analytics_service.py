"""
Options Analytics Service

Advanced options analytics: IV surface, skew, term structure analysis.
Bloomberg OMON equivalent - institutional-grade options metrics.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime, timedelta
import numpy as np
from scipy import interpolate
from scipy.stats import norm
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IVSurfacePoint:
    """Single point on the IV surface"""
    strike: float
    expiration: date
    days_to_expiry: int
    moneyness: float  # Strike / Spot
    delta: float
    implied_volatility: float
    option_type: str  # 'call' or 'put'


@dataclass
class IVSurface:
    """Complete IV surface"""
    symbol: str
    spot_price: float
    surface_points: List[IVSurfacePoint]
    calculation_time: datetime

    # Surface statistics
    min_iv: float
    max_iv: float
    atm_iv: float
    iv_range: float


@dataclass
class SkewMetrics:
    """Volatility skew metrics"""
    symbol: str
    expiration: date
    days_to_expiry: int
    spot_price: float

    # ATM metrics
    atm_strike: float
    atm_iv: float

    # Skew measurements
    put_skew: float  # IV(90% OTM put) - IV(ATM)
    call_skew: float  # IV(110% OTM call) - IV(ATM)
    skew_slope: float  # Linear regression slope
    risk_reversal_25delta: float  # 25-delta call IV - 25-delta put IV
    butterfly_25delta: float  # (25d call IV + 25d put IV) / 2 - ATM IV

    # By strike
    skew_by_strike: Dict[float, float]


@dataclass
class TermStructure:
    """Volatility term structure"""
    symbol: str
    spot_price: float
    structure_type: str  # 'atm', 'otm_put', 'otm_call'

    # Term structure points
    expirations: List[date]
    days_to_expiry: List[int]
    implied_volatilities: List[float]

    # Term structure shape
    front_month_iv: float
    back_month_iv: float
    term_structure_slope: float  # Contango or backwardation
    is_inverted: bool  # True if backwardation


class OptionsAnalyticsService:
    """Service for advanced options analytics"""

    def __init__(self):
        self.risk_free_rate = 0.045  # 4.5% risk-free rate

    async def calculate_iv_surface(
        self,
        symbol: str,
        options_data: List[Dict[str, Any]],
        spot_price: float
    ) -> IVSurface:
        """
        Calculate the complete IV surface.

        Args:
            symbol: Stock symbol
            options_data: List of option contracts with strike, expiry, IV, delta
            spot_price: Current spot price

        Returns:
            IVSurface with all calculated points
        """
        surface_points = []

        for option in options_data:
            if not option.get('implied_volatility'):
                continue

            strike = option['strike']
            expiration = option['expiration']

            # Calculate days to expiry
            if isinstance(expiration, str):
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            else:
                exp_date = expiration

            days_to_expiry = (exp_date - date.today()).days

            if days_to_expiry <= 0:
                continue

            # Calculate moneyness
            moneyness = strike / spot_price

            point = IVSurfacePoint(
                strike=strike,
                expiration=exp_date,
                days_to_expiry=days_to_expiry,
                moneyness=moneyness,
                delta=option.get('delta', 0),
                implied_volatility=option['implied_volatility'],
                option_type=option.get('option_type', 'call')
            )

            surface_points.append(point)

        # Calculate surface statistics
        if surface_points:
            ivs = [p.implied_volatility for p in surface_points]
            min_iv = min(ivs)
            max_iv = max(ivs)
            iv_range = max_iv - min_iv

            # Find ATM IV (closest to moneyness of 1.0)
            atm_point = min(surface_points, key=lambda p: abs(p.moneyness - 1.0))
            atm_iv = atm_point.implied_volatility
        else:
            min_iv = max_iv = atm_iv = iv_range = 0.0

        return IVSurface(
            symbol=symbol,
            spot_price=spot_price,
            surface_points=surface_points,
            calculation_time=datetime.now(),
            min_iv=min_iv,
            max_iv=max_iv,
            atm_iv=atm_iv,
            iv_range=iv_range
        )

    async def calculate_skew(
        self,
        symbol: str,
        expiration: date,
        options_data: List[Dict[str, Any]],
        spot_price: float
    ) -> SkewMetrics:
        """
        Calculate volatility skew for a specific expiration.

        Args:
            symbol: Stock symbol
            expiration: Option expiration date
            options_data: Options for this expiration
            spot_price: Current spot price

        Returns:
            SkewMetrics with skew calculations
        """
        days_to_expiry = (expiration - date.today()).days

        # Filter options for this expiration
        exp_options = [
            opt for opt in options_data
            if opt.get('expiration') == expiration or
            (isinstance(opt.get('expiration'), str) and
             datetime.strptime(opt['expiration'], '%Y-%m-%d').date() == expiration)
        ]

        if not exp_options:
            raise ValueError(f"No options found for expiration {expiration}")

        # Find ATM strike (closest to spot)
        atm_option = min(exp_options, key=lambda x: abs(x['strike'] - spot_price))
        atm_strike = atm_option['strike']
        atm_iv = atm_option.get('implied_volatility', 0)

        # Build skew by strike
        skew_by_strike = {}
        strikes = []
        ivs = []

        for opt in exp_options:
            if opt.get('implied_volatility'):
                strike = opt['strike']
                iv = opt['implied_volatility']
                skew_by_strike[strike] = iv
                strikes.append(strike)
                ivs.append(iv)

        # Calculate put skew (90% OTM put)
        put_strike = spot_price * 0.9
        put_iv = self._interpolate_iv(strikes, ivs, put_strike)
        put_skew = put_iv - atm_iv if put_iv else 0

        # Calculate call skew (110% OTM call)
        call_strike = spot_price * 1.1
        call_iv = self._interpolate_iv(strikes, ivs, call_strike)
        call_skew = call_iv - atm_iv if call_iv else 0

        # Calculate skew slope (linear regression)
        if len(strikes) >= 2:
            moneyness = [s / spot_price for s in strikes]
            slope, _ = np.polyfit(moneyness, ivs, 1)
            skew_slope = slope
        else:
            skew_slope = 0.0

        # Calculate risk reversal and butterfly (25-delta)
        # Risk reversal: 25d call IV - 25d put IV
        # Butterfly: (25d call IV + 25d put IV) / 2 - ATM IV

        # Find 25-delta options (approximate by moneyness)
        call_25d_strike = spot_price * 1.15  # Approximate
        put_25d_strike = spot_price * 0.85   # Approximate

        call_25d_iv = self._interpolate_iv(strikes, ivs, call_25d_strike)
        put_25d_iv = self._interpolate_iv(strikes, ivs, put_25d_strike)

        if call_25d_iv and put_25d_iv:
            risk_reversal_25delta = call_25d_iv - put_25d_iv
            butterfly_25delta = (call_25d_iv + put_25d_iv) / 2 - atm_iv
        else:
            risk_reversal_25delta = 0.0
            butterfly_25delta = 0.0

        return SkewMetrics(
            symbol=symbol,
            expiration=expiration,
            days_to_expiry=days_to_expiry,
            spot_price=spot_price,
            atm_strike=atm_strike,
            atm_iv=atm_iv,
            put_skew=put_skew,
            call_skew=call_skew,
            skew_slope=skew_slope,
            risk_reversal_25delta=risk_reversal_25delta,
            butterfly_25delta=butterfly_25delta,
            skew_by_strike=skew_by_strike
        )

    def _interpolate_iv(
        self,
        strikes: List[float],
        ivs: List[float],
        target_strike: float
    ) -> Optional[float]:
        """Interpolate IV at a target strike"""
        if len(strikes) < 2:
            return None

        try:
            # Sort by strike
            sorted_data = sorted(zip(strikes, ivs))
            strikes_sorted = [s for s, _ in sorted_data]
            ivs_sorted = [iv for _, iv in sorted_data]

            # Linear interpolation
            f = interpolate.interp1d(
                strikes_sorted,
                ivs_sorted,
                kind='linear',
                fill_value='extrapolate'
            )

            return float(f(target_strike))
        except Exception as e:
            logger.warning(f"IV interpolation failed: {e}")
            return None

    async def calculate_term_structure(
        self,
        symbol: str,
        options_data: List[Dict[str, Any]],
        spot_price: float,
        structure_type: str = 'atm'
    ) -> TermStructure:
        """
        Calculate volatility term structure.

        Args:
            symbol: Stock symbol
            options_data: All options data
            spot_price: Current spot price
            structure_type: 'atm', 'otm_put', or 'otm_call'

        Returns:
            TermStructure with term structure analysis
        """
        # Group options by expiration
        exp_groups: Dict[date, List[Dict]] = {}

        for opt in options_data:
            exp = opt.get('expiration')
            if isinstance(exp, str):
                exp = datetime.strptime(exp, '%Y-%m-%d').date()

            if exp not in exp_groups:
                exp_groups[exp] = []
            exp_groups[exp].append(opt)

        # Calculate IV for each expiration based on structure type
        expirations = []
        days_to_expiry_list = []
        ivs = []

        for exp in sorted(exp_groups.keys()):
            days = (exp - date.today()).days
            if days <= 0:
                continue

            opts = exp_groups[exp]

            # Select strike based on structure type
            if structure_type == 'atm':
                # ATM strike
                target_strike = spot_price
            elif structure_type == 'otm_put':
                # 90% OTM put
                target_strike = spot_price * 0.9
            elif structure_type == 'otm_call':
                # 110% OTM call
                target_strike = spot_price * 1.1
            else:
                target_strike = spot_price

            # Find IV at target strike
            closest_opt = min(opts, key=lambda x: abs(x['strike'] - target_strike))
            iv = closest_opt.get('implied_volatility')

            if iv:
                expirations.append(exp)
                days_to_expiry_list.append(days)
                ivs.append(iv)

        if not ivs:
            raise ValueError("No valid IV data for term structure")

        # Calculate term structure metrics
        front_month_iv = ivs[0] if ivs else 0
        back_month_iv = ivs[-1] if len(ivs) > 1 else front_month_iv

        # Calculate slope (linear regression over time)
        if len(days_to_expiry_list) >= 2:
            slope, _ = np.polyfit(days_to_expiry_list, ivs, 1)
            term_structure_slope = slope
        else:
            term_structure_slope = 0.0

        # Check if inverted (backwardation)
        is_inverted = front_month_iv > back_month_iv

        return TermStructure(
            symbol=symbol,
            spot_price=spot_price,
            structure_type=structure_type,
            expirations=expirations,
            days_to_expiry=days_to_expiry_list,
            implied_volatilities=ivs,
            front_month_iv=front_month_iv,
            back_month_iv=back_month_iv,
            term_structure_slope=term_structure_slope,
            is_inverted=is_inverted
        )

    async def calculate_all_analytics(
        self,
        symbol: str,
        options_data: List[Dict[str, Any]],
        spot_price: float
    ) -> Dict[str, Any]:
        """
        Calculate all options analytics in parallel.

        Returns complete analytics package.
        """
        # Calculate IV surface
        iv_surface = await self.calculate_iv_surface(symbol, options_data, spot_price)

        # Get unique expirations
        expirations = set()
        for opt in options_data:
            exp = opt.get('expiration')
            if isinstance(exp, str):
                exp = datetime.strptime(exp, '%Y-%m-%d').date()
            if exp and (exp - date.today()).days > 0:
                expirations.add(exp)

        # Calculate skew for each expiration (limit to first 5)
        skew_metrics = []
        for exp in sorted(expirations)[:5]:
            try:
                skew = await self.calculate_skew(symbol, exp, options_data, spot_price)
                skew_metrics.append(skew)
            except Exception as e:
                logger.warning(f"Failed to calculate skew for {exp}: {e}")

        # Calculate term structures
        term_structures = {}
        for struct_type in ['atm', 'otm_put', 'otm_call']:
            try:
                ts = await self.calculate_term_structure(
                    symbol, options_data, spot_price, struct_type
                )
                term_structures[struct_type] = ts
            except Exception as e:
                logger.warning(f"Failed to calculate {struct_type} term structure: {e}")

        return {
            'symbol': symbol,
            'spot_price': spot_price,
            'iv_surface': iv_surface,
            'skew_metrics': skew_metrics,
            'term_structures': term_structures,
            'calculation_time': datetime.now().isoformat()
        }
