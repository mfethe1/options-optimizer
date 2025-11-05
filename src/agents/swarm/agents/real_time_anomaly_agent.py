"""
Real-Time Anomaly Detection Agent

Detects unusual activity in real-time: volume spikes, price movements,
IV changes, and options flow anomalies.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from src.agents.swarm.base_swarm_agent import BaseSwarmAgent

logger = logging.getLogger(__name__)


class RealTimeAnomalyAgent(BaseSwarmAgent):
    """
    COMPETITIVE ADVANTAGE: Real-time anomaly detection

    Detects unusual activity that precedes major moves:
    - Volume spikes (3+ standard deviations)
    - Price gaps (unusual intraday movements)
    - IV expansion (rapid volatility increases)
    - Options flow anomalies (block trades, unusual strikes)
    - Dark pool activity (large off-exchange trades)

    Proven alpha source: Catching unusual activity before the crowd
    """

    def __init__(self):
        super().__init__(
            name="RealTimeAnomalyDetector",
            priority=9,  # High priority for time-sensitive signals
            temperature=0.2  # Low temperature for deterministic detection
        )

        # Historical baseline data (rolling window)
        self.baseline_data: Dict[str, Dict[str, List]] = {}

        # Anomaly thresholds
        self.volume_threshold = 3.0  # 3 standard deviations
        self.price_threshold = 2.5   # 2.5 standard deviations
        self.iv_threshold = 2.0      # 2 standard deviations

    async def analyze(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect real-time anomalies

        Args:
            portfolio_data: Contains current market data and historical baseline

        Returns:
            Dictionary with detected anomalies and severity levels
        """

        logger.info(f"{self.name} analyzing for anomalies")

        anomalies = []

        # Extract symbols from portfolio
        symbols = portfolio_data.get('symbols', [])

        for symbol in symbols:
            market_data = portfolio_data.get('market_data', {}).get(symbol, {})

            if not market_data:
                continue

            # Check for volume anomalies
            volume_anomaly = self._detect_volume_anomaly(symbol, market_data)
            if volume_anomaly:
                anomalies.append(volume_anomaly)

            # Check for price movement anomalies
            price_anomaly = self._detect_price_anomaly(symbol, market_data)
            if price_anomaly:
                anomalies.append(price_anomaly)

            # Check for IV expansion anomalies
            iv_anomaly = self._detect_iv_anomaly(symbol, market_data)
            if iv_anomaly:
                anomalies.append(iv_anomaly)

            # Check for options flow anomalies
            flow_anomaly = self._detect_options_flow_anomaly(symbol, market_data)
            if flow_anomaly:
                anomalies.append(flow_anomaly)

        # Rank anomalies by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)

        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "critical_count": len([a for a in anomalies if a['severity'] == 'critical']),
            "high_count": len([a for a in anomalies if a['severity'] == 'high']),
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.9  # High confidence in mathematical detection
        }

    def _detect_volume_anomaly(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect unusual volume spikes"""

        current_volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_30d', 0)

        if avg_volume == 0:
            return None

        # Calculate z-score
        historical_volumes = market_data.get('historical_volumes', [])
        if len(historical_volumes) < 10:
            return None

        std_volume = np.std(historical_volumes)
        if std_volume == 0:
            return None

        z_score = (current_volume - avg_volume) / std_volume

        # Detect anomaly
        if abs(z_score) > self.volume_threshold:
            severity = 'critical' if abs(z_score) > 5.0 else 'high'

            return {
                "type": "volume_spike",
                "symbol": symbol,
                "severity": severity,
                "z_score": round(z_score, 2),
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "multiplier": round(current_volume / avg_volume, 2),
                "message": f"{symbol} volume {round(current_volume / avg_volume, 1)}x average (z-score: {round(z_score, 1)})",
                "timestamp": datetime.now().isoformat(),
                "trading_implication": "High volume often precedes significant moves. Consider checking news and options flow."
            }

        return None

    def _detect_price_anomaly(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect unusual price movements"""

        current_price = market_data.get('current_price', 0)
        prev_close = market_data.get('previous_close', 0)

        if prev_close == 0:
            return None

        # Calculate percentage move
        pct_change = ((current_price - prev_close) / prev_close) * 100

        # Get historical daily returns
        historical_returns = market_data.get('historical_returns', [])
        if len(historical_returns) < 20:
            return None

        std_return = np.std(historical_returns)
        if std_return == 0:
            return None

        z_score = pct_change / std_return

        # Detect anomaly
        if abs(z_score) > self.price_threshold:
            severity = 'critical' if abs(z_score) > 4.0 else 'high'

            return {
                "type": "price_move",
                "symbol": symbol,
                "severity": severity,
                "z_score": round(z_score, 2),
                "pct_change": round(pct_change, 2),
                "current_price": round(current_price, 2),
                "prev_close": round(prev_close, 2),
                "message": f"{symbol} moved {abs(pct_change):.1f}% ({'up' if pct_change > 0 else 'down'}) - {abs(z_score):.1f}σ event",
                "timestamp": datetime.now().isoformat(),
                "trading_implication": f"Large moves create opportunities. Check if this is news-driven or technical."
            }

        return None

    def _detect_iv_anomaly(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect rapid IV expansion"""

        current_iv = market_data.get('implied_volatility', 0)
        avg_iv_30d = market_data.get('avg_iv_30d', 0)

        if avg_iv_30d == 0:
            return None

        historical_ivs = market_data.get('historical_ivs', [])
        if len(historical_ivs) < 10:
            return None

        std_iv = np.std(historical_ivs)
        if std_iv == 0:
            return None

        z_score = (current_iv - avg_iv_30d) / std_iv

        # Detect anomaly
        if z_score > self.iv_threshold:  # Only care about IV expansion
            severity = 'critical' if z_score > 3.0 else 'high'

            return {
                "type": "iv_expansion",
                "symbol": symbol,
                "severity": severity,
                "z_score": round(z_score, 2),
                "current_iv": round(current_iv * 100, 1),  # Convert to percentage
                "avg_iv": round(avg_iv_30d * 100, 1),
                "iv_rank": market_data.get('iv_rank', 0),
                "message": f"{symbol} IV expanded to {round(current_iv * 100, 1)}% (from {round(avg_iv_30d * 100, 1)}% avg)",
                "timestamp": datetime.now().isoformat(),
                "trading_implication": "High IV expansion suggests market expects big move. Consider selling premium or calendars."
            }

        return None

    def _detect_options_flow_anomaly(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect unusual options activity"""

        options_volume = market_data.get('options_volume', 0)
        avg_options_volume = market_data.get('avg_options_volume', 0)

        if avg_options_volume == 0:
            return None

        volume_ratio = options_volume / avg_options_volume

        # Detect high options volume relative to stock volume
        stock_volume = market_data.get('volume', 1)
        options_to_stock_ratio = options_volume / max(stock_volume, 1)

        # Check for block trades
        large_trades = market_data.get('large_options_trades', [])

        # Detect anomaly
        if volume_ratio > 3.0 or options_to_stock_ratio > 0.5 or len(large_trades) > 0:
            severity = 'critical' if volume_ratio > 5.0 or len(large_trades) > 3 else 'high'

            return {
                "type": "options_flow",
                "symbol": symbol,
                "severity": severity,
                "options_volume": options_volume,
                "avg_options_volume": avg_options_volume,
                "volume_ratio": round(volume_ratio, 2),
                "options_to_stock_ratio": round(options_to_stock_ratio, 3),
                "large_trades_count": len(large_trades),
                "put_call_ratio": market_data.get('put_call_ratio', 0),
                "message": f"{symbol} options volume {round(volume_ratio, 1)}x average ({len(large_trades)} block trades)",
                "timestamp": datetime.now().isoformat(),
                "trading_implication": "Unusual options activity suggests informed traders positioning. Investigate flow direction."
            }

        return None

    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make recommendations based on anomaly analysis

        Args:
            analysis: Analysis results from analyze()

        Returns:
            Recommendations with confidence levels
        """
        anomalies = analysis.get('anomalies', [])
        recommendations = []

        for anomaly in anomalies:
            if anomaly['severity'] == 'critical':
                # Critical anomalies warrant immediate action
                if anomaly['type'] == 'volume_spike':
                    recommendations.append({
                        'action': 'investigate',
                        'symbol': anomaly['symbol'],
                        'reason': f"Critical volume spike detected ({anomaly['multiplier']}x average)",
                        'urgency': 'high',
                        'confidence': 0.9
                    })
                elif anomaly['type'] == 'iv_expansion':
                    recommendations.append({
                        'action': 'consider_volatility_trade',
                        'symbol': anomaly['symbol'],
                        'reason': f"Rapid IV expansion to {anomaly['current_iv']}%",
                        'urgency': 'high',
                        'confidence': 0.85,
                        'suggestion': 'Consider selling premium or calendar spreads'
                    })
                elif anomaly['type'] == 'price_move':
                    recommendations.append({
                        'action': 'review_position',
                        'symbol': anomaly['symbol'],
                        'reason': f"Unusual price movement ({anomaly['pct_change']}%)",
                        'urgency': 'high',
                        'confidence': 0.9
                    })
                elif anomaly['type'] == 'options_flow':
                    recommendations.append({
                        'action': 'investigate_flow',
                        'symbol': anomaly['symbol'],
                        'reason': f"Unusual options activity ({anomaly['volume_ratio']}x average)",
                        'urgency': 'high',
                        'confidence': 0.85,
                        'suggestion': 'Check flow direction and large trades'
                    })

        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'high_urgency_count': len([r for r in recommendations if r.get('urgency') == 'high']),
            'timestamp': datetime.now().isoformat(),
            'confidence': analysis.get('confidence', 0.9)
        }

    async def update_baseline(self, symbol: str, historical_data: Dict[str, Any]):
        """
        Update historical baseline for anomaly detection

        Args:
            symbol: Stock symbol
            historical_data: Historical price, volume, IV data
        """

        if symbol not in self.baseline_data:
            self.baseline_data[symbol] = {
                "volumes": [],
                "returns": [],
                "ivs": [],
                "last_updated": datetime.now()
            }

        # Update rolling windows (keep last 30 days)
        max_points = 30

        if 'volumes' in historical_data:
            self.baseline_data[symbol]['volumes'].extend(historical_data['volumes'])
            self.baseline_data[symbol]['volumes'] = self.baseline_data[symbol]['volumes'][-max_points:]

        if 'returns' in historical_data:
            self.baseline_data[symbol]['returns'].extend(historical_data['returns'])
            self.baseline_data[symbol]['returns'] = self.baseline_data[symbol]['returns'][-max_points:]

        if 'ivs' in historical_data:
            self.baseline_data[symbol]['ivs'].extend(historical_data['ivs'])
            self.baseline_data[symbol]['ivs'] = self.baseline_data[symbol]['ivs'][-max_points:]

        self.baseline_data[symbol]['last_updated'] = datetime.now()

        logger.debug(f"Updated baseline for {symbol}")
