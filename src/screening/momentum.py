from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from src.data.providers.alpha_tech import AlphaTechProvider


@dataclass
class MomentumWeights:
    w_rsi14: float = 0.2
    w_rsi30: float = 0.2
    w_roc10: float = 0.2
    w_roc20: float = 0.2
    w_macd: float = 0.2


class MomentumComputer:
    def __init__(self):
        self.tech = AlphaTechProvider()

    def fetch_indicators(self, symbol: str, interval: str = "daily") -> Dict[str, Optional[float]]:
        """
        Fetch indicators via Alpha Vantage; fall back to local computation from yfinance
        to avoid long rate-limit sleeps when processing larger universes.
        """
        try:
            rsi14 = self._latest_value(self.tech.rsi(symbol, interval, 14))
            rsi30 = self._latest_value(self.tech.rsi(symbol, interval, 30))
            roc10 = self._latest_value(self.tech.roc(symbol, interval, 10))
            roc20 = self._latest_value(self.tech.roc(symbol, interval, 20))
            macd_vals = self.tech.macd(symbol, interval)
            macd = self._latest_macd(macd_vals)
            resp = {"rsi14": rsi14, "rsi30": rsi30, "roc10": roc10, "roc20": roc20, "macd": macd}
            present = sum(v is not None for v in resp.values())
            if present >= 3:
                return resp
        except Exception:
            pass
        # Fallback local computation from yfinance
        return self._compute_from_prices(symbol)

    @staticmethod
    def _latest_value(resp: Dict[str, Any]) -> Optional[float]:
        # Alpha Vantage response has "Technical Analysis: RSI" with date keys
        for key in resp.keys():
            if key.startswith("Technical Analysis"):
                series = resp[key]
                if not series:
                    return None
                latest = sorted(series.keys())[-1]
                val = list(series[latest].values())[0]
                try:
                    return float(val)
                except Exception:
                    return None
        return None

    @staticmethod
    def _latest_macd(resp: Dict[str, Any]) -> Optional[float]:
        for key in resp.keys():
            if key.startswith("Technical Analysis"):
                series = resp[key]
                if not series:
                    return None
                latest = sorted(series.keys())[-1]
                row = series[latest]
                # use macd - signal as normalized oscillator
                try:
                    return float(row.get("MACD", 0.0)) - float(row.get("MACD_Signal", 0.0))
                except Exception:
                    return None
        return None

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        if len(values) < period:
            return np.array([])
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros_like(values, dtype=float)
        ema[:period] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _compute_from_prices(self, symbol: str) -> Dict[str, Optional[float]]:
        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period="6mo")
            closes = hist["Close"].dropna().values.astype(float)
            if len(closes) < 60:
                return {"rsi14": None, "rsi30": None, "roc10": None, "roc20": None, "macd": None}
            # ROC
            roc10 = 100.0 * (closes[-1] / closes[-11] - 1.0) if len(closes) >= 11 else None
            roc20 = 100.0 * (closes[-1] / closes[-21] - 1.0) if len(closes) >= 21 else None
            # RSI (Wilder's smoothing approximated)
            def rsi_period(n: int) -> Optional[float]:
                if len(closes) < n + 1:
                    return None
                deltas = np.diff(closes)
                gains = np.clip(deltas, 0, None)
                losses = -np.clip(deltas, None, 0)
                roll_up = np.convolve(gains, np.ones(n), 'valid') / n
                roll_down = np.convolve(losses, np.ones(n), 'valid') / n
                rs = roll_up[-1] / roll_down[-1] if roll_down[-1] > 0 else np.inf
                return 100.0 - (100.0 / (1.0 + rs))
            rsi14 = rsi_period(14)
            rsi30 = rsi_period(30)
            # MACD
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            if len(ema12) == 0 or len(ema26) == 0:
                macd = None
            else:
                macd_line = ema12 - ema26
                signal = self._ema(macd_line, 9)
                macd = (macd_line[-1] - signal[-1]) if len(signal) == len(macd_line) else None
            return {"rsi14": rsi14, "rsi30": rsi30, "roc10": roc10, "roc20": roc20, "macd": macd}
        except Exception:
            return {"rsi14": None, "rsi30": None, "roc10": None, "roc20": None, "macd": None}

    @staticmethod
    def normalize_indicator(name: str, value: Optional[float]) -> float:
        if value is None:
            return 50.0  # neutral when missing
        if name.startswith("rsi"):
            return float(np.clip(value, 0.0, 100.0))
        if name.startswith("roc"):
            # map ROC% roughly into 0..100 with tanh scaling
            return float(50.0 + 25.0 * np.tanh(value / 5.0))
        if name == "macd":
            return float(50.0 + 25.0 * np.tanh(value))
        return 50.0

    def momentum_score(self, indicators: Dict[str, Optional[float]], w: MomentumWeights = MomentumWeights()) -> float:
        rsi14_n = self.normalize_indicator("rsi14", indicators.get("rsi14"))
        rsi30_n = self.normalize_indicator("rsi30", indicators.get("rsi30"))
        roc10_n = self.normalize_indicator("roc10", indicators.get("roc10"))
        roc20_n = self.normalize_indicator("roc20", indicators.get("roc20"))
        macd_n = self.normalize_indicator("macd", indicators.get("macd"))
        score = (
            w.w_rsi14 * rsi14_n + w.w_rsi30 * rsi30_n + w.w_roc10 * roc10_n + w.w_roc20 * roc20_n + w.w_macd * macd_n
        ) / (w.w_rsi14 + w.w_rsi30 + w.w_roc10 + w.w_roc20 + w.w_macd)
        return float(np.clip(score, 0.0, 100.0))

