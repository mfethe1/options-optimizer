from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import datetime as dt
import logging

from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.finnhub_provider import FinnhubProvider
from src.data.providers.fmp_provider import FMPProvider
from src.data.providers.marketstack_provider import MarketstackProvider

logger = logging.getLogger("screen")


@dataclass
class ProviderRouter:
    symbol: str
    fresh_chains: bool = False  # bypass read-through cache if True

    def __post_init__(self):
        # Defer yfinance import to avoid hard dependency during tests
        norm_sym = self._normalize_yf_symbol(self.symbol)
        try:
            from src.data.providers.yfinance_provider import YFinanceProvider
            self.yf = YFinanceProvider(norm_sym)
        except Exception:
            self.yf = None
        self.alpha = AlphaVantageProvider()
        self.finnhub = FinnhubProvider()
        self.fmp = FMPProvider()
        self.ms = MarketstackProvider()
        logger.debug(f"ProviderRouter init for {self.symbol} -> yf_symbol={norm_sym}, fresh_chains={self.fresh_chains}")

    @staticmethod
    def _normalize_yf_symbol(sym: str) -> str:
        # Strip leading '$', map dot to hyphen for class shares (e.g., BRK.B -> BRK-B)
        if not sym:
            return sym
        s = sym.strip().upper()
        if s.startswith("$"):
            s = s[1:]
        s = s.replace(".", "-")
        return s

    # Prefer yfinance for daily OHLCV to avoid rate limits; fallback to Alpha Vantage, then Marketstack
    def daily_prices(self) -> List[Dict[str, Any]]:
        # yfinance first (2 months to better cover adv window)
        try:
            if self.yf is not None:
                hist = self.yf._yf.history(period="2mo")
                out = []
                for idx, row in hist.iterrows():
                    out.append({"date": str(idx.date()), "close": float(row["Close"]), "volume": int(row.get("Volume", 0))})
                if out:
                    logger.debug(f"{self.symbol}: daily_prices via yfinance ({len(out)} rows)")
                    return out
        except Exception as e:
            logger.debug(f"{self.symbol}: yfinance daily_prices failed: {e}")
        # Alpha Vantage fallback
        try:
            out = self.alpha.daily_adjusted(self.symbol)
            if out:
                logger.debug(f"{self.symbol}: daily_prices via AlphaVantage ({len(out)} rows)")
            return out
        except Exception as e:
            logger.debug(f"{self.symbol}: AlphaVantage daily_prices failed: {e}")
        # Marketstack fallback
        try:
            data = self.ms.eod(self.symbol, limit=60)
            rows = data.get("data", []) if isinstance(data, dict) else []
            out: List[Dict[str, Any]] = []
            for r in rows:
                try:
                    out.append({"date": str(r.get("date", ""))[:10], "close": float(r["close"]), "volume": int(r.get("volume", 0))})
                except Exception:
                    continue
            out.sort(key=lambda x: x["date"])  # ascending
            if out:
                logger.debug(f"{self.symbol}: daily_prices via Marketstack ({len(out)} rows)")
            return out
        except Exception as e:
            logger.debug(f"{self.symbol}: Marketstack daily_prices failed: {e}")
        return []

    def _yf_period_for_days(self, days: int) -> str:
        if days <= 30:
            return "1mo"
        if days <= 60:
            return "3mo"
        if days <= 180:
            return "6mo"
        if days <= 365:
            return "1y"
        return "2y"

    def daily_prices_window(self, days: int) -> List[Dict[str, Any]]:
        # Try Marketstack first (fast bulk)
        try:
            data = self.ms.eod(self.symbol, limit=max(100, days + 10))
            rows = data.get("data", []) if isinstance(data, dict) else []
            if rows:
                out = []
                for r in rows:
                    try:
                        out.append({"date": str(r.get("date", ""))[:10], "close": float(r["close"]), "volume": int(r.get("volume", 0))})
                    except Exception:
                        continue
                out.sort(key=lambda x: x["date"])  # ascending
                logger.debug(f"{self.symbol}: daily_prices_window via Marketstack ({len(out)} rows)")
                return out[-(days + 1):]
        except Exception as e:
            logger.debug(f"{self.symbol}: Marketstack daily_prices_window failed: {e}")
        # Alpha Vantage fallback
        try:
            out = self.alpha.daily_adjusted(self.symbol)
            logger.debug(f"{self.symbol}: daily_prices_window via AlphaVantage ({len(out)} rows)")
            return out[-(days + 1):]
        except Exception as e:
            logger.debug(f"{self.symbol}: AlphaVantage daily_prices_window failed: {e}")
        # yfinance fallback with appropriate period
        try:
            if self.yf is None:
                return []
            hist = self.yf._yf.history(period=self._yf_period_for_days(days))
            out = []
            for idx, row in hist.iterrows():
                out.append({"date": str(idx.date()), "close": float(row["Close"]), "volume": int(row.get("Volume", 0))})
            logger.debug(f"{self.symbol}: daily_prices_window via yfinance ({len(out)} rows)")
            return out[-(days + 1):]
        except Exception as e:
            logger.debug(f"{self.symbol}: yfinance daily_prices_window failed: {e}")
            return []

    # Underlying price with fallback
    def underlying_price(self) -> Optional[float]:
        # Prefer yfinance last close; fallback to last close from daily_prices window
        try:
            if self.yf is not None:
                price = float(self.yf.underlying_price())
                logger.debug(f"{self.symbol}: underlying via yfinance {price}")
                return price
        except Exception as e:
            logger.debug(f"{self.symbol}: yfinance underlying failed: {e}")
        rows = self.daily_prices_window(5)
        if rows:
            try:
                price = float(rows[-1]["close"])
                logger.debug(f"{self.symbol}: underlying via OHLCV {price}")
                return price
            except Exception:
                return None
        return None

    # Option chains with read-through cache; Finnhub primary, yfinance fallback
    def options_chain(self, expiry, side: str) -> List[Dict[str, Any]]:
        # Read-through cache: if today's snapshot exists and not bypassed, load and filter by side
        try:
            if not self.fresh_chains:
                from src.data.datastore import DataStore
                ds = DataStore()
                as_of = dt.date.today()
                cached = ds.read_options_chain_snapshot(self.symbol, as_of)
                if cached:
                    out = []
                    from types import SimpleNamespace
                    for r in cached:
                        try:
                            if r.get("option_type") == side:
                                # Recreate provider-neutral structure with attribute access
                                out.append(SimpleNamespace(
                                    symbol=r.get("symbol"),
                                    strike=r.get("strike"),
                                    expiry=dt.date.fromisoformat(str(r.get("expiry"))[:10]) if r.get("expiry") else None,
                                    option_type=r.get("option_type"),
                                    last_price=r.get("last_price"),
                                    bid=r.get("bid"),
                                    ask=r.get("ask"),
                                    volume=r.get("volume"),
                                    open_interest=r.get("open_interest"),
                                    underlying_price=r.get("underlying_price"),
                                ))
                        except Exception:
                            continue
                    if out:
                        logger.debug(f"{self.symbol}: options_chain via CACHE ({side}) -> {len(out)} contracts")
                        return out
        except Exception as e:
            logger.debug(f"{self.symbol}: options_chain cache read failed: {e}")

        # Try Finnhub if available
        try:
            rows = self.finnhub.options_chain(self.symbol, expiry, side)
            if rows:
                logger.debug(f"{self.symbol}: options_chain via Finnhub ({side}) -> {len(rows)} contracts")
                return rows
        except Exception as e:
            logger.debug(f"{self.symbol}: Finnhub options_chain failed: {e}")
        # Fallback to yfinance
        try:
            if self.yf is None:
                return []
            rows = self.yf.options_chain(expiry, side)
            if rows:
                logger.debug(f"{self.symbol}: options_chain via yfinance ({side}) -> {len(rows)} contracts")
            return rows
        except Exception as e:
            logger.debug(f"{self.symbol}: yfinance options_chain failed: {e}")
            return []

    def expirations(self) -> List[dt.date]:
        # Prefer yfinance for expirations for now
        try:
            if self.yf is None:
                return []
            return self.yf.expirations()
        except Exception as e:
            logger.debug(f"{self.symbol}: yfinance expirations failed: {e}")
            return []

    # Earnings events (past and upcoming) with fallback and date filtering
    def earnings_events(self, as_of: dt.date, lookback_days: int = 365*2, lookahead_days: int = 365) -> List[dt.date]:
        start = as_of - dt.timedelta(days=lookback_days)
        end = as_of + dt.timedelta(days=lookahead_days)

        # Try Finnhub
        try:
            data = self.finnhub.earnings_calendar(self.symbol)
            dates = _extract_dates_generic(data)
            if dates:
                return [d for d in dates if start <= d <= end]
        except Exception as e:
            logger.debug(f"{self.symbol}: Finnhub earnings failed: {e}")
        # Fallback Alpha Vantage
        try:
            data = self.alpha.earnings(self.symbol)
            dates = _extract_dates_generic(data)
            if dates:
                return [d for d in dates if start <= d <= end]
        except Exception as e:
            logger.debug(f"{self.symbol}: AlphaVantage earnings failed: {e}")
        return []

    # Fundamentals
    def market_cap(self) -> Optional[float]:
        mc = self.fmp.market_cap(self.symbol)
        if mc is not None:
            return mc
        # yfinance fallback
        try:
            if self.yf is None:
                return None
            info = self.yf._yf.info
            mc2 = info.get("marketCap")
            return float(mc2) if mc2 else None
        except Exception:
            return None

    @staticmethod
    def list_all_symbols_via_fmp() -> List[str]:
        try:
            fmp = FMPProvider()
            rows = fmp.stock_list()
            # Basic US filter & common stock types; adjust as needed
            symbols = [r["symbol"] for r in rows if r.get("exchangeShortName") in {"NYSE", "NASDAQ"} and r.get("type") in {"stock", "Common Stock"}]
            # Filter out $-prefixed and long/odd symbols
            clean = []
            for s in symbols:
                if not s or s.startswith("$"):
                    continue
                if len(s) > 6:
                    continue
                clean.append(s)
            return list({s for s in clean})
        except Exception:
            return []

    # TODO: Implement option chains for Alpha Vantage / FMP when stable endpoints are mapped


def _extract_dates_generic(payload: Any) -> List[dt.date]:
    dates: List[dt.date] = []
    def try_parse(s: Any) -> Optional[dt.date]:
        try:
            if isinstance(s, str) and len(s) >= 10:
                return dt.date.fromisoformat(s[:10])
        except Exception:
            return None
        return None
    def walk(obj: Any):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        else:
            d = try_parse(obj)
            if d:
                dates.append(d)
    walk(payload)
    # de-duplicate and sort
    uniq = sorted({d for d in dates})
    return uniq

