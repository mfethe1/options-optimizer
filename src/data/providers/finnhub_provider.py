from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import logging

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

FINNHUB_URL = "https://finnhub.io/api/v1"
logger = logging.getLogger("screen")


@dataclass
class OptionContractFH:
    symbol: str
    strike: float
    expiry: dt.date
    option_type: str  # "call" | "put"
    last_price: float
    bid: float
    ask: float
    volume: Optional[int]
    open_interest: Optional[int]
    underlying_price: float


@dataclass
class FinnhubProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.finnhub_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.finnhub_rpm)

    def _get(self, path: str, params: Dict[str, Any], ttl_seconds: int = 60) -> Dict[str, Any]:
        p = {"token": self.api_key}
        p.update(params)
        url = f"{FINNHUB_URL}/{path}"
        return self.client.get_json(url, p, self.rate_limiter, ttl_seconds=ttl_seconds)

    # --------- Equities & Events ---------
    def earnings_calendar(self, symbol: str) -> Dict[str, Any]:
        return self._get("calendar/earnings", {"symbol": symbol}, ttl_seconds=3600)

    def news_sentiment(self, symbol: str) -> Dict[str, Any]:
        return self._get("news-sentiment", {"symbol": symbol}, ttl_seconds=1800)

    def quote(self, symbol: str) -> Dict[str, Any]:
        return self._get("quote", {"symbol": symbol}, ttl_seconds=30)

    def stock_candles(self, symbol: str, resolution: str, start: int, end: int) -> Dict[str, Any]:
        """
        Wrap Finnhub /stock/candle. resolution in { '1','5','15','30','60','D' }.
        start/end are UNIX epoch seconds (UTC). Returns raw Finnhub JSON.
        """
        return self._get("stock/candle", {"symbol": symbol, "resolution": resolution, "from": start, "to": end}, ttl_seconds=120)

    # --------- Options (chain & greeks) ---------
    def _coerce_row(self, r: Dict[str, Any], S: float, expiry: dt.date, side: str, symbol: str) -> OptionContractFH:
        exp_str = expiry.isoformat()
        # Finnhub fields are not always consistent across plans; handle common aliases
        strike = r.get("strike") or r.get("s") or r.get("strikePrice") or 0.0
        bid = r.get("bid") or r.get("b") or r.get("bidPrice") or 0.0
        ask = r.get("ask") or r.get("a") or r.get("askPrice") or 0.0
        last_price = r.get("lastPrice") or r.get("last") or r.get("lp") or 0.0
        vol = r.get("volume") if r.get("volume") is not None else r.get("v")
        oi = r.get("openInterest") if r.get("openInterest") is not None else r.get("oi")
        return OptionContractFH(
            symbol=str(r.get("symbol") or r.get("contractSymbol") or r.get("code") or f"{symbol}{strike}{side[0].upper()}{exp_str}"),
            strike=float(strike or 0.0),
            expiry=expiry,
            option_type=side,
            last_price=float(last_price or 0.0),
            bid=float(bid or 0.0),
            ask=float(ask or 0.0),
            volume=int(vol) if vol is not None else None,
            open_interest=int(oi) if oi is not None else None,
            underlying_price=float(S or 0.0),
        )

    def options_chain(self, symbol: str, expiry: dt.date, side: str) -> List[OptionContractFH]:
        """
        Fetch option chain for a symbol and expiry date.
        Strategy: try /stock/option-chain with date; if empty, try without date and filter by expiration.
        """
        exp_str = expiry.isoformat()
        try:
            q = self.quote(symbol)
            S = float(q.get("c")) if isinstance(q, dict) and q.get("c") is not None else 0.0
        except Exception:
            S = 0.0

        def parse_payload(payload: Dict[str, Any]) -> List[OptionContractFH]:
            out: List[OptionContractFH] = []
            if not isinstance(payload, dict):
                return out
            data = payload.get("data")
            # Some plans return { data: [ ... ] } where each element can be:
            #  - a contract row (has 'type'/'optionType'), or
            #  - an expiry entry with 'expirationDate' and nested 'options': {'CALL': [...], 'PUT': [...]}
            if isinstance(data, list):
                for entry in data:
                    try:
                        if isinstance(entry, dict) and ("options" in entry or "CALL" in entry or "PUT" in entry):
                            # expiry-level entry with nested options arrays
                            efield = entry.get("expirationDate") or entry.get("expiry") or entry.get("expDate") or entry.get("date")
                            efield_str = str(efield)[:10] if efield else None
                            if efield_str and efield_str != exp_str:
                                continue
                            opt_map = entry.get("options") if isinstance(entry.get("options"), dict) else entry
                            call_list = None
                            put_list = None
                            if isinstance(opt_map, dict):
                                call_list = opt_map.get("CALL") or opt_map.get("calls")
                                put_list = opt_map.get("PUT") or opt_map.get("puts")
                            if side == "call" and isinstance(call_list, list):
                                for r in call_list:
                                    try:
                                        out.append(self._coerce_row(r, S, expiry, "call", symbol))
                                    except Exception:
                                        continue
                            if side == "put" and isinstance(put_list, list):
                                for r in put_list:
                                    try:
                                        out.append(self._coerce_row(r, S, expiry, "put", symbol))
                                    except Exception:
                                        continue
                        else:
                            # contract-row shape
                            t = (entry.get("type") or entry.get("optionType") or entry.get("t") or "").lower()
                            row_side = "call" if t in ("call", "c") else ("put" if t in ("put", "p") else None)
                            if row_side == side:
                                out.append(self._coerce_row(entry, S, expiry, side, symbol))
                    except Exception:
                        continue
                return out
            # Some payloads return { calls: [...], puts: [...] }
            calls = payload.get("calls")
            puts = payload.get("puts")
            if side == "call" and isinstance(calls, list):
                for r in calls:
                    try:
                        out.append(self._coerce_row(r, S, expiry, "call", symbol))
                    except Exception:
                        continue
            if side == "put" and isinstance(puts, list):
                for r in puts:
                    try:
                        out.append(self._coerce_row(r, S, expiry, "put", symbol))
                    except Exception:
                        continue
            return out

        # First attempt: with date
        try:
            resp = self._get("stock/option-chain", {"symbol": symbol, "date": exp_str}, ttl_seconds=300)
            rows = parse_payload(resp)
            if rows:
                logger.debug(f"Finnhub options_chain {symbol} {exp_str} {side}: {len(rows)} rows (date-filtered)")
                return rows
            # Payload summary for debugging when zero rows
            if isinstance(resp, dict):
                keys = list(resp.keys())
                data = resp.get("data")
                dtype = type(data).__name__
                sample_keys = None
                if isinstance(data, list) and data:
                    sample_keys = list(data[0].keys())[:10]
                elif isinstance(data, dict):
                    sample_keys = list(data.keys())[:10]
                logger.debug(f"Finnhub option-chain 200 but empty parse: keys={keys}, data_type={dtype}, sample={sample_keys}")
        except Exception as e:
            logger.debug(f"Finnhub options_chain date call failed: {e}")

        # Second attempt: without date, filter by expiration field in rows
        try:
            resp = self._get("stock/option-chain", {"symbol": symbol}, ttl_seconds=300)
            out = []
            if isinstance(resp, dict):
                data = resp.get("data")
                if isinstance(data, list):
                    for r in data:
                        try:
                            # expiry field variants
                            efield = r.get("expirationDate") or r.get("expiry") or r.get("expDate") or r.get("date")
                            # normalize to YYYY-MM-DD if needed
                            efield_str = str(efield)[:10] if efield else None
                            t = (r.get("type") or r.get("optionType") or r.get("t") or "").lower()
                            row_side = "call" if t in ("call", "c") else ("put" if t in ("put", "p") else None)
                            if efield_str == exp_str and row_side == side:
                                out.append(self._coerce_row(r, S, expiry, side, symbol))
                        except Exception:
                            continue
                else:
                    # maybe calls/puts keys
                    for key in ("calls", "puts"):
                        arr = resp.get(key)
                        if isinstance(arr, list):
                            for r in arr:
                                try:
                                    efield = r.get("expirationDate") or r.get("expiry") or r.get("expDate") or r.get("date")
                                    efield_str = str(efield)[:10] if efield else None
                                    if efield_str == exp_str and ((key == "calls" and side == "call") or (key == "puts" and side == "put")):
                                        out.append(self._coerce_row(r, S, expiry, side, symbol))
                                except Exception:
                                    continue
            if out:
                logger.debug(f"Finnhub options_chain {symbol} {exp_str} {side}: {len(out)} rows (filtered from undated)")
            return out
        except Exception as e:
            logger.debug(f"Finnhub options_chain undated call failed: {e}")
            return []
