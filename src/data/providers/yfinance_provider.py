"""
YFinance provider for equities and options chains.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yfinance as yf


@dataclass
class OptionContract:
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


class YFinanceProvider:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf = yf.Ticker(self.ticker)

    def underlying_price(self) -> float:
        info = self._yf.history(period="1d")
        if info.empty:
            raise ValueError("No price data")
        return float(info["Close"].iloc[-1])

    def expirations(self) -> List[dt.date]:
        exps = self._yf.options
        return [dt.date.fromisoformat(x) for x in exps]

    def options_chain(self, expiry: dt.date, side: str) -> List[OptionContract]:
        exp_str = expiry.isoformat()
        data = self._yf.option_chain(exp_str)
        df = data.calls if side == "call" else data.puts
        S = self.underlying_price()
        out: List[OptionContract] = []
        for _, row in df.iterrows():
            out.append(
                OptionContract(
                    symbol=row.get("contractSymbol", f"{self.ticker}{row['strike']}{side[0].upper()}{exp_str}"),
                    strike=float(row["strike"]),
                    expiry=expiry,
                    option_type=side,
                    last_price=float(row.get("lastPrice", 0.0)),
                    bid=float(row.get("bid", 0.0)),
                    ask=float(row.get("ask", 0.0)),
                    volume=int(row.get("volume", 0)) if row.get("volume") == row.get("volume") else None,
                    open_interest=int(row.get("openInterest", 0)) if row.get("openInterest") == row.get("openInterest") else None,
                    underlying_price=S,
                )
            )
        return out

