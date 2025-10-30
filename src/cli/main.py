from __future__ import annotations
import argparse
import datetime as dt
from typing import List

from src.data.providers.yfinance_provider import YFinanceProvider
from src.pricing.iv.implied_vol import implied_vol
from src.pricing.black_scholes import bs_price


def rank_options_cli():
    parser = argparse.ArgumentParser(description="Options probability & ranking CLI")
    parser.add_argument("ticker", type=str)
    parser.add_argument("expiry", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()

    provider = YFinanceProvider(args.ticker)
    expiry = dt.date.fromisoformat(args.expiry)

    calls = provider.options_chain(expiry, "call")
    puts = provider.options_chain(expiry, "put")

    print(f"Loaded {len(calls)} calls and {len(puts)} puts for {args.ticker} @ {expiry}")


if __name__ == "__main__":
    rank_options_cli()

