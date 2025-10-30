from __future__ import annotations
import argparse
import csv
import datetime as dt
import logging

from src.screening.universe import build_universe_dynamic_from_fmp, UniverseConfig
from src.screening.full_pipeline import build_rows_for_universe


DEFAULT_COLUMNS = [
    "symbol","adv","market_cap","iv_rank","iv_percentile","iv_hv_ratio","volume_anomaly","oi_anomaly","pre_earnings_ratio","momentum_score","composite_score","rationale"
]


def run_screen_cli():
    parser = argparse.ArgumentParser(description="Daily screening: dynamic universe to CSV")
    parser.add_argument("output_csv", type=str)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--universe-size", type=int, default=200, help="Limit number of symbols to fetch and process")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--fresh-chains", action="store_true", help="Bypass read-through cache for options chains and fetch fresh from providers")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("screen")

    cfg = UniverseConfig()
    universe = build_universe_dynamic_from_fmp(cfg, limit=args.universe_size)
    logger.info(f"Universe size={len(universe)} (requested {args.universe_size})")
    logger.debug(f"Universe symbols: {universe}")

    rows = build_rows_for_universe(universe, as_of=dt.date.today(), top=args.top, fresh_chains=args.fresh_chains)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in DEFAULT_COLUMNS})

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    run_screen_cli()

