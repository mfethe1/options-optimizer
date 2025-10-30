from __future__ import annotations
import os
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import pandas as pd  # type: ignore
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    PARQUET_AVAILABLE = True
except Exception:
    PARQUET_AVAILABLE = False


@dataclass
class DataStore:
    base_dir: str = os.path.join("data")

    def _ensure_dir(self, *parts: str) -> str:
        path = os.path.join(self.base_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    # ---- Equities ----
    def write_eq_daily(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        path = self._ensure_dir("eq", "daily")
        fpath = os.path.join(path, f"{symbol.upper()}.parquet")
        if not PARQUET_AVAILABLE:
            # Fallback JSON
            jpath = fpath.replace(".parquet", ".json")
            import json
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            return
        df = pd.DataFrame(rows)
        df.to_parquet(fpath, index=False)

    def read_eq_daily(self, symbol: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.base_dir, "eq", "daily", f"{symbol.upper()}.parquet")
        if os.path.exists(path) and PARQUET_AVAILABLE:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        # JSON fallback
        jpath = path.replace(".parquet", ".json")
        if os.path.exists(jpath):
            import json
            with open(jpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def write_eq_intraday(self, symbol: str, interval: str, date: dt.date, rows: List[Dict[str, Any]]) -> None:
        # Partition by day for intraday
        path = self._ensure_dir("eq", f"{interval}", symbol.upper())
        fpath = os.path.join(path, f"{date.isoformat()}.parquet")
        if not PARQUET_AVAILABLE:
            jpath = fpath.replace(".parquet", ".json")
            import json
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            return
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(fpath, index=False)

    # ---- Options ----
    def write_options_chain_snapshot(self, symbol: str, as_of: dt.date, rows: List[Dict[str, Any]]) -> None:
        path = self._ensure_dir("opt", "chains", symbol.upper())
        fpath = os.path.join(path, f"{as_of.isoformat()}.parquet")
        if not PARQUET_AVAILABLE:
            jpath = fpath.replace(".parquet", ".json")
            import json
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            return
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(fpath, index=False)

    def read_options_chain_snapshot(self, symbol: str, as_of: dt.date) -> List[Dict[str, Any]]:
        path = os.path.join(self.base_dir, "opt", "chains", symbol.upper(), f"{as_of.isoformat()}.parquet")
        if os.path.exists(path) and PARQUET_AVAILABLE:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        # JSON fallback
        jpath = path.replace(".parquet", ".json")
        if os.path.exists(jpath):
            import json
            with open(jpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    # ---- Events ----
    def write_earnings(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        path = self._ensure_dir("events", "earnings")
        fpath = os.path.join(path, f"{symbol.upper()}.parquet")
        if not PARQUET_AVAILABLE:
            jpath = fpath.replace(".parquet", ".json")
            import json
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            return
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(fpath, index=False)

