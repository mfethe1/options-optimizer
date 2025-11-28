"""
Create mock cache file to test cache HIT functionality

This simulates having already fetched and cached data from yfinance
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Cache directory
cache_dir = Path("data/backtest_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Generate mock historical data
symbol = "AAPL"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

# Create mock DataFrame (simulates yfinance history data)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
n_days = len(dates)

# Generate realistic price data
base_price = 175.0
prices = base_price + np.cumsum(np.random.randn(n_days) * 2)

mock_data = pd.DataFrame({
    'Open': prices * (1 + np.random.randn(n_days) * 0.01),
    'High': prices * (1 + np.abs(np.random.randn(n_days)) * 0.015),
    'Low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.015),
    'Close': prices,
    'Volume': np.random.randint(50_000_000, 150_000_000, n_days),
}, index=dates)

# Save to cache
cache_key = f"{symbol}_{start_date}_{end_date}"
cache_file = cache_dir / f"{cache_key}.pkl"

with open(cache_file, 'wb') as f:
    pickle.dump(mock_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[OK] Created mock cache file: {cache_file}")
print(f"   - Rows: {len(mock_data)}")
print(f"   - Date range: {mock_data.index[0]} to {mock_data.index[-1]}")
print(f"   - File size: {cache_file.stat().st_size / 1024:.1f} KB")
print(f"\nFirst 5 rows:")
print(mock_data.head())
