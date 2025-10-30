# Design Overview

Directory structure: src/, tests/, docs/, data/, models/.

- src/data/providers: market data adapters (start with yfinance)
- src/pricing: pricing utils (BS, IV inversion, SABR/SVI later), RND extraction
- src/sentiment: FinBERT inference + aggregation (later)
- src/features: feature engineering for ranking
- src/models: ensembles (RF), timeseries/generative (later)
- src/cli: CLI entry points

Initial API targets:
- Probability engine: RND from IV surface, PoP/ITM, expected payoff
- Ranking: RandomForest score with market/sentiment features

Testing: pytest-based unit tests (IV roundtrip, RND sanity), rolling CV later.

