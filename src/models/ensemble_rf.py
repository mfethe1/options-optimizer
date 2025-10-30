"""
Random Forest ensemble for option ranking. Placeholder fit/score API.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class OptionRankerRF:
    n_estimators: int = 200
    random_state: int = 42
    model: Optional[RandomForestRegressor] = None

    def __post_init__(self):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if y is None:
            # unsupervised ranking fallback: predict own features' target like edge proxy
            y = X.mean(axis=1)
        self.model.fit(X, y)
        return self

    def score_options(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


__all__ = ["OptionRankerRF"]

