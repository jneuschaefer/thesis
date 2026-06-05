"""Common detector interface for several methods"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class OutlierDetector(ABC):
    """
    Base class for outlier scoring methods.

    """

    method_name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "OutlierDetector":
        ...

    @abstractmethod
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def fit_score(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.score_samples(X)


def make_ranking(
    scores: np.ndarray,
    *,
    metadata: pd.DataFrame | None = None,
    method: str,
    feature_space: str,
) -> pd.DataFrame:
    """create standardized ranking table."""
    result = pd.DataFrame({"score": scores})
    if metadata is not None:
        result = pd.concat([metadata.reset_index(drop=True), result], axis=1)

    result["method"] = method
    result["feature_space"] = feature_space
    result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)
    result = result.sort_values("rank").reset_index(drop=True)
    return result
