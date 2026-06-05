"""PCA reconstruction-error detector implemented with NumPy SVD."""

import numpy as np
import pandas as pd

from .base import OutlierDetector


class PCAReconstructionDetector(OutlierDetector):
    """
    PCA reconstruction-error outlier detector.

   xpects already imputed/scaled data.
    """

    method_name = "PCA_reconstruction"

    def __init__(self, n_components: int | None = None, variance_explained: float | None = 0.8):
        if n_components is None and variance_explained is None:
            raise ValueError("Either n_components or variance_explained must be provided.")
        if variance_explained is not None and not 0 < variance_explained <= 1:
            raise ValueError("variance_explained must be in (0, 1].")
        self.n_components = n_components
        self.variance_explained = variance_explained

    def fit(self, X: pd.DataFrame) -> "PCAReconstructionDetector":
        self.columns_ = list(X.columns)
        X_arr = X.astype(float).to_numpy()
        self.mean_ = X_arr.mean(axis=0, keepdims=True)
        X_centered = X_arr - self.mean_

        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        explained = s ** 2

        if self.n_components is None:
            cumulative = np.cumsum(explained) / explained.sum() if explained.sum() > 0 else np.ones_like(explained)
            k = int(np.searchsorted(cumulative, self.variance_explained) + 1)
        else:
            k = int(self.n_components)

        k = max(1, min(k, Vt.shape[0]))
        self.components_ = Vt[:k, :]
        self.singular_values_ = s[:k]
        self.n_components_ = k
        return self

    def reconstruct(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        X_arr = X.loc[:, self.columns_].astype(float).to_numpy()
        X_centered = X_arr - self.mean_
        scores = X_centered @ self.components_.T
        reconstruction = scores @ self.components_ + self.mean_
        return reconstruction

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        reconstruction = self.reconstruct(X)
        X_arr = X.loc[:, self.columns_].astype(float).to_numpy()
        residual = X_arr - reconstruction
        return np.sqrt((residual ** 2).sum(axis=1))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("not fitted.")
