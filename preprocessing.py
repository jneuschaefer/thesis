###cleaner thesis version

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MissingnessReport:
    column_missing_rate: pd.Series
    row_missing_rate: pd.Series


@dataclass
class PreprocessedMatrix:
    X: pd.DataFrame
    kept_rows: pd.Index
    kept_columns: pd.Index


def summarize_missingness(X: pd.DataFrame) -> MissingnessReport:
    return MissingnessReport(
        column_missing_rate=X.isna().mean(axis=0).sort_values(ascending=False),
        row_missing_rate=X.isna().mean(axis=1),
    )


def filter_missingness(
    X: pd.DataFrame,
    *,
    max_column_missing_rate: float = 0.5,
    max_row_missing_rate: float = 0.5,
) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    kept_columns = X.columns[X.isna().mean(axis=0) <= max_column_missing_rate]
    X_filtered = X.loc[:, kept_columns]

    kept_rows = X_filtered.index[X_filtered.isna().mean(axis=1) <= max_row_missing_rate]
    return X_filtered.loc[kept_rows].copy(), kept_rows, kept_columns


class RobustIQRScaler:
    """Median/IQR scaling with either median imputation or a -1 missing code."""

    def __init__(self, imputation: str = "median", eps: float = 1e-12):
        self.imputation = imputation
        self.eps = eps

    def fit(self, X: pd.DataFrame) -> "RobustIQRScaler":
        X_num = X.apply(pd.to_numeric, errors="coerce")
        self.columns_ = list(X_num.columns)
        self.median_ = X_num.median(axis=0, skipna=True).fillna(0.0)

        q25 = X_num.quantile(0.25, axis=0)
        q75 = X_num.quantile(0.75, axis=0)
        scale = (q75 - q25).replace(0, np.nan)
        self.scale_ = scale.where(scale.abs() > self.eps, np.nan).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.loc[:, self.columns_].apply(pd.to_numeric, errors="coerce")
        missing = X_num.isna()
        X_scaled = (X_num - self.median_) / self.scale_
        X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan)

        if self.imputation == "minus_one":
            X_scaled = X_scaled.fillna(0.0)
            X_scaled[missing] = -1.0
            return X_scaled

        return X_scaled.fillna(0.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


MedianIQRScaler = RobustIQRScaler


def preprocess_feature_matrix(
    X: pd.DataFrame,
    *,
    max_column_missing_rate: float = 0.5,
    max_row_missing_rate: float = 0.5,
    imputation: str = "median",
) -> PreprocessedMatrix:
    X_filtered, kept_rows, kept_columns = filter_missingness(
        X,
        max_column_missing_rate=max_column_missing_rate,
        max_row_missing_rate=max_row_missing_rate,
    )

    scaler = RobustIQRScaler(imputation=imputation)
    return PreprocessedMatrix(
        X=scaler.fit_transform(X_filtered),
        kept_rows=kept_rows,
        kept_columns=kept_columns,
    )
