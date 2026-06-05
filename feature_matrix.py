
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ID_COL = "gebernummer"
DATE_COL = "reference_date_int"
ASSET_COL = "F0101_0380_0010"
PARSED_DATE_COL = "__date"


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    metadata: pd.DataFrame
    feature_columns: list[str]


def parse_reference_date(value) -> pd.Timestamp:
    year, month = divmod(int(value), 100)
    return pd.Period(f"{year}-{month:02d}", freq="M").to_timestamp(how="end").normalize()


def parse_quarter(quarter) -> pd.Timestamp:
    if isinstance(quarter, (int, np.integer)):
        return parse_reference_date(quarter)
    if isinstance(quarter, str) and quarter.isdigit():
        return parse_reference_date(quarter)
    return pd.Timestamp(quarter).normalize()


def previous_quarter_end(date) -> pd.Timestamp:
    return (
        pd.Timestamp(date).normalize()
        - pd.DateOffset(months=3)
        + pd.offsets.MonthEnd(0)
    ).normalize()


def add_parsed_date(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    out[PARSED_DATE_COL] = out[date_col].apply(parse_reference_date)
    return out


def model_value_columns(
    value_columns: Iterable[str],
    *,
    id_col: str = ID_COL,
    date_col: str = DATE_COL,
    asset_col: str = ASSET_COL,
) -> list[str]:
    skip = {id_col, date_col, asset_col, PARSED_DATE_COL}
    return [c for c in value_columns if c not in skip]


def compute_total_asset_ratios(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    *,
    asset_col: str = ASSET_COL,
    id_col: str = ID_COL,
    date_col: str = DATE_COL,
) -> FeatureMatrix:
    """Compute ratio features R_{i,t,j} = X_{i,t,j} / A_{i,t}."""
    cols = model_value_columns(
        value_columns,
        id_col=id_col,
        date_col=date_col,
        asset_col=asset_col,
    )

    work = add_parsed_date(df, date_col)
    assets = pd.to_numeric(work[asset_col], errors="coerce")
    X = work[cols].apply(pd.to_numeric, errors="coerce").div(assets, axis=0)
    X.loc[assets.isna() | (assets == 0), :] = np.nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X.columns = [f"ratio__{c}" for c in cols]

    metadata = work[[id_col, date_col, PARSED_DATE_COL, asset_col]].copy()

    return FeatureMatrix(
        X=X.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        feature_columns=list(X.columns),
    )


def build_peer_ratio_matrix(
    df: pd.DataFrame,
    *,
    quarter,
    value_columns: Iterable[str],
    asset_col: str = ASSET_COL,
    id_col: str = ID_COL,
    date_col: str = DATE_COL,
    peer_group_col: str | None = "peer_group",
    peer_group_value=None,
) -> FeatureMatrix:
    """Build the one-quarter peer matrix R_{i,t}."""
    work = add_parsed_date(df, date_col)
    mask = work[PARSED_DATE_COL].eq(parse_quarter(quarter))

    if peer_group_col is not None and peer_group_value is not None:
        mask &= work[peer_group_col].eq(peer_group_value)

    selected = work.loc[mask].copy()
    out = compute_total_asset_ratios(
        selected,
        value_columns,
        asset_col=asset_col,
        id_col=id_col,
        date_col=date_col,
    )

    if peer_group_col is not None and peer_group_col in selected.columns:
        out.metadata[peer_group_col] = selected[peer_group_col].reset_index(drop=True)

    return out


def build_quarterly_delta_matrix(
    df: pd.DataFrame,
    *,
    quarter,
    value_columns: Iterable[str],
    asset_col: str = ASSET_COL,
    id_col: str = ID_COL,
    date_col: str = DATE_COL,
    peer_group_col: str | None = "peer_group",
    peer_group_value=None,
    require_previous_quarter: bool = True,
) -> FeatureMatrix:
    """Build the temporal matrix Delta R_{i,t} = R_{i,t} - R_{i,t-1}."""
    work = add_parsed_date(df, date_col)
    quarter_date = parse_quarter(quarter)

    if peer_group_col is not None and peer_group_value is not None:
        work = work.loc[work[peer_group_col].eq(peer_group_value)].copy()

    ratios = compute_total_asset_ratios(
        work,
        value_columns,
        asset_col=asset_col,
        id_col=id_col,
        date_col=date_col,
    )

    panel = pd.concat([ratios.metadata, ratios.X], axis=1)
    if peer_group_col is not None and peer_group_col in work.columns:
        panel[peer_group_col] = work[peer_group_col].reset_index(drop=True)

    panel = panel.sort_values([id_col, PARSED_DATE_COL]).reset_index(drop=True)
    ratio_cols = ratios.feature_columns

    previous_ratios = panel.groupby(id_col)[ratio_cols].shift(1)
    previous_dates = panel.groupby(id_col)[PARSED_DATE_COL].shift(1)

    X = panel[ratio_cols] - previous_ratios
    X.columns = [c.replace("ratio__", "delta_ratio__", 1) for c in ratio_cols]

    target = panel[PARSED_DATE_COL].eq(quarter_date)
    if require_previous_quarter:
        target &= previous_dates.eq(previous_quarter_end(quarter_date))

    metadata_cols = [id_col, date_col, PARSED_DATE_COL, asset_col]
    if peer_group_col is not None and peer_group_col in panel.columns:
        metadata_cols.append(peer_group_col)

    metadata = panel.loc[target, metadata_cols].reset_index(drop=True)
    metadata["previous_date"] = previous_dates.loc[target].reset_index(drop=True)

    return FeatureMatrix(
        X=X.loc[target].reset_index(drop=True),
        metadata=metadata,
        feature_columns=list(X.columns),
    )
